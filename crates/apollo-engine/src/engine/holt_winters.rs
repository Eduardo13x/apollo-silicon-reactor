//! Holt-Winters seasonal forecasting for memory pressure prediction.
//!
//! Decomposes the pressure signal into three components:
//!   - Level (L): current baseline pressure
//!   - Trend (T): is pressure drifting up or down over hours
//!   - Seasonal (S): hourly pattern (e.g., builds at 10am, light use at 6pm)
//!
//! Period: 24 hours (one seasonal cycle per day).
//! Requires ~2 days of data to produce useful forecasts; degrades gracefully
//! before that by returning the current level as the forecast.
//!
//! Reference: Holt, C.C. (1957) & Winters, P.R. (1960).
//! "Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages."

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Seasonal period: 24 hours.
const PERIOD: usize = 24;

/// Smoothing parameters (tuned conservatively for pressure 0.0-1.0):
///   α (level):    0.3 — moderately responsive to new data
///   β (trend):    0.05 — slow trend tracking (pressure trends shift slowly)
///   γ (seasonal): 0.15 — seasonal patterns update gradually
const ALPHA: f64 = 0.30;
const BETA: f64 = 0.05;
const GAMMA: f64 = 0.15;

/// Persisted state of the Holt-Winters model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoltWintersState {
    /// Current level (smoothed baseline).
    pub level: f64,
    /// Current trend (change per hour).
    pub trend: f64,
    /// Seasonal factors for each hour (0-23). Multiplicative model.
    pub seasonal: [f64; PERIOD],
    /// Total observations (hours observed).
    pub observations: u64,
    /// Last hour that was observed (0-23).
    pub last_hour: u8,
}

impl Default for HoltWintersState {
    fn default() -> Self {
        Self {
            level: 0.5, // Neutral starting point
            trend: 0.0,
            seasonal: [1.0; PERIOD], // Neutral seasonality
            observations: 0,
            last_hour: 0,
        }
    }
}

pub struct HoltWinters {
    state: HoltWintersState,
}

impl HoltWinters {
    pub fn new() -> Self {
        Self {
            state: HoltWintersState::default(),
        }
    }

    /// Load persisted state.
    pub fn from_persisted(state: HoltWintersState) -> Self {
        Self { state }
    }

    /// Observe the average pressure for the current hour.
    ///
    /// Call once per hour with the mean pressure during that hour.
    /// The model updates level, trend, and the seasonal factor for this hour.
    pub fn observe(&mut self, hour: u8, pressure: f64) {
        let h = (hour as usize) % PERIOD;
        let pressure = pressure.clamp(0.01, 1.0); // Avoid zero in multiplicative model

        if self.state.observations == 0 {
            // Cold start: initialize level to first observation.
            self.state.level = pressure;
            self.state.trend = 0.0;
            self.state.seasonal[h] = 1.0;
            self.state.observations = 1;
            self.state.last_hour = hour;
            return;
        }

        let s_prev = self.state.seasonal[h].max(0.01); // Guard against zero

        // Holt-Winters multiplicative update equations:
        // L(t) = α × y(t)/S(t-L) + (1-α) × (L(t-1) + T(t-1))
        let new_level =
            ALPHA * (pressure / s_prev) + (1.0 - ALPHA) * (self.state.level + self.state.trend);

        // T(t) = β × (L(t) - L(t-1)) + (1-β) × T(t-1)
        let new_trend = BETA * (new_level - self.state.level) + (1.0 - BETA) * self.state.trend;

        // S(t) = γ × y(t)/L(t) + (1-γ) × S(t-L)
        let new_seasonal = if new_level.abs() > 0.001 {
            GAMMA * (pressure / new_level) + (1.0 - GAMMA) * s_prev
        } else {
            s_prev
        };

        self.state.level = new_level.clamp(0.0, 1.0);
        self.state.trend = new_trend.clamp(-0.1, 0.1); // Max 10pp/hour drift
        self.state.seasonal[h] = new_seasonal.clamp(0.5, 2.0); // ±100% seasonal range
        self.state.observations += 1;
        self.state.last_hour = hour;
    }

    /// Forecast pressure `hours_ahead` hours from now.
    ///
    /// Returns `(forecast, confidence)`:
    ///   - forecast: predicted pressure [0.0, 1.0]
    ///   - confidence: [0.0, 1.0] based on observations (needs ~48h for full confidence)
    pub fn forecast(&self, current_hour: u8, hours_ahead: u8) -> (f64, f64) {
        let target_hour = ((current_hour as usize + hours_ahead as usize) % PERIOD);
        let seasonal = self.state.seasonal[target_hour];

        let forecast = (self.state.level + self.state.trend * hours_ahead as f64) * seasonal;

        // Confidence: ramp up from 0 to 1 over 48 observations (2 full days).
        let confidence = (self.state.observations as f64 / 48.0).min(1.0);

        (forecast.clamp(0.0, 1.0), confidence)
    }

    /// Get the seasonal factor for a given hour (1.0 = average).
    /// >1.0 = pressure typically higher than average at this hour.
    /// > <1.0 = pressure typically lower.
    pub fn seasonal_factor(&self, hour: u8) -> f64 {
        self.state.seasonal[(hour as usize) % PERIOD]
    }

    /// Current level (smoothed baseline pressure).
    pub fn level(&self) -> f64 {
        self.state.level
    }

    /// Current trend (pressure change per hour, + = rising).
    pub fn trend(&self) -> f64 {
        self.state.trend
    }

    /// Total hours observed.
    pub fn observations(&self) -> u64 {
        self.state.observations
    }

    /// Get state for persistence.
    pub fn state(&self) -> &HoltWintersState {
        &self.state
    }

    /// Persist state to disk.
    pub fn persist(&self, path: &Path) {
        if let Ok(json) = serde_json::to_string(&self.state) {
            let _ = std::fs::write(path, json);
        }
    }

    /// Load state from disk.
    pub fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        let state: HoltWintersState = serde_json::from_str(&data).ok()?;
        Some(Self { state })
    }
}

impl Default for HoltWinters {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_initializes_level() {
        let mut hw = HoltWinters::new();
        hw.observe(10, 0.65);
        assert!((hw.level() - 0.65).abs() < 0.01);
        assert_eq!(hw.observations(), 1);
    }

    #[test]
    fn stable_signal_converges() {
        let mut hw = HoltWinters::new();
        // Feed stable 0.50 pressure for 48 hours (2 full days).
        for hour in 0..48u8 {
            hw.observe(hour % 24, 0.50);
        }
        assert!(
            (hw.level() - 0.50).abs() < 0.05,
            "level {} should be ~0.50",
            hw.level()
        );
        assert!(hw.trend().abs() < 0.01, "trend {} should be ~0", hw.trend());
    }

    #[test]
    fn learns_daily_pattern() {
        let mut hw = HoltWinters::new();
        // Simulate: mornings (8-12) high pressure, afternoons low.
        for _day in 0..5 {
            for hour in 0..24u8 {
                let pressure = if (8..=12).contains(&hour) {
                    0.80 // Build time
                } else {
                    0.40 // Quiet time
                };
                hw.observe(hour, pressure);
            }
        }

        // Morning should have higher seasonal factor than afternoon.
        let morning = hw.seasonal_factor(10);
        let evening = hw.seasonal_factor(20);
        assert!(
            morning > evening,
            "morning factor {} should be > evening {}",
            morning,
            evening,
        );
    }

    #[test]
    fn forecast_returns_reasonable_values() {
        let mut hw = HoltWinters::new();
        for hour in 0..24u8 {
            hw.observe(hour, 0.60);
        }
        let (forecast, confidence) = hw.forecast(10, 2);
        assert!(
            forecast > 0.0 && forecast < 1.0,
            "forecast {} should be in (0,1)",
            forecast
        );
        assert!(confidence > 0.0, "confidence should be > 0 after 24 obs");
    }

    #[test]
    fn confidence_grows_with_observations() {
        let mut hw = HoltWinters::new();
        hw.observe(0, 0.50);
        let (_, c1) = hw.forecast(0, 1);

        for h in 1..48u8 {
            hw.observe(h % 24, 0.50);
        }
        let (_, c48) = hw.forecast(0, 1);

        assert!(
            c48 > c1,
            "confidence after 48h ({}) > after 1h ({})",
            c48,
            c1
        );
        assert!((c48 - 1.0).abs() < 0.01, "48h should reach full confidence");
    }

    /// Trend must be bounded to ±0.1 even with extreme inputs.
    /// [Holt 1957] damped trend prevents extrapolation explosion.
    #[test]
    fn trend_bounded_under_extreme_jumps() {
        let mut hw = HoltWinters::new();
        hw.observe(0, 0.10);
        hw.observe(1, 0.99);
        hw.observe(2, 0.10);
        hw.observe(3, 0.99);
        assert!(
            hw.trend().abs() <= 0.10,
            "trend {} must be bounded to ±0.10",
            hw.trend()
        );
    }

    /// Seasonal factors must stay in [0.5, 2.0] to prevent multiplicative explosion.
    #[test]
    fn seasonal_factors_bounded() {
        let mut hw = HoltWinters::new();
        // Extreme: one hour is always 0.01, another always 0.99.
        for _ in 0..10 {
            hw.observe(8, 0.01);
            hw.observe(20, 0.99);
        }
        for h in 0..24u8 {
            let s = hw.seasonal_factor(h);
            assert!(
                s >= 0.5 && s <= 2.0,
                "seasonal[{}]={} must be in [0.5, 2.0]",
                h,
                s
            );
        }
    }

    /// Forecast must be clamped to [0, 1] regardless of model state.
    #[test]
    fn forecast_clamped_to_valid_range() {
        let mut hw = HoltWinters::new();
        // Feed high-trending data.
        for h in 0..24u8 {
            hw.observe(h, 0.95);
        }
        // Forecast far ahead — trend + seasonal could push above 1.0.
        for hours_ahead in 0..24u8 {
            let (f, _) = hw.forecast(12, hours_ahead);
            assert!(
                f >= 0.0 && f <= 1.0,
                "forecast({})={} out of [0,1]",
                hours_ahead,
                f
            );
        }
    }

    /// Serde roundtrip must preserve model state for persistence.
    #[test]
    fn serde_roundtrip() {
        let mut hw = HoltWinters::new();
        for h in 0..48u8 {
            hw.observe(h % 24, 0.50 + (h as f64 % 5.0) * 0.02);
        }
        let json = serde_json::to_string(hw.state()).unwrap();
        let restored: HoltWintersState = serde_json::from_str(&json).unwrap();
        let hw2 = HoltWinters::from_persisted(restored);

        assert!((hw.level() - hw2.level()).abs() < 1e-10);
        assert!((hw.trend() - hw2.trend()).abs() < 1e-10);
        assert_eq!(hw.observations(), hw2.observations());
    }
}
