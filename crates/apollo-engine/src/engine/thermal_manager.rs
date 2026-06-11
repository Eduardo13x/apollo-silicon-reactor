//! Predictive thermal management
//!
//! Predicts thermal throttling and applies proactive cooling strategies.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct ThermalState {
    pub cpu_temp: f32, // Celsius
    pub gpu_temp: f32,
    pub mem_temp: f32,
    pub current_throttle_level: u8, // 0-100
    pub predicted_throttle_level: u8,
    pub thermal_trend: ThermalTrend,
    pub seconds_to_throttle: Option<i32>, // None = no forecast
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalTrend {
    Cooling,  // Temperature decreasing
    Stable,   // Temperature stable
    Warming,  // Temperature increasing slowly
    Critical, // Temperature increasing rapidly
}

pub struct ThermalManager {
    history: VecDeque<ThermalSnapshot>,
    max_history: usize,
    throttle_threshold: f32,
    shutdown_threshold: f32,
}

#[derive(Debug, Clone)]
struct ThermalSnapshot {
    timestamp: std::time::Instant,
    cpu_temp: f32,
    gpu_temp: f32,
    mem_temp: f32,
    throttle_level: u8,
    /// Schedule jitter en µs medido con cntvct_el0.
    /// Non-zero cuando hw_predictor detecta presión antes que los sensores.
    #[allow(dead_code)]
    jitter_us: u64,
}

impl ThermalManager {
    pub fn new() -> Self {
        // Thermal thresholds aligned with thermal_interrupt.rs SentinelConfig:
        // Moderate=90°C, Emergency=95°C, SuperEmergency=100°C
        Self {
            history: VecDeque::new(),
            max_history: 60,           // 60 samples (1 minute at 1Hz)
            throttle_threshold: 90.0,  // Start throttling at 90°C (matches sentinel Moderate phase)
            shutdown_threshold: 100.0, // Critical at 100°C (matches sentinel SuperEmergency phase)
        }
    }

    /// Record a thermal sample and update state.
    ///
    /// `jitter_us`: schedule jitter medido con cntvct_el0 (de hw_predictor).
    /// 0 si no disponible. >200 indica presión térmica antes que los sensores.
    pub fn update(
        &mut self,
        cpu_temp: f32,
        gpu_temp: f32,
        mem_temp: f32,
        throttle_level: u8,
        jitter_us: u64,
    ) -> ThermalState {
        // Si el jitter de assembly indica presión, adelantamos la temperatura
        // efectiva para que el predictor actúe antes de que el sensor lo confirme.
        let jitter_boost = match jitter_us {
            0..=200 => 0.0_f32,
            201..=600 => 3.0, // +3°C efectivos — Warning
            _ => 7.0,         // +7°C efectivos — Critical
        };
        let cpu_temp_eff = cpu_temp + jitter_boost;

        let snapshot = ThermalSnapshot {
            timestamp: std::time::Instant::now(),
            cpu_temp: cpu_temp_eff,
            gpu_temp,
            mem_temp,
            throttle_level,
            jitter_us,
        };

        self.history.push_back(snapshot.clone());
        if self.history.len() > self.max_history {
            let _ = self.history.pop_front();
        }

        let trend = self.calculate_trend();
        let predicted = self.predict_throttle_level(trend);
        let avg_temp = (cpu_temp + gpu_temp + mem_temp) / 3.0;
        let seconds_to_throttle = self.time_to_throttle(avg_temp);

        ThermalState {
            cpu_temp,
            gpu_temp,
            mem_temp,
            current_throttle_level: throttle_level,
            predicted_throttle_level: predicted,
            thermal_trend: trend,
            seconds_to_throttle,
        }
    }

    pub fn calculate_trend(&self) -> ThermalTrend {
        if self.history.len() < 3 {
            return ThermalTrend::Stable;
        }

        let recent = self.history.iter().rev().take(10).collect::<Vec<_>>();
        let mut deltas = Vec::new();

        for i in 1..recent.len() {
            let older = recent[i];
            let newer = recent[i - 1];
            // Use all three temp sensors for accurate trend (CPU, GPU, Memory)
            let temp_older = (older.cpu_temp + older.gpu_temp + older.mem_temp) / 3.0;
            let temp_newer = (newer.cpu_temp + newer.gpu_temp + newer.mem_temp) / 3.0;
            deltas.push(temp_newer - temp_older);
        }

        if deltas.is_empty() {
            return ThermalTrend::Stable;
        }

        let avg_delta: f32 = deltas.iter().sum::<f32>() / deltas.len() as f32;
        let rising_count = deltas.iter().filter(|&&d| d > 0.1).count();
        let rise_ratio = rising_count as f32 / deltas.len() as f32;

        match (avg_delta, rise_ratio) {
            (d, _) if d < -0.3 => ThermalTrend::Cooling,
            (d, r) if d > 0.3 && r > 0.6 => ThermalTrend::Critical,
            (d, _) if d > 0.1 => ThermalTrend::Warming,
            _ => ThermalTrend::Stable,
        }
    }

    fn predict_throttle_level(&self, trend: ThermalTrend) -> u8 {
        if self.history.is_empty() {
            return 0;
        }

        let last = &self.history[self.history.len() - 1];
        let avg_temp = (last.cpu_temp + last.gpu_temp + last.mem_temp) / 3.0;

        let base_throttle = if avg_temp > self.shutdown_threshold {
            100
        } else if avg_temp > self.throttle_threshold {
            ((avg_temp - self.throttle_threshold)
                / (self.shutdown_threshold - self.throttle_threshold)
                * 100.0) as u8
        } else {
            0
        };

        let trend_adjustment = match trend {
            ThermalTrend::Cooling => -5,
            ThermalTrend::Stable => 0,
            ThermalTrend::Warming => 10,
            ThermalTrend::Critical => 20,
        };

        // Blend with actual current throttle level for smoother prediction
        let predicted = ((base_throttle as i16 + trend_adjustment).max(0) as u8).min(100);
        ((predicted as u16 + last.throttle_level as u16) / 2) as u8
    }

    fn time_to_throttle(&self, current_temp: f32) -> Option<i32> {
        if current_temp >= self.throttle_threshold {
            return Some(0); // already at throttle threshold
        }
        if self.history.len() < 2 {
            return None;
        }

        let recent: Vec<_> = self.history.iter().rev().take(5).collect();
        if recent.len() < 2 {
            return None;
        }

        let mut temp_rise_per_sec = 0.0;
        for i in 1..recent.len() {
            let older = recent[i];
            let newer = recent[i - 1];
            let time_delta = newer
                .timestamp
                .duration_since(older.timestamp)
                .as_secs_f32()
                .max(0.1);
            let temp_newer = (newer.cpu_temp + newer.gpu_temp + newer.mem_temp) / 3.0;
            let temp_older = (older.cpu_temp + older.gpu_temp + older.mem_temp) / 3.0;
            let delta = temp_newer - temp_older;
            temp_rise_per_sec += delta / time_delta;
        }

        temp_rise_per_sec /= (recent.len() - 1) as f32;

        if temp_rise_per_sec <= 0.0 {
            return None; // cooling down — no throttle forecast
        }

        let temp_to_throttle = self.throttle_threshold - current_temp;
        let raw = temp_to_throttle / temp_rise_per_sec;
        Some(raw.clamp(0.0, 86400.0) as i32) // max 24h
    }

    /// Get recommended actions based on thermal state
    pub fn get_recommendations(&self) -> Vec<String> {
        if self.history.is_empty() {
            return vec![];
        }

        let state = {
            let last = &self.history[self.history.len() - 1];
            (last.cpu_temp + last.gpu_temp + last.mem_temp) / 3.0
        };

        let mut recommendations = Vec::new();

        match self.calculate_trend() {
            ThermalTrend::Critical => {
                recommendations
                    .push("🔴 CRITICAL: Applying emergency thermal throttling".to_string());
            }
            ThermalTrend::Warming
                if state > 75.0 => {
                    recommendations.push(
                        "🟡 Temperature rising: Consider reducing background load".to_string(),
                    );
                }
            _ => {}
        }

        if state > self.throttle_threshold {
            recommendations.push(format!(
                "⚠️  Temperature at {:.1}°C (throttling active)",
                state
            ));
        }

        recommendations
    }
}

impl Default for ThermalManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Threshold constants (must stay aligned w/ thermal_interrupt sentinel) ──
    #[test]
    fn thresholds_match_sentinel_phases() {
        let m = ThermalManager::new();
        assert_eq!(m.throttle_threshold, 90.0, "Moderate sentinel phase");
        assert_eq!(m.shutdown_threshold, 100.0, "SuperEmergency sentinel phase");
        assert_eq!(m.max_history, 60, "60 samples = 1 minute at 1Hz");
    }

    // ─── jitter_us → effective-temp boost mapping (hw_predictor pre-sensor) ──────
    // This is the assembly-jitter early-warning: it shifts the effective CPU temp
    // forward so the predictor acts before the sensor confirms. A wrong boundary
    // or value = either no early warning or a false thermal panic.
    #[test]
    fn jitter_boost_band_mapping() {
        let probe = |jitter: u64| -> f32 {
            let mut m = ThermalManager::new();
            // cpu_temp_eff is recorded as the first snapshot's cpu_temp.
            m.update(50.0, 40.0, 40.0, 0, jitter);
            m.history.back().unwrap().cpu_temp - 50.0
        };
        assert_eq!(probe(0), 0.0, "no jitter → no boost");
        assert_eq!(probe(200), 0.0, "upper edge of calm band → no boost");
        assert_eq!(probe(201), 3.0, "warning band lower edge → +3C");
        assert_eq!(probe(600), 3.0, "warning band upper edge → +3C");
        assert_eq!(probe(601), 7.0, "critical band → +7C");
        assert_eq!(probe(100_000), 7.0, "saturates at +7C");
    }

    // ─── Trend classification ───────────────────────────────────────────────────
    #[test]
    fn trend_stable_when_insufficient_history() {
        let mut m = ThermalManager::new();
        m.update(50.0, 50.0, 50.0, 0, 0);
        m.update(50.0, 50.0, 50.0, 0, 0);
        // <3 samples → forced Stable.
        assert_eq!(m.calculate_trend(), ThermalTrend::Stable);
    }

    #[test]
    fn trend_cooling_detected() {
        let mut m = ThermalManager::new();
        for t in [80.0_f32, 75.0, 70.0, 65.0, 60.0] {
            m.update(t, t, t, 0, 0);
        }
        assert_eq!(m.calculate_trend(), ThermalTrend::Cooling);
    }

    #[test]
    fn trend_critical_on_rapid_sustained_rise() {
        let mut m = ThermalManager::new();
        for t in [50.0_f32, 55.0, 60.0, 65.0, 70.0, 75.0] {
            m.update(t, t, t, 0, 0);
        }
        // avg_delta = 5.0 (>0.3) and every step rises (ratio 1.0 > 0.6).
        assert_eq!(m.calculate_trend(), ThermalTrend::Critical);
    }

    // ─── Throttle prediction clamping (u8 0..=100, no overflow/underflow) ────────
    #[test]
    fn predict_throttle_clamps_to_100() {
        let mut m = ThermalManager::new();
        // Well above shutdown threshold + rising → base 100 + trend → must clamp.
        for t in [95.0_f32, 100.0, 105.0, 110.0, 115.0] {
            m.update(t, t, t, 100, 0);
        }
        let p = m.predict_throttle_level(ThermalTrend::Critical);
        assert!(p <= 100, "predicted throttle must never exceed 100, got {p}");
    }

    #[test]
    fn predict_throttle_no_underflow_when_cooling_cold() {
        let mut m = ThermalManager::new();
        // Cold + cooling: base_throttle 0, trend_adjustment -5 → max(0) guards it.
        m.update(40.0, 40.0, 40.0, 0, 0);
        let p = m.predict_throttle_level(ThermalTrend::Cooling);
        assert_eq!(p, 0, "cold+cooling must floor at 0, not wrap u8");
    }

    #[test]
    fn predict_zero_with_empty_history() {
        let m = ThermalManager::new();
        assert_eq!(m.predict_throttle_level(ThermalTrend::Warming), 0);
    }

    // ─── time_to_throttle forecasting ───────────────────────────────────────────
    #[test]
    fn time_to_throttle_zero_when_already_hot() {
        let m = ThermalManager::new();
        assert_eq!(m.time_to_throttle(90.0), Some(0));
        assert_eq!(m.time_to_throttle(95.0), Some(0));
    }

    #[test]
    fn time_to_throttle_none_when_cooling() {
        let mut m = ThermalManager::new();
        for t in [80.0_f32, 75.0, 70.0] {
            m.update(t, t, t, 0, 0);
        }
        // Falling temps → no throttle forecast.
        assert_eq!(m.time_to_throttle(70.0), None);
    }

    #[test]
    fn state_fields_passthrough_raw_temps() {
        // update() must report the *raw* (un-jitter-boosted) temps in ThermalState
        // even though the snapshot stores the boosted cpu_temp.
        let mut m = ThermalManager::new();
        let st = m.update(50.0, 60.0, 55.0, 30, 700);
        assert_eq!(st.cpu_temp, 50.0, "ThermalState reports raw cpu_temp");
        assert_eq!(st.gpu_temp, 60.0);
        assert_eq!(st.mem_temp, 55.0);
        assert_eq!(st.current_throttle_level, 30);
        // But the snapshot kept the boosted value (+7 for jitter 700).
        assert_eq!(m.history.back().unwrap().cpu_temp, 57.0);
    }
}
