//! Predictive thermal management
//!
//! Predicts thermal throttling and applies proactive cooling strategies.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct ThermalState {
    pub cpu_temp: f32,           // Celsius
    pub gpu_temp: f32,
    pub mem_temp: f32,
    pub current_throttle_level: u8, // 0-100
    pub predicted_throttle_level: u8,
    pub thermal_trend: ThermalTrend,
    pub seconds_to_throttle: i32, // -1 if none
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalTrend {
    Cooling,      // Temperature decreasing
    Stable,       // Temperature stable
    Warming,      // Temperature increasing slowly
    Critical,     // Temperature increasing rapidly
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
}

impl ThermalManager {
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
            max_history: 60, // 60 samples (1 minute at 1Hz)
            throttle_threshold: 85.0, // Start throttling at 85°C
            shutdown_threshold: 100.0, // Critical at 100°C
        }
    }

    /// Record a thermal sample and update state
    pub fn update(
        &mut self,
        cpu_temp: f32,
        gpu_temp: f32,
        mem_temp: f32,
        throttle_level: u8,
    ) -> ThermalState {
        let snapshot = ThermalSnapshot {
            timestamp: std::time::Instant::now(),
            cpu_temp,
            gpu_temp,
            mem_temp,
            throttle_level,
        };

        self.history.push_back(snapshot.clone());
        if self.history.len() > self.max_history {
            let _ = self.history.pop_front();
        }

        let trend = self.calculate_trend();
        let predicted = self.predict_throttle_level();
        let seconds_to_throttle = self.time_to_throttle(cpu_temp);

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
            (d, r) if d > 0.5 && r > 0.7 => ThermalTrend::Critical,
            (d, _) if d > 0.1 => ThermalTrend::Warming,
            _ => ThermalTrend::Stable,
        }
    }

    fn predict_throttle_level(&self) -> u8 {
        if self.history.is_empty() {
            return 0;
        }

        let last = &self.history[self.history.len() - 1];
        let avg_temp = (last.cpu_temp + last.gpu_temp + last.mem_temp) / 3.0;

        let base_throttle = if avg_temp > self.shutdown_threshold {
            100
        } else if avg_temp > self.throttle_threshold {
            ((avg_temp - self.throttle_threshold) / (self.shutdown_threshold - self.throttle_threshold)
                * 100.0) as u8
        } else {
            0
        };

        // Adjust based on trend and current throttle level
        let trend_adjustment = match self.calculate_trend() {
            ThermalTrend::Cooling => -5,
            ThermalTrend::Stable => 0,
            ThermalTrend::Warming => 10,
            ThermalTrend::Critical => 20,
        };

        // Blend with actual current throttle level for smoother prediction
        let predicted = ((base_throttle as i16 + trend_adjustment).max(0) as u8).min(100);
        let smoothed = ((predicted as i16 + last.throttle_level as i16) / 2) as u8;
        smoothed
    }

    fn time_to_throttle(&self, current_temp: f32) -> i32 {
        if self.history.len() < 2 || current_temp >= self.throttle_threshold {
            return if current_temp >= self.throttle_threshold { 0 } else { -1 };
        }

        let recent: Vec<_> = self.history.iter().rev().take(5).collect();
        if recent.len() < 2 {
            return -1;
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
            let temp_newer = (newer.cpu_temp + newer.gpu_temp) / 2.0;
            let temp_older = (older.cpu_temp + older.gpu_temp) / 2.0;
            let delta = temp_newer - temp_older;
            temp_rise_per_sec += delta / time_delta;
        }

        temp_rise_per_sec /= (recent.len() - 1) as f32;

        if temp_rise_per_sec <= 0.0 {
            return -1; // Cooling down
        }

        let temp_to_throttle = self.throttle_threshold - current_temp;
        let seconds = (temp_to_throttle / temp_rise_per_sec) as i32;

        seconds.max(0)
    }

    /// Get recommended actions based on thermal state
    pub fn get_recommendations(&self) -> Vec<String> {
        if self.history.is_empty() {
            return vec![];
        }

        let state = {
            let last = &self.history[self.history.len() - 1];
            (last.cpu_temp + last.gpu_temp) / 2.0
        };

        let mut recommendations = Vec::new();

        match self.calculate_trend() {
            ThermalTrend::Critical => {
                recommendations.push("🔴 CRITICAL: Applying emergency thermal throttling".to_string());
            }
            ThermalTrend::Warming => {
                if state > 75.0 {
                    recommendations
                        .push("🟡 Temperature rising: Consider reducing background load".to_string());
                }
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
