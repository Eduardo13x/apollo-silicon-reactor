//! Analytics and Reporting Module
//!
//! Tracks and reports optimization impact and system metrics over time.

use std::collections::VecDeque;
use std::time::Instant;

use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct OptimizationMetric {
    pub timestamp: Instant,
    pub cpu_usage_before: f32,
    pub cpu_usage_after: f32,
    pub memory_before: u64,
    pub memory_after: u64,
    pub thermal_before: f32,
    pub thermal_after: f32,
    pub improvements_applied: u32,
}

#[derive(Debug, Clone)]
pub struct Analytics {
    pub total_optimizations: u64,
    pub avg_cpu_improvement_percent: f32,
    pub avg_memory_freed_mb: u64,
    pub avg_thermal_reduction_celsius: f32,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub title: String,
    pub generated_at: DateTime<Utc>,
    pub analytics: Analytics,
    pub top_optimizations: Vec<String>,
    pub energy_saved_wh: f32,
    pub co2_avoided_grams: f32,
}

pub struct AnalyticsEngine {
    metrics: VecDeque<OptimizationMetric>,
    max_history: usize,
    session_start: Instant,
}

impl AnalyticsEngine {
    pub fn new() -> Self {
        Self {
            metrics: VecDeque::new(),
            max_history: 1440, // 24 hours at 1 min granularity
            session_start: Instant::now(),
        }
    }

    /// Record an optimization cycle
    #[allow(clippy::too_many_arguments)]
    pub fn record_optimization(
        &mut self,
        cpu_before: f32,
        cpu_after: f32,
        mem_before: u64,
        mem_after: u64,
        thermal_before: f32,
        thermal_after: f32,
        improvements: u32,
    ) {
        let metric = OptimizationMetric {
            timestamp: Instant::now(),
            cpu_usage_before: cpu_before,
            cpu_usage_after: cpu_after,
            memory_before: mem_before,
            memory_after: mem_after,
            thermal_before,
            thermal_after,
            improvements_applied: improvements,
        };

        self.metrics.push_back(metric);
        if self.metrics.len() > self.max_history {
            let _ = self.metrics.pop_front();
        }
    }

    /// Calculate cumulative analytics
    pub fn calculate_analytics(&self) -> Analytics {
        if self.metrics.is_empty() {
            return Analytics {
                total_optimizations: 0,
                avg_cpu_improvement_percent: 0.0,
                avg_memory_freed_mb: 0,
                avg_thermal_reduction_celsius: 0.0,
                uptime_seconds: self.session_start.elapsed().as_secs(),
            };
        }

        let total = self.metrics.len() as u64;
        let mut total_cpu_improve = 0.0f32;
        let mut total_mem_freed = 0u64;
        let mut total_thermal_improve = 0.0f32;

        for metric in &self.metrics {
            total_cpu_improve += (metric.cpu_usage_before - metric.cpu_usage_after).max(0.0);
            total_mem_freed += metric.memory_before.saturating_sub(metric.memory_after);
            total_thermal_improve += (metric.thermal_before - metric.thermal_after).max(0.0);
        }

        Analytics {
            total_optimizations: total,
            avg_cpu_improvement_percent: total_cpu_improve / (total as f32).max(1.0),
            avg_memory_freed_mb: total_mem_freed / total.max(1) / 1024 / 1024,
            avg_thermal_reduction_celsius: total_thermal_improve / (total as f32).max(1.0),
            uptime_seconds: self.session_start.elapsed().as_secs(),
        }
    }

    /// Generate a comprehensive report
    pub fn generate_report(&self) -> Report {
        let analytics = self.calculate_analytics();

        let mut top_optimizations = vec![
            format!(
                "CPU optimized {:.1}% average",
                analytics.avg_cpu_improvement_percent
            ),
            format!(
                "Freed {:.0} MB of memory on average",
                analytics.avg_memory_freed_mb
            ),
            format!(
                "Reduced temperature {:.1}°C on average",
                analytics.avg_thermal_reduction_celsius
            ),
        ];

        if analytics.total_optimizations > 1000 {
            top_optimizations.push("High-frequency optimization successful".to_string());
        }

        // Estimate energy saved (1% CPU reduction ≈ 0.15W on Apple Silicon)
        let energy_saved_wh =
            (analytics.avg_cpu_improvement_percent * 0.15 * analytics.uptime_seconds as f32)
                / 3600.0;

        // CO2 avoided (0.39 g per Wh — US grid average, aligned with EnergyTracker)
        let co2_avoided_grams = energy_saved_wh * 0.39;

        Report {
            title: "Apollo Optimizer Performance Report".to_string(),
            generated_at: Utc::now(),
            analytics,
            top_optimizations,
            energy_saved_wh,
            co2_avoided_grams,
        }
    }

    /// Get trend data for last N cycles
    pub fn get_trend(&self, cycles: usize) -> Vec<(f32, f32, f32)> {
        self.metrics
            .iter()
            .rev()
            .take(cycles)
            .rev()
            .map(|m| {
                (
                    m.cpu_usage_before - m.cpu_usage_after,
                    (m.memory_before.saturating_sub(m.memory_after)) as f32 / 1024.0 / 1024.0,
                    m.thermal_before - m.thermal_after,
                )
            })
            .collect()
    }
}

impl Default for AnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_engine_returns_zero_analytics() {
        let engine = AnalyticsEngine::new();
        let analytics = engine.calculate_analytics();
        assert_eq!(analytics.total_optimizations, 0);
        assert_eq!(analytics.avg_cpu_improvement_percent, 0.0);
        assert_eq!(analytics.avg_memory_freed_mb, 0);
        assert_eq!(analytics.avg_thermal_reduction_celsius, 0.0);
    }

    #[test]
    fn single_optimization_recorded_correctly() {
        let mut engine = AnalyticsEngine::new();
        // 10% CPU improvement, 100 MB memory freed, 2°C thermal reduction
        engine.record_optimization(
            40.0,
            30.0,
            200 * 1024 * 1024,
            100 * 1024 * 1024,
            60.0,
            58.0,
            1,
        );

        let analytics = engine.calculate_analytics();
        assert_eq!(analytics.total_optimizations, 1);
        assert!(
            (analytics.avg_cpu_improvement_percent - 10.0).abs() < 0.01,
            "expected ~10.0, got {}",
            analytics.avg_cpu_improvement_percent
        );
        assert_eq!(analytics.avg_memory_freed_mb, 100);
        assert!(
            (analytics.avg_thermal_reduction_celsius - 2.0).abs() < 0.01,
            "expected ~2.0, got {}",
            analytics.avg_thermal_reduction_celsius
        );
    }

    #[test]
    fn multiple_optimizations_average_correctly() {
        let mut engine = AnalyticsEngine::new();
        // Cycle 1: 20% CPU, 200 MB freed, 4°C
        engine.record_optimization(
            50.0,
            30.0,
            300 * 1024 * 1024,
            100 * 1024 * 1024,
            70.0,
            66.0,
            2,
        );
        // Cycle 2: 0% CPU improvement (after >= before), 0 MB freed, 0°C
        engine.record_optimization(
            30.0,
            30.0,
            100 * 1024 * 1024,
            100 * 1024 * 1024,
            66.0,
            66.0,
            0,
        );

        let analytics = engine.calculate_analytics();
        assert_eq!(analytics.total_optimizations, 2);
        // avg CPU improvement = (20.0 + 0.0) / 2 = 10.0
        assert!(
            (analytics.avg_cpu_improvement_percent - 10.0).abs() < 0.01,
            "expected ~10.0, got {}",
            analytics.avg_cpu_improvement_percent
        );
        // avg memory freed = (200 + 0) / 2 = 100 MB
        assert_eq!(analytics.avg_memory_freed_mb, 100);
    }

    #[test]
    fn negative_cpu_improvement_clamped_to_zero() {
        let mut engine = AnalyticsEngine::new();
        // CPU went UP (after > before) — should contribute 0 to average
        engine.record_optimization(20.0, 30.0, 0, 0, 60.0, 60.0, 0);

        let analytics = engine.calculate_analytics();
        assert_eq!(
            analytics.avg_cpu_improvement_percent, 0.0,
            "negative improvement should be clamped to 0"
        );
    }

    #[test]
    fn memory_freed_saturates_instead_of_wrapping() {
        let mut engine = AnalyticsEngine::new();
        // memory_after > memory_before — saturating_sub yields 0
        engine.record_optimization(0.0, 0.0, 100 * 1024 * 1024, 200 * 1024 * 1024, 0.0, 0.0, 0);

        let analytics = engine.calculate_analytics();
        assert_eq!(
            analytics.avg_memory_freed_mb, 0,
            "memory_after > memory_before should not underflow"
        );
    }

    #[test]
    fn generate_report_has_correct_title_and_structure() {
        let engine = AnalyticsEngine::new();
        let report = engine.generate_report();

        assert_eq!(report.title, "Apollo Optimizer Performance Report");
        // Should have exactly 3 top-optimization strings for an empty engine
        assert_eq!(report.top_optimizations.len(), 3);
        assert!(report.top_optimizations[0].contains("CPU"));
        assert!(report.top_optimizations[1].contains("MB"));
        assert!(report.top_optimizations[2].contains("temperature"));
    }

    #[test]
    fn generate_report_adds_high_frequency_message_when_over_1000() {
        let mut engine = AnalyticsEngine::new();
        // Record enough metrics to trigger the >1000 branch.
        // max_history is 1440, so we need metrics.len() to report total_optimizations > 1000.
        for _ in 0..1001 {
            engine.record_optimization(
                50.0,
                40.0,
                200 * 1024 * 1024,
                100 * 1024 * 1024,
                65.0,
                63.0,
                1,
            );
        }
        let report = engine.generate_report();
        let has_high_freq = report
            .top_optimizations
            .iter()
            .any(|s| s.contains("High-frequency"));
        assert!(
            has_high_freq,
            "expected high-frequency message, got: {:?}",
            report.top_optimizations
        );
    }

    #[test]
    fn get_trend_returns_correct_count_and_values() {
        let mut engine = AnalyticsEngine::new();
        engine.record_optimization(
            40.0,
            30.0,
            200 * 1024 * 1024,
            100 * 1024 * 1024,
            65.0,
            63.0,
            1,
        );
        engine.record_optimization(
            50.0,
            40.0,
            300 * 1024 * 1024,
            200 * 1024 * 1024,
            70.0,
            68.0,
            1,
        );

        let trend = engine.get_trend(5);
        assert_eq!(trend.len(), 2);

        // First entry: CPU delta=10, mem=100MB, thermal=2
        assert!(
            (trend[0].0 - 10.0).abs() < 0.01,
            "cpu delta: {}",
            trend[0].0
        );
        assert!(
            (trend[0].1 - 100.0).abs() < 0.01,
            "mem freed MB: {}",
            trend[0].1
        );
        assert!(
            (trend[0].2 - 2.0).abs() < 0.01,
            "thermal delta: {}",
            trend[0].2
        );
    }

    #[test]
    fn get_trend_returns_empty_when_no_metrics() {
        let engine = AnalyticsEngine::new();
        assert!(engine.get_trend(10).is_empty());
    }

    #[test]
    fn history_is_capped_at_max_history() {
        let mut engine = AnalyticsEngine::new();
        // Push more entries than max_history (1440)
        for i in 0..1500u32 {
            engine.record_optimization(
                50.0,
                40.0,
                200 * 1024 * 1024,
                100 * 1024 * 1024,
                65.0,
                63.0,
                i,
            );
        }
        let analytics = engine.calculate_analytics();
        // Should be capped at 1440, not 1500
        assert_eq!(analytics.total_optimizations, 1440);
    }

    #[test]
    fn energy_and_co2_are_non_negative() {
        let mut engine = AnalyticsEngine::new();
        engine.record_optimization(50.0, 30.0, 0, 0, 65.0, 63.0, 1);
        let report = engine.generate_report();
        assert!(report.energy_saved_wh >= 0.0);
        assert!(report.co2_avoided_grams >= 0.0);
    }
}
