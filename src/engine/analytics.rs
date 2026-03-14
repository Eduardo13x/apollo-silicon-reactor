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
