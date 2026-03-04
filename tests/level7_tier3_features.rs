//! Level 7: TIER 3 Features Tests
//! Tests for analytics and power_management (including battery)

use apollo_optimizer::engine::analytics::AnalyticsEngine;
use apollo_optimizer::engine::power_management::{
    PowerManager, PowerMode, BatteryMode, BatteryStatus,
};

// ============================================================================
// Analytics Engine Tests
// ============================================================================

#[test]
fn test_analytics_engine_new() {
    let engine = AnalyticsEngine::new();
    let analytics = engine.calculate_analytics();
    assert_eq!(analytics.total_optimizations, 0);
}

#[test]
fn test_analytics_engine_record_optimization() {
    let mut engine = AnalyticsEngine::new();
    engine.record_optimization(50.0, 40.0, 8 * 1024 * 1024 * 1024, 7 * 1024 * 1024 * 1024, 80.0, 75.0, 5);

    let analytics = engine.calculate_analytics();
    assert_eq!(analytics.total_optimizations, 1);
    assert!(analytics.avg_cpu_improvement_percent > 0.0);
}

#[test]
fn test_analytics_engine_multiple_optimizations() {
    let mut engine = AnalyticsEngine::new();

    for _ in 0..10 {
        engine.record_optimization(
            50.0, 40.0,
            8 * 1024 * 1024 * 1024, 7 * 1024 * 1024 * 1024,
            80.0, 75.0,
            5,
        );
    }

    let analytics = engine.calculate_analytics();
    assert_eq!(analytics.total_optimizations, 10);
}

#[test]
fn test_analytics_engine_cpu_improvement() {
    let mut engine = AnalyticsEngine::new();
    engine.record_optimization(50.0, 30.0, 8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 80.0, 80.0, 1);

    let analytics = engine.calculate_analytics();
    assert!(analytics.avg_cpu_improvement_percent >= 20.0);
}

#[test]
fn test_analytics_engine_memory_freed() {
    let mut engine = AnalyticsEngine::new();
    let mem_freed = 1024 * 1024 * 1024; // 1GB
    engine.record_optimization(
        50.0, 50.0,
        8 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024 - mem_freed,
        80.0, 80.0,
        1,
    );

    let analytics = engine.calculate_analytics();
    assert_eq!(analytics.avg_memory_freed_mb, 1024);
}

#[test]
fn test_analytics_engine_thermal_reduction() {
    let mut engine = AnalyticsEngine::new();
    engine.record_optimization(
        50.0, 50.0,
        8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024,
        90.0, 70.0,
        1,
    );

    let analytics = engine.calculate_analytics();
    assert!(analytics.avg_thermal_reduction_celsius >= 20.0);
}

#[test]
fn test_analytics_engine_generate_report() {
    let mut engine = AnalyticsEngine::new();
    for _ in 0..100 {
        engine.record_optimization(50.0, 40.0, 8 * 1024 * 1024 * 1024, 7 * 1024 * 1024 * 1024, 80.0, 75.0, 5);
    }

    let report = engine.generate_report();
    assert!(!report.title.is_empty());
    assert!(report.analytics.total_optimizations > 0);
    assert!(!report.top_optimizations.is_empty());
}

#[test]
fn test_analytics_engine_energy_calculation() {
    let mut engine = AnalyticsEngine::new();
    engine.record_optimization(50.0, 30.0, 8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024, 80.0, 80.0, 1);

    let report = engine.generate_report();
    assert!(report.energy_saved_wh >= 0.0);
    assert!(report.co2_avoided_grams >= 0.0);
}

#[test]
fn test_analytics_engine_trend() {
    let mut engine = AnalyticsEngine::new();
    for _ in 0..5 {
        engine.record_optimization(50.0, 40.0, 8 * 1024 * 1024 * 1024, 7 * 1024 * 1024 * 1024, 80.0, 75.0, 5);
    }

    let trend = engine.get_trend(3);
    assert!(!trend.is_empty());
}

#[test]
fn test_analytics_engine_next_optimization_time() {
    let mut engine = AnalyticsEngine::new();
    for _ in 0..50 {
        engine.record_optimization(
            50.0, 40.0,
            8 * 1024 * 1024 * 1024,
            6 * 1024 * 1024 * 1024,
            80.0, 75.0,
            5,
        );
    }

    let next_time = engine.estimate_next_optimization();
    assert!(next_time > 0);
}

// ============================================================================
// Power Manager Tests
// ============================================================================

#[test]
fn test_power_manager_new() {
    let _manager = PowerManager::new();
}

#[test]
fn test_power_manager_performance_mode() {
    let manager = PowerManager::new();
    let rec = manager.get_recommendation();
    assert!(rec.target_frequency < 4000);
}

#[test]
fn test_power_manager_set_mode() {
    let mut manager = PowerManager::new();
    manager.set_mode(PowerMode::Battery);
}

#[test]
fn test_power_manager_performance_recommendation() {
    let mut manager = PowerManager::new();
    manager.set_mode(PowerMode::Performance);
    let rec = manager.get_recommendation();
    assert_eq!(rec.mode, PowerMode::Performance);
    assert!(rec.target_frequency > 3000);
    assert!(!rec.deep_sleep_enabled);
}

#[test]
fn test_power_manager_balanced_recommendation() {
    let mut manager = PowerManager::new();
    manager.set_mode(PowerMode::Balanced);
    let rec = manager.get_recommendation();
    assert_eq!(rec.mode, PowerMode::Balanced);
    assert!(rec.deep_sleep_enabled);
}

#[test]
fn test_power_manager_battery_recommendation() {
    let mut manager = PowerManager::new();
    manager.set_mode(PowerMode::Battery);
    let rec = manager.get_recommendation();
    assert_eq!(rec.mode, PowerMode::Battery);
    assert!(rec.target_frequency < 2000);
    assert!(rec.deep_sleep_enabled);
}

#[test]
fn test_power_manager_optimize_idle_states_high_idle() {
    let mut manager = PowerManager::new();
    manager.power_state.idle_percentage = 85.0;
    manager.optimize_idle_states();
}

#[test]
fn test_power_manager_optimize_idle_states_low_idle() {
    let mut manager = PowerManager::new();
    manager.power_state.idle_percentage = 10.0;
    manager.optimize_idle_states();
}

#[test]
fn test_power_manager_estimate_power() {
    let manager = PowerManager::new();
    let power = manager.estimate_power(2400, 8, 50.0);
    assert!(power > 0.0);
}

#[test]
fn test_power_manager_power_increases_with_frequency() {
    let manager = PowerManager::new();
    let power_low = manager.estimate_power(1200, 4, 50.0);
    let power_high = manager.estimate_power(3600, 8, 50.0);
    assert!(power_high > power_low);
}

#[test]
fn test_power_manager_frequency_scaling_needed() {
    let mut manager = PowerManager::new();
    manager.power_state.idle_percentage = 80.0;
    manager.power_state.cpu_frequency_mhz = 2400;
    assert!(manager.needs_frequency_scaling());
}

#[test]
fn test_power_manager_sysctl_recommendations() {
    let manager = PowerManager::new();
    let recs = manager.get_sysctl_recommendations(PowerMode::Performance);
    assert!(!recs.is_empty());
}

// ============================================================================
// Battery (merged into PowerManager) Tests
// ============================================================================

#[test]
fn test_battery_mode_normal() {
    let manager = PowerManager::new();
    let mode = manager.get_battery_mode(75);
    assert_eq!(mode, BatteryMode::Normal);
}

#[test]
fn test_battery_mode_low_power() {
    let manager = PowerManager::new();
    let mode = manager.get_battery_mode(35);
    assert_eq!(mode, BatteryMode::LowPower);
}

#[test]
fn test_battery_mode_critical() {
    let manager = PowerManager::new();
    let mode = manager.get_battery_mode(10);
    assert_eq!(mode, BatteryMode::Critical);
}

#[test]
fn test_battery_optimization_normal() {
    let mut manager = PowerManager::new();
    let status = BatteryStatus {
        percentage: 75,
        time_remaining_minutes: 300,
        is_charging: false,
        charge_rate_percent_per_hour: 50.0,
        discharge_rate_percent_per_hour: 10.0,
    };
    manager.update_battery_status(status);
    let opt = manager.get_battery_optimization();
    assert_eq!(opt.mode, BatteryMode::Normal);
    assert!(!opt.cpu_throttle);
    assert!(!opt.disable_background_apps);
}

#[test]
fn test_battery_optimization_low_power() {
    let mut manager = PowerManager::new();
    let status = BatteryStatus {
        percentage: 35,
        time_remaining_minutes: 120,
        is_charging: false,
        charge_rate_percent_per_hour: 50.0,
        discharge_rate_percent_per_hour: 10.0,
    };
    manager.update_battery_status(status);
    let opt = manager.get_battery_optimization();
    assert_eq!(opt.mode, BatteryMode::LowPower);
    assert!(opt.cpu_throttle);
    assert!(opt.disable_background_apps);
}

#[test]
fn test_battery_optimization_critical() {
    let mut manager = PowerManager::new();
    let status = BatteryStatus {
        percentage: 15,
        time_remaining_minutes: 30,
        is_charging: false,
        charge_rate_percent_per_hour: 50.0,
        discharge_rate_percent_per_hour: 10.0,
    };
    manager.update_battery_status(status);
    let opt = manager.get_battery_optimization();
    assert_eq!(opt.mode, BatteryMode::Critical);
    assert!(opt.cpu_throttle);
    assert!(opt.disable_background_apps);
}

#[test]
fn test_battery_critical_actions() {
    let manager = PowerManager::new();
    let actions = manager.get_critical_actions();
    assert!(actions.len() >= 5);
    assert!(actions.iter().any(|a| a.contains("CPU")));
}

#[test]
fn test_battery_power_savings() {
    let manager = PowerManager::new();
    let normal = manager.estimate_power_savings_percent(BatteryMode::Normal);
    let low = manager.estimate_power_savings_percent(BatteryMode::LowPower);
    let critical = manager.estimate_power_savings_percent(BatteryMode::Critical);

    assert_eq!(normal, 0.0);
    assert!(low > normal);
    assert!(critical > low);
}

#[test]
fn test_battery_apps_to_disable() {
    let manager = PowerManager::new();
    let normal_apps = manager.get_apps_to_disable(BatteryMode::Normal);
    let critical_apps = manager.get_apps_to_disable(BatteryMode::Critical);

    assert!(normal_apps.is_empty());
    assert!(!critical_apps.is_empty());
}

#[test]
fn test_battery_emergency_intervention() {
    let mut manager = PowerManager::new();
    let status = BatteryStatus {
        percentage: 3,
        time_remaining_minutes: 1,
        is_charging: false,
        charge_rate_percent_per_hour: 50.0,
        discharge_rate_percent_per_hour: 10.0,
    };
    manager.update_battery_status(status);
    assert!(manager.needs_emergency_intervention());
}

#[test]
fn test_battery_time_to_critical() {
    let mut manager = PowerManager::new();
    let status = BatteryStatus {
        percentage: 25,
        time_remaining_minutes: 100,
        is_charging: false,
        charge_rate_percent_per_hour: 50.0,
        discharge_rate_percent_per_hour: 10.0,
    };
    manager.update_battery_status(status);
    let time = manager.time_to_critical();
    assert!(time > 0);
}

#[test]
fn test_battery_predict_time_with_optimization() {
    let mut manager = PowerManager::new();
    let status = BatteryStatus {
        percentage: 50,
        time_remaining_minutes: 300,
        is_charging: false,
        charge_rate_percent_per_hour: 50.0,
        discharge_rate_percent_per_hour: 10.0,
    };
    manager.update_battery_status(status);
    let without_opt = manager.predict_time_remaining(false);
    let with_opt = manager.predict_time_remaining(true);

    assert!(with_opt > without_opt);
}

// ============================================================================
// Integration: Analytics + Power
// ============================================================================

#[test]
fn test_analytics_battery_integration() {
    let mut analytics = AnalyticsEngine::new();
    let mut manager = PowerManager::new();

    for _ in 0..10 {
        analytics.record_optimization(
            50.0, 40.0,
            8 * 1024 * 1024 * 1024, 7 * 1024 * 1024 * 1024,
            80.0, 75.0,
            5,
        );
    }

    let status = BatteryStatus {
        percentage: 50,
        time_remaining_minutes: 300,
        is_charging: false,
        charge_rate_percent_per_hour: 50.0,
        discharge_rate_percent_per_hour: 10.0,
    };
    manager.update_battery_status(status);

    let analytics_report = analytics.generate_report();
    let battery_opt = manager.get_battery_optimization();

    assert!(analytics_report.analytics.total_optimizations > 0);
    assert_eq!(battery_opt.mode, BatteryMode::Normal);
}

#[test]
fn test_power_analytics_integration() {
    let mut power = PowerManager::new();
    let mut analytics = AnalyticsEngine::new();

    power.set_mode(PowerMode::Performance);
    let rec = power.get_recommendation();

    analytics.record_optimization(
        80.0, 60.0,
        8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024,
        80.0, 75.0,
        1,
    );

    assert!(rec.target_frequency > 0);
    assert!(analytics.calculate_analytics().total_optimizations > 0);
}
