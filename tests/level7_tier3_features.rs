//! Level 7: TIER 3 Features Tests
//! Tests for analytics and power_management (including battery)

use apollo_engine::engine::analytics::AnalyticsEngine;
use apollo_engine::engine::power_management::{
    BatteryMode, BatteryStatus, PowerManager, PowerMode,
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
    engine.record_optimization(
        50.0,
        40.0,
        8 * 1024 * 1024 * 1024,
        7 * 1024 * 1024 * 1024,
        80.0,
        75.0,
        5,
    );

    let analytics = engine.calculate_analytics();
    assert_eq!(analytics.total_optimizations, 1);
    assert!(analytics.avg_cpu_improvement_percent > 0.0);
}

#[test]
fn test_analytics_engine_multiple_optimizations() {
    let mut engine = AnalyticsEngine::new();

    for _ in 0..10 {
        engine.record_optimization(
            50.0,
            40.0,
            8 * 1024 * 1024 * 1024,
            7 * 1024 * 1024 * 1024,
            80.0,
            75.0,
            5,
        );
    }

    let analytics = engine.calculate_analytics();
    assert_eq!(analytics.total_optimizations, 10);
}

#[test]
fn test_analytics_engine_cpu_improvement() {
    let mut engine = AnalyticsEngine::new();
    engine.record_optimization(
        50.0,
        30.0,
        8 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
        80.0,
        80.0,
        1,
    );

    let analytics = engine.calculate_analytics();
    assert!(analytics.avg_cpu_improvement_percent >= 20.0);
}

#[test]
fn test_analytics_engine_memory_freed() {
    let mut engine = AnalyticsEngine::new();
    let mem_freed = 1024 * 1024 * 1024; // 1GB
    engine.record_optimization(
        50.0,
        50.0,
        8 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024 - mem_freed,
        80.0,
        80.0,
        1,
    );

    let analytics = engine.calculate_analytics();
    assert_eq!(analytics.avg_memory_freed_mb, 1024);
}

#[test]
fn test_analytics_engine_thermal_reduction() {
    let mut engine = AnalyticsEngine::new();
    engine.record_optimization(
        50.0,
        50.0,
        8 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
        90.0,
        70.0,
        1,
    );

    let analytics = engine.calculate_analytics();
    assert!(analytics.avg_thermal_reduction_celsius >= 20.0);
}

#[test]
fn test_analytics_engine_generate_report() {
    let mut engine = AnalyticsEngine::new();
    for _ in 0..100 {
        engine.record_optimization(
            50.0,
            40.0,
            8 * 1024 * 1024 * 1024,
            7 * 1024 * 1024 * 1024,
            80.0,
            75.0,
            5,
        );
    }

    let report = engine.generate_report();
    assert!(!report.title.is_empty());
    assert!(report.analytics.total_optimizations > 0);
    assert!(!report.top_optimizations.is_empty());
}

#[test]
fn test_analytics_engine_energy_calculation() {
    let mut engine = AnalyticsEngine::new();
    engine.record_optimization(
        50.0,
        30.0,
        8 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
        80.0,
        80.0,
        1,
    );

    let report = engine.generate_report();
    assert!(report.energy_saved_wh >= 0.0);
    assert!(report.co2_avoided_grams >= 0.0);
}

#[test]
fn test_analytics_engine_trend() {
    let mut engine = AnalyticsEngine::new();
    for _ in 0..5 {
        engine.record_optimization(
            50.0,
            40.0,
            8 * 1024 * 1024 * 1024,
            7 * 1024 * 1024 * 1024,
            80.0,
            75.0,
            5,
        );
    }

    let trend = engine.get_trend(3);
    assert!(!trend.is_empty());
}

// ============================================================================
// Power Manager Tests
// ============================================================================

#[test]
fn test_power_manager_new() {
    let manager = PowerManager::new();
    // PowerState should come from real sysctl detection
    assert!(
        manager.power_state.core_count_active >= 1,
        "core_count_active must be >= 1"
    );
    // power_draw_watts should be 0.0 (not invented)
    assert_eq!(manager.power_state.power_draw_watts, 0.0);
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
    // Performance uses 100% of detected frequency
    assert_eq!(rec.target_frequency, manager.power_state.cpu_frequency_mhz);
    assert_eq!(rec.active_cores, manager.power_state.core_count_active);
    assert!(!rec.deep_sleep_enabled);
}

#[test]
fn test_power_manager_balanced_recommendation() {
    let mut manager = PowerManager::new();
    manager.set_mode(PowerMode::Balanced);
    let rec = manager.get_recommendation();
    assert_eq!(rec.mode, PowerMode::Balanced);
    assert!(rec.deep_sleep_enabled);
    // Balanced uses 75% of max frequency
    assert!(rec.target_frequency <= manager.power_state.cpu_frequency_mhz);
}

#[test]
fn test_power_manager_battery_recommendation() {
    let mut manager = PowerManager::new();
    manager.set_mode(PowerMode::Battery);
    let rec = manager.get_recommendation();
    assert_eq!(rec.mode, PowerMode::Battery);
    // Battery uses 30% of max — should be less than max
    assert!(rec.target_frequency <= manager.power_state.cpu_frequency_mhz);
    assert!(rec.deep_sleep_enabled);
}

#[test]
fn test_power_manager_recommendation_ordering() {
    // Performance > Balanced > Efficiency > Battery in target_frequency
    let mut manager = PowerManager::new();

    manager.set_mode(PowerMode::Performance);
    let perf = manager.get_recommendation();

    manager.set_mode(PowerMode::Balanced);
    let balanced = manager.get_recommendation();

    manager.set_mode(PowerMode::Efficiency);
    let efficiency = manager.get_recommendation();

    manager.set_mode(PowerMode::Battery);
    let battery = manager.get_recommendation();

    assert!(perf.target_frequency >= balanced.target_frequency);
    assert!(balanced.target_frequency >= efficiency.target_frequency);
    assert!(efficiency.target_frequency >= battery.target_frequency);
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
fn test_power_manager_update_thermal_headroom() {
    let mut manager = PowerManager::new();
    // Initial headroom is 100.0
    assert_eq!(manager.power_state.thermal_headroom, 100.0);
    // Moderate thermal pressure (0.5 = 50% thermal)
    manager.update_thermal_headroom(0.5);
    assert!((manager.power_state.thermal_headroom - 50.0).abs() < 0.01);
    // Critical thermal (1.0 = 100% thermal)
    manager.update_thermal_headroom(1.0);
    assert!((manager.power_state.thermal_headroom - 0.0).abs() < 0.01);
}

#[test]
fn test_power_manager_update_power_draw() {
    let mut manager = PowerManager::new();
    assert_eq!(manager.power_state.power_draw_watts, 0.0);
    manager.update_power_draw(8.5);
    assert!((manager.power_state.power_draw_watts - 8.5).abs() < 0.01);
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
fn test_battery_mode_covers_high_percentage() {
    // Verify percentages > 100 map to Normal (open-ended range)
    let manager = PowerManager::new();
    assert_eq!(manager.get_battery_mode(150), BatteryMode::Normal);
    assert_eq!(manager.get_battery_mode(255), BatteryMode::Normal);
}

#[test]
fn test_update_battery_calibrates_baseline() {
    let mut manager = PowerManager::new();
    let status = BatteryStatus {
        percentage: 60,
        time_remaining_minutes: 180,
        is_charging: false,
        charge_rate_percent_per_hour: 0.0,
        discharge_rate_percent_per_hour: 15.0,
    };
    manager.update_battery_status(status);
    // baseline_discharge_rate should now be 15.0
    // Verify via time_to_critical: (60 - 20) / 15.0 * 60 = 160
    let ttc = manager.time_to_critical();
    assert_eq!(ttc, 160);
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
            50.0,
            40.0,
            8 * 1024 * 1024 * 1024,
            7 * 1024 * 1024 * 1024,
            80.0,
            75.0,
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
        80.0,
        60.0,
        8 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
        80.0,
        75.0,
        1,
    );

    assert!(rec.target_frequency > 0);
    assert!(analytics.calculate_analytics().total_optimizations > 0);
}
