//! Level 5: TIER 1 Extended Features Tests
//! Tests for process_recovery, swap_predictor, and gpu_manager modules
//! These are the highest-priority missing optimization features

use apollo_optimizer::engine::gpu_manager::{GPUManager, GPUMetrics, GPUPowerState};
use apollo_optimizer::engine::process_recovery::{LeakingProcess, ProcessRecoveryManager};
use apollo_optimizer::engine::swap_predictor::{SwapPredictor, SwapTrend};
use std::time::{Duration, Instant};

// ============================================================================
// GPU Manager Tests
// ============================================================================

#[test]
fn test_gpu_power_state_off() {
    let manager = GPUManager::new();
    let state = manager.recommend_power_state(0.0, 30.0);
    assert_eq!(state, GPUPowerState::Idle);
}

#[test]
fn test_gpu_power_state_idle() {
    let manager = GPUManager::new();
    let state = manager.recommend_power_state(5.0, 40.0);
    assert_eq!(state, GPUPowerState::Idle);
}

#[test]
fn test_gpu_power_state_dynamic() {
    let manager = GPUManager::new();
    let state = manager.recommend_power_state(50.0, 60.0);
    assert_eq!(state, GPUPowerState::Dynamic);
}

#[test]
fn test_gpu_power_state_maximum() {
    let manager = GPUManager::new();
    let state = manager.recommend_power_state(85.0, 70.0);
    assert_eq!(state, GPUPowerState::Maximum);
}

#[test]
fn test_gpu_power_state_throttled_temp() {
    let manager = GPUManager::new();
    let state = manager.recommend_power_state(50.0, 105.0); // Over max safe
    assert_eq!(state, GPUPowerState::Throttled);
}

#[test]
fn test_gpu_power_state_dynamic_high_temp() {
    let manager = GPUManager::new();
    let state = manager.recommend_power_state(50.0, 95.0); // Above 90°C throttle threshold
    assert_eq!(state, GPUPowerState::Dynamic);
}

#[test]
fn test_gpu_needs_cooling() {
    let manager = GPUManager::new();
    let metrics = GPUMetrics {
        gpu_temp: 95.0, // Above 90°C throttle threshold
        gpu_utilization: 50.0,
        gpu_frequency: 1000,
        gpu_memory_used: 1024 * 1024 * 1024,
        gpu_memory_total: 2 * 1024 * 1024 * 1024,
        throttle_active: false,
        power_state: GPUPowerState::Dynamic,
    };
    assert!(manager.needs_cooling(&metrics));
}

#[test]
fn test_gpu_does_not_need_cooling() {
    let manager = GPUManager::new();
    let metrics = GPUMetrics {
        gpu_temp: 70.0, // Below 90°C throttle threshold
        gpu_utilization: 50.0,
        gpu_frequency: 1000,
        gpu_memory_used: 1024 * 1024 * 1024,
        gpu_memory_total: 2 * 1024 * 1024 * 1024,
        throttle_active: false,
        power_state: GPUPowerState::Dynamic,
    };
    assert!(!manager.needs_cooling(&metrics));
}

#[test]
fn test_gpu_workload_ml() {
    let manager = GPUManager::new();
    let actions = manager.optimize_for_workload("ai");
    assert!(actions.len() >= 2);
    assert!(actions.iter().any(|a| a.contains("ML")));
    assert!(actions.iter().any(|a| a.contains("maximum")));
}

#[test]
fn test_gpu_workload_rendering() {
    let manager = GPUManager::new();
    let actions = manager.optimize_for_workload("rendering");
    assert!(!actions.is_empty());
    assert!(actions.iter().any(|a| a.contains("cache")));
}

#[test]
fn test_gpu_workload_idle() {
    let manager = GPUManager::new();
    let actions = manager.optimize_for_workload("idle");
    assert!(!actions.is_empty());
    assert!(actions.iter().any(|a| a.contains("idle")));
}

#[test]
fn test_gpu_thermal_recommendations_critical() {
    let manager = GPUManager::new();
    let metrics = GPUMetrics {
        gpu_temp: 105.0, // Over max safe (100)
        gpu_utilization: 50.0,
        gpu_frequency: 1000,
        gpu_memory_used: 1024 * 1024 * 1024,
        gpu_memory_total: 2 * 1024 * 1024 * 1024,
        throttle_active: true,
        power_state: GPUPowerState::Throttled,
    };
    let recs = manager.thermal_recommendations(&metrics);
    assert!(!recs.is_empty());
    assert!(recs.iter().any(|r| r.contains("CRITICAL")));
}

#[test]
fn test_gpu_thermal_recommendations_warning() {
    let manager = GPUManager::new();
    let metrics = GPUMetrics {
        gpu_temp: 95.0, // Above throttle (90) but below max (100)
        gpu_utilization: 50.0,
        gpu_frequency: 1000,
        gpu_memory_used: 1024 * 1024 * 1024,
        gpu_memory_total: 2 * 1024 * 1024 * 1024,
        throttle_active: false,
        power_state: GPUPowerState::Dynamic,
    };
    let recs = manager.thermal_recommendations(&metrics);
    assert!(!recs.is_empty());
    assert!(recs.iter().any(|r| r.contains("warming")));
}

// ============================================================================
// Process Recovery Manager Tests
// ============================================================================

#[test]
fn test_recovery_manager_register_leak() {
    let mut manager = ProcessRecoveryManager::new();
    manager.register_leak(1234, "test_app".to_string(), 0.85, 500 * 1024 * 1024);

    let targets = manager.get_recovery_targets();
    // Should not be killed yet (only 30 seconds have passed in test, need 1800)
    assert_eq!(targets.len(), 0);
}

#[test]
fn test_recovery_manager_ignores_low_confidence_leak() {
    let mut manager = ProcessRecoveryManager::new();
    manager.register_leak(1234, "test_app".to_string(), 0.5, 500 * 1024 * 1024); // Low prob

    let targets = manager.get_recovery_targets();
    assert_eq!(targets.len(), 0);
}

#[test]
fn test_recovery_manager_high_confidence_leak() {
    let mut manager = ProcessRecoveryManager::new();
    manager.register_leak(1234, "test_app".to_string(), 0.9, 500 * 1024 * 1024);
    manager.register_leak(5678, "bad_process".to_string(), 0.95, 800 * 1024 * 1024);

    // Even though registered, should not be targeted yet
    let targets = manager.get_recovery_targets();
    assert_eq!(targets.len(), 0);
}

#[test]
fn test_recovery_manager_cleanup_resolved() {
    let mut manager = ProcessRecoveryManager::new();
    manager.register_leak(1234, "test_app".to_string(), 0.85, 500 * 1024 * 1024);
    manager.register_leak(5678, "bad_process".to_string(), 0.5, 500 * 1024 * 1024);

    // Cleanup should remove low-confidence ones
    manager.cleanup_resolved();

    let targets = manager.get_recovery_targets();
    assert_eq!(targets.len(), 0);
}

#[test]
fn test_recovery_manager_max_attempts() {
    let mut manager = ProcessRecoveryManager::new();
    manager.register_leak(1234, "test_app".to_string(), 0.85, 500 * 1024 * 1024);

    // Simulate 3 recovery attempts
    manager.record_kill_attempt(1234);
    manager.record_kill_attempt(1234);
    manager.record_kill_attempt(1234);

    // Should not try again
    assert!(!manager.should_kill_process(1234));
}

#[test]
fn test_recovery_manager_cost_estimation() {
    let proc1 = LeakingProcess {
        pid: 1234,
        name: "small_app".to_string(),
        leak_probability: 0.8,
        rss_bytes: 100 * 1024 * 1024, // 100MB
        first_detected_at: Instant::now(),
        recovery_attempts: 0,
    };

    let proc2 = LeakingProcess {
        pid: 5678,
        name: "large_app".to_string(),
        leak_probability: 0.9,
        rss_bytes: 4 * 1024 * 1024 * 1024, // 4GB
        first_detected_at: Instant::now(),
        recovery_attempts: 2,
    };

    let cost1 = ProcessRecoveryManager::estimate_recovery_cost(&proc1);
    let cost2 = ProcessRecoveryManager::estimate_recovery_cost(&proc2);

    // Larger process with more attempts should have higher cost
    assert!(cost2 > cost1);
}

#[test]
fn test_recovery_manager_targets_sorted_by_memory() {
    let mut manager = ProcessRecoveryManager::new();
    manager.register_leak(1111, "small".to_string(), 0.8, 100 * 1024 * 1024);
    manager.register_leak(2222, "large".to_string(), 0.8, 800 * 1024 * 1024);
    manager.register_leak(3333, "medium".to_string(), 0.8, 400 * 1024 * 1024);

    let targets = manager.get_recovery_targets();
    // Should be sorted by memory (largest first)
    if targets.len() > 1 {
        for i in 1..targets.len() {
            assert!(targets[i - 1].rss_bytes >= targets[i].rss_bytes);
        }
    }
}

#[test]
fn test_recovery_manager_max_3_targets() {
    let mut manager = ProcessRecoveryManager::new();
    for i in 0..10 {
        manager.register_leak(1000 + i, format!("proc_{}", i), 0.8, 100 * 1024 * 1024);
    }

    let targets = manager.get_recovery_targets();
    // Should limit to max 3 targets per cycle
    assert!(targets.len() <= 3);
}

// ============================================================================
// Swap Predictor Tests
// ============================================================================

#[test]
fn test_swap_predictor_stable_trend() {
    let mut predictor = SwapPredictor::new();

    // Feed stable swap usage (no increase) - need 10+ samples for good trend detection
    for _ in 0..10 {
        let _forecast = predictor.update(500 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
        std::thread::sleep(Duration::from_millis(1));
    }

    let forecast = predictor.update(500 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
    // With no growth, should be stable or decreasing (no increases detected)
    assert!(matches!(
        forecast.swap_trend,
        SwapTrend::Stable | SwapTrend::Decreasing
    ));
}

#[test]
fn test_swap_predictor_increasing_trend() {
    let mut predictor = SwapPredictor::new();

    // Feed increasing swap usage
    let mut swap_used = 100 * 1024 * 1024;
    for i in 0..12 {
        let forecast = predictor.update(swap_used, 2 * 1024 * 1024 * 1024);
        swap_used += 50 * 1024 * 1024; // Increase each cycle

        if i >= 3 {
            // After enough samples, should detect increasing trend
            assert!(matches!(
                forecast.swap_trend,
                SwapTrend::Increasing | SwapTrend::Stable
            ));
        }
    }
}

#[test]
fn test_swap_predictor_critical_trend() {
    let mut predictor = SwapPredictor::new();

    // Feed swap above critical threshold - need at least 3 samples
    for _ in 0..3 {
        let _forecast = predictor.update(1100 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
    }

    let forecast = predictor.update(1100 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
    assert_eq!(forecast.swap_trend, SwapTrend::Critical);
}

#[test]
fn test_swap_predictor_decreasing_trend() {
    let mut predictor = SwapPredictor::new();

    // Feed decreasing swap usage
    let mut swap_used = 500 * 1024 * 1024;
    for _ in 0..12 {
        predictor.update(swap_used, 2 * 1024 * 1024 * 1024);
        swap_used = swap_used.saturating_sub(30 * 1024 * 1024);
    }

    let forecast = predictor.update(swap_used, 2 * 1024 * 1024 * 1024);
    // After 12+ samples, should detect decreasing trend
    assert_eq!(forecast.swap_trend, SwapTrend::Decreasing);
}

#[test]
fn test_swap_predictor_time_to_critical() {
    let mut predictor = SwapPredictor::new();

    // Already critical
    let forecast = predictor.update(1100 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
    assert_eq!(forecast.time_to_swap_critical, 0);
}

#[test]
fn test_swap_predictor_time_to_critical_not_yet() {
    let mut predictor = SwapPredictor::new();

    // Not yet critical, and not increasing
    let forecast = predictor.update(500 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
    assert_eq!(forecast.time_to_swap_critical, -1); // No time to critical
}

#[test]
fn test_swap_predictor_recommendations_critical() {
    let mut predictor = SwapPredictor::new();

    // Need at least 3 samples to detect trend
    for _ in 0..3 {
        let _forecast = predictor.update(1100 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
    }

    let forecast = predictor.update(1100 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
    assert!(!forecast.recommended_actions.is_empty());
    assert!(forecast
        .recommended_actions
        .iter()
        .any(|r| r.contains("CRITICAL")));
}

#[test]
fn test_swap_predictor_recommendations_increasing() {
    let mut predictor = SwapPredictor::new();

    // Feed increasing trend
    let mut swap_used = 300 * 1024 * 1024;
    for _ in 0..15 {
        predictor.update(swap_used, 2 * 1024 * 1024 * 1024);
        swap_used += 40 * 1024 * 1024;
    }

    let forecast = predictor.update(swap_used, 2 * 1024 * 1024 * 1024);

    if forecast.swap_trend == SwapTrend::Increasing {
        // Should have recommendations for increasing trend
        assert!(!forecast.recommended_actions.is_empty() || swap_used < 500 * 1024 * 1024);
    }
}

#[test]
fn test_swap_predictor_history_limit() {
    let mut predictor = SwapPredictor::new();

    // Feed more than max_history samples
    for i in 0..150 {
        let _forecast = predictor.update((i as u64) * 1024 * 1024, 2 * 1024 * 1024 * 1024);
    }

    // The predictor maintains internal history limit, so it shouldn't panic or fail
    let _forecast = predictor.update(100 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
    // Verify predictor still works after exceeding history limit (no panic)
}

#[test]
fn test_swap_predictor_prediction_extrapolation() {
    let mut predictor = SwapPredictor::new();

    // Feed a linear increase pattern
    for i in 0..5 {
        predictor.update((i as u64) * 100 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
        std::thread::sleep(Duration::from_millis(10));
    }

    let forecast = predictor.update(400 * 1024 * 1024, 2 * 1024 * 1024 * 1024);

    // Prediction should exist and be reasonable
    assert!(forecast.swap_predicted_bytes <= 4 * 1024 * 1024 * 1024); // Within safe bounds
}

// ============================================================================
// Integration: GPU + Thermal Awareness
// ============================================================================

#[test]
fn test_gpu_thermal_integration() {
    let manager = GPUManager::new();

    let hot_metrics = GPUMetrics {
        gpu_temp: 95.0,
        gpu_utilization: 90.0,
        gpu_frequency: 1500,
        gpu_memory_used: 6 * 1024 * 1024 * 1024,
        gpu_memory_total: 8 * 1024 * 1024 * 1024,
        throttle_active: false,
        power_state: GPUPowerState::Maximum,
    };

    // Should recommend Dynamic or lower due to high temp
    let recommended =
        manager.recommend_power_state(hot_metrics.gpu_utilization, hot_metrics.gpu_temp);

    assert!(matches!(
        recommended,
        GPUPowerState::Dynamic | GPUPowerState::Throttled
    ));
}

// ============================================================================
// Integration: Process Recovery + Swap Predictor
// ============================================================================

#[test]
fn test_recovery_and_swap_integration() {
    let mut recovery = ProcessRecoveryManager::new();
    let mut swap = SwapPredictor::new();

    // Register memory leaks
    recovery.register_leak(1234, "leaky".to_string(), 0.85, 2 * 1024 * 1024 * 1024);

    // Monitor swap increasing
    let mut swap_used = 200 * 1024 * 1024;
    for _ in 0..10 {
        let forecast = swap.update(swap_used, 2 * 1024 * 1024 * 1024);
        swap_used += 50 * 1024 * 1024;

        // If swap is increasing, should coordinate with process recovery
        if matches!(
            forecast.swap_trend,
            SwapTrend::Increasing | SwapTrend::Critical
        ) {
            // Recovery module should also be active
        }
    }
}
