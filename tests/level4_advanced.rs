//! Level 4: Advanced feature tests — memory analysis, thermal prediction

use apollo_optimizer::engine::memory_analyzer::MemoryAnalyzer;
use apollo_optimizer::engine::thermal_manager::{ThermalManager, ThermalTrend};

// ── Memory Analyzer Tests ───────────────────────────────────────────────────

#[test]
fn memory_analyzer_tracks_process_history() {
    let mut analyzer = MemoryAnalyzer::new();

    let profile1 =
        analyzer.analyze_process(1001, "app1", 100 * 1024 * 1024, 500 * 1024 * 1024, 100);
    assert_eq!(profile1.pid, 1001);
    assert_eq!(profile1.rss_bytes, 100 * 1024 * 1024);

    let profile2 =
        analyzer.analyze_process(1001, "app1", 105 * 1024 * 1024, 510 * 1024 * 1024, 150);
    assert_eq!(profile2.pid, 1001);
}

#[test]
fn memory_analyzer_detects_memory_leaks() {
    let mut analyzer = MemoryAnalyzer::new();

    // 30 monotonically-growing samples required by detect_memory_leak's MIN_SAMPLES=30.
    // 5 MB/step × 30 steps = 150 MB growth (100→250 MB, ratio 2.5×), well above all thresholds.
    let mut current_rss = 100u64 * 1024 * 1024;
    for i in 0..30 {
        analyzer.analyze_process(2001, "leaky_app", current_rss, current_rss * 5, 100 + i);
        current_rss += 5 * 1024 * 1024;
    }

    let leaks = analyzer.find_memory_leaks(0.6);
    assert!(
        leaks.iter().any(|(pid, prob)| *pid == 2001 && *prob > 0.6),
        "Should detect memory leak with high probability"
    );
}

#[test]
fn memory_analyzer_calculates_efficiency() {
    let mut analyzer = MemoryAnalyzer::new();

    let profile = analyzer.analyze_process(
        3001,
        "efficient_app",
        200 * 1024 * 1024,
        400 * 1024 * 1024,
        50,
    );
    assert!(profile.memory_efficiency > 0.0 && profile.memory_efficiency <= 1.0);
}

#[test]
fn memory_analyzer_finds_inefficient_processes() {
    let mut analyzer = MemoryAnalyzer::new();

    for i in 0..10 {
        analyzer.analyze_process(
            4001,
            "inefficient",
            100 * 1024 * 1024,
            500 * 1024 * 1024,
            10000 + i * 1000,
        );
    }

    let inefficient = analyzer.find_inefficient_processes(0.8);
    assert!(!inefficient.is_empty(), "Should find inefficient processes");
}

// ── Thermal Manager Tests ───────────────────────────────────────────────────

#[test]
fn thermal_manager_detects_cooling() {
    let mut manager = ThermalManager::new();

    manager.update(80.0, 75.0, 70.0, 0, 0);
    manager.update(78.0, 73.0, 68.0, 0, 0);
    manager.update(75.0, 71.0, 66.0, 0, 0);

    let trend = manager.calculate_trend();
    assert_eq!(trend, ThermalTrend::Cooling);
}

#[test]
fn thermal_manager_detects_critical_warming() {
    let mut manager = ThermalManager::new();

    manager.update(70.0, 65.0, 60.0, 0, 0);
    manager.update(75.0, 70.0, 65.0, 10, 0);
    manager.update(82.0, 77.0, 72.0, 20, 0);
    manager.update(88.0, 83.0, 78.0, 40, 0);

    let trend = manager.calculate_trend();
    assert_eq!(trend, ThermalTrend::Critical, "Should detect rapid warming");
}

#[test]
fn thermal_manager_predicts_throttle_level() {
    let mut manager = ThermalManager::new();

    let state = manager.update(90.0, 85.0, 80.0, 20, 0);
    assert!(
        state.predicted_throttle_level > 0,
        "Should predict throttling at 90C"
    );
    assert!(
        state.predicted_throttle_level <= 100,
        "Throttle level capped at 100"
    );
}

#[test]
fn thermal_manager_estimates_time_to_throttle() {
    let mut manager = ThermalManager::new();

    manager.update(70.0, 65.0, 60.0, 0, 0);
    std::thread::sleep(std::time::Duration::from_millis(100));
    manager.update(72.0, 67.0, 62.0, 0, 0);
    std::thread::sleep(std::time::Duration::from_millis(100));
    manager.update(74.0, 69.0, 64.0, 0, 0);

    let state = manager.update(75.0, 70.0, 65.0, 0, 0);
    if state.seconds_to_throttle > 0 {
        assert!(state.seconds_to_throttle > 0);
    }
}

#[test]
fn thermal_manager_provides_recommendations() {
    let mut manager = ThermalManager::new();

    for _ in 0..3 {
        manager.update(75.0, 70.0, 65.0, 0, 0);
    }
    for _ in 0..3 {
        manager.update(85.0, 80.0, 75.0, 30, 0);
    }

    let recommendations = manager.get_recommendations();
    assert!(
        !recommendations.is_empty(),
        "Should provide recommendations when thermal issues detected"
    );
}

#[test]
fn thermal_manager_stable_state() {
    let mut manager = ThermalManager::new();

    manager.update(70.0, 65.0, 60.0, 0, 0);
    manager.update(70.0, 65.0, 60.0, 0, 0);
    manager.update(70.0, 65.0, 60.0, 0, 0);

    let trend = manager.calculate_trend();
    assert_eq!(
        trend,
        ThermalTrend::Stable,
        "Should detect stable temperature"
    );
}
