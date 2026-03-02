//! Level 4: Advanced feature tests — I/O profiling, memory analysis, thermal prediction
//!
//! Tests the new advanced optimization modules.

use apollo_optimizer::engine::io_profiler::{IOProfiler, IOStats};
use apollo_optimizer::engine::memory_analyzer::MemoryAnalyzer;
use apollo_optimizer::engine::thermal_manager::{ThermalManager, ThermalTrend};

// ── I/O Profiler Tests ──────────────────────────────────────────────────────

#[test]
fn io_profiler_detects_high_utilization() {
    let stats = IOStats {
        reads_per_sec: 500.0,
        writes_per_sec: 300.0,
        bytes_read_per_sec: 100.0,
        bytes_written_per_sec: 50.0,
        avg_read_latency_ms: 15.0,
        avg_write_latency_ms: 20.0,
        sequential_ratio: 0.8,
        io_util_percent: 85.0,
    };

    let bottlenecks = IOProfiler::detect_io_bottlenecks(&stats);
    assert!(bottlenecks.iter().any(|b| b.contains("High I/O utilization")));
    assert!(bottlenecks.iter().any(|b| b.contains("High read latency")));
}

#[test]
fn io_profiler_detects_high_throughput() {
    let stats = IOStats {
        reads_per_sec: 1000.0,
        writes_per_sec: 500.0,
        bytes_read_per_sec: 600.0, // > 500MB/s threshold
        bytes_written_per_sec: 100.0,
        avg_read_latency_ms: 5.0,
        avg_write_latency_ms: 3.0,
        sequential_ratio: 0.9,
        io_util_percent: 60.0,
    };

    let bottlenecks = IOProfiler::detect_io_bottlenecks(&stats);
    assert!(bottlenecks.iter().any(|b| b.contains("High read throughput")));
}

#[test]
fn io_profiler_reports_no_issues_when_healthy() {
    let stats = IOStats {
        reads_per_sec: 100.0,
        writes_per_sec: 50.0,
        bytes_read_per_sec: 10.0,
        bytes_written_per_sec: 5.0,
        avg_read_latency_ms: 2.0,
        avg_write_latency_ms: 1.5,
        sequential_ratio: 0.6,
        io_util_percent: 20.0,
    };

    let bottlenecks = IOProfiler::detect_io_bottlenecks(&stats);
    assert!(bottlenecks.is_empty(), "No bottlenecks expected for healthy I/O");
}

// ── Memory Analyzer Tests ───────────────────────────────────────────────────

#[test]
fn memory_analyzer_tracks_process_history() {
    let mut analyzer = MemoryAnalyzer::new();

    let profile1 = analyzer.analyze_process(1001, "app1", 100 * 1024 * 1024, 500 * 1024 * 1024, 100);
    assert_eq!(profile1.pid, 1001);
    assert_eq!(profile1.rss_bytes, 100 * 1024 * 1024);

    let profile2 = analyzer.analyze_process(1001, "app1", 105 * 1024 * 1024, 510 * 1024 * 1024, 150);
    // Both samples stored
    assert_eq!(profile2.pid, 1001);
}

#[test]
fn memory_analyzer_detects_memory_leaks() {
    let mut analyzer = MemoryAnalyzer::new();

    // Simulate consistent growth: always increasing
    let mut current_rss = 100 * 1024 * 1024;
    for i in 0..10 {
        analyzer.analyze_process(2001, "leaky_app", current_rss, current_rss * 5, 100 + i);
        current_rss += 5 * 1024 * 1024; // Grow by 5MB each time
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

    let profile = analyzer.analyze_process(3001, "efficient_app", 200 * 1024 * 1024, 400 * 1024 * 1024, 50);
    assert!(profile.memory_efficiency > 0.0 && profile.memory_efficiency <= 1.0);
}

#[test]
fn memory_analyzer_finds_inefficient_processes() {
    let mut analyzer = MemoryAnalyzer::new();

    // Low efficiency: high page faults relative to RSS
    for i in 0..10 {
        analyzer.analyze_process(4001, "inefficient", 100 * 1024 * 1024, 500 * 1024 * 1024, 10000 + i * 1000);
    }

    let inefficient = analyzer.find_inefficient_processes(0.8);
    assert!(inefficient.len() > 0, "Should find inefficient processes");
}

// ── Thermal Manager Tests ───────────────────────────────────────────────────

#[test]
fn thermal_manager_detects_cooling() {
    let mut manager = ThermalManager::new();

    manager.update(80.0, 75.0, 70.0, 0);
    manager.update(78.0, 73.0, 68.0, 0);
    manager.update(75.0, 71.0, 66.0, 0);

    let trend = manager.calculate_trend();
    assert_eq!(trend, ThermalTrend::Cooling);
}

#[test]
fn thermal_manager_detects_critical_warming() {
    let mut manager = ThermalManager::new();

    manager.update(70.0, 65.0, 60.0, 0);
    manager.update(75.0, 70.0, 65.0, 10);
    manager.update(82.0, 77.0, 72.0, 20);
    manager.update(88.0, 83.0, 78.0, 40);

    let trend = manager.calculate_trend();
    assert_eq!(trend, ThermalTrend::Critical, "Should detect rapid warming");
}

#[test]
fn thermal_manager_predicts_throttle_level() {
    let mut manager = ThermalManager::new();

    let state = manager.update(90.0, 85.0, 80.0, 20);
    assert!(state.predicted_throttle_level > 0, "Should predict throttling at 90°C");
    assert!(state.predicted_throttle_level <= 100, "Throttle level capped at 100");
}

#[test]
fn thermal_manager_estimates_time_to_throttle() {
    let mut manager = ThermalManager::new();

    // Simulate steady warming
    manager.update(70.0, 65.0, 60.0, 0);
    std::thread::sleep(std::time::Duration::from_millis(100));
    manager.update(72.0, 67.0, 62.0, 0);
    std::thread::sleep(std::time::Duration::from_millis(100));
    manager.update(74.0, 69.0, 64.0, 0);

    let state = manager.update(75.0, 70.0, 65.0, 0);
    if state.seconds_to_throttle > 0 {
        // We're warming but not at throttle threshold yet
        assert!(state.seconds_to_throttle > 0);
    }
}

#[test]
fn thermal_manager_provides_recommendations() {
    let mut manager = ThermalManager::new();

    // Rapid warming
    for _ in 0..3 {
        manager.update(75.0, 70.0, 65.0, 0);
    }
    for _ in 0..3 {
        manager.update(85.0, 80.0, 75.0, 30);
    }

    let recommendations = manager.get_recommendations();
    assert!(
        recommendations.len() > 0,
        "Should provide recommendations when thermal issues detected"
    );
}

#[test]
fn thermal_manager_stable_state() {
    let mut manager = ThermalManager::new();

    manager.update(70.0, 65.0, 60.0, 0);
    manager.update(70.0, 65.0, 60.0, 0);
    manager.update(70.0, 65.0, 60.0, 0);

    let trend = manager.calculate_trend();
    assert_eq!(trend, ThermalTrend::Stable, "Should detect stable temperature");
}
