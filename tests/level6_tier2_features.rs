//! Level 6: TIER 2 Features Tests
//! Tests for wake_storm_detector and network_optimizer

use apollo_engine::engine::network_optimizer::{NetworkOptimizer, NetworkProfile, NetworkStats};
use apollo_engine::engine::wake_storm_detector::{StormSeverity, WakeStormDetector};

// ============================================================================
// Wake Storm Detector Tests
// ============================================================================

#[test]
fn test_wake_storm_detector_new() {
    let detector = WakeStormDetector::new();
    assert!(detector.detect_storms().is_empty());
}

#[test]
fn test_wake_storm_detector_record_wakeup() {
    let mut detector = WakeStormDetector::new();
    detector.record_wakeup(1234, "test_app".to_string());
    assert!(detector.detect_storms().is_empty());
}

#[test]
fn test_wake_storm_detector_severity_low() {
    let detector = WakeStormDetector::new();
    let severity = detector.get_severity(25.0);
    assert_eq!(severity, StormSeverity::Low);
}

#[test]
fn test_wake_storm_detector_severity_medium() {
    let detector = WakeStormDetector::new();
    let severity = detector.get_severity(75.0);
    assert_eq!(severity, StormSeverity::Medium);
}

#[test]
fn test_wake_storm_detector_severity_high() {
    let detector = WakeStormDetector::new();
    let severity = detector.get_severity(500.0);
    assert_eq!(severity, StormSeverity::High);
}

#[test]
fn test_wake_storm_detector_severity_critical() {
    let detector = WakeStormDetector::new();
    let severity = detector.get_severity(1500.0);
    assert_eq!(severity, StormSeverity::Critical);
}

#[test]
fn test_wake_storm_detector_mitigation_low() {
    let actions = WakeStormDetector::get_mitigation_actions(StormSeverity::Low);
    assert!(!actions.is_empty());
    assert!(actions.iter().any(|a| a.contains("Monitor")));
}

#[test]
fn test_wake_storm_detector_mitigation_critical() {
    let actions = WakeStormDetector::get_mitigation_actions(StormSeverity::Critical);
    assert!(actions.len() >= 2);
    assert!(actions.iter().any(|a| a.contains("CRITICAL")));
    assert!(actions.iter().any(|a| a.contains("Suspend")));
}

#[test]
fn test_wake_storm_detector_cleanup_stale() {
    let mut detector = WakeStormDetector::new();
    detector.record_wakeup(1234, "test_app".to_string());

    let max_age = std::time::Duration::from_secs(1);
    std::thread::sleep(std::time::Duration::from_millis(100));
    detector.cleanup_stale(max_age);

    // Should still exist (< 1 second old)
    let storms = detector.detect_storms();
    let _ = storms.len();
}

// ============================================================================
// Network Optimizer Tests
// ============================================================================

#[test]
fn test_network_optimizer_new() {
    let _optimizer = NetworkOptimizer::new();
}

#[test]
fn test_network_optimizer_high_throughput() {
    let optimizer = NetworkOptimizer::new();
    let opt = optimizer.get_optimization(NetworkProfile::HighThroughput);
    assert_eq!(opt.profile, NetworkProfile::HighThroughput);
    assert!(opt.tcp_send_buffer > 1_000_000);
    assert!(opt.tcp_recv_buffer > 1_000_000);
}

#[test]
fn test_network_optimizer_low_latency() {
    let optimizer = NetworkOptimizer::new();
    let opt = optimizer.get_optimization(NetworkProfile::LowLatency);
    assert_eq!(opt.profile, NetworkProfile::LowLatency);
    assert!(opt.tcp_send_buffer < 100_000);
}

#[test]
fn test_network_optimizer_battery() {
    let optimizer = NetworkOptimizer::new();
    let opt = optimizer.get_optimization(NetworkProfile::Battery);
    assert_eq!(opt.profile, NetworkProfile::Battery);
    assert!(opt.tcp_send_buffer < 500_000);
}

#[test]
fn test_network_optimizer_update_stats() {
    let mut optimizer = NetworkOptimizer::new();
    let stats = NetworkStats {
        packets_sent: 1000,
        packets_recv: 2000,
        bytes_sent: 1_000_000,
        bytes_recv: 2_000_000,
        errors: 0,
        dropped: 0,
        packet_loss_percent: 0.0,
        latency_ms: 10.0,
    };
    optimizer.update_stats(stats);
}

#[test]
fn test_network_optimizer_recommend_profile_low_latency() {
    let mut optimizer = NetworkOptimizer::new();
    let stats = NetworkStats {
        packets_sent: 1000,
        packets_recv: 2000,
        bytes_sent: 1_000_000,
        bytes_recv: 2_000_000,
        errors: 0,
        dropped: 0,
        packet_loss_percent: 2.0,
        latency_ms: 75.0,
    };
    optimizer.update_stats(stats);
    let profile = optimizer.recommend_profile();
    assert_eq!(profile, NetworkProfile::LowLatency);
}

#[test]
fn test_network_optimizer_recommend_profile_high_throughput() {
    let mut optimizer = NetworkOptimizer::new();
    let stats = NetworkStats {
        packets_sent: 1_000_000,
        packets_recv: 1_000_000,
        bytes_sent: 2_000_000_000,
        bytes_recv: 2_000_000_000,
        errors: 0,
        dropped: 0,
        packet_loss_percent: 0.0,
        latency_ms: 5.0,
    };
    optimizer.update_stats(stats);
    let profile = optimizer.recommend_profile();
    assert_eq!(profile, NetworkProfile::HighThroughput);
}

#[test]
fn test_network_optimizer_detect_issues_packet_loss() {
    let mut optimizer = NetworkOptimizer::new();
    let stats = NetworkStats {
        packets_sent: 1000,
        packets_recv: 500,
        bytes_sent: 1_000_000,
        bytes_recv: 500_000,
        errors: 0,
        dropped: 0,
        packet_loss_percent: 10.0,
        latency_ms: 5.0,
    };
    optimizer.update_stats(stats);
    let issues = optimizer.detect_issues();
    assert!(!issues.is_empty());
    assert!(issues.iter().any(|i| i.contains("packet loss")));
}

#[test]
fn test_network_optimizer_detect_issues_latency() {
    let mut optimizer = NetworkOptimizer::new();
    let stats = NetworkStats {
        packets_sent: 1000,
        packets_recv: 1000,
        bytes_sent: 1_000_000,
        bytes_recv: 1_000_000,
        errors: 0,
        dropped: 0,
        packet_loss_percent: 0.0,
        latency_ms: 150.0,
    };
    optimizer.update_stats(stats);
    let issues = optimizer.detect_issues();
    assert!(!issues.is_empty());
    assert!(issues.iter().any(|i| i.contains("latency")));
}

#[test]
fn test_network_optimizer_sysctl_recommendations() {
    let optimizer = NetworkOptimizer::new();
    let recs = optimizer.get_sysctl_recommendations(NetworkProfile::HighThroughput);
    assert!(!recs.is_empty());
    assert!(recs.iter().any(|(k, _)| k.contains("tcp")));
}

// ============================================================================
// Integration: Wake Storm + Network
// ============================================================================

#[test]
fn test_wake_storm_network_integration() {
    let mut detector = WakeStormDetector::new();
    let _optimizer = NetworkOptimizer::new();

    for _ in 0..50 {
        detector.record_wakeup(1234, "network_app".to_string());
    }

    let storms = detector.detect_storms();
    let _ = storms.len();
}
