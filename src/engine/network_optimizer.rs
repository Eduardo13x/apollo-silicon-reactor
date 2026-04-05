//! Network I/O Optimization for macOS
//!
//! Optimizes network parameters using macOS-specific sysctl names.

#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub packets_sent: u64,
    pub packets_recv: u64,
    pub bytes_sent: u64,
    pub bytes_recv: u64,
    pub errors: u64,
    pub dropped: u64,
    pub packet_loss_percent: f32,
    pub latency_ms: f32,
}

impl Default for NetworkStats {
    fn default() -> Self {
        Self {
            packets_sent: 0,
            packets_recv: 0,
            bytes_sent: 0,
            bytes_recv: 0,
            errors: 0,
            dropped: 0,
            packet_loss_percent: 0.0,
            latency_ms: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkProfile {
    HighThroughput, // Maximize bandwidth (video streaming, downloads)
    LowLatency,     // Minimize delay (interactive apps, gaming)
    Balanced,       // General purpose
    Battery,        // Minimize power consumption
}

/// macOS-appropriate network optimization settings.
///
/// macOS uses single-value sysctls (`net.inet.tcp.sendspace`, `net.inet.tcp.recvspace`)
/// rather than Linux min/default/max tuples.
#[derive(Debug, Clone)]
pub struct NetworkOptimization {
    pub profile: NetworkProfile,
    /// net.inet.tcp.sendspace — per-socket send buffer size
    pub tcp_send_buffer: u32,
    /// net.inet.tcp.recvspace — per-socket receive buffer size
    pub tcp_recv_buffer: u32,
    /// net.inet.tcp.win_scale_factor — window scaling (0 = disabled)
    pub tcp_window_scale: u32,
    /// net.inet.tcp.delayed_ack — 0 = no delay (low latency), 3 = combined (throughput)
    pub tcp_delayed_ack: u32,
    /// net.inet.tcp.mssdflt — default MSS
    pub tcp_mss_default: u32,
}

pub struct NetworkOptimizer {
    stats: NetworkStats,
}

impl NetworkOptimizer {
    pub fn new() -> Self {
        Self {
            stats: NetworkStats::default(),
        }
    }

    /// Get optimization settings for a profile
    pub fn get_optimization(&self, profile: NetworkProfile) -> NetworkOptimization {
        match profile {
            NetworkProfile::HighThroughput => NetworkOptimization {
                profile,
                tcp_send_buffer: 4_194_304, // 4MB
                tcp_recv_buffer: 4_194_304,
                tcp_window_scale: 8,
                tcp_delayed_ack: 3, // Combine ACKs for throughput
                tcp_mss_default: 1460,
            },
            NetworkProfile::LowLatency => NetworkOptimization {
                profile,
                tcp_send_buffer: 65_536, // 64KB — small buffers reduce latency
                tcp_recv_buffer: 65_536,
                tcp_window_scale: 4,
                tcp_delayed_ack: 0, // No delayed ACKs
                tcp_mss_default: 1460,
            },
            NetworkProfile::Balanced => NetworkOptimization {
                profile,
                tcp_send_buffer: 1_048_576, // 1MB
                tcp_recv_buffer: 1_048_576,
                tcp_window_scale: 6,
                tcp_delayed_ack: 1,
                tcp_mss_default: 1460,
            },
            NetworkProfile::Battery => NetworkOptimization {
                profile,
                tcp_send_buffer: 262_144, // 256KB — smaller buffers, less wake
                tcp_recv_buffer: 262_144,
                tcp_window_scale: 4,
                tcp_delayed_ack: 3, // Combine ACKs to reduce CPU wake
                tcp_mss_default: 1460,
            },
        }
    }

    /// Update network statistics
    pub fn update_stats(&mut self, stats: NetworkStats) {
        self.stats = stats;
    }

    /// Recommend profile based on current network characteristics
    pub fn recommend_profile(&self) -> NetworkProfile {
        if self.stats.latency_ms > 50.0 {
            if self.stats.packet_loss_percent > 1.0 {
                NetworkProfile::LowLatency
            } else {
                NetworkProfile::HighThroughput
            }
        } else if self.stats.latency_ms > 20.0 {
            NetworkProfile::LowLatency
        } else if self.stats.bytes_recv + self.stats.bytes_sent > 1_000_000_000 {
            NetworkProfile::HighThroughput
        } else {
            NetworkProfile::Balanced
        }
    }

    /// Detect network issues
    pub fn detect_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.stats.packet_loss_percent > 5.0 {
            issues.push(format!(
                "High packet loss: {:.2}%",
                self.stats.packet_loss_percent
            ));
        }

        if self.stats.latency_ms > 100.0 {
            issues.push(format!("High latency: {:.1}ms", self.stats.latency_ms));
        }

        if self.stats.errors > 1000 {
            issues.push(format!("Network errors: {}", self.stats.errors));
        }

        if self.stats.dropped > 100 {
            issues.push(format!("Dropped packets: {}", self.stats.dropped));
        }

        issues
    }

    /// Get macOS sysctl recommendations for network tuning.
    /// Only emits keys on the safety allowlist.
    pub fn get_sysctl_recommendations(&self, profile: NetworkProfile) -> Vec<(String, String)> {
        let opt = self.get_optimization(profile);
        vec![
            (
                "net.inet.tcp.sendspace".to_string(),
                opt.tcp_send_buffer.to_string(),
            ),
            (
                "net.inet.tcp.recvspace".to_string(),
                opt.tcp_recv_buffer.to_string(),
            ),
            (
                "net.inet.tcp.win_scale_factor".to_string(),
                opt.tcp_window_scale.to_string(),
            ),
            (
                "net.inet.tcp.delayed_ack".to_string(),
                opt.tcp_delayed_ack.to_string(),
            ),
        ]
    }
}

impl Default for NetworkOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_optimizer(latency_ms: f32, packet_loss: f32, bytes: u64) -> NetworkOptimizer {
        let mut opt = NetworkOptimizer::new();
        opt.update_stats(NetworkStats {
            latency_ms,
            packet_loss_percent: packet_loss,
            bytes_sent: bytes / 2,
            bytes_recv: bytes / 2,
            ..NetworkStats::default()
        });
        opt
    }

    // ── NetworkStats default ────────────────────────────────────────────────

    #[test]
    fn network_stats_default_is_zeroed() {
        let s = NetworkStats::default();
        assert_eq!(s.packets_sent, 0);
        assert_eq!(s.errors, 0);
        assert_eq!(s.dropped, 0);
        assert!((s.packet_loss_percent).abs() < 1e-6);
        assert!((s.latency_ms).abs() < 1e-6);
    }

    // ── get_optimization profiles ────────────────────────────────────────────

    #[test]
    fn high_throughput_has_large_buffers() {
        let opt = NetworkOptimizer::new();
        let cfg = opt.get_optimization(NetworkProfile::HighThroughput);
        assert_eq!(cfg.tcp_send_buffer, 4_194_304);
        assert_eq!(cfg.tcp_recv_buffer, 4_194_304);
        assert_eq!(cfg.tcp_delayed_ack, 3);
    }

    #[test]
    fn low_latency_has_small_buffers_no_delayed_ack() {
        let opt = NetworkOptimizer::new();
        let cfg = opt.get_optimization(NetworkProfile::LowLatency);
        assert_eq!(cfg.tcp_send_buffer, 65_536);
        assert_eq!(cfg.tcp_delayed_ack, 0, "LowLatency must disable delayed ACK");
    }

    #[test]
    fn balanced_is_between_high_and_low() {
        let opt = NetworkOptimizer::new();
        let hi = opt.get_optimization(NetworkProfile::HighThroughput);
        let lo = opt.get_optimization(NetworkProfile::LowLatency);
        let bal = opt.get_optimization(NetworkProfile::Balanced);
        assert!(bal.tcp_send_buffer > lo.tcp_send_buffer);
        assert!(bal.tcp_send_buffer < hi.tcp_send_buffer);
    }

    #[test]
    fn battery_profile_has_combined_ack() {
        let opt = NetworkOptimizer::new();
        let cfg = opt.get_optimization(NetworkProfile::Battery);
        assert_eq!(cfg.tcp_delayed_ack, 3, "Battery combines ACKs to reduce CPU wakeups");
        assert!(cfg.tcp_send_buffer < 1_048_576, "Battery uses smaller buffers");
    }

    // ── recommend_profile ────────────────────────────────────────────────────

    #[test]
    fn recommend_high_throughput_for_high_latency_no_loss() {
        let opt = make_optimizer(60.0, 0.5, 0);
        assert_eq!(opt.recommend_profile(), NetworkProfile::HighThroughput);
    }

    #[test]
    fn recommend_low_latency_for_high_latency_with_loss() {
        let opt = make_optimizer(60.0, 2.0, 0);
        assert_eq!(opt.recommend_profile(), NetworkProfile::LowLatency);
    }

    #[test]
    fn recommend_low_latency_for_moderate_latency() {
        let opt = make_optimizer(30.0, 0.0, 0);
        assert_eq!(opt.recommend_profile(), NetworkProfile::LowLatency);
    }

    #[test]
    fn recommend_high_throughput_for_large_data_low_latency() {
        let opt = make_optimizer(10.0, 0.0, 2_000_000_000);
        assert_eq!(opt.recommend_profile(), NetworkProfile::HighThroughput);
    }

    #[test]
    fn recommend_balanced_for_idle_low_latency() {
        let opt = make_optimizer(5.0, 0.0, 0);
        assert_eq!(opt.recommend_profile(), NetworkProfile::Balanced);
    }

    // ── detect_issues ────────────────────────────────────────────────────────

    #[test]
    fn no_issues_for_clean_network() {
        let opt = make_optimizer(5.0, 0.5, 0);
        assert!(opt.detect_issues().is_empty());
    }

    #[test]
    fn detects_high_packet_loss() {
        let opt = make_optimizer(0.0, 6.0, 0);
        let issues = opt.detect_issues();
        assert!(issues.iter().any(|s| s.contains("packet loss")));
    }

    #[test]
    fn detects_high_latency() {
        let opt = make_optimizer(150.0, 0.0, 0);
        let issues = opt.detect_issues();
        assert!(issues.iter().any(|s| s.contains("latency")));
    }

    #[test]
    fn detects_errors_and_drops() {
        let mut optimizer = NetworkOptimizer::new();
        optimizer.update_stats(NetworkStats {
            errors: 2000,
            dropped: 200,
            ..NetworkStats::default()
        });
        let issues = optimizer.detect_issues();
        assert!(issues.iter().any(|s| s.contains("errors")));
        assert!(issues.iter().any(|s| s.contains("Dropped")));
    }

    // ── get_sysctl_recommendations ───────────────────────────────────────────

    #[test]
    fn sysctl_recommendations_contain_expected_keys() {
        let opt = NetworkOptimizer::new();
        let recs = opt.get_sysctl_recommendations(NetworkProfile::Balanced);
        let keys: Vec<&str> = recs.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"net.inet.tcp.sendspace"));
        assert!(keys.contains(&"net.inet.tcp.recvspace"));
        assert!(keys.contains(&"net.inet.tcp.delayed_ack"));
        assert!(keys.contains(&"net.inet.tcp.win_scale_factor"));
    }

    #[test]
    fn sysctl_values_match_profile_settings() {
        let opt = NetworkOptimizer::new();
        let recs = opt.get_sysctl_recommendations(NetworkProfile::HighThroughput);
        let sendspace = recs.iter().find(|(k, _)| k == "net.inet.tcp.sendspace")
            .map(|(_, v)| v.parse::<u32>().unwrap());
        assert_eq!(sendspace, Some(4_194_304));
    }

    // ── Default impl ─────────────────────────────────────────────────────────

    #[test]
    fn network_optimizer_default_same_as_new() {
        let d = NetworkOptimizer::default();
        let n = NetworkOptimizer::new();
        // Both should recommend Balanced with zeroed stats.
        assert_eq!(d.recommend_profile(), n.recommend_profile());
        assert_eq!(d.recommend_profile(), NetworkProfile::Balanced);
    }
}
