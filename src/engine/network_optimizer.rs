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
    HighThroughput,  // Maximize bandwidth (video streaming, downloads)
    LowLatency,      // Minimize delay (interactive apps, gaming)
    Balanced,        // General purpose
    Battery,         // Minimize power consumption
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
                tcp_send_buffer: 65_536,    // 64KB — small buffers reduce latency
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
                tcp_send_buffer: 262_144,   // 256KB — smaller buffers, less wake
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

    /// Get macOS sysctl recommendations for network tuning
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
            (
                "net.inet.tcp.mssdflt".to_string(),
                opt.tcp_mss_default.to_string(),
            ),
        ]
    }
}

impl Default for NetworkOptimizer {
    fn default() -> Self {
        Self::new()
    }
}
