//! TCP statistics collector for macOS
//!
//! Parses `netstat -s -p tcp` output to track retransmissions, listen-queue
//! drops, resets, and throughput.  Maintains a 120-entry ring buffer of
//! per-tick deltas and exposes EMA-smoothed rates for use by the sysctl
//! governor.

use std::collections::VecDeque;
use std::process::Command;
use std::time::{Duration, Instant};

// ── Public types ─────────────────────────────────────────────────────────────

/// Per-tick delta TCP statistics (computed from consecutive `netstat` snapshots).
#[derive(Debug, Clone)]
pub struct TcpStats {
    /// TCP segments retransmitted since last tick.
    pub retransmissions: u64,
    /// Listen-queue overflows since last tick.
    pub listen_drops: u64,
    /// Connection resets received since last tick.
    pub resets: u64,
    /// Total segments (packets) sent since last tick.
    pub segments_sent: u64,
    /// Total segments (packets) received since last tick.
    pub segments_recv: u64,
    /// Active + passive connections opened since last tick.
    pub connections: u64,
    /// Data bytes sent since last tick.
    pub bytes_sent: u64,
    /// Data bytes received since last tick.
    pub bytes_recv: u64,
    /// Wall-clock duration since previous tick.
    pub elapsed: std::time::Duration,
}

impl Default for TcpStats {
    fn default() -> Self {
        Self {
            retransmissions: 0,
            listen_drops: 0,
            resets: 0,
            segments_sent: 0,
            segments_recv: 0,
            connections: 0,
            bytes_sent: 0,
            bytes_recv: 0,
            elapsed: std::time::Duration::from_secs(1),
        }
    }
}

// ── Raw cumulative counters (internal) ───────────────────────────────────────

/// Cumulative counters straight from `netstat -s -p tcp`.
#[derive(Debug, Clone)]
struct RawTcpCounters {
    packets_sent: u64,
    data_bytes_sent: u64,
    retransmitted_packets: u64,
    packets_received: u64,
    data_bytes_received: u64,
    resets_received: u64,
    listen_queue_overflows: u64,
    connections_opened: u64,
    timestamp: Instant,
}

impl Default for RawTcpCounters {
    fn default() -> Self {
        Self {
            packets_sent: 0,
            data_bytes_sent: 0,
            retransmitted_packets: 0,
            packets_received: 0,
            data_bytes_received: 0,
            resets_received: 0,
            listen_queue_overflows: 0,
            connections_opened: 0,
            timestamp: Instant::now(),
        }
    }
}

// ── NetworkMonitor ───────────────────────────────────────────────────────────

/// Collects TCP statistics from the macOS `netstat` utility and maintains
/// EMA-smoothed rates for the sysctl governor.
pub struct NetworkMonitor {
    /// Ring buffer of per-tick deltas (max 120 entries, ~2 min at 1 Hz).
    pub(crate) history: VecDeque<TcpStats>,
    /// Previous raw snapshot for computing deltas.
    prev_raw: Option<RawTcpCounters>,
    /// EMA-smoothed retransmission rate (retransmissions per 1000 segments).
    pub(crate) ema_retransmission_rate: f64,
    /// EMA-smoothed listen-queue drop rate (drops per second).
    pub(crate) ema_listen_drop_rate: f64,
    /// EMA time constant in seconds.  Controls the smoothing half-life
    /// independently of the tick period via `alpha = 1 - exp(-elapsed / tau)`.
    tau: f64,
    /// Maximum ring-buffer size.
    max_history: usize,
    /// True until the first delta has been computed.  Used to seed the EMA
    /// with the actual first measurement instead of blending with 0.0.
    first_sample: bool,
}

impl NetworkMonitor {
    /// Create a new monitor with default settings.
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
            prev_raw: None,
            ema_retransmission_rate: 0.0,
            ema_listen_drop_rate: 0.0,
            tau: 60.0,
            max_history: 120,
            first_sample: true,
        }
    }

    /// Run one collection cycle: parse `netstat`, compute deltas, update EMAs.
    ///
    /// Returns the per-tick delta stats.  On the very first call the deltas
    /// will be zero because there is no previous snapshot to diff against.
    pub fn tick(&mut self) -> TcpStats {
        let raw = parse_netstat();

        let stats = if let Some(prev) = &self.prev_raw {
            // Use checked_duration_since to handle clock going backwards
            // during NTP adjustments or sleep/wake transitions.
            let elapsed = raw
                .timestamp
                .checked_duration_since(prev.timestamp)
                .unwrap_or(Duration::from_secs(1));
            let elapsed_secs = elapsed.as_secs_f64().max(0.001);

            let delta = TcpStats {
                retransmissions: raw
                    .retransmitted_packets
                    .saturating_sub(prev.retransmitted_packets),
                listen_drops: raw
                    .listen_queue_overflows
                    .saturating_sub(prev.listen_queue_overflows),
                resets: raw.resets_received.saturating_sub(prev.resets_received),
                segments_sent: raw.packets_sent.saturating_sub(prev.packets_sent),
                segments_recv: raw.packets_received.saturating_sub(prev.packets_received),
                connections: raw
                    .connections_opened
                    .saturating_sub(prev.connections_opened),
                bytes_sent: raw.data_bytes_sent.saturating_sub(prev.data_bytes_sent),
                bytes_recv: raw
                    .data_bytes_received
                    .saturating_sub(prev.data_bytes_received),
                elapsed,
            };

            // Retransmission rate: retransmissions per 1000 segments sent.
            // Guard against NaN/Infinity which would permanently contaminate the EMA.
            let instant_retx_rate = if delta.segments_sent > 0 {
                let rate = (delta.retransmissions as f64 / delta.segments_sent as f64) * 1000.0;
                if rate.is_finite() {
                    rate
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Listen-drop rate: drops per second.
            let raw_drop_rate = delta.listen_drops as f64 / elapsed_secs;
            let instant_drop_rate = if raw_drop_rate.is_finite() {
                raw_drop_rate
            } else {
                0.0
            };

            // Compute a dynamic smoothing factor that normalises the EMA
            // response regardless of the tick period.  With a fixed alpha the
            // half-life measured in *wall-clock* time changes when the daemon
            // switches between fast-tick (5 s) and normal-tick (30 s).
            //
            //   alpha = 1 - exp(-elapsed / tau)
            //
            // With tau = 60 s:
            //   dt =  5 s  =>  alpha ≈ 0.080  (gentle, as expected in fast-tick)
            //   dt = 30 s  =>  alpha ≈ 0.393  (stronger, compensating for the gap)
            let dynamic_alpha = 1.0 - (-elapsed_secs / self.tau).exp();
            let alpha = dynamic_alpha.clamp(0.01, 0.95);

            // On the first delta, seed the EMA directly instead of blending
            // with the initial 0.0 which causes a cold-start undercount.
            if self.first_sample {
                self.ema_retransmission_rate = instant_retx_rate;
                self.ema_listen_drop_rate = instant_drop_rate;
                self.first_sample = false;
            } else {
                self.ema_retransmission_rate =
                    alpha * instant_retx_rate + (1.0 - alpha) * self.ema_retransmission_rate;
                self.ema_listen_drop_rate =
                    alpha * instant_drop_rate + (1.0 - alpha) * self.ema_listen_drop_rate;
            }

            delta
        } else {
            // First tick: no previous data, return zeroes.
            TcpStats::default()
        };

        self.prev_raw = Some(raw);

        self.history.push_back(stats.clone());
        if self.history.len() > self.max_history {
            let _ = self.history.pop_front();
        }

        stats
    }

    /// EMA-smoothed retransmission rate (retransmissions per 1000 segments sent).
    pub fn retransmission_rate(&self) -> f64 {
        self.ema_retransmission_rate
    }

    /// EMA-smoothed listen-queue drop rate (drops per second).
    pub fn listen_drop_rate(&self) -> f64 {
        self.ema_listen_drop_rate
    }

    /// Estimated send and receive throughput in bytes per second, averaged
    /// over the most recent 10 ticks.
    pub fn throughput_bps(&self) -> (u64, u64) {
        let window: Vec<_> = self.history.iter().rev().take(10).collect();
        if window.is_empty() {
            return (0, 0);
        }

        let total_elapsed: f64 = window.iter().map(|s| s.elapsed.as_secs_f64()).sum();
        if total_elapsed < 0.001 {
            return (0, 0);
        }

        let total_sent: u64 = window.iter().map(|s| s.bytes_sent).sum();
        let total_recv: u64 = window.iter().map(|s| s.bytes_recv).sum();

        let send_bps = (total_sent as f64 / total_elapsed) as u64;
        let recv_bps = (total_recv as f64 / total_elapsed) as u64;

        (send_bps, recv_bps)
    }

    /// Number of collected samples.
    pub fn sample_count(&self) -> usize {
        self.history.len()
    }
}

impl Default for NetworkMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ── netstat parser ───────────────────────────────────────────────────────────

/// Parse `netstat -s -p tcp` on macOS into cumulative counters.
///
/// The output contains lines like:
/// ```text
///     12345 packets sent
///     67890 data packets (12345678 bytes)
///     100 data packets (50000 bytes) retransmitted
///     54321 packets received
///     5 connection resets received
///     2 listen queue overflows
///     99 connections established (including accepts)
/// ```
///
/// We use simple `contains` + `split` matching -- no regex needed.
fn parse_netstat() -> RawTcpCounters {
    let mut counters = RawTcpCounters {
        timestamp: Instant::now(),
        ..Default::default()
    };

    let output = match Command::new("/usr/sbin/netstat")
        .args(["-s", "-p", "tcp"])
        .output()
    {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).to_string(),
        _ => return counters,
    };

    for line in output.lines() {
        let trimmed = line.trim();

        // Order matters: more specific patterns first to avoid false matches.

        // "100 data packets (50000 bytes) retransmitted" or singular "1 data packet ..."
        if trimmed.contains("retransmitted")
            && (trimmed.contains("data packets") || trimmed.contains("data packet"))
        {
            if let Some(n) = first_number(trimmed) {
                counters.retransmitted_packets = n;
            }
            continue;
        }

        // "67890 data packets (12345678 bytes)" — sent data with byte count
        // Must come after the retransmitted check.
        // Handle singular "data packet" and "byte" forms from macOS netstat.
        // The "byte"/"bytes" check prevents false matches with
        // "data packet sent after flow control" which has no byte count.
        if (trimmed.contains("data packets") || trimmed.contains("data packet"))
            && (trimmed.contains("bytes") || trimmed.contains("byte"))
            && !trimmed.contains("received")
        {
            if let Some(bytes) = extract_parenthesized_bytes(trimmed) {
                counters.data_bytes_sent = bytes;
            }
            continue;
        }

        // "12345 packets sent" or "1 packet sent"
        if (trimmed.contains("packets sent") || trimmed.contains("packet sent"))
            && !trimmed.contains("data")
        {
            if let Some(n) = first_number(trimmed) {
                counters.packets_sent = n;
            }
            continue;
        }

        // Data bytes received: "NNNNN data packets (MMMMM bytes)" under the received section.
        // macOS netstat groups sent and received data; the received line often reads:
        //   "12345 data packets (67890 bytes)"
        // We need a secondary pass or rely on ordering.  For robustness we also
        // check for "received in-sequence" or simply parse "packets received" for
        // segment count.  Byte-level recv is handled below.
        if trimmed.contains("received in-sequence")
            && (trimmed.contains("bytes") || trimmed.contains("byte"))
        {
            if let Some(bytes) = extract_parenthesized_bytes(trimmed) {
                counters.data_bytes_received = bytes;
            }
            continue;
        }

        // "54321 packets received" or "1 packet received"
        if trimmed.contains("packets received") || trimmed.contains("packet received") {
            if let Some(n) = first_number(trimmed) {
                counters.packets_received = n;
            }
            continue;
        }

        // "5 connection resets received" or "5 resets" or "5 bad resets"
        if trimmed.contains("connection reset")
            || trimmed.contains("resets received")
            || trimmed.contains("bad reset")
        {
            if let Some(n) = first_number(trimmed) {
                counters.resets_received = n;
            }
            continue;
        }

        // "2 listen queue overflows"
        if trimmed.contains("listen queue overflow") {
            if let Some(n) = first_number(trimmed) {
                counters.listen_queue_overflows = n;
            }
            continue;
        }

        // "99 connections established (including accepts)" or "1 connection established"
        if trimmed.contains("connections established") || trimmed.contains("connection established")
        {
            if let Some(n) = first_number(trimmed) {
                counters.connections_opened = n;
            }
            continue;
        }
    }

    counters
}

/// Extract the first decimal number from a string.
fn first_number(s: &str) -> Option<u64> {
    s.split_whitespace()
        .find_map(|word| word.parse::<u64>().ok())
}

/// Extract the number inside parentheses from a pattern like "(12345 bytes)".
/// Uses `"byte"` which matches both the singular "byte" and plural "bytes".
fn extract_parenthesized_bytes(s: &str) -> Option<u64> {
    let start = s.find('(')? + 1;
    let end = s.find("byte")?;
    let inner = s.get(start..end)?.trim();
    inner.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_first_number() {
        assert_eq!(first_number("  12345 packets sent"), Some(12345));
        assert_eq!(first_number("  0 listen queue overflows"), Some(0));
        assert_eq!(first_number("no numbers here"), None);
    }

    #[test]
    fn parse_parenthesized_bytes() {
        assert_eq!(
            extract_parenthesized_bytes("67890 data packets (12345678 bytes)"),
            Some(12345678)
        );
        assert_eq!(
            extract_parenthesized_bytes("100 data packets (50000 bytes) retransmitted"),
            Some(50000)
        );
        assert_eq!(extract_parenthesized_bytes("no parens here"), None);
    }

    #[test]
    fn default_tcp_stats() {
        let stats = TcpStats::default();
        assert_eq!(stats.retransmissions, 0);
        assert_eq!(stats.segments_sent, 0);
        assert_eq!(stats.bytes_sent, 0);
    }

    #[test]
    fn network_monitor_initial_state() {
        let monitor = NetworkMonitor::new();
        assert_eq!(monitor.sample_count(), 0);
        assert_eq!(monitor.retransmission_rate(), 0.0);
        assert_eq!(monitor.listen_drop_rate(), 0.0);
        assert_eq!(monitor.throughput_bps(), (0, 0));
    }

    #[test]
    fn network_monitor_first_tick_returns_zeroes() {
        let mut monitor = NetworkMonitor::new();
        let stats = monitor.tick();
        // First tick has no previous snapshot, so deltas are zero.
        assert_eq!(stats.retransmissions, 0);
        assert_eq!(stats.segments_sent, 0);
        assert_eq!(monitor.sample_count(), 1);
    }

    #[test]
    fn ema_converges_toward_zero_with_zero_input() {
        let mut monitor = NetworkMonitor::new();
        // Seed with a non-zero EMA to verify convergence.
        monitor.ema_retransmission_rate = 10.0;
        monitor.ema_listen_drop_rate = 5.0;

        // Simulate zero-rate ticks using the dynamic alpha formula.
        // Each tick is 30 seconds apart (tau = 60 s by default).
        // alpha = 1 - exp(-30/60) ≈ 0.393, so after 20 iterations:
        // (1 - 0.393)^20 ≈ 5e-6 — well below the assertion thresholds.
        let dt = 30.0_f64;
        for _ in 0..20 {
            let alpha = (1.0 - (-dt / monitor.tau).exp()).clamp(0.01, 0.95);
            monitor.ema_retransmission_rate =
                alpha * 0.0 + (1.0 - alpha) * monitor.ema_retransmission_rate;
            monitor.ema_listen_drop_rate =
                alpha * 0.0 + (1.0 - alpha) * monitor.ema_listen_drop_rate;
        }

        assert!(monitor.ema_retransmission_rate < 0.1);
        assert!(monitor.ema_listen_drop_rate < 0.05);
    }
}
