//! I/O profiling and optimization
//!
//! Monitors disk read/write patterns and provides optimization hints.

use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone)]
pub struct IOStats {
    pub reads_per_sec: f64,
    pub writes_per_sec: f64,
    pub bytes_read_per_sec: f64,
    pub bytes_written_per_sec: f64,
    pub avg_read_latency_ms: f64,
    pub avg_write_latency_ms: f64,
    pub sequential_ratio: f64,  // 0.0-1.0
    pub io_util_percent: f64,   // 0-100
}

impl Default for IOStats {
    fn default() -> Self {
        Self {
            reads_per_sec: 0.0,
            writes_per_sec: 0.0,
            bytes_read_per_sec: 0.0,
            bytes_written_per_sec: 0.0,
            avg_read_latency_ms: 0.0,
            avg_write_latency_ms: 0.0,
            sequential_ratio: 0.5,
            io_util_percent: 0.0,
        }
    }
}

pub struct IOProfiler {
    prev_read_count: u64,
    prev_write_count: u64,
    prev_bytes_read: u64,
    prev_bytes_written: u64,
    process_io_stats: HashMap<u32, ProcessIOStats>,
}

#[derive(Debug, Clone)]
pub struct ProcessIOStats {
    pub pid: u32,
    pub name: String,
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub read_ops: u64,
    pub write_ops: u64,
    pub io_wait_time_ms: u64,
}

impl IOProfiler {
    pub fn new() -> Self {
        Self {
            prev_read_count: 0,
            prev_write_count: 0,
            prev_bytes_read: 0,
            prev_bytes_written: 0,
            process_io_stats: HashMap::new(),
        }
    }

    /// Profile system I/O from /proc/diskstats (macOS uses different APIs)
    pub fn profile_system_io(&mut self) -> IOStats {
        // On macOS, read from /proc/diskstats if available (Linux VMs) or use defaults
        if let Ok(diskstats) = fs::read_to_string("/proc/diskstats") {
            self.parse_diskstats(&diskstats)
        } else {
            // Fallback: estimate from iostat or use defaults
            self.estimate_from_iostat()
        }
    }

    fn parse_diskstats(&mut self, content: &str) -> IOStats {
        let mut total_reads = 0u64;
        let mut total_writes = 0u64;
        let mut total_bytes_read = 0u64;
        let mut total_bytes_written = 0u64;
        let mut total_read_time = 0u64;
        let mut total_write_time = 0u64;

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 14 {
                continue;
            }

            // diskstats format: ... reads_completed bytes_read reads_time writes_completed bytes_written writes_time ...
            if let (Ok(reads), Ok(read_bytes), Ok(read_time), Ok(writes), Ok(write_bytes), Ok(write_time)) = (
                parts[3].parse::<u64>(),
                parts[5].parse::<u64>(),
                parts[6].parse::<u64>(),
                parts[7].parse::<u64>(),
                parts[9].parse::<u64>(),
                parts[10].parse::<u64>(),
            ) {
                total_reads += reads;
                total_bytes_read += read_bytes;
                total_read_time += read_time;
                total_writes += writes;
                total_bytes_written += write_bytes;
                total_write_time += write_time;
            }
        }

        let read_delta = total_reads.saturating_sub(self.prev_read_count);
        let write_delta = total_writes.saturating_sub(self.prev_write_count);
        let bytes_read_delta = total_bytes_read.saturating_sub(self.prev_bytes_read);
        let bytes_write_delta = total_bytes_written.saturating_sub(self.prev_bytes_written);

        self.prev_read_count = total_reads;
        self.prev_write_count = total_writes;
        self.prev_bytes_read = total_bytes_read;
        self.prev_bytes_written = total_bytes_written;

        IOStats {
            reads_per_sec: read_delta as f64,
            writes_per_sec: write_delta as f64,
            bytes_read_per_sec: bytes_read_delta as f64 / 1024.0 / 1024.0, // MB/s
            bytes_written_per_sec: bytes_write_delta as f64 / 1024.0 / 1024.0,
            avg_read_latency_ms: if read_delta > 0 {
                total_read_time as f64 / read_delta as f64
            } else {
                0.0
            },
            avg_write_latency_ms: if write_delta > 0 {
                total_write_time as f64 / write_delta as f64
            } else {
                0.0
            },
            sequential_ratio: 0.5, // Placeholder: would need more data
            io_util_percent: ((read_delta + write_delta) as f64 / 10000.0).min(100.0),
        }
    }

    fn estimate_from_iostat(&self) -> IOStats {
        // Fallback estimation for macOS
        IOStats::default()
    }

    /// Check for I/O bottlenecks
    pub fn detect_io_bottlenecks(stats: &IOStats) -> Vec<String> {
        let mut issues = Vec::new();

        if stats.io_util_percent > 80.0 {
            issues.push(format!("🔴 High I/O utilization: {:.1}%", stats.io_util_percent));
        }

        if stats.avg_read_latency_ms > 10.0 {
            issues.push(format!("🟡 High read latency: {:.1}ms", stats.avg_read_latency_ms));
        }

        if stats.avg_write_latency_ms > 10.0 {
            issues.push(format!("🟡 High write latency: {:.1}ms", stats.avg_write_latency_ms));
        }

        if stats.bytes_read_per_sec > 500.0 {
            issues.push(format!("🟡 High read throughput: {:.0}MB/s", stats.bytes_read_per_sec));
        }

        issues
    }

    /// Get top I/O processes (mock - would need /proc/[pid]/io)
    pub fn top_io_processes(&self, limit: usize) -> Vec<ProcessIOStats> {
        let mut processes: Vec<_> = self.process_io_stats.values().cloned().collect();
        processes.sort_by(|a, b| {
            (b.read_bytes + b.write_bytes).cmp(&(a.read_bytes + a.write_bytes))
        });
        processes.truncate(limit);
        processes
    }
}
