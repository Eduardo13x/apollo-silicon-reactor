//! Background pressure collectors — cached system pressure data.
//!
//! Moves blocking subprocesses (`memory_pressure -Q`, `sysctl vm.swapusage`)
//! out of the main daemon loop into a dedicated background thread that polls
//! at a configurable interval.  The main loop reads cached data in <1 μs.

use crate::engine::host_vm_info;
use crate::engine::sysctl_direct;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::engine::lock_ext::LockRecover;

/// Cached memory/swap pressure data.
#[derive(Debug, Clone)]
pub struct PressureData {
    /// Memory pressure ratio 0.0–1.0 (1.0 = fully pressured).
    pub memory_pressure: f64,
    /// Swap bytes currently in use.
    pub swap_used_bytes: u64,
    /// Total swap capacity.
    pub swap_total_bytes: u64,
    /// Swap growth rate in bytes/sec (positive = growing).
    pub swap_delta_bps: f64,
    /// When this data was last refreshed.
    pub updated_at: Instant,
}

impl Default for PressureData {
    fn default() -> Self {
        Self {
            memory_pressure: 0.0,
            swap_used_bytes: 0,
            swap_total_bytes: 0,
            swap_delta_bps: 0.0,
            updated_at: Instant::now(),
        }
    }
}

/// Background thread that polls memory pressure and swap usage.
pub struct PressureCollector {
    cache: Arc<Mutex<PressureData>>,
    /// Heartbeat: epoch millis of the last successful collection.
    heartbeat: Arc<AtomicU64>,
}

impl PressureCollector {
    /// Spawn a background thread that polls pressure data every `interval`.
    ///
    /// The thread runs until the process exits.
    pub fn spawn(interval: Duration) -> Self {
        let cache = Arc::new(Mutex::new(PressureData::default()));
        let heartbeat = Arc::new(AtomicU64::new(0));
        let c = cache.clone();
        let hb = heartbeat.clone();

        if let Err(e) = thread::Builder::new()
            .name("pressure-collector".into())
            .spawn(move || {
                let mut prev_swap_used: Option<u64> = None;
                let mut prev_swap_at: Option<Instant> = None;

                loop {
                    let (mem_pressure, swap_used, swap_total) = collect_pressure_facts();
                    let now = Instant::now();
                    let swap_delta = match (prev_swap_used, prev_swap_at) {
                        (Some(prev), Some(at)) => {
                            let dt = now.duration_since(at).as_secs_f64().max(0.001);
                            (swap_used.saturating_sub(prev) as f64) / dt
                        }
                        _ => 0.0,
                    };
                    prev_swap_used = Some(swap_used);
                    prev_swap_at = Some(now);

                    *c.lock_recover() = PressureData {
                        memory_pressure: mem_pressure,
                        swap_used_bytes: swap_used,
                        swap_total_bytes: swap_total,
                        swap_delta_bps: swap_delta,
                        updated_at: now,
                    };

                    // Update heartbeat after successful collection.
                    hb.store(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64,
                        Ordering::Release,
                    );

                    thread::sleep(interval);
                }
            })
        {
            eprintln!("warning: failed to spawn pressure-collector: {}", e);
        }

        Self { cache, heartbeat }
    }

    /// Get the latest cached pressure data (<1 μs).
    pub fn latest(&self) -> PressureData {
        self.cache.lock_recover().clone()
    }

    /// Age of the cached data.
    pub fn data_age(&self) -> Duration {
        self.cache.lock_recover().updated_at.elapsed()
    }

    /// Get a clone of the inner Arc for sharing with other threads.
    pub fn cache_arc(&self) -> Arc<Mutex<PressureData>> {
        self.cache.clone()
    }

    /// Returns `true` if the background thread has updated within `max_stale_secs`.
    ///
    /// Returns `true` if the thread has not started yet (heartbeat == 0),
    /// since the thread may simply be in its first collection cycle.
    pub fn is_alive(&self, max_stale_secs: u64) -> bool {
        let hb = self.heartbeat.load(Ordering::Acquire);
        if hb == 0 {
            return true; // Not yet started
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now.saturating_sub(hb) < max_stale_secs * 1000
    }
}

fn collect_pressure_facts() -> (f64, u64, u64) {
    // Memory pressure via Mach host_statistics64 (~1µs vs 50ms for subprocess).
    let memory_pressure = host_vm_info::read_vm_stats()
        .map(|s| s.pressure())
        .unwrap_or(0.0);

    // Swap usage via direct sysctl struct read (~1µs vs 10ms for subprocess).
    let (swap_total_bytes, swap_used_bytes) =
        sysctl_direct::read_swap_usage().unwrap_or((0, 0));

    (memory_pressure, swap_used_bytes, swap_total_bytes)
}

#[cfg(test)]
fn parse_sysctl_size(s: &str, key: &str) -> Option<u64> {
    let needle = format!("{key} =");
    let idx = s.find(&needle)?;
    let rest = s[idx + needle.len()..].trim_start();
    let mut num = String::new();
    let mut unit = None;
    for ch in rest.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            num.push(ch);
        } else if ch.is_ascii_alphabetic() {
            unit = Some(ch);
            break;
        } else if !num.is_empty() {
            break;
        }
    }
    let val = num.parse::<f64>().ok()?;
    let mul = match unit.unwrap_or('B') {
        'K' | 'k' => 1024_f64,
        'M' | 'm' => 1024_f64 * 1024_f64,
        'G' | 'g' => 1024_f64 * 1024_f64 * 1024_f64,
        _ => 1_f64,
    };
    Some((val * mul) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_data_defaults() {
        let data = PressureData::default();
        assert!((data.memory_pressure - 0.0).abs() < f64::EPSILON);
        assert_eq!(data.swap_used_bytes, 0);
        assert_eq!(data.swap_total_bytes, 0);
        assert!((data.swap_delta_bps - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_sysctl_size_megabytes() {
        let input = "vm.swapusage: total = 3072.00M  used = 2251.25M  free = 820.75M  (encrypted)";
        assert_eq!(parse_sysctl_size(input, "total"), Some(3_221_225_472));
        // 2251.25 * 1024 * 1024 = 2360606720 (f64 truncation)
        let used = parse_sysctl_size(input, "used").unwrap();
        assert!(
            (used as f64 - 2251.25 * 1024.0 * 1024.0).abs() < 1024.0,
            "used bytes {used} too far from expected"
        );
        let free = parse_sysctl_size(input, "free").unwrap();
        assert!(
            (free as f64 - 820.75 * 1024.0 * 1024.0).abs() < 1024.0,
            "free bytes {free} too far from expected"
        );
    }

    #[test]
    fn parse_sysctl_size_gigabytes() {
        let input = "vm.swapusage: total = 4.00G  used = 1.50G  free = 2.50G";
        assert_eq!(parse_sysctl_size(input, "total"), Some(4_294_967_296));
        assert_eq!(parse_sysctl_size(input, "used"), Some(1_610_612_736));
    }

    #[test]
    fn parse_sysctl_size_kilobytes() {
        let input = "vm.swapusage: total = 1024.00K  used = 512.00K  free = 512.00K";
        assert_eq!(parse_sysctl_size(input, "total"), Some(1_048_576));
        assert_eq!(parse_sysctl_size(input, "used"), Some(524_288));
    }

    #[test]
    fn parse_sysctl_size_missing_key() {
        let input = "vm.swapusage: total = 3072.00M  used = 2251.25M  free = 820.75M";
        assert_eq!(parse_sysctl_size(input, "nonexistent"), None);
    }

    #[test]
    fn parse_sysctl_size_zero() {
        let input = "vm.swapusage: total = 0.00M  used = 0.00M  free = 0.00M";
        assert_eq!(parse_sysctl_size(input, "total"), Some(0));
        assert_eq!(parse_sysctl_size(input, "used"), Some(0));
    }

    #[test]
    fn pressure_collector_spawn_and_read() {
        // Spawn a real collector — it should produce data within a few seconds.
        let collector = PressureCollector::spawn(Duration::from_millis(500));
        // Give the background thread time to complete at least one collection.
        std::thread::sleep(Duration::from_secs(2));

        let data = collector.latest();
        // memory_pressure should be between 0 and 1 on any running system.
        assert!(
            data.memory_pressure >= 0.0 && data.memory_pressure <= 1.0,
            "memory_pressure out of range: {}",
            data.memory_pressure
        );

        let age = collector.data_age();
        assert!(age < Duration::from_secs(5), "data_age too old: {:?}", age);
    }

    #[test]
    fn pressure_collector_cache_arc_is_shared() {
        let collector = PressureCollector::spawn(Duration::from_millis(500));
        let arc1 = collector.cache_arc();
        let arc2 = collector.cache_arc();
        // Both Arcs point to the same allocation.
        assert!(Arc::ptr_eq(&arc1, &arc2));
    }
}
