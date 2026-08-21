//! Background pressure collectors — cached system pressure data.
//!
//! Moves blocking subprocesses (`memory_pressure -Q`, `sysctl vm.swapusage`)
//! out of the main daemon loop into a dedicated background thread that polls
//! at a configurable interval.  The main loop reads cached data in <1 μs.

use crate::engine::cpu_saturation::{self as cpu_sat, CpuSaturation, PerCoreTicks};
use crate::engine::host_vm_info::{self, VmPageStats, VmRate};
use crate::engine::sysctl_direct;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::engine::lock_ext::LockRecover;

/// Fast enough to observe short memory-flow bursts without polling subprocesses.
pub const RESPONSIVE_PRESSURE_INTERVAL: Duration = Duration::from_millis(500);
pub const RESPONSIVE_PRESSURE_MAX_AGE: Duration = Duration::from_secs(2);

/// Cached memory/swap pressure data.
#[derive(Debug, Clone)]
pub struct PressureData {
    /// Monotonic publication generation. Consumers use this to avoid
    /// recomputing derived state on faster daemon cycles.
    pub generation: u64,
    /// Memory pressure ratio 0.0–1.0 (1.0 = fully pressured).
    pub memory_pressure: f64,
    /// Swap bytes currently in use.
    pub swap_used_bytes: u64,
    /// Total swap capacity.
    pub swap_total_bytes: u64,
    /// Swap change rate in bytes/sec (positive = growing, negative = shrinking).
    pub swap_delta_bps: f64,
    /// Native VM page size for converting page rates into comparable byte rates.
    pub page_size_bytes: u64,
    /// When this data was last refreshed.
    pub updated_at: Instant,
    /// Per-second VM flow rates derived from host_statistics64 cumulative
    /// counters. Populated by `PressureCollector` using the previous sample
    /// as baseline. Zero-filled on the very first collection (no prev).
    ///
    /// This is the "flow" view of memory pressure: pressure_percentage tells
    /// you the water level, vm_rate tells you whether water is pouring in
    /// or draining out.
    pub vm_rate: VmRate,
    /// Composite thrashing score from `VmRate::thrashing_score()`.
    /// 0 ≈ quiet, 1_000+ ≈ mild churn, 10_000+ ≈ active thrashing.
    /// Cached here so consumers never have to re-derive it.
    pub thrashing_score: f64,
    /// Per-core CPU busy ratios derived from host_processor_info tick deltas
    /// between two successive samples. On the first cycle this is
    /// `CpuSaturation::default()` (empty per_core_busy, all-zero scalars);
    /// subsequent cycles have real data.
    ///
    /// Apollo used to have no per-core load signal at all — only per-process
    /// CPU% and the aggregate runnable_time_ns counters. Surfacing it here
    /// keeps the "read PressureData, get every resource pressure axis" API
    /// uniform so consumers don't have to juggle collectors.
    pub cpu_saturation: CpuSaturation,
}

impl Default for PressureData {
    fn default() -> Self {
        Self {
            generation: 0,
            memory_pressure: 0.0,
            swap_used_bytes: 0,
            swap_total_bytes: 0,
            swap_delta_bps: 0.0,
            page_size_bytes: 0,
            updated_at: Instant::now(),
            vm_rate: VmRate::default(),
            thrashing_score: 0.0,
            cpu_saturation: CpuSaturation::default(),
        }
    }
}

impl PressureData {
    pub fn is_fresh(&self, max_age: Duration) -> bool {
        self.updated_at.elapsed() <= max_age
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
                let mut swap_tracker = SwapRateTracker::default();
                // Previous VM sample + its wall-clock timestamp for rate
                // derivation. Separate from swap bookkeeping because the
                // VM stats come from a different kernel call and we want
                // the rate to be computed from the exact dt between the
                // two host_statistics64 reads, not from the loop period.
                let mut prev_vm: Option<(VmPageStats, Instant)> = None;
                // Previous per-core tick sample for CpuSaturation derivation.
                // The compute() helper handles empty / mismatched-length
                // samples on the first cycle, so no special-casing here.
                let mut prev_cpu_ticks: Vec<PerCoreTicks> = Vec::new();
                let mut generation = 0_u64;

                loop {
                    let (mem_pressure, vm_sample, swap_sample) = collect_pressure_facts();
                    let curr_cpu_ticks = cpu_sat::read_per_core_ticks();
                    let now = Instant::now();
                    let (swap_used, swap_total, swap_delta) =
                        swap_tracker.observe(swap_sample, now);

                    // VM flow rates: derive from prev sample if we have one,
                    // zero-filled on first iteration.
                    let page_size_bytes = vm_sample.as_ref().map_or(0, |sample| sample.page_size);
                    let (vm_rate, thrashing_score) = match (&vm_sample, &prev_vm) {
                        (Some(curr), Some((prev, prev_at))) => {
                            let dt = now.duration_since(*prev_at).as_secs_f64();
                            let rate = VmRate::compute(prev, curr, dt);
                            let score = rate.thrashing_score();
                            (rate, score)
                        }
                        _ => (VmRate::default(), 0.0),
                    };
                    if let Some(s) = vm_sample {
                        prev_vm = Some((s, now));
                    }

                    // CPU saturation: compute vs prev sample, then update prev.
                    // The compute() helper returns Default on empty/mismatched
                    // samples, so the first cycle naturally yields no signal.
                    let cpu_saturation = CpuSaturation::compute(&prev_cpu_ticks, &curr_cpu_ticks);
                    prev_cpu_ticks = curr_cpu_ticks;

                    generation = generation.saturating_add(1);
                    *c.lock_recover() = PressureData {
                        generation,
                        memory_pressure: mem_pressure,
                        swap_used_bytes: swap_used,
                        swap_total_bytes: swap_total,
                        swap_delta_bps: swap_delta,
                        page_size_bytes,
                        updated_at: now,
                        vm_rate,
                        thrashing_score,
                        cpu_saturation,
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

fn signed_byte_rate(previous: u64, current: u64, dt_secs: f64) -> f64 {
    if !dt_secs.is_finite() || dt_secs <= 0.0 {
        return 0.0;
    }
    (current as i128 - previous as i128) as f64 / dt_secs
}

#[derive(Debug, Default)]
struct SwapRateTracker {
    used_bytes: Option<u64>,
    total_bytes: Option<u64>,
    observed_at: Option<Instant>,
}

impl SwapRateTracker {
    fn observe(&mut self, sample: Option<(u64, u64)>, now: Instant) -> (u64, u64, f64) {
        let Some((total_bytes, used_bytes)) = sample else {
            return (
                self.used_bytes.unwrap_or(0),
                self.total_bytes.unwrap_or(0),
                0.0,
            );
        };
        let delta_bps = match (self.used_bytes, self.observed_at) {
            (Some(previous), Some(observed_at)) => signed_byte_rate(
                previous,
                used_bytes,
                now.saturating_duration_since(observed_at).as_secs_f64(),
            ),
            _ => 0.0,
        };
        self.used_bytes = Some(used_bytes);
        self.total_bytes = Some(total_bytes);
        self.observed_at = Some(now);
        (used_bytes, total_bytes, delta_bps)
    }
}

/// Collect a raw sample of kernel memory+swap facts for the collector thread.
///
/// Returns the pressure percentage, the full VmPageStats sample (so the
/// caller can also compute flow rates from it), and swap used/total. The
/// VmPageStats is returned as `Option` because host_statistics64 can
/// theoretically fail; the caller's rate computation already handles
/// the None case by zero-filling.
fn collect_pressure_facts() -> (f64, Option<VmPageStats>, Option<(u64, u64)>) {
    // Memory pressure via Mach host_statistics64 (~1µs vs 50ms for subprocess).
    let vm_stats = host_vm_info::read_vm_stats();
    let memory_pressure = vm_stats.as_ref().map(|s| s.pressure()).unwrap_or(0.0);

    // Swap usage via direct sysctl struct read (~1µs vs 10ms for subprocess).
    let swap_usage = sysctl_direct::read_swap_usage();

    (memory_pressure, vm_stats, swap_usage)
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
        assert_eq!(data.generation, 0);
        assert_eq!(data.swap_used_bytes, 0);
        assert_eq!(data.swap_total_bytes, 0);
        assert!((data.swap_delta_bps - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn swap_delta_rate_preserves_growth_and_recovery() {
        assert_eq!(signed_byte_rate(1_000, 1_500, 0.5), 1_000.0);
        assert_eq!(signed_byte_rate(1_500, 1_000, 0.5), -1_000.0);
    }

    #[test]
    fn swap_delta_rate_rejects_invalid_windows() {
        assert_eq!(signed_byte_rate(1_000, 2_000, 0.0), 0.0);
        assert_eq!(signed_byte_rate(1_000, 2_000, f64::NAN), 0.0);
    }

    #[test]
    fn failed_swap_sample_preserves_last_valid_baseline() {
        let started = Instant::now();
        let mut tracker = SwapRateTracker::default();
        assert_eq!(
            tracker.observe(Some((8_000, 4_000)), started),
            (4_000, 8_000, 0.0)
        );
        assert_eq!(
            tracker.observe(None, started + Duration::from_millis(500)),
            (4_000, 8_000, 0.0)
        );
        assert_eq!(
            tracker.observe(Some((8_000, 5_000)), started + Duration::from_secs(1)),
            (5_000, 8_000, 1_000.0)
        );
    }

    #[test]
    fn pressure_sample_freshness_is_bounded() {
        let mut data = PressureData::default();
        assert!(data.is_fresh(Duration::from_secs(2)));
        data.updated_at = Instant::now() - Duration::from_secs(3);
        assert!(!data.is_fresh(Duration::from_secs(2)));
    }

    #[test]
    fn responsive_pressure_interval_catches_short_bursts() {
        assert!(RESPONSIVE_PRESSURE_INTERVAL <= Duration::from_millis(500));
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
