//! Background SMC/Powermetrics Reader — cached hardware telemetry.
//!
//! The IOKitSensorReader blocks for ~500 ms per call (powermetrics sample).
//! This module wraps it in a background thread that polls at a configurable
//! interval and caches the latest `HardwareSnapshot` behind an `Arc<Mutex>`.
//!
//! Benefits:
//!   - Main daemon loop reads cached data in <1 μs instead of blocking 500 ms.
//!   - No more "every 5th cycle" polling — data is always fresh.
//!   - If powermetrics hangs, the background thread absorbs the timeout
//!     without affecting the daemon's optimization loop.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::engine::iokit_sensors::{HardwareSnapshot, IOKitSensorReader};
use crate::engine::lock_ext::LockRecover;

/// Cached sensor reader with background polling thread.
pub struct SmcReader {
    /// Latest successful snapshot.
    cache: Arc<Mutex<Option<HardwareSnapshot>>>,
    /// Timestamp of last successful read.
    last_read_at: Arc<Mutex<Option<Instant>>>,
    /// Number of successful reads.
    success_count: Arc<Mutex<u64>>,
    /// Number of failed reads.
    error_count: Arc<Mutex<u64>>,
    /// Last error message.
    last_error: Arc<Mutex<Option<String>>>,
    /// Heartbeat: epoch millis of the last successful snapshot.
    heartbeat: Arc<AtomicU64>,
}

impl SmcReader {
    /// Spawn a background thread that polls powermetrics every `interval`.
    ///
    /// Returns immediately; the thread runs until the process exits.
    pub fn spawn(interval: Duration) -> Self {
        let cache: Arc<Mutex<Option<HardwareSnapshot>>> = Arc::new(Mutex::new(None));
        let last_read_at: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));
        let success_count: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
        let error_count: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
        let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let heartbeat = Arc::new(AtomicU64::new(0));

        let c = cache.clone();
        let lr = last_read_at.clone();
        let sc = success_count.clone();
        let ec = error_count.clone();
        let le = last_error.clone();
        let hb = heartbeat.clone();

        if let Err(e) = thread::Builder::new()
            .name("smc-reader".into())
            .spawn(move || {
                let reader = IOKitSensorReader::new();
                loop {
                    match reader.snapshot() {
                        Ok(hw) => {
                            *c.lock_recover() = Some(hw);
                            *lr.lock_recover() = Some(Instant::now());
                            *sc.lock_recover() += 1;

                            // Update heartbeat after successful snapshot.
                            hb.store(
                                SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis() as u64,
                                Ordering::Release,
                            );
                        }
                        Err(e) => {
                            *ec.lock_recover() += 1;
                            *le.lock_recover() = Some(e);
                        }
                    }
                    thread::sleep(interval);
                }
            })
        {
            eprintln!("warning: failed to spawn smc-reader: {}", e);
        }

        Self {
            cache,
            last_read_at,
            success_count,
            error_count,
            last_error,
            heartbeat,
        }
    }

    /// Get the latest cached hardware snapshot (<1 μs).
    ///
    /// Returns `None` if no successful read has occurred yet.
    pub fn latest(&self) -> Option<HardwareSnapshot> {
        self.cache.lock_recover().clone()
    }

    /// Age of the cached data.  Returns `None` if no data yet.
    pub fn data_age(&self) -> Option<Duration> {
        self.last_read_at.lock_recover().map(|t| t.elapsed())
    }

    /// True if the cached data is stale (older than `max_age`).
    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.data_age().map(|age| age > max_age).unwrap_or(true)
    }

    /// Number of successful reads.
    pub fn success_count(&self) -> u64 {
        *self.success_count.lock_recover()
    }

    /// Number of failed reads.
    pub fn error_count(&self) -> u64 {
        *self.error_count.lock_recover()
    }

    /// Get a clone of the cache Arc for sharing with other threads (e.g. resource sentinel).
    pub fn cache_arc(&self) -> Arc<Mutex<Option<HardwareSnapshot>>> {
        self.cache.clone()
    }

    /// Last error message, if any.
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock_recover().clone()
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
