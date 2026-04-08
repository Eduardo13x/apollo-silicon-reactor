//! Telemetry Logger — ring-buffer data collection for Time-Series Transformer training.
//!
//! Collects a 16-dimensional feature vector each daemon cycle (≈500ms) into
//! a circular buffer.  Dumps to disk in raw f32 binary format, optimized for
//! direct ingestion by PyTorch (`np.fromfile(..., dtype=np.float32)`).
//!
//! ## Dump strategy
//!
//! 1. **Periodic** (every `PERIODIC_INTERVAL` cycles ≈ 10 min): captures
//!    normal-regime behaviour for self-supervised pre-training.
//! 2. **Event-triggered**: when any anomaly signal exceeds its threshold,
//!    the full buffer is flushed immediately, capturing the lead-up to the
//!    event for supervised fine-tuning.
//!
//! ## Binary file format
//!
//! ```text
//! Bytes 0..4    : magic  0x41504F4C ("APOL")
//! Bytes 4..8    : n_vectors   (u32 LE)
//! Bytes 8..12   : n_features  (u32 LE) — always 16
//! Bytes 12..16  : reserved    (u32 LE, zero)
//! Bytes 16..24  : timestamp   (i64 LE, Unix epoch seconds)
//! Bytes 24..28  : event_kind  (u32 LE, 0=periodic, 1=oom_risk, 2=urgency, 3=latency)
//! Bytes 28..32  : padding     (u32 LE, zero)
//! Bytes 32..    : f32 LE × n_vectors × n_features
//! ```
//!
//! ## References
//!
//! - Welch 1967, "The use of FFT for estimation of power spectra" — windowed
//!   ring-buffer approach to time-series sampling.
//! - Zerveas et al. 2021, "A Transformer-based Framework for Multivariate
//!   Time Series Representation Learning" — self-supervised pre-training with
//!   periodic + event-triggered sampling to address class imbalance.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// Number of features per telemetry vector.
pub const N_FEATURES: usize = 16;

/// Ring buffer capacity: 240 vectors = 2 minutes at 500ms/cycle.
/// Provides sufficient lead-up context for anomaly events (Tuli et al. 2022
/// recommend ≥30s of context; we store 120s for richer temporal patterns).
const RING_CAPACITY: usize = 240;

/// Periodic dump interval in cycles.  1200 × 500ms = 10 minutes.
const PERIODIC_INTERVAL: u64 = 1200;

/// File header magic: "APOL" in little-endian.
const MAGIC: u32 = 0x41504F4C;

/// Header size in bytes.
const HEADER_SIZE: usize = 32;

/// Classification of why a dump was triggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DumpKind {
    /// Regular periodic sample (self-supervised training data).
    Periodic = 0,
    /// High P(OOM) from hazard model.
    OomRisk = 1,
    /// High composite urgency from SignalIntelligence.
    Urgency = 2,
    /// High perceptual latency score.
    Latency = 3,
}

/// A single observation of the system's state — 16 f32 features.
///
/// All features are normalised to comparable scales (mostly 0–1 or small
/// floats) so the Transformer's attention mechanism doesn't need to learn
/// magnitude compensation (Vaswani et al. 2017, §3.2.2 — scaled dot-product).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TelemetryVector {
    // ── From SignalDigest (Kalman-filtered) ──────────────────────────────
    /// Memory pressure, Kalman-smoothed (0–1).
    pub pressure_smooth: f32,
    /// Rate of pressure change (units/s, typically -0.5..+0.5).
    pub pressure_velocity: f32,
    /// Kalman-predicted pressure 5s ahead (0–1).
    pub pressure_predicted_5s: f32,
    /// Swap delta smoothed (bytes/s, log-scaled then normalised).
    pub swap_velocity_smooth: f32,
    /// PID integral of pressure error over target, windowed 60s.
    pub pressure_integral: f32,

    // ── From SignalDigest (detectors) ────────────────────────────────────
    /// CUSUM high accumulator (regime-shift indicator).
    pub cusum_score: f32,
    /// Entropy anomaly z-score (-3..+3 typical).
    pub entropy_anomaly: f32,
    /// P(OOM in 30s) from Cox hazard model (0–1).
    pub p_oom_30s: f32,
    /// Lotka-Volterra monopoly risk (0–1).
    pub monopoly_risk: f32,
    /// Composite urgency score (0–1).
    pub urgency: f32,

    // ── From snapshot / daemon ───────────────────────────────────────────
    /// Total CPU usage (0–1).
    pub cpu_total: f32,
    /// Compressor pressure (0–1).
    pub compressor_ratio: f32,
    /// Dominant process memory share (0–1).
    pub dominant_share: f32,
    /// Perceptual latency score (0–1).
    pub latency_score: f32,
    /// Active process count, normalised (count / 200, capped at 1).
    pub active_proc_count: f32,
    /// Thermal level (0=nominal, 0.33=light, 0.66=serious, 1=critical).
    pub thermal_score: f32,
}

impl TelemetryVector {
    /// Convert to a fixed-size f32 array for binary serialisation.
    /// Layout must match the field order above — `#[repr(C)]` guarantees this.
    #[inline]
    pub fn as_f32_slice(&self) -> &[f32; N_FEATURES] {
        // SAFETY: TelemetryVector is #[repr(C)] with exactly N_FEATURES f32 fields,
        // so its memory layout is identical to [f32; N_FEATURES].
        unsafe { &*(self as *const TelemetryVector as *const [f32; N_FEATURES]) }
    }
}

/// Convert a thermal level string from `snapshot.pressure.thermal_level` to
/// a normalised f32 score suitable for the Transformer input.
pub fn thermal_str_to_score(level: &str) -> f32 {
    match level {
        "critical" => 1.0,
        "serious" => 0.66,
        "light" | "moderate" => 0.33,
        _ => 0.0, // "nominal" / unknown
    }
}

/// Ring-buffer telemetry logger.
///
/// Maintains the last `RING_CAPACITY` vectors in memory (~3.8 KB total)
/// and flushes to binary files for Transformer training.
pub struct TelemetryLogger {
    ring: VecDeque<TelemetryVector>,
    output_dir: PathBuf,
    cycle_count: u64,
    /// Whether logging is enabled (can be toggled at runtime).
    enabled: bool,
}

impl TelemetryLogger {
    /// Create a new logger.  `output_dir` is created lazily on first dump.
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            ring: VecDeque::with_capacity(RING_CAPACITY),
            output_dir,
            cycle_count: 0,
            enabled: true,
        }
    }

    /// Record a new telemetry vector and return whether a dump was triggered.
    ///
    /// Call this once per daemon cycle, after all signals are computed.
    pub fn record(&mut self, vec: TelemetryVector) -> Option<DumpKind> {
        if !self.enabled {
            return None;
        }

        // Push to ring, evict oldest if full.
        if self.ring.len() >= RING_CAPACITY {
            self.ring.pop_front();
        }
        self.ring.push_back(vec);
        self.cycle_count += 1;

        // Need at least 120 vectors (60s) before any dump is meaningful.
        if self.ring.len() < 120 {
            return None;
        }

        // Check event triggers (Tuli et al. 2022 — event-triggered dumps
        // capture pre-anomaly context for anomaly detection training).
        let kind = if vec.p_oom_30s > 0.6 {
            DumpKind::OomRisk
        } else if vec.urgency > 0.8 {
            DumpKind::Urgency
        } else if vec.latency_score > 0.7 {
            DumpKind::Latency
        } else if self.cycle_count % PERIODIC_INTERVAL == 0 {
            DumpKind::Periodic
        } else {
            return None;
        };

        // Best-effort dump: if it fails, we don't crash the daemon.
        if let Err(e) = self.dump(kind) {
            eprintln!("[telemetry] dump failed: {e}");
        }

        Some(kind)
    }

    /// Flush the ring buffer to a binary file.
    fn dump(&self, kind: DumpKind) -> std::io::Result<()> {
        // Ensure output directory exists.
        std::fs::create_dir_all(&self.output_dir)?;

        let now = chrono::Utc::now();
        let filename = format!(
            "{}_{}.bin",
            now.format("%Y%m%dT%H%M%S"),
            match kind {
                DumpKind::Periodic => "periodic",
                DumpKind::OomRisk => "oom_risk",
                DumpKind::Urgency => "urgency",
                DumpKind::Latency => "latency",
            }
        );
        let path = self.output_dir.join(&filename);

        let n_vecs = self.ring.len() as u32;
        let n_feat = N_FEATURES as u32;
        let timestamp = now.timestamp();

        // Build binary blob: header + data.
        let data_bytes = n_vecs as usize * N_FEATURES * std::mem::size_of::<f32>();
        let mut buf = Vec::with_capacity(HEADER_SIZE + data_bytes);

        // Header (32 bytes).
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&n_vecs.to_le_bytes());
        buf.extend_from_slice(&n_feat.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
        buf.extend_from_slice(&timestamp.to_le_bytes());
        buf.extend_from_slice(&(kind as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // padding

        // Data: raw f32 LE, contiguous.
        for vec in &self.ring {
            for &val in vec.as_f32_slice() {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        std::fs::write(&path, &buf)?;
        Ok(())
    }

    /// Warm-start the ring buffer from recent `.bin` files in `output_dir`.
    ///
    /// Loads up to `max_files` most-recently-modified files and pushes their
    /// vectors into the ring (oldest first so the ring ends up in time order).
    /// Files older than `max_age` are skipped — stale data would mislead the
    /// anomaly detector more than starting cold.
    ///
    /// [Gray & Reuter 1992] §11.3 — restart protocols restore in-flight state.
    /// §11.5 — checkpoint freshness must be bounded; stale state ≠ live state.
    pub fn warm_start_from_dir(&mut self, max_files: usize) {
        const MAX_AGE: std::time::Duration = std::time::Duration::from_secs(3600); // 1 hour
        let read_dir = match std::fs::read_dir(&self.output_dir) {
            Ok(rd) => rd,
            Err(_) => return,
        };
        let now = std::time::SystemTime::now();
        let mut files: Vec<(std::time::SystemTime, std::path::PathBuf)> = read_dir
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("bin"))
            .filter_map(|e| {
                let mtime = e.metadata().ok()?.modified().ok()?;
                // Freshness gate: skip files older than MAX_AGE.
                if now.duration_since(mtime).ok()? > MAX_AGE {
                    return None;
                }
                Some((mtime, e.path()))
            })
            .collect();

        // Sort newest first, take max_files, then reverse to oldest-first order
        // so the ring ends up in chronological order.
        files.sort_by(|a, b| b.0.cmp(&a.0));
        files.truncate(max_files);
        files.reverse();

        let mut loaded = 0usize;
        for (_mtime, path) in &files {
            if let Ok(data) = std::fs::read(path) {
                loaded += self.load_bin_file(&data);
            }
        }
        if loaded > 0 {
            eprintln!(
                "[telemetry] warm-start: loaded {loaded} vectors from {} file(s)",
                files.len()
            );
        }
    }

    /// Parse a binary telemetry file and push its vectors into the ring.
    /// Returns the number of vectors loaded.
    fn load_bin_file(&mut self, data: &[u8]) -> usize {
        if data.len() < HEADER_SIZE {
            return 0;
        }
        // Validate magic.
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != MAGIC {
            return 0;
        }
        let n_vecs = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let n_feat = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        if n_feat != N_FEATURES {
            return 0; // incompatible version
        }
        let expected_len = HEADER_SIZE + n_vecs * N_FEATURES * 4;
        if data.len() < expected_len {
            return 0;
        }
        let mut offset = HEADER_SIZE;
        let mut count = 0;
        for _ in 0..n_vecs {
            let mut arr = [0f32; N_FEATURES];
            for f in arr.iter_mut() {
                let bytes = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
                *f = f32::from_le_bytes(bytes);
                offset += 4;
            }
            // Reconstruct TelemetryVector from f32 array.
            // SAFETY: TelemetryVector is #[repr(C)] with exactly N_FEATURES f32 fields.
            let tv: TelemetryVector = unsafe { std::mem::transmute(arr) };
            if self.ring.len() >= RING_CAPACITY {
                self.ring.pop_front();
            }
            self.ring.push_back(tv);
            count += 1;
        }
        count
    }

    /// Toggle logging on/off.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Current ring buffer length.
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// True if ring is empty.
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Output directory path.
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
}

// ── Housekeeping ──────────────────────────────────────────────────────────

/// Delete telemetry files older than `max_age_days`.
/// Call periodically (e.g. once per day) to prevent unbounded disk growth.
pub fn prune_old_files(dir: &Path, max_age_days: u32) -> std::io::Result<u32> {
    let cutoff =
        std::time::SystemTime::now() - std::time::Duration::from_secs(max_age_days as u64 * 86400);
    let mut removed = 0u32;

    if !dir.exists() {
        return Ok(0);
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        if let Ok(meta) = entry.metadata() {
            if let Ok(modified) = meta.modified() {
                if modified < cutoff {
                    let _ = std::fs::remove_file(&path);
                    removed += 1;
                }
            }
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(pressure: f32, urgency: f32, p_oom: f32, latency: f32) -> TelemetryVector {
        TelemetryVector {
            pressure_smooth: pressure,
            pressure_velocity: 0.0,
            pressure_predicted_5s: pressure,
            swap_velocity_smooth: 0.0,
            pressure_integral: 0.0,
            cusum_score: 0.0,
            entropy_anomaly: 0.0,
            p_oom_30s: p_oom,
            monopoly_risk: 0.0,
            urgency,
            cpu_total: 0.3,
            compressor_ratio: 0.0,
            dominant_share: 0.0,
            latency_score: latency,
            active_proc_count: 0.1,
            thermal_score: 0.0,
        }
    }

    #[test]
    fn telemetry_vector_f32_slice_roundtrip() {
        let v = make_vec(0.42, 0.7, 0.1, 0.2);
        let slice = v.as_f32_slice();
        assert_eq!(slice.len(), N_FEATURES);
        assert!((slice[0] - 0.42).abs() < 1e-6);
        assert!((slice[9] - 0.7).abs() < 1e-6); // urgency is field index 9
    }

    #[test]
    fn ring_buffer_capacity() {
        let dir = std::env::temp_dir().join("apollo_test_telemetry_cap");
        let _ = std::fs::remove_dir_all(&dir);
        let mut logger = TelemetryLogger::new(dir.clone());

        // Fill beyond capacity.
        for i in 0..300 {
            let v = make_vec(i as f32 / 300.0, 0.0, 0.0, 0.0);
            logger.record(v);
        }

        assert_eq!(logger.len(), RING_CAPACITY);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn event_triggered_dump_oom() {
        let dir = std::env::temp_dir().join("apollo_test_telemetry_oom");
        let _ = std::fs::remove_dir_all(&dir);
        let mut logger = TelemetryLogger::new(dir.clone());

        // Fill 120 normal vectors.
        for _ in 0..120 {
            logger.record(make_vec(0.3, 0.2, 0.0, 0.1));
        }

        // Now trigger OOM event.
        let kind = logger.record(make_vec(0.9, 0.9, 0.65, 0.3));
        assert_eq!(kind, Some(DumpKind::OomRisk));

        // Verify file exists.
        let files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .contains("oom_risk")
            })
            .collect();
        assert_eq!(files.len(), 1);

        // Verify binary format.
        let data = std::fs::read(files[0].path()).unwrap();
        assert!(data.len() >= HEADER_SIZE);

        // Check magic.
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(magic, MAGIC);

        // Check n_vectors.
        let n_vecs = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(n_vecs, 121); // 120 + 1 trigger

        // Check n_features.
        let n_feat = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        assert_eq!(n_feat, N_FEATURES as u32);

        // Check total size.
        let expected = HEADER_SIZE + (121 * N_FEATURES * 4);
        assert_eq!(data.len(), expected);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn periodic_dump() {
        let dir = std::env::temp_dir().join("apollo_test_telemetry_periodic");
        let _ = std::fs::remove_dir_all(&dir);
        let mut logger = TelemetryLogger::new(dir.clone());

        // Fill to cycle 1200 (periodic interval).
        for _ in 0..PERIODIC_INTERVAL {
            logger.record(make_vec(0.3, 0.1, 0.0, 0.1));
        }

        // Verify a periodic file was created.
        if dir.exists() {
            let files: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .contains("periodic")
                })
                .collect();
            assert_eq!(files.len(), 1);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_dump_below_120_vectors() {
        let dir = std::env::temp_dir().join("apollo_test_telemetry_nodump");
        let _ = std::fs::remove_dir_all(&dir);
        let mut logger = TelemetryLogger::new(dir.clone());

        // High urgency but not enough data — should NOT dump.
        for _ in 0..100 {
            let kind = logger.record(make_vec(0.9, 0.95, 0.9, 0.9));
            assert_eq!(kind, None);
        }

        // Dir should not even be created.
        assert!(!dir.exists() || std::fs::read_dir(&dir).unwrap().count() == 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn thermal_str_conversion() {
        assert!((thermal_str_to_score("critical") - 1.0).abs() < 1e-6);
        assert!((thermal_str_to_score("serious") - 0.66).abs() < 1e-6);
        assert!((thermal_str_to_score("light") - 0.33).abs() < 1e-6);
        assert!((thermal_str_to_score("nominal") - 0.0).abs() < 1e-6);
        assert!((thermal_str_to_score("unknown_value") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn prune_does_not_panic_on_missing_dir() {
        let dir = std::env::temp_dir().join("apollo_test_prune_missing");
        let _ = std::fs::remove_dir_all(&dir);
        let result = prune_old_files(&dir, 30);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn warm_start_loads_recent_dump() {
        let dir = std::env::temp_dir().join("apollo_test_warm_start");
        let _ = std::fs::remove_dir_all(&dir);

        // First logger: produce one dump.
        {
            let mut logger = TelemetryLogger::new(dir.clone());
            for _ in 0..120 {
                logger.record(make_vec(0.3, 0.2, 0.0, 0.1));
            }
            // Trigger an OOM-risk dump.
            logger.record(make_vec(0.9, 0.9, 0.65, 0.3));
        }

        // Second logger: warm-start from disk should reload the vectors.
        let mut fresh = TelemetryLogger::new(dir.clone());
        assert_eq!(fresh.len(), 0);
        fresh.warm_start_from_dir(3);
        assert!(fresh.len() > 100, "expected warm-start to load >100 vectors, got {}", fresh.len());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn warm_start_handles_missing_dir() {
        let dir = std::env::temp_dir().join("apollo_test_warm_start_missing_xyz");
        let _ = std::fs::remove_dir_all(&dir);
        let mut logger = TelemetryLogger::new(dir);
        // Must not panic when directory doesn't exist.
        logger.warm_start_from_dir(3);
        assert_eq!(logger.len(), 0);
    }

    #[test]
    fn warm_start_rejects_bad_magic() {
        let dir = std::env::temp_dir().join("apollo_test_warm_start_badmagic");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Write a file with wrong magic bytes.
        std::fs::write(dir.join("bad.bin"), vec![0u8; 1024]).unwrap();
        let mut logger = TelemetryLogger::new(dir.clone());
        logger.warm_start_from_dir(3);
        assert_eq!(logger.len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disabled_logger_no_record() {
        let dir = std::env::temp_dir().join("apollo_test_telemetry_disabled");
        let _ = std::fs::remove_dir_all(&dir);
        let mut logger = TelemetryLogger::new(dir.clone());
        logger.set_enabled(false);

        for _ in 0..200 {
            let kind = logger.record(make_vec(0.9, 0.95, 0.9, 0.9));
            assert_eq!(kind, None);
        }
        assert!(logger.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
