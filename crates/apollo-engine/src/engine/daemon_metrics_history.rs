//! # Daemon Metrics History — per-cycle telemetry archive
//!
//! Append-only JSONL archive of a 16-d feature vector per daemon cycle.
//! Unblocks the MLP router PR (`.plan/PR-feature-MLP-router.md`) which
//! failed Phase 1 CV (0.4990 < 0.55 gate, see `.plan/PR-feature-MLP-router-DEFERRED.md`)
//! because `runtime_metrics.json` is a single current snapshot, not a per-cycle
//! time series. This writer emits one JSON line per cycle to
//! `/var/lib/apollo/runtime_metrics_history.jsonl` so an offline trainer can
//! replay the 16-d feature vector at every cycle.
//!
//! ## Wire format (≈ 250 bytes/line)
//!
//! ```json
//! {"t":1719456123,"c":4242,"f":[0.5,...16 floats...],"w":0.01,"n":0.7,"l":12345678901234}
//! ```
//!
//! - `t` — unix timestamp (i64 seconds).
//! - `c` — daemon cycle count (u64).
//! - `f` — 16-d feature vector per `.plan/PR-feature-MLP-router.md §4a`.
//! - `w` — world-model natural drift (Rubin counterfactual baseline, f32).
//! - `n` — NARS top-belief confidence (`belief("compile").confidence`, 0.5 if None).
//! - `l` — stable hash of current LearnableParams snapshot (u64).
//!
//! ## Invariants
//!
//! - **Write + fsync per line** — the 250 B line is amortised into the
//!   existing per-cycle fsync budget; one extra fsync is invisible to the
//!   cycle timing (Apollo already does multiple fsync per cycle).
//! - **Rotation at `rotation_max_bytes`** — rename to `.jsonl.1`, start fresh.
//!   Never grows past `rotation_max_files` (default 2 = 200 MB on disk).
//! - **Startup cap** — if the live file exceeds `startup_cap_bytes` (default
//!   1 GB) at the moment of a write attempt, the writer is no-op for that
//!   line; rotation continues. Bounded disk usage no matter the daemon
//!   lifetime.
//! - **Never blocks the cycle on failure** — failed writes log a warn + bump
//!   `failed_writes_total()`. Caller's cycle proceeds unaffected.
//! - **Symlink guard** — refuses to write through a symlinked path
//!   (TOCTOU [Lampson 1974], matches `journal::append_journal`).
//!
//! ## Why a per-cycle archive is the right shape
//!
//! `runtime_metrics.json` is overwritten with the current snapshot every
//! cycle, so any historical query loses N-1 of N observations. The archive
//! is the canonical time series: 60 cycles/min × 250 B = ~15 KB/minute =
//! ~22 MB/day. At default rotation (100 MB → 2 files), the writer never
//! holds more than 200 MB on disk and a 24h training window has ~528 MB
//! worth of two file pairs (rotated 4-5×/day).
//!
//! [Pei Wang 2013] NARS beliefs are read-only on this path; the writer does
//! NOT mutate `DriftDetector`. [Sutton & Barto 2018] §11.7: a snapshot that
//! is never written cannot be learned from.
use std::collections::hash_map::DefaultHasher;
use std::fs::{self, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::engine::learned_state::LearnableParams;
use crate::engine::nars_belief::DriftDetector;
use crate::engine::types::RuntimeMetrics;
use crate::engine::world_model::WorldModel;

// ── Constants (defaults) ─────────────────────────────────────────────────────

/// Default rotation size: 100 MB. Rotated once a day at ~22 MB/day.
pub const DEFAULT_ROTATION_MAX_BYTES: u64 = 100 * 1024 * 1024;

/// Default startup cap: 10× rotation = 1 GB. If a previous daemon run left
/// a larger archive, the writer refuses to grow it past this size.
pub const DEFAULT_STARTUP_CAP_BYTES: u64 = 1024 * 1024 * 1024;

/// Default file count cap: 2 (current + 1 rotated).
pub const DEFAULT_ROTATION_MAX_FILES: u32 = 2;

// ── HistoryConfig ────────────────────────────────────────────────────────────

/// Mirrors the `[history]` section in `apollo-optimizer.toml`. All fields
/// are `Option<…>` so the section is fully optional; missing values fall
/// back to the defaults above. Reads via `load_repo_config` in
/// [`crate::engine::llm`].
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(default)]
pub struct HistoryConfig {
    pub enabled: Option<bool>,
    pub rotation_max_bytes: Option<u64>,
    pub rotation_max_files: Option<u32>,
    pub startup_cap_bytes: Option<u64>,
}

impl HistoryConfig {
    pub fn enabled(&self) -> bool {
        self.enabled.unwrap_or(true)
    }
    pub fn rotation_max_bytes(&self) -> u64 {
        self.rotation_max_bytes
            .unwrap_or(DEFAULT_ROTATION_MAX_BYTES)
            .max(1) // never let a config typo disable rotation entirely
    }
    pub fn rotation_max_files(&self) -> u32 {
        self.rotation_max_files
            .unwrap_or(DEFAULT_ROTATION_MAX_FILES)
            .max(1)
    }
    /// Cap on TOTAL on-disk bytes (live + rotated) above which the writer
    /// becomes no-op. Matches the spec: "if it has 1+ GB of history, the
    /// writer becomes no-op (rotation continues; never grows past 2 files)".
    /// Total is measured live + rotated so rotation does NOT reset the cap
    /// — a daemon that filled its cap cannot grow it again by rotating.
    pub fn startup_cap_bytes(&self) -> u64 {
        self.startup_cap_bytes
            .unwrap_or(DEFAULT_STARTUP_CAP_BYTES)
            .max(1)
    }
}

// ── Failed-write counter (lock-free, exposed for diagnostics) ────────────────

static FAILED_WRITES: AtomicU64 = AtomicU64::new(0);

/// Total failed append attempts (lock-free). Visible via `runtime_metrics.json`
/// only if a future PR adds a `#[serde(default)]` mirror; today it lives in
/// the daemon log + this counter for unit tests.
pub fn failed_writes_total() -> u64 {
    FAILED_WRITES.load(Ordering::Relaxed)
}

#[cfg(test)]
pub(crate) fn reset_failed_writes_for_test() {
    FAILED_WRITES.store(0, Ordering::Relaxed);
}

// ── 16-d feature extraction (per SPEC §4a) ───────────────────────────────────

/// Extract the canonical 16-d feature vector. Pure, no I/O, no syscalls.
/// Returns `[f32; 16]` indexed per `.plan/PR-feature-MLP-router.md §4a`:
///
/// ```text
/// f[0]  memory_pressure
/// f[1]  swap_used_gb / 4.0
/// f[2]  swap_delta_bps / 524288
/// f[3]  thrashing_score / 10000
/// f[4]  cpu_max_busy
/// f[5]  thermal_predicted_throttle / 100
/// f[6]  sigmoid(secs_to_throttle / 60)
/// f[7]  cycles_high_pressure / 30
/// f[8]  refault_delta_per_sec / 5000
/// f[9]  humble_mode (0 or 1)
/// f[10] meta_cognition.subsystem_debias_multiplier(CausalGraph)
/// f[11] 1.0 - adversarial_pass_rate (specialist-disagreement surrogate)
/// f[12] max(world_model predicted delta - natural drift, 0)
/// f[13] NARS top belief ("compile") confidence
/// f[14] interactivity = interactive_pids / max(cycles, 1) [SPEC pseudocode]
/// f[15] user_call_in_progress (0 or 1)
/// ```
///
/// `causal_subsystem_debias` is read via
/// [`crate::engine::meta_cognition::MetaCognition::subsystem_debias_multiplier`]
/// passed in by the caller — the writer must NOT take a `MetaCognition`
/// directly to keep the dependency surface narrow. The caller threads the
/// `f32` value through the inputs bundle.
pub fn extract_features(
    metrics: &RuntimeMetrics,
    causal_subsystem_debias: f32,
    world_model: &WorldModel,
    drift_detector: &DriftDetector,
) -> [f32; 16] {
    // f[1]: swap GB normalised to 4 GB ceiling. swap_used_bytes is u64.
    let swap_gb_norm = (metrics.swap_used_bytes as f32 / (4.0 * 1024.0 * 1024.0 * 1024.0))
        .clamp(0.0, 1.0);
    // f[2]: swap delta bytes-per-second / 524288 (one 512 KiB write/sec).
    let swap_delta_norm = (metrics.swap_delta_bps as f32 / 524_288.0).clamp(0.0, 1.0);
    // f[3]: thrashing score / 10000. Scale matches `decide_actions.rs:1362`.
    let thrashing_norm = (metrics.thrashing_score as f32 / 10_000.0).clamp(0.0, 1.0);
    // f[5]: thermal prediction percent / 100.
    let thermal_pred_norm = (metrics.thermal_predicted_throttle as f32 / 100.0).clamp(0.0, 1.0);
    // f[6]: sigmoid(-secs / 60). >0 means throttling coming; 1.0 = imminent.
    // We use sigmoid(secs / 60) and invert via 1.0 - x so the semantics match
    // the SPEC's "1.0 = imminent" framing.
    let thermal_secs_norm = metrics
        .thermal_seconds_to_throttle
        .map(|s| 1.0 / (1.0 + (s as f32 / 60.0).exp()))
        .unwrap_or(0.0);
    // f[7]: 30 cycles = 30s sustained high pressure.
    let cycles_high_norm = (metrics.cycles_high_pressure as f32 / 30.0).clamp(0.0, 1.0);
    // f[8]: refault rate normalised to 5000 pages/sec = "storm".
    let refault_norm = (metrics.refault_delta_per_sec as f32 / 5_000.0).clamp(0.0, 1.0);
    // f[9]: humble mode is a 0/1 signal.
    let humble_norm = if metrics.humble_mode { 1.0 } else { 0.0 };
    // f[11]: 1.0 - adversarial_pass_rate. Lower pass rate ⇒ more disagreement.
    // adversarial_pass_rate lives in RuntimeMetrics (Phase 0 lockfree drain)
    // as f32; no cast needed.
    let disagreement_ema = (1.0 - metrics.adversarial_pass_rate).clamp(0.0, 1.0);
    // f[12]: world-model predicted-drop minus natural drift, clamped ≥0.
    // The model exposes a single margin per action; for the trainer we
    // take the max across all known actions via the public accessor
    // `max_predicted_margin()`. Empty model → 0.0.
    let predicted_margin = (world_model.max_predicted_margin() as f32).clamp(0.0, 1.0);
    // f[13]: NARS top belief. SPEC canonical lookup is "compile"; None → 0.5
    // (cold / neutral — never return 0 to avoid poisoning the trainer).
    let nars_compile_conf = drift_detector
        .belief("compile")
        .map(|tv| tv.confidence)
        .unwrap_or(0.5);
    // f[14]: interactive pid count / max(cycles, 1). SPEC pseudocode uses
    // `cycles` as a conservative proxy for total pid count.
    let interactivity = (metrics.behavior_interactive_pid_count as f32
        / metrics.cycles.max(1) as f32)
        .clamp(0.0, 1.0);
    // f[15]: realtime call is a 0/1 signal.
    let call_active = if metrics.user_call_in_progress { 1.0 } else { 0.0 };

    [
        metrics.memory_pressure as f32,
        swap_gb_norm,
        swap_delta_norm,
        thrashing_norm,
        metrics.cpu_max_busy as f32,
        thermal_pred_norm,
        thermal_secs_norm,
        cycles_high_norm,
        refault_norm,
        humble_norm,
        causal_subsystem_debias.clamp(0.25, 1.5),
        disagreement_ema,
        predicted_margin,
        nars_compile_conf,
        interactivity,
        call_active,
    ]
}

// ── Learned-policy hash ──────────────────────────────────────────────────────

/// Stable hash of the current `LearnableParams`. Tuned to flip when key
/// adaptive parameters change (zone_alpha, RL bands, tuning_cycles).
/// NOT a cryptographic hash — just a correlation ID for the trainer.
fn learned_hash(lp: &LearnableParams) -> u64 {
    let mut h = DefaultHasher::new();
    lp.zone_alpha.to_bits().hash(&mut h);
    lp.rl_pressure_bands[0].to_bits().hash(&mut h);
    lp.rl_pressure_bands[1].to_bits().hash(&mut h);
    lp.rl_pressure_bands[2].to_bits().hash(&mut h);
    lp.nars_drift_threshold.to_bits().hash(&mut h);
    lp.tuning_cycles.hash(&mut h);
    h.finish()
}

// ── Wire format ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct HistoryLine {
    t: i64,
    c: u64,
    f: [f32; 16],
    w: f32,
    n: f32,
    l: u64,
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Append ONE compact JSON line per cycle.
///
/// # Failure mode
///
/// `append_history_snapshot` NEVER blocks the cycle. On any I/O failure
/// (symlink, missing parent, EACCES, ENOSPC, write refused) it:
///   1. Increments `FAILED_WRITES` (lock-free counter).
///   2. Emits `tracing::warn!` with the path + cycle + error.
///   3. Returns `Err(anyhow::Error)` to the caller for visibility.
///
/// The daemon's main loop calls this function at the END of
/// `wire_enriched_telemetry`; a failed append cannot influence any
/// decision in the current cycle (the function is post-decision).
#[allow(clippy::too_many_arguments)] // path + cfg + 6 engine refs is the SPEC signature.
pub fn append_history_snapshot(
    path: &Path,
    cfg: &HistoryConfig,
    metrics: &RuntimeMetrics,
    cycle: u64,
    world_model: &WorldModel,
    drift_detector: &DriftDetector,
    learnable_params: &LearnableParams,
    causal_subsystem_debias: f32,
) -> anyhow::Result<()> {
    if !cfg.enabled() {
        return Ok(());
    }

    // Symlink guard — refuse to write through a symlinked path. Matches the
    // protection in `journal::append_journal_batch` (TOCTOU [Lampson 1974]).
    if let Ok(meta) = fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            anyhow::bail!(
                "metrics_history: refusing to write through symlink {}",
                path.display()
            );
        }
    }

    // Compute total on-disk bytes (live + rotated). The startup cap is on
    // TOTAL — rotation does NOT reset the cap because the rotated file is
    // still on disk. Spec: "never grows past 2 files" + "if it has 1+ GB
    // of history, the writer becomes no-op (rotation continues)".
    let rotated_path = path.with_extension("jsonl.1");
    let live_size = fs::symlink_metadata(path).map(|m| m.len()).unwrap_or(0);
    let rotated_size = fs::symlink_metadata(&rotated_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let total_size = live_size.saturating_add(rotated_size);
    if total_size > cfg.startup_cap_bytes() {
        // No-op: don't bump FAILED_WRITES (this is not an error, it's policy).
        return Ok(());
    }

    // Rotation: if the live file exceeds rotation_max_bytes, rename it to
    // .jsonl.1 and start fresh. If the rotated file already exists, remove
    // it first (we keep at most `rotation_max_files` generations).
    if live_size > cfg.rotation_max_bytes() {
        // Best-effort: removing the rotated file is fine if it doesn't
        // exist or is owned by us. Errors here are logged but do not
        // abort the rotation attempt — we still try to rename.
        if let Err(e) = fs::remove_file(&rotated_path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(
                    target: "apollo.metrics_history",
                    path = %rotated_path.display(),
                    error = %e,
                    "remove of stale rotated file failed (continuing)"
                );
            }
        }
        if let Err(e) = fs::rename(path, &rotated_path) {
            FAILED_WRITES.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                target: "apollo.metrics_history",
                path = %path.display(),
                rotated = %rotated_path.display(),
                error = %e,
                "rotation rename failed — cycle NOT blocked, will retry next call"
            );
            return Err(e.into());
        }
    }

    // Build the line.
    let features = extract_features(metrics, causal_subsystem_debias, world_model, drift_detector);
    let line = HistoryLine {
        t: chrono::Utc::now().timestamp(),
        c: cycle,
        f: features,
        w: world_model.natural_drift as f32,
        n: drift_detector
            .belief("compile")
            .map(|tv| tv.confidence)
            .unwrap_or(0.5),
        l: learned_hash(learnable_params),
    };

    // Atomic append + fsync. OpenOptions::append(true) positions at EOF
    // under POSIX; multiple writers on the same file would interleave, but
    // Apollo is the only writer (no other process touches this path).
    let mut buf = serde_json::to_vec(&line)?;
    buf.push(b'\n');

    let result = (|| -> std::io::Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        f.write_all(&buf)?;
        f.sync_all()?;
        Ok(())
    })();

    if let Err(e) = result {
        FAILED_WRITES.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            target: "apollo.metrics_history",
            path = %path.display(),
            cycle,
            error = %e,
            "append_history_snapshot failed (cycle NOT blocked)"
        );
        return Err(e.into());
    }
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::nars_belief::DriftDetector;
    use crate::engine::world_model::WorldModel;

    /// Tiny cfg for tests: rotation at 200 B, cap at 400 B. Forces rotation
    /// without writing 100 MB.
    fn tiny_cfg() -> HistoryConfig {
        HistoryConfig {
            enabled: Some(true),
            rotation_max_bytes: Some(200),
            rotation_max_files: Some(2),
            startup_cap_bytes: Some(400),
        }
    }

    fn tiny_metrics() -> RuntimeMetrics {
        RuntimeMetrics {
            memory_pressure: 0.5,
            swap_used_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
            swap_delta_bps: 262_144.0,               // 0.5 → norm 0.5
            thrashing_score: 5_000.0,                // → norm 0.5
            cpu_max_busy: 0.7,
            thermal_predicted_throttle: 50,
            thermal_seconds_to_throttle: Some(120), // 2 min headroom
            cycles_high_pressure: 15,
            refault_delta_per_sec: 2_500.0, // → norm 0.5
            humble_mode: false,
            adversarial_pass_rate: 0.5, // disagreement_ema = 0.5
            behavior_interactive_pid_count: 100,
            user_call_in_progress: false,
            cycles: 200,
            ..RuntimeMetrics::default()
        }
    }

    #[test]
    fn single_write_produces_line_with_all_16_features_and_expected_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history.jsonl");
        reset_failed_writes_for_test();

        let metrics = tiny_metrics();
        let wm = WorldModel::default();
        let dd = DriftDetector::default();
        let lp = LearnableParams::default();

        append_history_snapshot(
            &path,
            &tiny_cfg(),
            &metrics,
            4242,
            &wm,
            &dd,
            &lp,
            1.0, // causal_subsystem_debias
        )
        .expect("append ok");

        let content = std::fs::read_to_string(&path).expect("read");
        assert_eq!(content.lines().count(), 1, "exactly one line written");
        let v: serde_json::Value = serde_json::from_str(content.trim_end())
            .expect("valid JSON line");
        assert!(v.get("t").is_some(), "key 't' present");
        assert!(v.get("c").is_some(), "key 'c' present");
        assert!(v.get("f").is_some(), "key 'f' present");
        assert!(v.get("w").is_some(), "key 'w' present");
        assert!(v.get("n").is_some(), "key 'n' present");
        assert!(v.get("l").is_some(), "key 'l' present");
        let f = v.get("f").and_then(|x| x.as_array()).expect("f is array");
        assert_eq!(f.len(), 16, "16-d feature vector");
        assert_eq!(v.get("c").and_then(|x| x.as_u64()), Some(4242));
        // Spot-check a few features against the input values.
        assert!((f[0].as_f64().unwrap() - 0.5).abs() < 1e-6, "f[0] = memory_pressure");
        assert!((f[1].as_f64().unwrap() - 0.5).abs() < 1e-6, "f[1] = swap GB / 4");
        assert!((f[4].as_f64().unwrap() - 0.7).abs() < 1e-6, "f[4] = cpu_max_busy");
        assert!((f[9].as_f64().unwrap() - 0.0).abs() < 1e-6, "f[9] = humble_mode");
        assert!((f[10].as_f64().unwrap() - 1.0).abs() < 1e-6, "f[10] = debias");
        assert!((f[11].as_f64().unwrap() - 0.5).abs() < 1e-6, "f[11] = disagreement");
        assert!((f[15].as_f64().unwrap() - 0.0).abs() < 1e-6, "f[15] = call_inactive");
    }

    #[test]
    fn rotation_triggers_when_file_exceeds_max_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history.jsonl");
        reset_failed_writes_for_test();

        // Pre-grow the live file past the rotation threshold (200 B).
        let pre = vec![b'x'; 250];
        std::fs::write(&path, &pre).expect("write seed");

        let metrics = tiny_metrics();
        let wm = WorldModel::default();
        let dd = DriftDetector::default();
        let lp = LearnableParams::default();

        append_history_snapshot(
            &path,
            &tiny_cfg(),
            &metrics,
            1,
            &wm,
            &dd,
            &lp,
            1.0,
        )
        .expect("append after rotation");

        let rotated = path.with_extension("jsonl.1");
        assert!(rotated.exists(), "rotated file .jsonl.1 exists");
        // Live file should now contain exactly the new JSON line, not the seed.
        let content = std::fs::read_to_string(&path).expect("read live");
        assert!(!content.starts_with("xxxx"), "live file is the new line, not seed");
        assert!(content.contains("\"c\":1"), "live file has the new cycle's line");
    }

    #[test]
    fn write_failure_returns_err_and_does_not_panic() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Path whose parent directory does NOT exist. OpenOptions::create(true)
        // creates the file but not the parent → open() fails. We expect Err,
        // NO panic, and the failed-writes counter increments.
        let path = dir.path().join("nonexistent_subdir_xyz").join("history.jsonl");
        reset_failed_writes_for_test();

        let metrics = tiny_metrics();
        let wm = WorldModel::default();
        let dd = DriftDetector::default();
        let lp = LearnableParams::default();

        let result = append_history_snapshot(
            &path,
            &tiny_cfg(),
            &metrics,
            1,
            &wm,
            &dd,
            &lp,
            1.0,
        );
        assert!(result.is_err(), "write to non-existent parent must fail");
        let msg = format!("{}", result.unwrap_err());
        // We don't pin the exact error kind (EACCES vs ENOENT varies by
        // platform) — only that we surfaced a non-empty error chain.
        assert!(!msg.is_empty(), "error chain is non-empty");
        assert!(
            failed_writes_total() >= 1,
            "FAILED_WRITES counter incremented (got {})",
            failed_writes_total()
        );
    }

    #[test]
    fn startup_cap_makes_writer_noop_above_threshold() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history.jsonl");
        reset_failed_writes_for_test();

        // tiny_cfg cap = 200 * 2 * 2 = 800 B. Pre-grow live file past that.
        std::fs::write(&path, vec![b'y'; 900]).expect("write oversized");

        let metrics = tiny_metrics();
        let wm = WorldModel::default();
        let dd = DriftDetector::default();
        let lp = LearnableParams::default();

        append_history_snapshot(
            &path,
            &tiny_cfg(),
            &metrics,
            99,
            &wm,
            &dd,
            &lp,
            1.0,
        )
        .expect("no-op return Ok");

        // Live file is unchanged — the no-op must not have written.
        let content = std::fs::read_to_string(&path).expect("read");
        assert!(content.starts_with("yyyy"), "live file untouched above cap");
        // No-op is NOT a failure; counter stays at 0.
        assert_eq!(
            failed_writes_total(),
            0,
            "no-op must not bump FAILED_WRITES (got {})",
            failed_writes_total()
        );
    }

    #[test]
    fn config_defaults_match_constants() {
        let cfg = HistoryConfig::default();
        assert!(cfg.enabled());
        assert_eq!(cfg.rotation_max_bytes(), DEFAULT_ROTATION_MAX_BYTES);
        assert_eq!(cfg.rotation_max_files(), DEFAULT_ROTATION_MAX_FILES);
        assert_eq!(cfg.startup_cap_bytes(), DEFAULT_STARTUP_CAP_BYTES);
    }

    #[test]
    fn config_partial_overrides_fall_back_to_defaults() {
        let cfg = HistoryConfig {
            enabled: Some(false),
            rotation_max_bytes: None,
            rotation_max_files: None,
            startup_cap_bytes: None,
        };
        assert!(!cfg.enabled());
        assert_eq!(cfg.rotation_max_bytes(), DEFAULT_ROTATION_MAX_BYTES);
        assert_eq!(cfg.rotation_max_files(), DEFAULT_ROTATION_MAX_FILES);
        assert_eq!(cfg.startup_cap_bytes(), DEFAULT_STARTUP_CAP_BYTES);
    }

    #[test]
    fn extract_features_handles_zero_cycles() {
        // f[14] divides by max(cycles, 1) — guard against div-by-zero.
        let mut metrics = tiny_metrics();
        metrics.cycles = 0;
        metrics.behavior_interactive_pid_count = 0;
        let wm = WorldModel::default();
        let dd = DriftDetector::default();
        let f = extract_features(&metrics, 1.0, &wm, &dd);
        assert!(f[14].is_finite(), "f[14] finite when cycles == 0");
        assert!(f[14] >= 0.0 && f[14] <= 1.0, "f[14] in [0,1]");
    }

    #[test]
    fn learned_hash_changes_when_zone_alpha_changes() {
        let mut lp_a = LearnableParams::default();
        let mut lp_b = LearnableParams::default();
        lp_a.zone_alpha = 0.10;
        lp_b.zone_alpha = 0.20;
        assert_ne!(
            learned_hash(&lp_a),
            learned_hash(&lp_b),
            "hash must change when zone_alpha changes"
        );
    }
}
