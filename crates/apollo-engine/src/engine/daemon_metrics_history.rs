//! # Daemon Metrics History — per-cycle telemetry archive
//!
//! Append-only JSONL archive of action features plus live World Model context.
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
//! {"v":4,"t":1719456123,"c":4242,"f":[...16...],"x":[...74...],"w":0.01,"n":0.7,"l":12345678901234}
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
//! - **Persistent append descriptor** — steady-state cycles issue one write;
//!   path metadata, reopen, and `sync_data` are reconciled every 30 cycles.
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
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::engine::learned_state::LearnableParams;
use crate::engine::nars_belief::DriftDetector;
use crate::engine::types::RuntimeMetrics;
use crate::engine::world_model::WorldModel;

/// Fixed-width live-context vector written beside the original action vector.
/// The stable shape lets offline sequence models consume old `f` data while
/// newer trainers opt into richer `x` context.
pub const N_CONTEXT_FEATURES: usize = 74;

// ── Constants (defaults) ─────────────────────────────────────────────────────

/// Default rotation size: 100 MB. Rotated once a day at ~22 MB/day.
pub const DEFAULT_ROTATION_MAX_BYTES: u64 = 100 * 1024 * 1024;

/// Default startup cap: 10× rotation = 1 GB. If a previous daemon run left
/// a larger archive, the writer refuses to grow it past this size.
pub const DEFAULT_STARTUP_CAP_BYTES: u64 = 1024 * 1024 * 1024;

/// Default file count cap: 2 (current + 1 rotated).
pub const DEFAULT_ROTATION_MAX_FILES: u32 = 2;

/// Force history data to stable storage periodically rather than on every
/// cycle. Every line is still appended immediately; only the expensive APFS
/// flush is batched. A power loss can drop at most one short telemetry window,
/// never policy or learned-state files.
const HISTORY_SYNC_EVERY_CYCLES: u64 = 30;

fn should_sync_history(cycle: u64) -> bool {
    cycle <= 1 || cycle.is_multiple_of(HISTORY_SYNC_EVERY_CYCLES)
}

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

use crate::engine::lse_counters::LSE_COUNTERS;

/// Total failed append attempts (lock-free). Visible via `runtime_metrics.json`.
pub fn failed_writes_total() -> u64 {
    LSE_COUNTERS.snapshot().failed_history_writes
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
    let swap_gb_norm =
        (metrics.swap_used_bytes as f32 / (4.0 * 1024.0 * 1024.0 * 1024.0)).clamp(0.0, 1.0);
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
    let call_active = if metrics.user_call_in_progress {
        1.0
    } else {
        0.0
    };

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

/// Extract the broader live context used to train a state-aware World Model.
/// This stays separate from `f`: descriptive context must not be interpreted
/// as causal evidence that an actuator changed the machine.
pub fn extract_context_features(
    metrics: &RuntimeMetrics,
    world_model: &WorldModel,
) -> [f32; N_CONTEXT_FEATURES] {
    let ctx = world_model.latest_context();
    let ctx_f64 =
        |read: fn(&crate::engine::telemetry_medallion::TelemetryContextSummary) -> f64,
         fallback: f64| { ctx.map(read).unwrap_or(fallback) };
    let ctx_u64 =
        |read: fn(&crate::engine::telemetry_medallion::TelemetryContextSummary) -> u64,
         fallback: u64| { ctx.map(read).unwrap_or(fallback) };

    let total_ram = ctx_u64(|c| c.total_ram_bytes, 0).max(1) as f64;
    let swap_ratio = if metrics.swap_total_bytes > 0 {
        metrics.swap_used_bytes as f64 / metrics.swap_total_bytes as f64
    } else {
        metrics.swap_used_bytes as f64 / (4.0 * 1024.0 * 1024.0 * 1024.0)
    };
    let workload_code = match metrics.current_workload.as_str() {
        "idle" => 0.0,
        "browsing" => 0.33,
        "llm-inference" => 0.66,
        "build" => 1.0,
        _ => 0.5,
    };
    let profile_code = match metrics.effective_profile.as_str() {
        "safe-root" => 0.25,
        "balanced-root" => 0.5,
        "aggressive-root" => 1.0,
        _ => 0.0,
    };
    let thermal = ctx_f64(|c| c.thermal_score, 0.0);
    let action_activity = ctx.map_or(0.0, |c| {
        (c.boosts_applied
            .saturating_add(c.throttles_applied)
            .saturating_add(c.freezes_applied)
            .saturating_add(c.paging_hints_applied) as f64
            / 4.0)
            .clamp(0.0, 1.0)
    });
    let disk_available_ratio = ctx.map_or(0.0, |c| {
        if c.disk_total_bytes == 0 {
            0.0
        } else {
            c.disk_available_bytes as f64 / c.disk_total_bytes as f64
        }
    });
    let log_norm =
        |value: u64, ceiling: f64| ((value as f64).ln_1p() / ceiling.ln_1p()).clamp(0.0, 1.0);
    let markov_resolved = metrics
        .markov_prewarm_hits
        .saturating_add(metrics.markov_prewarm_misses);
    let markov_hit_rate = if markov_resolved == 0 {
        0.0
    } else {
        metrics.markov_prewarm_hits as f64 / markov_resolved as f64
    };

    [
        ctx_f64(|c| c.memory_pressure, metrics.memory_pressure).clamp(0.0, 1.0) as f32,
        ctx_f64(|c| c.memory_pressure_raw, metrics.memory_pressure).clamp(0.0, 1.0) as f32,
        ctx_f64(|c| c.compressor_pressure, metrics.compressed_memory_ratio).clamp(0.0, 1.0) as f32,
        ctx_f64(|c| c.used_ram_fraction, 0.0).clamp(0.0, 1.0) as f32,
        swap_ratio.clamp(0.0, 1.0) as f32,
        (metrics.swap_delta_bps / (64.0 * 1024.0 * 1024.0)).clamp(-1.0, 1.0) as f32,
        (metrics.thrashing_score / 50_000.0).clamp(0.0, 1.0) as f32,
        (metrics.refault_delta_per_sec / 10_000.0).clamp(0.0, 1.0) as f32,
        ctx_f64(|c| c.cpu_global_usage, metrics.cpu_mean_busy).clamp(0.0, 1.0) as f32,
        metrics.cpu_mean_busy.clamp(0.0, 1.0) as f32,
        metrics.cpu_max_busy.clamp(0.0, 1.0) as f32,
        metrics.cpu_pegged_fraction.clamp(0.0, 1.0) as f32,
        metrics.stall_fraction.clamp(0.0, 1.0) as f32,
        (ctx.map_or(0, |c| c.process_count) as f64 / 200.0).clamp(0.0, 1.0) as f32,
        ctx_f64(|c| c.top_process_cpu, 0.0).clamp(0.0, 1.0) as f32,
        (ctx_u64(|c| c.top_process_rss_bytes, 0) as f64 / total_ram).clamp(0.0, 1.0) as f32,
        thermal.clamp(0.0, 1.0) as f32,
        (metrics.energy_package_watts.unwrap_or(0.0) / 50.0).clamp(0.0, 1.0) as f32,
        (metrics.energy_gpu_watts.unwrap_or(0.0) / 30.0).clamp(0.0, 1.0) as f32,
        (metrics.energy_ane_watts.unwrap_or(0.0) / 20.0).clamp(0.0, 1.0) as f32,
        (metrics.energy_ane_util_pct.unwrap_or(0.0) / 100.0).clamp(0.0, 1.0) as f32,
        metrics.fluidity_score.clamp(0.0, 1.0),
        (metrics.windowserver_cpu_pct as f64 / 100.0).clamp(0.0, 1.0) as f32,
        (metrics.user_idle_secs / 600.0).clamp(0.0, 1.0) as f32,
        u8::from(metrics.user_has_sleep_assertion) as f32,
        u8::from(metrics.user_call_in_progress) as f32,
        u8::from(metrics.user_audio_active) as f32,
        u8::from(metrics.app_launching) as f32,
        u8::from(metrics.window_op_active) as f32,
        (metrics.context_switches_5min as f64 / 20.0).clamp(0.0, 1.0) as f32,
        u8::from(metrics.context_switch_burst) as f32,
        u8::from(metrics.foreground_idle) as f32,
        workload_code,
        profile_code,
        metrics.markov_prediction_confidence.clamp(0.0, 1.0) as f32,
        (metrics.markov_prediction_eta_secs / 60.0).clamp(0.0, 1.0) as f32,
        u8::from(metrics.markov_prewarm_active) as f32,
        u8::from(metrics.markov_prewarm_eligible) as f32,
        ctx_f64(|c| c.nars_drift_score, metrics.nars_drift_score).clamp(0.0, 1.0) as f32,
        (ctx.map_or(metrics.nars_beliefs_total as u64, |c| c.nars_beliefs_total) as f64 / 3_000.0)
            .clamp(0.0, 1.0) as f32,
        ctx_f64(|c| c.arousal_level, metrics.arousal_level as f64).clamp(0.0, 1.0) as f32,
        ctx_f64(|c| c.signal_pressure_smooth, metrics.si_pressure_smooth).clamp(0.0, 1.0) as f32,
        ctx_f64(|c| c.signal_pressure_velocity, metrics.si_pressure_velocity).clamp(-1.0, 1.0)
            as f32,
        ctx_f64(|c| c.signal_p_oom_30s, metrics.si_p_oom_30s).clamp(0.0, 1.0) as f32,
        ctx_f64(|c| c.signal_urgency, metrics.si_urgency).clamp(0.0, 1.0) as f32,
        (ctx_f64(|c| c.signal_entropy_anomaly, metrics.si_entropy_anomaly) / 3.0).clamp(-1.0, 1.0)
            as f32,
        ctx_f64(|c| c.signal_transformer_anomaly, 0.0).clamp(0.0, 1.0) as f32,
        world_model.context_quality().clamp(0.0, 1.0) as f32,
        action_activity as f32,
        disk_available_ratio.clamp(0.0, 1.0) as f32,
        log_norm(ctx_u64(|c| c.network_received_bytes, 0), 1024.0_f64.powi(4)) as f32,
        log_norm(
            ctx_u64(|c| c.network_transmitted_bytes, 0),
            1024.0_f64.powi(4),
        ) as f32,
        (metrics.ais_score / 100.0).clamp(0.0, 1.0) as f32,
        metrics.ais_learning.clamp(0.0, 1.0) as f32,
        u8::from(metrics.predictive_agent_active) as f32,
        markov_hit_rate.clamp(0.0, 1.0) as f32,
        log_norm(metrics.world_model_actuator_issued_total, 1_000_000.0) as f32,
        log_norm(metrics.world_model_actuator_bronze_total, 1_000_000.0) as f32,
        (metrics.world_model_actuator_pending_total as f64 / 192.0).clamp(0.0, 1.0) as f32,
        if metrics.world_model_actuator_bronze_total == 0 {
            0.0
        } else {
            (metrics.world_model_actuator_gold_total as f64
                / metrics.world_model_actuator_bronze_total as f64)
                .clamp(0.0, 1.0) as f32
        },
        if metrics.world_model_actuator_bronze_total == 0 {
            0.0
        } else {
            (metrics.world_model_actuator_effective_total as f64
                / metrics.world_model_actuator_bronze_total as f64)
                .clamp(0.0, 1.0) as f32
        },
        metrics.world_model_actuator_quality.clamp(0.0, 1.0) as f32,
        metrics.world_model_actuator_mean_utility.clamp(-1.0, 1.0) as f32,
        (metrics.world_model_actuator_ready_models as f64 / 64.0).clamp(0.0, 1.0) as f32,
        u8::from(ctx.is_some_and(|context| context.daemon_is_root)) as f32,
        u8::from(ctx.is_some_and(|context| context.kernel_taskpolicy_available)) as f32,
        u8::from(ctx.is_some_and(|context| context.kernel_sysctl_available)) as f32,
        u8::from(ctx.is_some_and(|context| context.kernel_memorystatus_available)) as f32,
        u8::from(ctx.is_some_and(|context| context.kernel_pressure_send_available)) as f32,
        (ctx.map_or(0, |context| context.p_core_count) as f64 / 16.0).clamp(0.0, 1.0) as f32,
        (ctx.map_or(0, |context| context.e_core_count) as f64 / 16.0).clamp(0.0, 1.0) as f32,
        (ctx.map_or(0, |context| context.unavailable_capability_count) as f64 / 16.0)
            .clamp(0.0, 1.0) as f32,
        u8::from(ctx.is_some_and(|context| context.memorystatus_probe_ok)) as f32,
        u8::from(ctx.is_some_and(|context| context.task_for_pid_probe_ok)) as f32,
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
struct HistoryLine<'a> {
    v: u8,
    t: i64,
    c: u64,
    f: [f32; 16],
    x: &'a [f32],
    w: f32,
    n: f32,
    l: u64,
}

/// Allocation-free history payload prepared while the metrics snapshot is
/// consistent. Keeping only the compact wire fields avoids cloning the large
/// `RuntimeMetrics` structure under the daemon's metrics lock.
#[derive(Debug, Clone, Copy)]
pub struct PreparedHistorySnapshot {
    features: [f32; 16],
    context_features: [f32; N_CONTEXT_FEATURES],
    natural_drift: f32,
    nars_compile_confidence: f32,
    learned_hash: u64,
}

pub fn prepare_history_snapshot(
    metrics: &RuntimeMetrics,
    causal_subsystem_debias: f32,
    world_model: &WorldModel,
    drift_detector: &DriftDetector,
    learnable_params: &LearnableParams,
) -> PreparedHistorySnapshot {
    PreparedHistorySnapshot {
        features: extract_features(
            metrics,
            causal_subsystem_debias,
            world_model,
            drift_detector,
        ),
        context_features: extract_context_features(metrics, world_model),
        natural_drift: world_model.natural_drift as f32,
        nars_compile_confidence: drift_detector
            .belief("compile")
            .map(|tv| tv.confidence)
            .unwrap_or(0.5),
        learned_hash: learned_hash(learnable_params),
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Backward-compatible convenience API for callers that do not hold a metrics
/// lock. Daemon hot paths should prepare the compact snapshot first and append
/// it after releasing their lock.
#[allow(clippy::too_many_arguments)]
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
    let snapshot = prepare_history_snapshot(
        metrics,
        causal_subsystem_debias,
        world_model,
        drift_detector,
        learnable_params,
    );
    append_prepared_history_snapshot(path, cfg, cycle, &snapshot)
}

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
pub fn append_prepared_history_snapshot(
    path: &Path,
    cfg: &HistoryConfig,
    cycle: u64,
    snapshot: &PreparedHistorySnapshot,
) -> anyhow::Result<()> {
    MetricsHistoryWriter::new(path.to_path_buf(), *cfg).append(cycle, snapshot)
}

/// Stateful history writer for the daemon hot path.
///
/// Keeps the append descriptor and byte counts across cycles. Filesystem state
/// is reconciled at startup, every sync window, and after an I/O error; normal
/// cycles therefore need one `write(2)` instead of metadata checks + open +
/// write + close.
pub struct MetricsHistoryWriter {
    path: PathBuf,
    cfg: HistoryConfig,
    file: Option<File>,
    live_size: u64,
    rotated_size: u64,
    initialized: bool,
    #[cfg(test)]
    filesystem_refreshes: u64,
    #[cfg(test)]
    file_opens: u64,
}

impl MetricsHistoryWriter {
    pub fn new(path: PathBuf, cfg: HistoryConfig) -> Self {
        Self {
            path,
            cfg,
            file: None,
            live_size: 0,
            rotated_size: 0,
            initialized: false,
            #[cfg(test)]
            filesystem_refreshes: 0,
            #[cfg(test)]
            file_opens: 0,
        }
    }

    fn rotated_path(&self) -> PathBuf {
        self.path.with_extension("jsonl.1")
    }

    fn refresh_filesystem_state(&mut self) -> std::io::Result<()> {
        #[cfg(test)]
        {
            self.filesystem_refreshes += 1;
        }
        if let Ok(meta) = fs::symlink_metadata(&self.path) {
            if meta.file_type().is_symlink() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!(
                        "metrics_history: refusing to write through symlink {}",
                        self.path.display()
                    ),
                ));
            }
            self.live_size = meta.len();
        } else {
            self.live_size = 0;
        }
        self.rotated_size = fs::symlink_metadata(self.rotated_path())
            .map(|m| m.len())
            .unwrap_or(0);
        self.initialized = true;
        Ok(())
    }

    fn ensure_open(&mut self) -> std::io::Result<()> {
        if !self.initialized {
            self.refresh_filesystem_state()?;
        }
        if self.file.is_none() {
            self.file = Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path)?,
            );
            #[cfg(test)]
            {
                self.file_opens += 1;
            }
        }
        Ok(())
    }

    fn rotate_if_needed(&mut self) -> std::io::Result<bool> {
        if self.live_size <= self.cfg.rotation_max_bytes() {
            return Ok(false);
        }
        self.file.take();
        let rotated_path = self.rotated_path();
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
        fs::rename(&self.path, &rotated_path)?;
        self.rotated_size = self.live_size;
        self.live_size = 0;
        self.ensure_open()?;
        Ok(true)
    }

    pub fn append(&mut self, cycle: u64, snapshot: &PreparedHistorySnapshot) -> anyhow::Result<()> {
        if !self.cfg.enabled() {
            return Ok(());
        }

        let result = self.append_inner(cycle, snapshot);
        if let Err(e) = result {
            self.file.take();
            self.initialized = false;
            LSE_COUNTERS.inc_failed_history_writes();
            tracing::warn!(
                target: "apollo.metrics_history",
                path = %self.path.display(),
                cycle,
                error = %e,
                "append_history_snapshot failed (cycle NOT blocked)"
            );
            return Err(e.into());
        }
        Ok(())
    }

    fn append_inner(
        &mut self,
        cycle: u64,
        snapshot: &PreparedHistorySnapshot,
    ) -> std::io::Result<()> {
        if self.initialized && should_sync_history(cycle) {
            // Reopen at the durability boundary so external replacement,
            // truncation, or unlink cannot leave us writing an orphaned inode.
            self.file.take();
        }
        if !self.initialized || should_sync_history(cycle) {
            self.refresh_filesystem_state()?;
        }
        if self.live_size.saturating_add(self.rotated_size) > self.cfg.startup_cap_bytes() {
            return Ok(());
        }

        self.ensure_open()?;
        let rotated = self.rotate_if_needed()?;

        let line = HistoryLine {
            v: 4,
            t: chrono::Utc::now().timestamp(),
            c: cycle,
            f: snapshot.features,
            x: &snapshot.context_features,
            w: snapshot.natural_drift,
            n: snapshot.nars_compile_confidence,
            l: snapshot.learned_hash,
        };
        let mut buf = serde_json::to_vec(&line)?;
        buf.push(b'\n');

        let file = self.file.as_mut().expect("history file opened");
        file.write_all(&buf)?;
        self.live_size = self.live_size.saturating_add(buf.len() as u64);
        if rotated || should_sync_history(cycle) {
            file.sync_data()?;
        }
        Ok(())
    }
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
    fn single_write_produces_action_and_context_vectors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history.jsonl");
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
        let v: serde_json::Value =
            serde_json::from_str(content.trim_end()).expect("valid JSON line");
        assert!(v.get("t").is_some(), "key 't' present");
        assert!(v.get("c").is_some(), "key 'c' present");
        assert!(v.get("f").is_some(), "key 'f' present");
        assert_eq!(v.get("v").and_then(|x| x.as_u64()), Some(4));
        assert!(v.get("x").is_some(), "key 'x' present");
        assert!(v.get("w").is_some(), "key 'w' present");
        assert!(v.get("n").is_some(), "key 'n' present");
        assert!(v.get("l").is_some(), "key 'l' present");
        let f = v.get("f").and_then(|x| x.as_array()).expect("f is array");
        assert_eq!(f.len(), 16, "16-d feature vector");
        let x = v
            .get("x")
            .and_then(|value| value.as_array())
            .expect("x is array");
        assert_eq!(x.len(), N_CONTEXT_FEATURES, "live context vector");
        assert_eq!(v.get("c").and_then(|x| x.as_u64()), Some(4242));
        // Spot-check a few features against the input values.
        assert!(
            (f[0].as_f64().unwrap() - 0.5).abs() < 1e-6,
            "f[0] = memory_pressure"
        );
        assert!(
            (f[1].as_f64().unwrap() - 0.5).abs() < 1e-6,
            "f[1] = swap GB / 4"
        );
        assert!(
            (f[4].as_f64().unwrap() - 0.7).abs() < 1e-6,
            "f[4] = cpu_max_busy"
        );
        assert!(
            (f[9].as_f64().unwrap() - 0.0).abs() < 1e-6,
            "f[9] = humble_mode"
        );
        assert!(
            (f[10].as_f64().unwrap() - 1.0).abs() < 1e-6,
            "f[10] = debias"
        );
        assert!(
            (f[11].as_f64().unwrap() - 0.5).abs() < 1e-6,
            "f[11] = disagreement"
        );
        assert!(
            (f[15].as_f64().unwrap() - 0.0).abs() < 1e-6,
            "f[15] = call_inactive"
        );
    }

    #[test]
    fn prepared_snapshot_preserves_feature_vector_and_wire_shape() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("prepared-history.jsonl");
        let metrics = tiny_metrics();
        let wm = WorldModel::default();
        let dd = DriftDetector::default();
        let lp = LearnableParams::default();

        let prepared = prepare_history_snapshot(&metrics, 1.0, &wm, &dd, &lp);
        assert_eq!(prepared.features, extract_features(&metrics, 1.0, &wm, &dd));
        assert_eq!(
            prepared.context_features,
            extract_context_features(&metrics, &wm)
        );

        append_prepared_history_snapshot(&path, &tiny_cfg(), 77, &prepared)
            .expect("prepared append ok");
        let content = std::fs::read_to_string(path).expect("read prepared line");
        let value: serde_json::Value = serde_json::from_str(content.trim()).expect("valid JSON");

        assert_eq!(value.get("c").and_then(|v| v.as_u64()), Some(77));
        assert_eq!(
            value.get("f").and_then(|v| v.as_array()).map(|v| v.len()),
            Some(16)
        );
        assert_eq!(
            value.get("x").and_then(|v| v.as_array()).map(|v| v.len()),
            Some(N_CONTEXT_FEATURES)
        );
        assert!(value.get("w").is_some());
        assert!(value.get("n").is_some());
        assert!(value.get("l").is_some());
    }

    #[test]
    fn rotation_triggers_when_file_exceeds_max_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history.jsonl");
        // Pre-grow the live file past the rotation threshold (200 B).
        let pre = vec![b'x'; 250];
        std::fs::write(&path, &pre).expect("write seed");

        let metrics = tiny_metrics();
        let wm = WorldModel::default();
        let dd = DriftDetector::default();
        let lp = LearnableParams::default();

        append_history_snapshot(&path, &tiny_cfg(), &metrics, 1, &wm, &dd, &lp, 1.0)
            .expect("append after rotation");

        let rotated = path.with_extension("jsonl.1");
        assert!(rotated.exists(), "rotated file .jsonl.1 exists");
        // Live file should now contain exactly the new JSON line, not the seed.
        let content = std::fs::read_to_string(&path).expect("read live");
        assert!(
            !content.starts_with("xxxx"),
            "live file is the new line, not seed"
        );
        assert!(
            content.contains("\"c\":1"),
            "live file has the new cycle's line"
        );
    }

    #[test]
    fn write_failure_returns_err_and_does_not_panic() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Path whose parent directory does NOT exist. OpenOptions::create(true)
        // creates the file but not the parent → open() fails. We expect Err,
        // NO panic, and the failed-writes counter increments.
        let path = dir
            .path()
            .join("nonexistent_subdir_xyz")
            .join("history.jsonl");
        let failures_before = failed_writes_total();

        let metrics = tiny_metrics();
        let wm = WorldModel::default();
        let dd = DriftDetector::default();
        let lp = LearnableParams::default();

        let result = append_history_snapshot(&path, &tiny_cfg(), &metrics, 1, &wm, &dd, &lp, 1.0);
        assert!(result.is_err(), "write to non-existent parent must fail");
        let msg = format!("{}", result.unwrap_err());
        // We don't pin the exact error kind (EACCES vs ENOENT varies by
        // platform) — only that we surfaced a non-empty error chain.
        assert!(!msg.is_empty(), "error chain is non-empty");
        assert!(
            failed_writes_total() > failures_before,
            "FAILED_WRITES counter did not increment (before={}, after={})",
            failures_before,
            failed_writes_total()
        );
    }

    #[test]
    fn startup_cap_makes_writer_noop_above_threshold() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history.jsonl");

        // tiny_cfg cap = 200 * 2 * 2 = 800 B. Pre-grow live file past that.
        std::fs::write(&path, vec![b'y'; 900]).expect("write oversized");

        let metrics = tiny_metrics();
        let wm = WorldModel::default();
        let dd = DriftDetector::default();
        let lp = LearnableParams::default();

        append_history_snapshot(&path, &tiny_cfg(), &metrics, 99, &wm, &dd, &lp, 1.0)
            .expect("no-op return Ok");

        // Live file is unchanged — the no-op must not have written.
        let content = std::fs::read_to_string(&path).expect("read");
        assert!(content.starts_with("yyyy"), "live file untouched above cap");
        // The process-global failure counter may be changed by another test;
        // unchanged file contents are the direct, race-free no-op proof.
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
    fn history_sync_cadence_batches_apfs_flushes() {
        assert!(should_sync_history(0));
        assert!(should_sync_history(1));
        assert!(!should_sync_history(2));
        assert!(!should_sync_history(29));
        assert!(should_sync_history(30));
        assert!(should_sync_history(60));
        assert!(!should_sync_history(61));
    }

    #[test]
    fn persistent_writer_amortizes_metadata_and_open_syscalls() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history.jsonl");
        let mut writer = MetricsHistoryWriter::new(path, HistoryConfig::default());
        let snapshot = PreparedHistorySnapshot {
            features: [0.0; 16],
            context_features: [0.0; N_CONTEXT_FEATURES],
            natural_drift: 0.0,
            nars_compile_confidence: 0.5,
            learned_hash: 0,
        };

        for cycle in 2..102 {
            writer.append(cycle, &snapshot).expect("append");
        }

        assert_eq!(writer.filesystem_refreshes, 4);
        assert_eq!(writer.file_opens, 4);
        let content = std::fs::read_to_string(&writer.path).expect("read history");
        assert_eq!(content.lines().count(), 100);
    }

    #[cfg(unix)]
    #[test]
    fn persistent_writer_rejects_path_replaced_by_symlink() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("history.jsonl");
        let target = dir.path().join("target.jsonl");
        let mut writer = MetricsHistoryWriter::new(path.clone(), HistoryConfig::default());
        let snapshot = PreparedHistorySnapshot {
            features: [0.0; 16],
            context_features: [0.0; N_CONTEXT_FEATURES],
            natural_drift: 0.0,
            nars_compile_confidence: 0.5,
            learned_hash: 0,
        };
        writer.append(2, &snapshot).expect("initial append");
        std::fs::remove_file(&path).expect("remove live path");
        std::fs::write(&target, b"untouched").expect("seed target");
        symlink(&target, &path).expect("replace with symlink");

        assert!(writer.append(30, &snapshot).is_err());
        assert_eq!(std::fs::read(&target).expect("read target"), b"untouched");
    }

    #[test]
    fn persistent_writer_recovers_after_transient_open_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let parent = dir.path().join("late-parent");
        let path = parent.join("history.jsonl");
        let mut writer = MetricsHistoryWriter::new(path.clone(), HistoryConfig::default());
        let snapshot = PreparedHistorySnapshot {
            features: [0.0; 16],
            context_features: [0.0; N_CONTEXT_FEATURES],
            natural_drift: 0.0,
            nars_compile_confidence: 0.5,
            learned_hash: 0,
        };

        assert!(writer.append(2, &snapshot).is_err());
        std::fs::create_dir(&parent).expect("create parent");
        writer.append(3, &snapshot).expect("retry succeeds");

        let content = std::fs::read_to_string(path).expect("read recovered history");
        assert_eq!(content.lines().count(), 1);
        assert!(content.contains("\"c\":3"));
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
