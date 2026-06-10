//! # Daemon Signal Tick
//!
//! Per-cycle signal intelligence assembly extracted from main.rs (Wave 14).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Run SignalIntelligence.tick() (Kalman + CUSUM + Entropy + Hazard + LV + MPC)
//! - Apply Darwin-Boltzmann anomaly scoring and record TelemetryVector
//! - Wire fluidity signals into SignalDigest
//! - Compute last_pressure_velocity (D-term) and cache entropy_anomaly
//!
//! ## Ordering invariant
//! Must run AFTER fluidity_state, thermal_emergency, cycle_hw_snap, and
//! cycle_dt_secs are computed for this cycle. Uses lctx.signal_intel (&mut).
//! [NLM warning: pass cycle_dt_secs as parameter — never recalculate it here
//!  to avoid the mid-loop reset bug (dac6de9) that corrupted ODE models.]

use apollo_engine::collector::SystemSnapshot;
use apollo_engine::engine::daemon_helpers::audit_log;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::evolved_anomaly::EvolvedAnomalyDetector;
use apollo_engine::engine::fluidity::FluidityState;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::signal_intelligence::{SignalDigest, SignalIntelligence};
use apollo_engine::engine::telemetry_logger::{TelemetryLogger, TelemetryVector};

pub struct SignalTickOutput {
    pub signal_digest: SignalDigest,
    pub last_pressure_velocity: f64,
    pub entropy_anomaly: f64,
}

/// Per-cycle signal intelligence pass.
///
/// # Parameters
/// - `signal_intel` — mutable ref to lctx.signal_intel
/// - `snapshot` — system snapshot for this cycle
/// - `cycle_dt_secs` — inter-cycle delta (MUST be passed in, never re-measured here)
/// - `battery_percentage` / `battery_is_charging` — from power_mgr.battery_status
/// - `thermal_emergency` — from thermal_action.force_ecores
/// - `package_watts` — from cycle_hw_snap (None if IOKit unavailable)
/// - `hour_of_day` — for workload-aware bias
/// - `state` — SharedState (policy lock for workload bias)
/// - `darwin_anomaly` — Darwin-Boltzmann anomaly detector (mutable)
/// - `telemetry_logger` — ring-buffer logger (mutable)
/// - `fluidity_state` — fluidity state (read-only after fluidity tick ran)
/// - `cycle_count` — for DBAD audit rate gate
#[allow(clippy::too_many_arguments)]
pub fn run_signal_tick(
    signal_intel: &mut SignalIntelligence,
    snapshot: &SystemSnapshot,
    cycle_dt_secs: f64,
    battery_percentage: u32,
    battery_is_charging: bool,
    thermal_emergency: bool,
    package_watts: Option<f32>,
    hour_of_day: u8,
    state: &SharedState,
    darwin_anomaly: &mut EvolvedAnomalyDetector,
    telemetry_logger: &mut TelemetryLogger,
    fluidity_state: &FluidityState,
    cycle_count: u64,
) -> SignalTickOutput {
    // ── Signal intelligence tick ─────────────────────────────────────────────
    // Kalman + CUSUM + Entropy + Hazard + LV + MPC.
    let signal_digest = {
        let cpu_vals: Vec<f64> = snapshot
            .top_processes
            .iter()
            .map(|p| p.cpu_usage as f64)
            .collect();
        let mem_vals: Vec<f64> = snapshot
            .top_processes
            .iter()
            .map(|p| p.memory_usage as f64)
            .collect();
        let (dom_name, dom_bytes) = snapshot
            .top_processes
            .iter()
            .max_by_key(|p| p.memory_usage)
            .map(|p| (p.name.as_str(), p.memory_usage))
            .unwrap_or(("", 0));
        let total_used: u64 = snapshot.top_processes.iter().map(|p| p.memory_usage).sum();
        let swap_ratio = if snapshot.pressure.swap_total_bytes > 0 {
            snapshot.pressure.swap_used_bytes as f64 / snapshot.pressure.swap_total_bytes as f64
        } else {
            0.0
        };
        // Energy-aware routing: shift subsystem thresholds by battery/thermal.
        signal_intel.set_energy_bias(battery_percentage, battery_is_charging, thermal_emergency);
        // Power-aware bias: when real watts are high, engage optimizer earlier.
        // M1 Air TDP ~15W; >8W = active load, >12W = stressed.
        if let Some(pkg_w) = package_watts {
            signal_intel.adjust_bias_for_power(pkg_w);
        }
        // Workload-aware bias: heavy workloads (Coding/VideoEdit) spike pressure
        // fast — engage optimizer 2pp earlier during those hours.
        {
            let wl = state
                .policy
                .lock_recover()
                .adaptive_governor
                .user_profile
                .likely_workload_at_hour(hour_of_day);
            signal_intel.adjust_bias_for_workload(wl);
        }
        // UserProfile hygiene — once per ~hour (720 cycles ≈ 60 min @ 5s/cycle).
        // Apps idle >30d decay (total_foreground_secs *= 0.5); idle >90d evict
        // unless their lifetime usage is high (>50h grace window). Keeps
        // process_relevance reflecting current habits and prevents the model
        // from protecting ghosts of uninstalled apps.
        if cycle_count.is_multiple_of(720) {
            const DECAY_AFTER_SECS: u64 = 30 * 86_400; // 30 days
            const EVICT_AFTER_SECS: u64 = 90 * 86_400; // 90 days
            const DECAY_FACTOR: f32 = 0.5;
            const HIGH_USAGE_GRACE_SECS: u64 = 50 * 3_600; // 50 hours lifetime
            let evicted = state
                .policy
                .lock_recover()
                .adaptive_governor
                .user_profile
                .prune_stale(
                    DECAY_AFTER_SECS,
                    EVICT_AFTER_SECS,
                    DECAY_FACTOR,
                    HIGH_USAGE_GRACE_SECS,
                );
            if evicted > 0 {
                audit_log(&serde_json::json!({
                    "event": "user_profile_prune",
                    "evicted": evicted,
                }));
            }
        }
        signal_intel.tick(
            snapshot.pressure.memory_pressure,
            snapshot.pressure.swap_delta_bytes_per_sec,
            swap_ratio,
            snapshot.pressure.memory_pressure, // compressor proxy
            &cpu_vals,
            &mem_vals,
            dom_name,
            dom_bytes,
            total_used,
            snapshot.memory.total_ram,
            cycle_dt_secs,
        )
    };

    // ── Darwin-Boltzmann anomaly scoring + TelemetryVector ───────────────────
    // Feed signal digest into Hopfield memory + evolving SAE population for
    // learned anomaly detection. [Tuli et al. 2022]
    let signal_digest = {
        let mut d = signal_digest;
        if d.pressure_smooth >= 0.30 {
            d.memory_scan_available = true;
        }
        let dom_share = {
            let max_mem = snapshot
                .top_processes
                .iter()
                .map(|p| p.memory_usage)
                .max()
                .unwrap_or(0) as f64;
            let total = snapshot.memory.total_ram as f64;
            if total > 0.0 {
                (max_mem / total) as f32
            } else {
                0.0
            }
        };
        let thermal_score = match snapshot.pressure.thermal_level.as_str() {
            "nominal" => 0.0f32,
            "light" => 0.33,
            "serious" => 0.66,
            "critical" => 1.0,
            _ => 0.0,
        };
        let cpu_total = snapshot
            .top_processes
            .iter()
            .map(|p| p.cpu_usage)
            .sum::<f32>()
            / 100.0;
        let active_count = (snapshot.top_processes.len() as f32 / 200.0).min(1.0);
        let tv = TelemetryVector {
            pressure_smooth: d.pressure_smooth as f32,
            pressure_velocity: d.pressure_velocity as f32,
            pressure_predicted_5s: d.pressure_predicted_5s as f32,
            swap_velocity_smooth: (d.swap_velocity_smooth as f32).clamp(-5.0, 5.0),
            pressure_integral: d.pressure_integral as f32,
            cusum_score: d.cusum_score as f32,
            entropy_anomaly: d.entropy_anomaly as f32,
            p_oom_30s: d.p_oom_30s as f32,
            monopoly_risk: d.monopoly_risk as f32,
            urgency: d.urgency as f32,
            cpu_total: cpu_total.min(1.0),
            compressor_ratio: snapshot.pressure.memory_pressure as f32,
            dominant_share: dom_share,
            latency_score: 0.0,
            active_proc_count: active_count,
            thermal_score,
        };
        d.transformer_anomaly = darwin_anomaly.score(tv.as_f32_slice(), d.pressure_smooth as f32);
        // Record to TelemetryLogger ring buffer. record() self-triggers disk dumps
        // (event-triggered at OOM/urgency/latency thresholds, periodic ~10 min).
        // [Welch 1967, Tuli et al. 2022]
        telemetry_logger.record(tv);
        if d.transformer_anomaly > 0.3 || cycle_count.is_multiple_of(60) {
            audit_log(&serde_json::json!({
                "event": "dbad_score",
                "score": (d.transformer_anomaly * 1000.0).round() / 1000.0,
                "alpha": (darwin_anomaly.alpha() * 100.0).round() / 100.0,
                "samples": darwin_anomaly.sample_count(),
                "ready": darwin_anomaly.is_ready(),
                "pressure": (d.pressure_smooth * 1000.0).round() / 1000.0,
            }));
        }
        // Fluidity signals → SignalDigest.
        // [Jain 1991] Composite urgency includes fluidity degradation.
        d.fluidity_score = fluidity_state.fluidity_ema;
        d.window_op_active = fluidity_state.window_op_active();
        d.app_launching = fluidity_state.launch_active;
        if fluidity_state.fluidity_degraded {
            let fluidity_urgency = ((0.65 - fluidity_state.fluidity_ema as f64) * 0.4).max(0.0);
            d.urgency = (d.urgency + fluidity_urgency).min(1.0);
        }
        d
    };

    // ── D-term: last_pressure_velocity ──────────────────────────────────────
    // Use MV8 velocity when P[1,1] ≤ Q[1]=0.005 (converged).
    // Fallback to 1D KF until MV8 velocity covariance has stabilized.
    // [NotebookLM: gated switch prevents 200× velocity noise from cold start]
    let last_pressure_velocity = if signal_intel.kf_mv_velocity_converged() {
        signal_intel.kf_mv_pressure_velocity()
    } else {
        signal_digest.pressure_velocity
    };

    // G12: cache entropy_anomaly for next cycle's DRAM BP proxy.
    let entropy_anomaly = signal_digest.entropy_anomaly;

    SignalTickOutput {
        signal_digest,
        last_pressure_velocity,
        entropy_anomaly,
    }
}
