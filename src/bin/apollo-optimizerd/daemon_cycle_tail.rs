//! # Daemon Cycle Tail
//!
//! End-of-cycle blocks extracted from the daemon main loop as part of
//! the V1.1.0 Strangler Fig pass (Wave 10) [Fowler 2004].
//!
//! ## Ordering invariant (peer-review 2026-04-18)
//!
//! `Fluidity QoS → Enriched telemetry (incl. UCHS) → Periodic stage →
//! status broadcast`.
//!
//! - Fluidity QoS elevation must land BEFORE telemetry wiring so the
//!   cognitive metrics reflect this cycle's decision to prioritize UI
//!   fluidity (NotebookLM peer review §1).
//! - UCHS fields are merged into the same `state.metrics` lock guard as
//!   enriched telemetry; the two stages share one critical section to
//!   avoid a second round-trip through the mutex (NotebookLM §1, §3).
//! - Periodic stage (% 100 / % 500 / % 7200 gates) runs LAST so GC and
//!   persistence see a consistent `runtime_metrics.json` snapshot.
//!
//! ## Purity
//!
//! All four functions are shallow glue: they mutate through the locks /
//! `&mut` handles they already owned inline. No new allocations, no
//! new I/O, no new ordering.
//!
//! ## Shared-state carry-overs
//!
//! `frozen_state` and `mach_qos` remain **flat** `Arc<Mutex<…>>` fields on
//! `SharedState` — the thermal sentinel holds independent `Arc`s into
//! them. Do not bundle them into a sub-struct (NotebookLM §"Advertencia
//! de Bloqueo").

use std::collections::HashSet;
use std::path::Path;

use apollo_engine::collector::{SystemCollector, SystemSnapshot};
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::fluidity::FluidityState;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::lse_counters::CycleStage;
// Switch-4b: SchedulingTier no longer needed — mediator::MachPolicyKind drives dispatch
use apollo_engine::engine::overflow_guard::OverflowThresholds;
use apollo_engine::engine::pipeline::learning_context::LearningContext;
use apollo_engine::engine::pipeline::periodic_stage::{
    run_periodic, PeriodicContext, PeriodicResult,
};
use apollo_engine::engine::swap_predictor::SwapForecast;
use apollo_engine::engine::thermal_bailout::ThermalAction;

use crate::cognitive_tick::{CognitiveDecision, CognitiveState};

/// Elevate the foreground app to Foreground (P-Core) tier when a
/// window operation or app launch is in progress.
///
/// Skipped during thermal emergency (`force_ecores = true`) — the
/// P-cluster is already parked for survival.
///
/// Pre-conditions:
/// - `fluidity_state` has been updated this cycle.
/// - `thermal_action` has been evaluated this cycle.
///
/// Post-conditions:
/// - `state.mach_qos` may have one new Foreground-tier entry.
///
/// [Apple QoS Programming Guide 2014] user-interactive QoS =
/// render-frame priority on P-Cores (Firestorm).
pub fn apply_fluidity_qos(
    state: &SharedState,
    fluidity_state: &FluidityState,
    thermal_action: &ThermalAction,
    foreground_pid: Option<u32>,
) {
    if (fluidity_state.window_op_active() || fluidity_state.app_launching())
        && !thermal_action.force_ecores
    {
        if let Some(fg_pid) = foreground_pid {
            // Switch-4b (2026-06-03): route through MachPolicyEffector.
            let mach_effector =
                apollo_engine::engine::mediator::MachPolicyEffector::new(state.mach_qos.clone());
            let mach_eff = apollo_engine::engine::mediator::Effect::SetMachPolicy {
                pid: fg_pid,
                start_sec: 0,
                policy: apollo_engine::engine::mediator::MachPolicyKind::UserInteractive,
            };
            if apollo_engine::engine::mediator::mediate(
                &mach_eff,
                &apollo_engine::engine::mediator::PreCondition::default(),
                &mach_effector,
            )
            .is_ok()
            {
                tracing::debug!(
                    pid = fg_pid,
                    window_op = fluidity_state.window_op_active(),
                    launching = fluidity_state.launch_active,
                    "fluidity: elevated foreground to P-Core (Foreground QoS)"
                );
            }
        }
    }
}

/// Context bundle for [`wire_enriched_telemetry`].
///
/// Grouped because the function mutates ~20 fields of `RuntimeMetrics`
/// under a single `state.metrics` lock guard.
pub struct EnrichedTelemetryInputs<'a> {
    pub snapshot: &'a SystemSnapshot,
    pub swap_forecast: &'a SwapForecast,
    pub fluidity_state: &'a FluidityState,
    pub overflow_thresholds: &'a OverflowThresholds,
    pub behavior_interactive_pids: &'a HashSet<u32>,
    pub cog_decision: &'a CognitiveDecision,
    pub cognitive_state: &'a CognitiveState,
    pub lctx: &'a LearningContext<'a>,
    pub causal_qos_upgrades_cycle: u32,
    pub thermal_predicted_throttle: u8,
    pub thermal_seconds_to_throttle: Option<i32>,
    pub thermal_trend_predicted: &'a str,
    /// Number of recent foreground coalitions in the active envelope
    /// (Sprint Coalition 2026-05-10). 0 when nothing is foreground;
    /// 1-3 in steady-state app-switching.
    pub active_coalitions_count: u32,
    /// Lock-free metrics for Phase 0 lock-decomp instrumentation.
    pub lf_metrics: &'a apollo_engine::engine::lse_counters::LockFreeMetrics,
}

/// Wire enriched telemetry + UCHS neurocognitive metrics into
/// `RuntimeMetrics` under a single `state.metrics` lock guard.
///
/// Fields written here can only be computed in the main loop where
/// `swap_forecast`, `sys`, and per-cycle cognitive state are in scope.
///
/// Pre-conditions:
/// - `fluidity_state.windowserver_cpu_ema` has been updated this cycle
///   (via `fluidity_state.observe()` inside the proc-snapshot block).
/// - `cog_decision` is this cycle's fresh neurocognitive decision.
/// - `run_neurocognitive_tick` has already mutated `cognitive_state`.
///
/// Post-conditions:
/// - `state.metrics.metrics.*` has ~20 fields refreshed.
/// - `state.frozen_state` is briefly read-locked to sum frozen RAM.
///
/// Note: `collector.system()` is passed separately because we need the
/// `sysinfo::System` reference for per-PID RSS lookups.
pub fn wire_enriched_telemetry(
    state: &SharedState,
    collector: &SystemCollector,
    inputs: &EnrichedTelemetryInputs<'_>,
) {
    let mut m = state.metrics.lock_recover();
    // SwapTrend — previously computed but never exposed.
    m.metrics.swap_trend = format!("{:?}", inputs.swap_forecast.swap_trend);
    // WindowServer CPU — use EMA from FluidityState (already computed
    // each cycle in the proc_snaps block). More stable than raw sample.
    m.metrics.windowserver_cpu_pct = inputs.fluidity_state.windowserver_cpu_ema;
    // Compression signal from the EMA-smoothed compressor_pressure already
    // computed by the collector (ratio of compressor pages to total physical
    // pages × 0.85). The old formula used_ram - (total - free) was wrong:
    // on macOS total ≠ used + free (inactive/wired/speculative pages exist),
    // producing saturating_sub underflow → always 0 or nonsense.
    m.metrics.compressed_memory_ratio =
        inputs.snapshot.pressure.compressor_pressure.clamp(0.0, 1.0);
    // Frozen RAM: sum of RSS of currently frozen PIDs.
    let sys = collector.system();
    let frozen_pids = state.frozen_state.lock_recover().clone();
    m.metrics.frozen_ram_mb = frozen_pids
        .keys()
        .filter_map(|pid| sys.process(sysinfo::Pid::from_u32(*pid)))
        .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
        .sum::<f64>()
        .max(0.0);
    // cycles_high_pressure — consecutive cycles above bg_pressure.
    let bg_threshold = inputs.overflow_thresholds.bg_pressure;
    if inputs.snapshot.pressure.memory_pressure > bg_threshold {
        m.metrics.cycles_high_pressure = m.metrics.cycles_high_pressure.saturating_add(1);
    } else {
        m.metrics.cycles_high_pressure = 0;
    }
    // behavior_interactive_pid_count — how many PIDs learned dynamically.
    m.metrics.behavior_interactive_pid_count = inputs.behavior_interactive_pids.len();
    // rl_threshold_current — absolute threshold (bg_pressure + rl_adj).
    m.metrics.rl_threshold_current = bg_threshold + m.metrics.rl_adjustment_pp as f64 / 100.0;
    // ── UCHS / Neurocognitive metrics (8 cognitive modules) ──────────
    m.metrics.uchs_composite = inputs.cog_decision.uchs_composite;
    m.metrics.uchs_grade = inputs.cognitive_state.health.grade.clone();
    m.metrics.uchs_recovery_mode = inputs.cognitive_state.health.recovery_mode;
    m.metrics.epistemic_uncertainty = inputs.cognitive_state.epistemic.composite;
    m.metrics.epistemic_level = inputs.cognitive_state.epistemic.level_label().to_string();
    // Sprint Coalition 2026-05-10 metrics — guard-tower over-protection
    // signal (6th component of epistemic composite) + active-coalition
    // envelope size. Surfaces whether the new layered protection from
    // commits a381c6b..1ab6bdb is actually firing in production.
    m.metrics.guard_overprotection = inputs.cognitive_state.epistemic.guard_overprotection;
    m.metrics.active_coalitions_count = inputs.active_coalitions_count;
    // Phase 0 lock-decomp baseline (2026-05-10). Average over all
    // record_metrics_lock() observations since daemon start; max is
    // monotonic. If avg_wait << avg_held in steady state, the metrics
    // god-lock is held-time-bound not contention-bound, so
    // lock-decomposition would shift the bottleneck rather than eliminate it.
    let lf = inputs.lf_metrics;
    let wc = lf
        .metrics_lock_wait_count
        .load(std::sync::atomic::Ordering::Relaxed);
    let ws = lf
        .metrics_lock_wait_total_ns
        .load(std::sync::atomic::Ordering::Relaxed);
    let hc = lf
        .metrics_lock_held_count
        .load(std::sync::atomic::Ordering::Relaxed);
    let hs = lf
        .metrics_lock_held_total_ns
        .load(std::sync::atomic::Ordering::Relaxed);
    let (wm, hm) = lf.drain_metrics_lock_max_ns();
    m.metrics.metrics_lock_wait_avg_us = if wc > 0 {
        (ws as f64 / wc as f64) / 1000.0
    } else {
        0.0
    };
    m.metrics.metrics_lock_wait_max_us = wm / 1000;
    m.metrics.metrics_lock_held_avg_us = if hc > 0 {
        (hs as f64 / hc as f64) / 1000.0
    } else {
        0.0
    };
    m.metrics.metrics_lock_held_max_us = hm / 1000;
    // Phase 0b stage split.
    //
    // Windowed avg + windowed max — both drained per publish so producer
    // and consumer agree on the same time horizon. Previously the avg
    // divided a lifetime cumulative `stage_*_total_ns` by lifetime
    // `stage_count`, while the max was per-interval drained — this
    // structurally produced `avg_ms > max_ms` on tail-light stages
    // (esp. Persist) and leaked stale lifetime values into dashboards.
    // Sprint 9 `4b13a39` rule: producer + consumer agree on horizon.
    // [Welford 1962] online statistics windowing.
    let sc_window = lf.drain_stage_count_window();
    let to_avg_ms = |total_window: u64| -> f64 {
        if sc_window > 0 {
            (total_window as f64 / sc_window as f64) / 1_000_000.0
        } else {
            0.0
        }
    };
    let ns_to_ms = |ns: u64| -> f64 { ns as f64 / 1_000_000.0 };
    m.metrics.stage_sense_avg_ms = to_avg_ms(lf.drain_stage_total_ns(CycleStage::Sense));
    m.metrics.stage_sense_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::Sense));
    m.metrics.stage_reason_avg_ms = to_avg_ms(lf.drain_stage_total_ns(CycleStage::Reason));
    m.metrics.stage_reason_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::Reason));
    m.metrics.stage_execute_avg_ms = to_avg_ms(lf.drain_stage_total_ns(CycleStage::Execute));
    m.metrics.stage_execute_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::Execute));
    m.metrics.stage_learn_avg_ms = to_avg_ms(lf.drain_stage_total_ns(CycleStage::Learn));
    m.metrics.stage_learn_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::Learn));
    m.metrics.stage_persist_avg_ms = to_avg_ms(lf.drain_stage_total_ns(CycleStage::Persist));
    m.metrics.stage_persist_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::Persist));
    // REASON sub-stages (Phase 0c).
    m.metrics.stage_reason_signal_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonSignalTick));
    m.metrics.stage_reason_signal_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonSignalTick));
    m.metrics.stage_reason_neuro_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonNeuro));
    m.metrics.stage_reason_neuro_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonNeuro));
    m.metrics.stage_reason_decide_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonDecide));
    m.metrics.stage_reason_decide_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonDecide));
    m.metrics.stage_reason_usercontext_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonUserContext));
    m.metrics.stage_reason_usercontext_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonUserContext));
    m.metrics.stage_reason_holtwinters_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonHoltWinters));
    m.metrics.stage_reason_holtwinters_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonHoltWinters));
    m.metrics.stage_reason_pagereclaim_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonPageReclaim));
    m.metrics.stage_reason_pagereclaim_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonPageReclaim));
    m.metrics.stage_reason_chromium_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonChromium));
    m.metrics.stage_reason_chromium_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonChromium));
    m.metrics.stage_reason_enrich_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonEnrich));
    m.metrics.stage_reason_enrich_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonEnrich));
    // Additive instrumentation (2026-06-23): untimed enrich→decide ops.
    m.metrics.stage_reason_procscan_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonProcScan));
    m.metrics.stage_reason_procscan_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonProcScan));
    m.metrics.stage_reason_rusage_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonRusage));
    m.metrics.stage_reason_rusage_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonRusage));
    m.metrics.stage_reason_signalintel_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonSignalIntel));
    m.metrics.stage_reason_signalintel_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonSignalIntel));
    m.metrics.meta_confidence = inputs.cognitive_state.meta_cognition.meta_confidence;
    m.metrics.humble_mode = inputs.cog_decision.humble_mode;
    m.metrics.adversarial_pass_rate =
        inputs.cognitive_state.adversarial.lifetime_pass_rate() as f32;
    m.metrics.adversarial_safety_alert = inputs.cog_decision.safety_alert;
    m.metrics.cognitive_snr = inputs.cognitive_state.reward_bus.signal_to_noise();
    m.metrics.self_eval_quality = inputs.cognitive_state.self_evaluator.evaluator_trust();
    m.metrics.reptile_cached_workloads = inputs.cognitive_state.reptile.cached_workloads();
    m.metrics.drift_early_warning = inputs.lctx.outcome_tracker.drift_detector.early_warning();
    // Causal QoS upgrades this cycle (FreezeProcess → ThrottleProcess).
    m.metrics.causal_qos_upgrades_cycle = inputs.causal_qos_upgrades_cycle;
    // Predictive thermal state from ThermalManager (previously discarded).
    // seconds_to_throttle: null = no forecast, 0 = throttling now, >0 = seconds of headroom.
    m.metrics.thermal_predicted_throttle = inputs.thermal_predicted_throttle;
    m.metrics.thermal_seconds_to_throttle = inputs.thermal_seconds_to_throttle;
    m.metrics.thermal_trend_predicted = inputs.thermal_trend_predicted.to_string();
    // Phase-1 stall-candidate F2 (audit 2026-06-24): the metrics god-lock
    // covers a `sysinfo::System` walk for frozen-PID RSS lookups (~150us in
    // steady state, scales nonlinearly under pressure). Steady-state
    // `metrics_lock_held_max_us` is ~452us; firing only when it crosses
    // 5000us (10x headroom) keeps noise out of prod logs. Zero behavior
    // change — only log emission. [F2 MED-HIGH] per
    // /Users/eduardocortez/hardening-audit-2026-06-24/main-loop-stall-candidates.md
    //
    // Reading the freshly-stored field on `m` before drop is safe; this
    // value is the per-cycle peak (drained from lock-free atomics above).
    if m.metrics.metrics_lock_held_max_us > 5000 {
        tracing::warn!(
            target: "apollo.stall_candidate",
            held_max_us = m.metrics.metrics_lock_held_max_us,
            "stall_candidate_F2: metrics lock held >5ms this cycle (sysinfo walk under lock?)"
        );
    }
}

/// Context bundle for [`run_periodic_stage`].
///
/// A thin wrapper over [`PeriodicContext`]'s owned (non-`lctx`) fields —
/// keeps the main-loop call-site from re-listing every path and counter.
pub struct PeriodicStageInputs<'a> {
    pub cycle_count: u64,
    pub current_pressure: f64,
    pub workload_mode: &'a str,
    pub skills_path: &'a Path,
    pub hop_groups_path: &'a Path,
    pub signal_intel_path: &'a Path,
    pub learned_state_path: &'a Path,
    pub persist_generations: u32,
    pub last_restore_quality: Option<f64>,
    pub pending_trial_skill: Option<(String, f64)>,
}

/// Run the periodic maintenance stage (% 100 / % 500 / % 7200 gates).
///
/// Delegates to [`run_periodic`] with a freshly-constructed
/// [`PeriodicContext`]. The % 500 GC (experience compression, weight
/// prune, skill GC + persist) runs here; the % 100 persist and
/// rule-induction remain inline in main.rs above this call (they need
/// SharedState access); the % 7200 hourly GC also remains inline
/// (binary-local types: `cache_warmer`, `io_shaper`,
/// `temporal_predictor`).
///
/// Side effect: persists `optimization_skills.json` when the % 500
/// gate fires and new GC work occurred.
pub fn run_periodic_stage<'a>(
    inputs: PeriodicStageInputs<'a>,
    lctx: &mut LearningContext<'a>,
) -> PeriodicResult {
    let mut pctx = PeriodicContext {
        cycle_count: inputs.cycle_count,
        current_pressure: inputs.current_pressure,
        workload_mode: inputs.workload_mode,
        skills_path: inputs.skills_path,
        hop_groups_path: inputs.hop_groups_path,
        signal_intel_path: inputs.signal_intel_path,
        learned_state_path: inputs.learned_state_path,
        persist_generations: inputs.persist_generations,
        last_restore_quality: inputs.last_restore_quality,
        pending_trial_skill: inputs.pending_trial_skill,
        lctx,
    };
    run_periodic(&mut pctx)
}

/// S10 consumer: drain expired effect-decay observations, re-read each
/// observable, bump `effect_decay_detected_total` on disagreement.
///
/// Called once per main-loop cycle from the daemon tail. Bounded by
/// RING_CAP=64 (effect_decay module-level constant).
///
/// Wake-grace: callers MUST pass a `seconds_since_wake` value > 30
/// (or skip the call entirely) since immediately after wake the
/// kernel may not have reapplied tier hints — false-positive
/// disagreements would inflate the counter. The daemon's wake
/// tracking is in main.rs; this function trusts the caller.
///
/// FIX-3 wire (2026-06-07): forwards the observation into
/// `report_disagreement_with` so hard-protected disagreements feed
/// the 5-minute sliding window, then consults
/// `poke_rollback_guard_via_decay` once per cycle. When the window
/// crosses `HARD_PROTECTED_DECAY_THRESHOLD` and the rollback guard
/// has eligible shifts + no active cooldown, this is the path that
/// auto-reverts `zone_alpha` / `rl_pressure_bands[2]` to their
/// pre-shift values. Without this caller, the wire was dormant —
/// `poke_rollback_guard_via_decay` had zero invocations in the daemon.
pub fn drain_effect_decay(
    state: &SharedState,
    lp: &mut apollo_engine::engine::learned_state::LearnableParams,
) {
    let expired = {
        let mut w = state.effect_decay.lock_recover();
        w.drain_expired(std::time::Instant::now())
    };
    if expired.is_empty() {
        // Still consult the rollback guard — a previously-recorded
        // hard-protected disagreement window may have just crossed the
        // threshold even on a cycle with no new expirations.
        let (hp_count, hp_pids) = {
            let mut w = state.effect_decay.lock_recover();
            let now = std::time::Instant::now();
            (
                w.hard_protected_decay_count_5min(now),
                w.hard_protected_decay_pids(now),
            )
        };
        apollo_engine::engine::learned_state::poke_rollback_guard_via_decay(lp, hp_count, &hp_pids);
        return;
    }
    {
        let mut watchdog = state.effect_decay.lock_recover();
        for obs in expired {
            use apollo_engine::engine::effect_decay::ObsKind;
            // FIX-3-v2 (Round 3, Option B): MachPolicy attempts on
            // hard-protected processes ARE the disagreement signal — Apollo
            // trying to mutate a Chromium-protected process under pressure is
            // itself anomalous; no Mach FFI re-read needed.
            //
            // Round-4 (2026-06-07): route through `record_hp_mach_attempt`
            // (NOT report_disagreement_with) so the HP MachPolicy path bumps
            // its dedicated counter `effect_decay_hp_mach_attempts_total`,
            // leaving `effect_decay_detected_total` reserved for the
            // Jetsam/Sysctl re-read-disagreement baseline 27. Without the
            // split, baseline comparisons in metrics_to_watch are invalidated
            // because the same counter would mix two distinct signals.
            if matches!(obs.kind, ObsKind::MachPolicy) {
                if obs.hard_protected {
                    watchdog.record_hp_mach_attempt(&obs);
                }
                // Non-hard-protected MachPolicy: producer-side re-read
                // deferred — see banner. Skip.
                continue;
            }
            let live = match obs.kind {
                ObsKind::JetsamTier => {
                    apollo_engine::engine::jetsam_control::get_priority(obs.pid).map(|p| p as i64)
                }
                ObsKind::Sysctl => obs
                    .key
                    .as_deref()
                    .and_then(apollo_engine::engine::sysctl_direct::read_i32)
                    .map(|v| v as i64),
                ObsKind::MachPolicy => unreachable!("handled above"),
            };
            if let Some(actual) = live {
                if actual != obs.value_post {
                    watchdog.report_disagreement_with(&obs);
                }
            }
        }
    }
    // FIX-3 wire: consult the rollback guard once per cycle AFTER the
    // drain loop has updated the hard-protected sliding window. The
    // watchdog borrow is released above so we can re-lock it here for
    // the count/pids snapshot without deadlock.
    let (hp_count, hp_pids) = {
        let mut w = state.effect_decay.lock_recover();
        let now = std::time::Instant::now();
        (
            w.hard_protected_decay_count_5min(now),
            w.hard_protected_decay_pids(now),
        )
    };
    apollo_engine::engine::learned_state::poke_rollback_guard_via_decay(lp, hp_count, &hp_pids);
}
