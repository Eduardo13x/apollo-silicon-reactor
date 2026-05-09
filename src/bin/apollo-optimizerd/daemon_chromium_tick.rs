//! # Daemon Chromium Tick
//!
//! Per-cycle Chromium Renderer Manager update extracted from the daemon main loop.
//! Wave 11 Strangler Fig extraction [Fowler 2004].
//!
//! ## Responsibilities
//! - Prime ChromiumManager context (Markov, pressure, arousal, build-preemption, fluidity)
//! - Refresh CGWindowList visibility every 10 cycles (~5s)
//! - Run `chromium_mgr.update()` to compute freeze/thaw/demotion actions
//! - Publish metrics unconditionally (observability decoupled from actuation)
//! - Execute ChromiumActions when `CHROMIUM_FREEZE_DISABLED = false`
//!
//! ## Ordering invariant
//! Must run AFTER `win_workload_intent` and `arousal_state` are computed for this cycle,
//! and AFTER main-loop freeze decisions (so excluded PIDs are already in `frozen_state`).
//! [NotebookLM peer review — 2026-04-22]

use std::collections::HashSet;

use apollo_engine::engine::chromium_manager::{ChromiumAction, ChromiumManager};
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::fluidity::FluidityState;
use apollo_engine::engine::focus_markov::FocusMarkov;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::mach_qos::SchedulingTier;
use apollo_engine::engine::nars_belief::ArousalState;
use apollo_engine::engine::process_classifier::ProcessSnapshot;
use apollo_engine::engine::process_identity::ProcessIdentity;
use apollo_engine::engine::types::{FreezeSource, FrozenEntry};
use apollo_engine::engine::window_sensor::WorkloadIntent;

/// Per-cycle Chromium renderer management tick.
///
/// Set `CHROMIUM_FREEZE_DISABLED = false` to re-enable renderer SIGSTOP/SIGCONT.
/// Inventory update and metrics run unconditionally regardless of the flag.
///
/// # Parameters
/// - `chromium_mgr` — ChromiumManager owning renderer state.
/// - `focus_markov` — FocusMarkov for predictive pre-thaw context.
/// - `foreground_app` — Current foreground app name (for Markov + update()).
/// - `foreground_pid` — Current foreground PID (for update()).
/// - `proc_snaps` — Process snapshot slice (CPU%, RSS, name).
/// - `state` — SharedState (metrics, frozen_state, mach_qos).
/// - `win_workload_intent` — Workload intent for build-preemption gate.
/// - `arousal_state` — Global arousal level (thresholds adapt to system stress).
/// - `fluidity_state` — Fluidity state (window-op + launch gates).
/// - `pressure_smooth` — Smoothed memory pressure [0,1].
/// - `memory_pressure_at_freeze` — Raw pressure at freeze time (stored in FrozenEntry).
/// - `cycle_count` — Current cycle number (CGWindowList refresh rate gate).
/// - `swap_velocity_bps` — Kalman-smoothed swap I/O velocity (bytes/sec) from
///   `signal_digest.swap_velocity_smooth`. Drives anticipatory E-core demotion of
///   invisible fg-browser tabs before pressure peaks.
#[allow(clippy::too_many_arguments)]
pub fn run_chromium_tick(
    chromium_mgr: &mut ChromiumManager,
    focus_markov: &FocusMarkov,
    foreground_app: Option<&str>,
    foreground_pid: Option<u32>,
    proc_snaps: &[ProcessSnapshot],
    state: &SharedState,
    win_workload_intent: WorkloadIntent,
    arousal_state: &ArousalState,
    fluidity_state: &FluidityState,
    pressure_smooth: f32,
    memory_pressure_at_freeze: f64,
    cycle_count: u64,
    swap_velocity_bps: f32,
) {
    // Set to false to re-enable Chromium renderer freezing.
    const CHROMIUM_FREEZE_DISABLED: bool = true;

    // Context setters prime thresholds used by update()'s freeze logic.
    // Run unconditionally so state is warm when CHROMIUM_FREEZE_DISABLED is set to false.
    {
        let preds: Vec<(String, f64, f64)> = focus_markov
            .predict_top_n(foreground_app.unwrap_or(""), 5)
            .into_iter()
            .map(|p| (p.app_name, p.probability, p.avg_dwell_secs))
            .collect();
        let elapsed = focus_markov.elapsed_dwell_secs();
        chromium_mgr.set_markov_context(&preds, elapsed);
    }
    chromium_mgr.set_pressure_context(pressure_smooth);
    chromium_mgr.set_arousal_context(arousal_state.level);
    chromium_mgr.set_build_preemption(win_workload_intent == WorkloadIntent::BuildSession);
    chromium_mgr.set_velocity_context(swap_velocity_bps);
    chromium_mgr.set_fluidity_context(
        fluidity_state.window_op_active(),
        fluidity_state.launch_active,
    );

    // F3: CGWindowList visibility gate — refresh at most every 10 cycles (~5s).
    // Syscall costs ~1-3ms; window ownership rarely changes faster than that.
    if cycle_count % 10 == 0 {
        let visible = apollo_engine::engine::cg_window::visible_pids();
        chromium_mgr.set_visible_pids(visible);
    }

    let chromium_assertion_pids =
        apollo_engine::engine::activity_sensor::pids_with_assertions();
    let main_frozen_set: HashSet<u32> =
        state.frozen_state.lock_recover().keys().copied().collect();
    let proc_list: Vec<(u32, &str, f32, u64)> = proc_snaps
        .iter()
        .map(|p| (p.pid, p.name.as_str(), p.cpu_percent, p.rss_bytes))
        .collect();

    let chromium_actions = chromium_mgr.update(
        &proc_list,
        foreground_pid,
        foreground_app,
        &chromium_assertion_pids,
        &main_frozen_set,
    );

    // Metrics always populated — renderer count visible even with freeze off.
    // [Jones 2011] observability must not be gated behind actuation flags.
    {
        let cm = chromium_mgr.metrics();
        let mut m = state.metrics.lock_recover();
        m.metrics.chromium_renderers_total = cm.total_renderers;
        m.metrics.chromium_renderers_frozen = cm.frozen_renderers;
        m.metrics.chromium_renderers_ecore = cm.ecore_renderers;
        m.metrics.chromium_freed_mb = cm.estimated_freed_mb;
        m.metrics.chromium_browsers_managed = cm.browsers_managed;
    }

    // Non-SIGSTOP actions run UNCONDITIONALLY. DemoteToEcores uses
    // PRIO_DARWIN_BG (turnstile-compatible, commit 97410cd) and PurgePurgeable
    // marks pages volatile via mach_vm_purgable_control — neither triggers the
    // Brave IPC timeout that motivated CHROMIUM_FREEZE_DISABLED.
    for action in &chromium_actions {
        match action {
            ChromiumAction::DemoteToEcores { pid, name } => {
                tracing::debug!(
                    pid = pid,
                    name = name.as_str(),
                    "chromium: E-core demotion for background renderer"
                );
                let mut qos = state.mach_qos.lock_recover();
                let _ = qos.set_tier(*pid, SchedulingTier::Background);
            }
            ChromiumAction::PurgePurgeable { pid, name } => {
                let purged =
                    apollo_engine::engine::compressor_aware::purge_purgeable_regions(*pid)
                        .unwrap_or(0);
                tracing::debug!(
                    pid = pid,
                    name = name.as_str(),
                    regions_purged = purged,
                    "chromium: velocity-anticipatory purgeable hint"
                );
            }
            _ => {} // SIGSTOP-related actions handled in the gated block below.
        }
    }

    if !CHROMIUM_FREEZE_DISABLED {
        for action in &chromium_actions {
            match action {
                ChromiumAction::FreezeRenderer {
                    pid,
                    name,
                    estimated_mb,
                } => {
                    tracing::info!(
                        pid = pid,
                        name = name.as_str(),
                        estimated_mb = estimated_mb,
                        "chromium: freezing idle renderer"
                    );
                    let ok = ChromiumManager::freeze_renderer(*pid);
                    // Confirm or roll back optimistic internal state from update().
                    // Keeps chromium_manager in sync with reality when SIGSTOP fails.
                    chromium_mgr.confirm_freeze(*pid, ok);
                    if ok {
                        let mut fs = state.frozen_state.lock_recover();
                        fs.entry(*pid).or_insert(FrozenEntry {
                            frozen_at: chrono::Utc::now(),
                            source: FreezeSource::ChromiumManager,
                            pressure_at_freeze: memory_pressure_at_freeze,
                            process_name: Some(name.clone()),
                            start_sec: ProcessIdentity::from_pid(*pid)
                                .map(|pi| pi.start_sec)
                                .unwrap_or(0),
                            original_jetsam_priority: None,
                        });
                    }
                }
                ChromiumAction::ThawRenderer { pid, name } => {
                    tracing::info!(
                        pid = pid,
                        name = name.as_str(),
                        "chromium: thawing renderer (became active)"
                    );
                    ChromiumManager::thaw_renderer(*pid);
                    // Restore Mach scheduling to Normal (P-cores) so renderer resumes fast.
                    {
                        let mut qos = state.mach_qos.lock_recover();
                        let _ = qos.set_tier(*pid, SchedulingTier::Normal);
                    }
                    state.frozen_state.lock_recover().remove(pid);
                    // NARS belief update: observe whether renderer survived freeze.
                    // [Pei Wang 2013] Revision rule accumulates evidence over time.
                    let alive = proc_snaps.iter().any(|p| p.pid == *pid);
                    chromium_mgr.observe_freeze_outcome(
                        name.as_str(),
                        alive,
                        if alive { 0.3 } else { 0.8 },
                    );
                }
                ChromiumAction::DemoteToEcores { .. }
                | ChromiumAction::PurgePurgeable { .. } => {
                    // Already handled in the unconditional block above.
                }
            }
        }
    }
}
