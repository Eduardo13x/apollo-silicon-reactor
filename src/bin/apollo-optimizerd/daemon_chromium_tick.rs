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
    thrashing_score: f64,
    media_active: bool,
    build_active: bool,
    call_in_progress: bool,
    llm_active: bool,
    external_4k_attached: bool,
) {
    // Step 2 (2026-05-11): conditional SIGSTOP enablement on crisis only.
    // Historical context (commit 712b927, Apr 21): SIGSTOP on chromium renderers
    // caused Brave IPC timeouts / beachballs. Kept globally disabled.
    // Step 2 reframes as "emergency-only": SIGSTOP fires only when memory is
    // genuinely tight (pressure ≥ 0.75 OR thrashing ≥ 10k). All other safety
    // gates (visibility, MAX_FREEZE_RATIO, assertion check, inet sockets,
    // NEW_RENDERER_GRACE) remain active. The pressure-floor (Step 1)
    // DemoteToEcores + PurgePurgeable continue to run unconditionally at
    // pressure ≥ 0.55, providing the soft escalation.
    //
    // Thrashing escalation (2026-05-11 evening): user reported swap saturation
    // at pressure ≈ 0.62 (below the 0.75 gate) — Apollo paralyzed while system
    // crawled. Compressor 0.77, thrashing 94k (9× crisis threshold), swap 88%.
    // Pressure-smooth alone misses sticky-swap states where the kernel has
    // already swapped out hot pages and pressure relaxes. Thrashing >10k is an
    // independent flow-crisis signal that justifies SIGSTOP on its own.
    // Workload-aware threshold relaxation. Priority strict-max — NOT additive,
    // because compounding gates (call+build+media → 0.40 SIGSTOP floor) would
    // breach the freeze_protected invariant Apollo trusts elsewhere. The chain
    // chooses ONE regime per cycle.
    //
    //   call_in_progress : strictest. Codec needs RAM+bandwidth, beachball =
    //                      unacceptable. Lower gate → more aggressive on
    //                      invisible browser tabs. freeze_protected still
    //                      vetoes SIGSTOP downstream, so effect is via
    //                      DemoteToEcores + PurgePurgeable only.
    //   build_active      : 2-4 GB rustc heap imminent. Preempt invisible tabs
    //                      BEFORE rustc spikes — Nygard 2018 bulkheading.
    //   media_active      : 4K decode + audio buffer compete with background
    //                      tabs. Same logic as commits 0f2c99c/8ab7d28.
    //   default           : crisis-only.
    //
    // CGWindowList visibility gate (chromium_manager.rs:919) ensures
    // foreground/visible tabs are NEVER demoted/frozen regardless of regime.
    // 2026-05-13 FINAL: SIGSTOP path PERMANENTLY DISABLED for chromium
    // renderers. User confirmed (multiple report cycles in this session)
    // that ANY threshold tuning still produces "tabs frozen, won't
    // resume, eventually crashed" — the historical regression that
    // motivated commit 712b927 (2026-04-17 permanent disable).
    //
    // Root cause is architectural per NotebookLM institutional memory:
    //   Brave's main process retries IPC to frozen renderers with no
    //   way for Apollo to intercept the timeout. The browser's own
    //   watchdog kills the tab before SIGCONT can land. No threshold
    //   makes this safe.
    //
    // Step 1 paths (DemoteToEcores + PurgePurgeable) remain ACTIVE —
    // they are scheduling/RAM hints, not freezes, and have no IPC
    // interaction risk. jetsam BACKGROUND demote under survival mode
    // (commit 59b449d) still handles real OOM via kernel-managed
    // discard. Maintenance Purge Gate (commit 41cf1161) cleans
    // pre-emptively at moderate pressure to prevent reaching the
    // crisis zone where Step 2 used to fire.
    //
    // Gate values below are kept (as comment) for future reference
    // if a non-SIGSTOP "freeze" mechanism becomes available (e.g.,
    // browser cooperation via CDP, Accessibility API).
    let _regime_history = (
        ("call",        0.55, 25_000.0),
        ("llm",         0.60, 30_000.0),
        ("media",       0.60, 30_000.0),
        ("build",       0.65, 35_000.0),
        ("external_4k", 0.65, 35_000.0),
        ("default",     0.75, 50_000.0),
    );
    let _ = (call_in_progress, llm_active, media_active, build_active, external_4k_attached);
    // Effective: chromium SIGSTOP unconditionally disabled.
    let (pressure_gate, thrashing_gate, regime): (f64, f64, &str) =
        (f64::INFINITY, f64::INFINITY, "disabled");
    // Surface regime for observability — silent regression to "default"
    // under crisis would otherwise be invisible.
    {
        let mut m = state.metrics.lock_recover();
        if m.metrics.chromium_gate_regime != regime {
            m.metrics.chromium_gate_regime = regime.to_string();
        }
    }
    let chromium_freeze_disabled = pressure_smooth < pressure_gate as f32
        && memory_pressure_at_freeze < pressure_gate
        && thrashing_score < thrashing_gate;

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
    // Step 2 (2026-05-11): hard-disable freeze emit when pressure is low,
    // so chromium_manager doesn't optimistically mark info.frozen for
    // actions the daemon-side gate would block.
    chromium_mgr.set_freeze_globally_disabled(chromium_freeze_disabled);

    // F3: CGWindowList visibility gate — refresh at most every 10 cycles (~5s).
    // Syscall costs ~1-3ms; window ownership rarely changes faster than that.
    if cycle_count.is_multiple_of(10) {
        let visible = apollo_engine::engine::cg_window::visible_pids();
        chromium_mgr.set_visible_pids(visible);
    }

    let chromium_assertion_pids = apollo_engine::engine::activity_sensor::pids_with_assertions();
    let main_frozen_set: HashSet<u32> = state.frozen_state.lock_recover().keys().copied().collect();
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
                // RAM Switch-4 (2026-06-03): route chromium E-core demotion
                // through MachPolicyEffector. Mediator chokepoint accounts
                // for the dispatch via mediator_blocks/postcondition counters
                // when applicable. The effector internally locks the manager
                // once; no outer guard needed at this site.
                let effector = apollo_engine::engine::mediator::MachPolicyEffector::new(
                    state.mach_qos.clone(),
                );
                let eff = apollo_engine::engine::mediator::Effect::SetMachPolicy {
                    pid: *pid,
                    start_sec: 0,
                    policy: apollo_engine::engine::mediator::MachPolicyKind::Background,
                };
                let pre = apollo_engine::engine::mediator::PreCondition::default();
                let _ = apollo_engine::engine::mediator::mediate(&eff, &pre, &effector);
            }
            ChromiumAction::PurgePurgeable { pid, name } => {
                // RAM Switch-5 (2026-06-03): route through PurgeableEffector
                // via mediator chokepoint. The effector internally walks
                // purgeable VM regions and issues madvise; the no_op signal
                // counts hits against the NLM-confirmed Brave Renderer
                // pressure_no_change pattern (corpus 2026-05-30).
                //
                // Region-count plumbing through the Receipt is deferred so
                // we avoid double-walking. The log line below loses the
                // exact region count; the mediator counters
                // (mediator_noop_writes_total) are the new ground truth.
                let effector = apollo_engine::engine::mediator::PurgeableEffector;
                let eff = apollo_engine::engine::mediator::Effect::PurgeHint {
                    pid: *pid,
                    start_sec: 0,
                    target_bytes: 0,
                };
                let pre = apollo_engine::engine::mediator::PreCondition::default();
                let receipt = apollo_engine::engine::mediator::mediate(&eff, &pre, &effector);
                let purged: u32 = match &receipt {
                    Ok(r) if !r.no_op => 1, // 1 = "at least one region purged"
                    _ => 0,
                };
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

    if !chromium_freeze_disabled {
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
                ChromiumAction::DemoteToEcores { .. } | ChromiumAction::PurgePurgeable { .. } => {
                    // Already handled in the unconditional block above.
                }
            }
        }
    }
}
