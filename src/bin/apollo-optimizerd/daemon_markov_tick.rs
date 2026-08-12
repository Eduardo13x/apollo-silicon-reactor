//! # Daemon Markov Tick
//!
//! FocusMarkov prediction + temporal predictor per-cycle tick extracted from
//! main.rs (Wave 29). [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - FocusMarkov miss check: score last high-confidence prediction
//! - Markov observe + predicted-app pre-warm (unfreeze + QoS + cache)
//! - Universal pre-thaw: categories matching predicted next app
//! - Temporal predictor: observe fg transitions, blend Markov + temporal, cache-warm
//!
//! ## Ordering invariant
//! Must run AFTER foreground detection (fg_state → foreground_app/pid) and BEFORE
//! the context-switch burst detector (which updates last_fg_name).

use std::path::Path;
use std::time::{Duration, Instant};

use apollo_engine::collector::SystemCollector;
use apollo_engine::engine::cache_warmer::CacheWarmer;
use apollo_engine::engine::coalition::CoalitionTracker;
use apollo_engine::engine::daemon_helpers::{unfreeze_pids_verified_outcome, write_frozen_state};
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::decision_ledger::{
    ActuatorDecisionEvent, ActuatorDecisionOutcome, CycleDecisionEvents,
};
use apollo_engine::engine::exploration_scheduler::{
    ActionClass, CommitEvidence, ExplorationArm, ExplorationCandidate, ExplorationContext,
    ExplorationGates, ExplorationMetadata, ExplorationMode, ExplorationScheduler, TimePoint,
};
use apollo_engine::engine::focus_markov::{FocusMarkov, PrewarmAdmission, PrewarmContext};
use apollo_engine::engine::freeze_intelligence::FreezeIntelligence;
use apollo_engine::engine::jetsam_control;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::mach_qos::{LatencyTier, SchedulingTier, ThroughputTier};
use apollo_engine::engine::process_identity::ProcessIdentity;
use apollo_engine::engine::process_tree::ProcessTree;
use apollo_engine::engine::safety::hard_protected_contains;
use apollo_engine::engine::telemetry_medallion::ActuatorFamily;
use apollo_engine::engine::temporal_predictor::TemporalPredictor;
use apollo_engine::engine::world_model::{ContextualActionBias, WorldModel};
use chrono::{Timelike, Utc};

const TEMPORAL_PREWARM_COOLDOWN_SECS: u64 = 120;

pub struct MarkovTickOutput {
    pub temporal_hour: u8,
    pub temporal_weekday: u8,
    pub decision_events: CycleDecisionEvents,
}

fn markov_event(
    target: impl Into<String>,
    cycle: u64,
    outcome: ActuatorDecisionOutcome,
    detail: impl Into<String>,
) -> ActuatorDecisionEvent {
    ActuatorDecisionEvent::local(
        "markov_prewarm:predicted_app",
        target,
        cycle,
        outcome,
        "focus-markov",
        detail,
    )
}

/// One speculative acceleration lease. A prediction is evaluated on an
/// actual foreground transition or a wall-clock deadline, never merely on
/// the next daemon cycle.
#[derive(Debug)]
pub struct MarkovPrewarmLease {
    source_app: String,
    predicted_app: String,
    acquired_at: Instant,
    members: Vec<PrewarmedMember>,
    cache_bytes: u64,
    expires_at: Instant,
    activated: bool,
    activated_at: Option<Instant>,
    settle_recorded: bool,
    calibration_probe: bool,
    calibration_context: PrewarmContext,
    exploration: Option<ExplorationMetadata>,
}

/// Passive prediction tracking. It never owns an actuator or kernel resource;
/// it only scores the next foreground transition against one Markov forecast.
/// `sampled_pair` prevents repeated timeout samples while the user remains in
/// the same foreground episode.
#[derive(Debug, Default)]
pub struct MarkovShadowTracker {
    active: Option<MarkovShadowLease>,
    sampled_pair: Option<(String, String)>,
    temporal_last_app: Option<String>,
    temporal_last_at: Option<Instant>,
}

#[derive(Debug)]
struct MarkovShadowLease {
    source_app: String,
    predicted_app: String,
    expires_at: Instant,
    calibration_context: PrewarmContext,
}

#[derive(Debug)]
struct PrewarmedMember {
    pid: u32,
    name: String,
    prior_jetsam: i32,
    jetsam_applied: bool,
    tier_applied: bool,
    task_qos_applied: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkovMemberEffect {
    Jetsam,
    MachTier,
    TaskQos,
    Cache,
}

fn markov_member_effect_event(
    pid: u32,
    name: &str,
    effect: MarkovMemberEffect,
    cycle: u64,
    outcome: ActuatorDecisionOutcome,
    detail: impl Into<String>,
) -> ActuatorDecisionEvent {
    let suffix = match effect {
        MarkovMemberEffect::Jetsam => "jetsam",
        MarkovMemberEffect::MachTier => "mach_tier",
        MarkovMemberEffect::TaskQos => "task_qos",
        MarkovMemberEffect::Cache => "cache",
    };
    ActuatorDecisionEvent::local(
        format!("markov_prewarm:{suffix}"),
        format!("{name}:pid:{pid}"),
        cycle,
        outcome,
        "focus-markov",
        detail,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeaseResolution {
    Pending,
    Hit,
    Completed,
    Miss,
}

fn markov_exploration_admission(
    admission: PrewarmAdmission,
    scheduler_approved: bool,
) -> (bool, bool) {
    match admission {
        PrewarmAdmission::Probe => (scheduler_approved, false),
        PrewarmAdmission::Ready => (true, true),
        PrewarmAdmission::Quarantined => (false, false),
    }
}

/// Run FocusMarkov + temporal predictor for this cycle.
///
/// # Parameters
/// - `foreground_app` — current foreground app name (from ForegroundDetector)
/// - `foreground_pid` — current foreground PID
/// - `last_fg_name` — fg name from previous cycle (for transition detection)
/// - `cycle_count` — current cycle index
/// - `focus_markov` — mutable FocusMarkov (observe + predict)
/// - `temporal_predictor` — mutable TemporalPredictor (observe + predict)
/// - `markov_prewarm` — active prediction lease being tracked for resolution
/// - `markov_hit_count` — cumulative hit counter for accuracy audit
/// - `markov_miss_count` — cumulative miss counter for accuracy audit
/// - `state` — SharedState (frozen_state, mach_qos, metrics)
/// - `collector` — SystemCollector (process table for PID lookup)
/// - `process_tree` — current-cycle hierarchy used to bound coalition candidates
/// - `coalition_tracker` — XNU resource-coalition identity verifier
/// - `cache_warmer` — for cache pre-warming predicted apps
/// - `frozen_state_path` — for write_frozen_state after pre-thaw
#[allow(clippy::too_many_arguments)]
pub fn run_markov_tick(
    foreground_app: Option<&str>,
    foreground_pid: Option<u32>,
    last_fg_name: Option<&str>,
    cycle_count: u64,
    focus_markov: &mut FocusMarkov,
    temporal_predictor: &mut TemporalPredictor,
    markov_prewarm: &mut Option<MarkovPrewarmLease>,
    markov_shadow: &mut MarkovShadowTracker,
    markov_hit_count: &mut u32,
    markov_miss_count: &mut u32,
    state: &SharedState,
    collector: &SystemCollector,
    process_tree: &ProcessTree,
    coalition_tracker: &CoalitionTracker,
    cache_warmer: &mut CacheWarmer,
    frozen_state_path: &Path,
    world_model: &WorldModel,
    exploration_scheduler: &mut ExplorationScheduler,
    exploration_gates: ExplorationGates,
    exploration_now: TimePoint,
) -> MarkovTickOutput {
    let mut decision_events = CycleDecisionEvents::default();
    // Fight-hunt fix (2026-06-10): prefetch is a luxury. Under pressure the
    // maintenance/survival paths are EVICTING file cache while these
    // warm_pid calls fault pages back in — Apollo fighting itself,
    // amplifying thrashing. Gate all speculative cache warming on pressure.
    let now = Instant::now();
    let (cache_warm_allowed, fluidity_degraded, app_launching, calibration_context, workload) = {
        let m = state.metrics.lock_recover();
        let pressure = m.metrics.memory_pressure;
        (
            cache_warm_allowed_at(pressure) && m.metrics.apollo_overhead_speculation_allowed,
            m.metrics.fluidity_degraded,
            m.metrics.app_launching,
            PrewarmContext::new(
                &m.metrics.current_workload,
                Utc::now().hour() as u8,
                pressure,
                m.metrics.app_launching || m.metrics.window_op_active,
            ),
            m.metrics.current_workload.clone(),
        )
    };

    // A foreground transition starts a new sampling episode. Resolve the old
    // passive forecast first, then permit one forecast for the new source app.
    let foreground_changed = foreground_app != last_fg_name;
    let shadow_resolution = markov_shadow
        .active
        .as_ref()
        .map(|lease| shadow_lease_resolution(lease, foreground_app, now))
        .unwrap_or(LeaseResolution::Pending);
    if matches!(
        shadow_resolution,
        LeaseResolution::Hit | LeaseResolution::Miss
    ) {
        if let Some(lease) = markov_shadow.active.take() {
            let hit = shadow_resolution == LeaseResolution::Hit;
            focus_markov.record_prewarm_outcome_with_context(
                &lease.source_app,
                &lease.predicted_app,
                hit,
                lease.calibration_context,
            );
            let mut metrics = state.metrics.lock_recover();
            metrics.metrics.markov_shadow_active = false;
            metrics.metrics.markov_shadow_resolved_total = metrics
                .metrics
                .markov_shadow_resolved_total
                .saturating_add(1);
            if hit {
                metrics.metrics.markov_shadow_hits =
                    metrics.metrics.markov_shadow_hits.saturating_add(1);
            } else {
                metrics.metrics.markov_shadow_misses =
                    metrics.metrics.markov_shadow_misses.saturating_add(1);
            }
        }
    }
    if foreground_changed {
        markov_shadow.sampled_pair = None;
    }

    if let Some(lease) = markov_prewarm.as_mut() {
        let target_is_foreground = foreground_app
            .map(|app| app_names_match(app, &lease.predicted_app))
            .unwrap_or(false);
        if maybe_record_settle(
            lease,
            target_is_foreground,
            fluidity_degraded,
            app_launching,
            now,
        ) {
            let settle_ms = lease
                .activated_at
                .map(|started| now.duration_since(started).as_millis() as u64)
                .unwrap_or(0);
            let mut metrics = state.metrics.lock_recover();
            metrics.metrics.markov_prewarm_last_settle_ms = settle_ms;
            metrics.metrics.markov_prewarm_settle_observations += 1;
        }
    }

    // ── Resolve the existing lease ──────────────────────────────────────────
    // The old implementation scored every prediction one daemon cycle later.
    // A user who simply stayed in the same app for two seconds produced a
    // false miss, followed by another identical kernel pre-warm. Resolve on
    // actual focus transitions or the prediction's wall-clock deadline.
    let resolution = markov_prewarm
        .as_ref()
        .map(|lease| lease_resolution(lease, foreground_app, now))
        .unwrap_or(LeaseResolution::Pending);
    match resolution {
        LeaseResolution::Hit => {
            if let Some(lease) = markov_prewarm.as_ref() {
                focus_markov.record_prewarm_outcome_with_context(
                    &lease.source_app,
                    &lease.predicted_app,
                    true,
                    lease.calibration_context,
                );
            }
            if let Some(lease) = markov_prewarm.as_mut() {
                lease.activated = true;
                lease.activated_at = Some(now);
                *markov_hit_count += 1;
                let mut metrics = state.metrics.lock_recover();
                metrics.metrics.markov_prewarm_hits += 1;
                metrics.metrics.markov_prewarm_last_lead_ms =
                    now.duration_since(lease.acquired_at).as_millis() as u64;
            }
        }
        LeaseResolution::Completed | LeaseResolution::Miss => {
            if resolution == LeaseResolution::Miss {
                if let Some(lease) = markov_prewarm.as_ref() {
                    let before = focus_markov.prewarm_admission_with_context(
                        &lease.source_app,
                        &lease.predicted_app,
                        lease.calibration_context,
                    );
                    focus_markov.record_prewarm_outcome_with_context(
                        &lease.source_app,
                        &lease.predicted_app,
                        false,
                        lease.calibration_context,
                    );
                    let after = focus_markov.prewarm_admission_with_context(
                        &lease.source_app,
                        &lease.predicted_app,
                        lease.calibration_context,
                    );
                    let mut metrics = state.metrics.lock_recover();
                    if before != PrewarmAdmission::Quarantined
                        && after == PrewarmAdmission::Quarantined
                    {
                        metrics.metrics.markov_prewarm_quarantines_total = metrics
                            .metrics
                            .markov_prewarm_quarantines_total
                            .saturating_add(1);
                    }
                }
                *markov_miss_count += 1;
                state.metrics.lock_recover().metrics.markov_prewarm_misses += 1;
            }
            if let Some(lease) = markov_prewarm.take() {
                let target = lease.predicted_app.clone();
                if now >= lease.expires_at {
                    decision_events.push(markov_event(
                        target.clone(),
                        cycle_count,
                        ActuatorDecisionOutcome::Expired,
                        "prediction lease reached its deadline",
                    ));
                }
                let release = release_markov_prewarm(lease, state, cycle_count);
                decision_events.extend_buffer(&release.decision_events);
            }
        }
        LeaseResolution::Pending => {}
    }

    let total = *markov_hit_count + *markov_miss_count;
    if matches!(resolution, LeaseResolution::Hit | LeaseResolution::Miss)
        && total > 0
        && total.is_multiple_of(50)
    {
        let accuracy = *markov_hit_count as f64 / total as f64;
        apollo_engine::engine::daemon_helpers::audit_log(&serde_json::json!({
            "event": "markov_prediction_accuracy",
            "hits": markov_hit_count,
            "misses": markov_miss_count,
            "accuracy": (accuracy * 1000.0).round() / 1000.0,
        }));
    }

    // ── Markov observe + predicted-app pre-warm ──────────────────────────────
    let markov_prediction = focus_markov.observe(foreground_app);
    let mut markov_acceleration_allowed = false;
    if let Some(ref pred) = markov_prediction {
        let elapsed = focus_markov.elapsed_dwell_secs();
        let time_to_switch = pred.avg_dwell_secs - elapsed;
        let horizon_5s = pred.confidence_within(elapsed, 5.0);
        let horizon_30s = pred.confidence_within(elapsed, 30.0);
        let horizon_2m = pred.confidence_within(elapsed, 120.0);
        let horizon_10m = pred.confidence_within(elapsed, 600.0);
        let source_app = foreground_app.unwrap_or_default();
        let admission = focus_markov.prewarm_admission_with_context(
            source_app,
            &pred.app_name,
            calibration_context,
        );
        let contextual_bias =
            world_model.contextual_action_bias("markov_prewarm:predicted_app", &workload);
        let probability_floor = contextual_prewarm_probability_floor(contextual_bias);
        let prediction_eligible = prediction_window_open(pred.probability, time_to_switch);
        let base_eligible = prewarm_window_open(
            pred.probability,
            time_to_switch,
            cache_warm_allowed,
            probability_floor,
        );
        let predicted_pid = find_running_pid(collector, &pred.app_name);
        let mut gates = exploration_gates;
        gates.markov_quarantined = admission == PrewarmAdmission::Quarantined;
        let identity = predicted_pid.and_then(ProcessIdentity::from_pid);
        gates.identity_present = identity.is_some();
        gates.identity_start_nonzero = identity
            .as_ref()
            .is_some_and(|identity| identity.start_sec > 0 || identity.start_usec > 0);
        gates.identity_recheck_ok =
            predicted_pid
                .zip(identity.as_ref())
                .is_some_and(|(pid, identity)| {
                    ProcessIdentity::verify(
                        pid,
                        Some(&pred.app_name),
                        identity.start_sec,
                        identity.start_usec,
                    )
                });
        gates.identity_recycled = !gates.identity_recheck_ok;
        gates.target_protected = predicted_pid.is_none_or(|pid| {
            hard_protected_contains(&pred.app_name)
                || apollo_engine::engine::apple_owned::is_apple_owned(pid)
        });
        gates.target_apple_owned =
            predicted_pid.is_none_or(apollo_engine::engine::apple_owned::is_apple_owned);
        let mut exploration_approval = if admission == PrewarmAdmission::Probe && base_eligible {
            ExplorationCandidate::new(
                ActuatorFamily::MarkovPrewarm,
                ExplorationMode::Treatment,
                ExplorationArm::MarkovCacheOnly,
                ActionClass::MarkovPredictedApp,
                ExplorationContext::Background,
                exploration_scheduler.origin(),
            )
            .ok()
            .and_then(|candidate| {
                exploration_scheduler
                    .request(&candidate, &gates, exploration_now)
                    .ok()
            })
        } else {
            None
        };
        if exploration_approval
            .as_ref()
            .is_some_and(|approval| exploration_scheduler.recheck(approval, &gates).is_err())
        {
            if let Some(approval) = exploration_approval.take() {
                exploration_scheduler.cancel(
                    approval.metadata.correlation,
                    apollo_engine::engine::exploration_scheduler::TerminalDiagnostic::Cancelled,
                );
            }
        }
        let (specialist_allowed, allow_kernel_acceleration) =
            markov_exploration_admission(admission, exploration_approval.is_some());
        let prewarm_eligible = base_eligible && specialist_allowed;
        // Cache-only probes may test a cold/new context. Reversible acceleration
        // remains reserved for mature transition evidence.
        markov_acceleration_allowed = allow_kernel_acceleration && cache_warm_allowed;
        let probe_transitions_remaining = focus_markov.prewarm_probe_transitions_remaining(
            source_app,
            &pred.app_name,
            calibration_context,
        );
        let mut blocker = prewarm_blocker(
            pred.probability,
            time_to_switch,
            cache_warm_allowed,
            probability_floor,
            admission,
        );
        {
            let mut metrics = state.metrics.lock_recover();
            metrics.metrics.markov_prediction_app = pred.app_name.clone();
            metrics.metrics.markov_prediction_confidence = pred.probability;
            metrics.metrics.markov_prediction_eta_secs = time_to_switch;
            metrics.metrics.markov_prediction_5s = horizon_5s;
            metrics.metrics.markov_prediction_30s = horizon_30s;
            metrics.metrics.markov_prediction_2m = horizon_2m;
            metrics.metrics.markov_prediction_10m = horizon_10m;
            metrics.metrics.markov_prediction_dwell_observations = pred.dwell_observations;
            metrics.metrics.markov_prediction_dwell_deviation_secs = pred.dwell_deviation_secs;
            metrics.metrics.markov_prewarm_eligible = prewarm_eligible;
            metrics.metrics.markov_prewarm_quarantined = admission == PrewarmAdmission::Quarantined;
            metrics.metrics.markov_prewarm_admission = admission.as_str().to_string();
            metrics.metrics.markov_prewarm_blocker = blocker.to_string();
            metrics.metrics.markov_prewarm_reliability = focus_markov
                .prewarm_reliability_with_context(source_app, &pred.app_name, calibration_context);
            metrics.metrics.markov_prewarm_context_trials = focus_markov.prewarm_context_trials(
                source_app,
                &pred.app_name,
                calibration_context,
            );
            metrics.metrics.markov_prewarm_probe_transitions_remaining =
                probe_transitions_remaining;
            if contextual_bias.is_informative() {
                metrics.metrics.world_model_contextual_markov_total = metrics
                    .metrics
                    .world_model_contextual_markov_total
                    .saturating_add(1);
                metrics.metrics.world_model_contextual_last_action =
                    "markov_prewarm:predicted_app".to_string();
                metrics.metrics.world_model_contextual_last_bias = contextual_bias.score;
                if contextual_bias.has_gpu_influence() {
                    metrics.metrics.record_gpu_contextual_influence(
                        "markov-prewarm",
                        "markov_prewarm:predicted_app",
                        contextual_bias.gpu_context_support,
                    );
                }
            }
            if base_eligible && admission == PrewarmAdmission::Quarantined {
                metrics.metrics.markov_prewarm_quarantine_skips_total = metrics
                    .metrics
                    .markov_prewarm_quarantine_skips_total
                    .saturating_add(1);
            }
        }

        if markov_prewarm.is_none() && prewarm_eligible {
            if let Some(pid) = predicted_pid {
                if markov_shadow.active.take().is_some() {
                    let mut metrics = state.metrics.lock_recover();
                    metrics.metrics.markov_shadow_active = false;
                    metrics.metrics.markov_shadow_superseded_total = metrics
                        .metrics
                        .markov_shadow_superseded_total
                        .saturating_add(1);
                }
                let (members, cache_bytes, unfrozen_count, conflict_skips, member_events) =
                    acquire_coalition_prewarm(
                        pid,
                        state,
                        collector,
                        process_tree,
                        coalition_tracker,
                        cache_warmer,
                        frozen_state_path,
                        allow_kernel_acceleration,
                        cycle_count,
                    );
                decision_events.extend_buffer(&member_events);
                let members_applied = members.len() as u64;
                let kernel_members = members
                    .iter()
                    .filter(|member| {
                        member.jetsam_applied || member.tier_applied || member.task_qos_applied
                    })
                    .count();
                let applied = !members.is_empty();
                let lease_secs = contextual_prewarm_lease_secs(
                    (time_to_switch.max(0.0) + 5.0).clamp(5.0, 20.0),
                    contextual_bias,
                );
                let acquired_at = Instant::now();
                let exploration = exploration_approval.and_then(|approval| {
                    if applied {
                        exploration_scheduler.commit_metadata(
                            approval.metadata.correlation,
                            exploration_now,
                            CommitEvidence::MutationApplied,
                        )
                    } else {
                        exploration_scheduler.commit(
                            approval.metadata.correlation,
                            exploration_now,
                            CommitEvidence::NoOp,
                        );
                        None
                    }
                });
                *markov_prewarm = Some(MarkovPrewarmLease {
                    source_app: foreground_app.unwrap_or_default().to_string(),
                    predicted_app: pred.app_name.clone(),
                    acquired_at,
                    members,
                    cache_bytes,
                    expires_at: acquired_at + Duration::from_secs_f64(lease_secs),
                    activated: false,
                    activated_at: None,
                    settle_recorded: false,
                    calibration_probe: admission == PrewarmAdmission::Probe,
                    calibration_context,
                    exploration: exploration.clone(),
                });
                if let Some(metadata) = exploration {
                    decision_events.push(
                        markov_event(
                            pred.app_name.clone(),
                            cycle_count,
                            ActuatorDecisionOutcome::Pending,
                            "bounded cache-only exploration started",
                        )
                        .with_exploration(metadata),
                    );
                }
                let mut metrics = state.metrics.lock_recover();
                metrics.metrics.markov_prewarm_attempts += 1;
                metrics.metrics.markov_prewarm_probes_total +=
                    u64::from(admission == PrewarmAdmission::Probe);
                metrics.metrics.markov_prewarm_applied += u64::from(applied);
                metrics.metrics.markov_prewarm_cache_only_total +=
                    u64::from(admission == PrewarmAdmission::Probe && applied);
                metrics.metrics.markov_prewarm_conflict_skips_total = metrics
                    .metrics
                    .markov_prewarm_conflict_skips_total
                    .saturating_add(conflict_skips);
                metrics.metrics.markov_prewarm_active = true;
                metrics.metrics.markov_prewarm_members = members_applied as u32;
                metrics.metrics.markov_prewarm_members_applied += members_applied;
                metrics.metrics.markov_prewarm_cache_bytes = metrics
                    .metrics
                    .markov_prewarm_cache_bytes
                    .saturating_add(cache_bytes);
                metrics.metrics.unfreezes_applied += unfrozen_count;
                tracing::debug!(
                    cycle = cycle_count,
                    pid,
                    app = pred.app_name.as_str(),
                    confidence = pred.probability,
                    time_to_switch,
                    cache_bytes,
                    members = members_applied,
                    kernel_members,
                    "markov: acquired predictive coalition acceleration lease"
                );
            } else {
                blocker = "target-not-running";
                state.metrics.lock_recover().metrics.markov_prewarm_blocker = blocker.to_string();
                decision_events.push(markov_event(
                    pred.app_name.clone(),
                    cycle_count,
                    ActuatorDecisionOutcome::Blocked,
                    blocker,
                ));
            }
        } else if markov_prewarm.is_none() {
            decision_events.push(markov_event(
                pred.app_name.clone(),
                cycle_count,
                markov_blocker_outcome(blocker),
                blocker,
            ));
        } else if prewarm_eligible {
            decision_events.push(markov_event(
                pred.app_name.clone(),
                cycle_count,
                ActuatorDecisionOutcome::NoOp,
                "matching prewarm lease already active",
            ));
        }

        // Score one passive prediction when no real lease is available. This
        // includes quarantined transitions and targets that are not running;
        // no cache, QoS, jetsam, signal, or process mutation occurs here.
        let pair = (source_app.to_string(), pred.app_name.clone());
        if markov_prewarm.is_none()
            && markov_shadow.active.is_none()
            && prediction_eligible
            && !source_app.is_empty()
            && markov_shadow.sampled_pair.as_ref() != Some(&pair)
        {
            let lease_secs = (time_to_switch.max(0.0) + 5.0).clamp(5.0, 20.0);
            markov_shadow.active = Some(MarkovShadowLease {
                source_app: pair.0.clone(),
                predicted_app: pair.1.clone(),
                expires_at: now + Duration::from_secs_f64(lease_secs),
                calibration_context,
            });
            markov_shadow.sampled_pair = Some(pair);
            let mut metrics = state.metrics.lock_recover();
            metrics.metrics.markov_shadow_predictions_total = metrics
                .metrics
                .markov_shadow_predictions_total
                .saturating_add(1);
            metrics.metrics.markov_shadow_active = true;
        }
    } else {
        let mut metrics = state.metrics.lock_recover();
        metrics.metrics.markov_prediction_app.clear();
        metrics.metrics.markov_prediction_confidence = 0.0;
        metrics.metrics.markov_prediction_eta_secs = 0.0;
        metrics.metrics.markov_prediction_5s = 0.0;
        metrics.metrics.markov_prediction_30s = 0.0;
        metrics.metrics.markov_prediction_2m = 0.0;
        metrics.metrics.markov_prediction_10m = 0.0;
        metrics.metrics.markov_prediction_dwell_observations = 0;
        metrics.metrics.markov_prediction_dwell_deviation_secs = 0.0;
        metrics.metrics.markov_prewarm_eligible = false;
        metrics.metrics.markov_prewarm_quarantined = false;
        metrics.metrics.markov_prewarm_admission.clear();
        metrics.metrics.markov_prewarm_blocker = "no-prediction".to_string();
        metrics.metrics.markov_prewarm_reliability = 0.5;
        metrics.metrics.markov_prewarm_context_trials = 0;
        metrics.metrics.markov_prewarm_probe_transitions_remaining = 0;
        decision_events.push(markov_event(
            "none",
            cycle_count,
            ActuatorDecisionOutcome::NoOp,
            "no-prediction",
        ));
    }

    // ── Universal pre-thaw: FocusMarkov → pre-thaw ALL frozen processes ──────
    // whose category matches the hint for the predicted next app.
    // [Altmann & Trafton 2002] Pre-activate resources before predicted task switch.
    if let Some(ref pred) = markov_prediction {
        if pred.probability >= 0.35 && markov_acceleration_allowed {
            let elapsed = focus_markov.elapsed_dwell_secs();
            let time_to_switch = pred.avg_dwell_secs - elapsed;
            if time_to_switch > -5.0 && time_to_switch < 10.0 {
                let hint_categories = FreezeIntelligence::pre_thaw_hint(&pred.app_name);
                let mut frozen_guard = state.frozen_state.lock_recover();
                let candidates: std::collections::HashMap<
                    u32,
                    apollo_engine::engine::types::FrozenEntry,
                > = frozen_guard
                    .iter()
                    .filter_map(|(&pid, entry)| {
                        let pname = entry.process_name.as_deref().unwrap_or("");
                        if !pname.is_empty() {
                            let cat = FreezeIntelligence::classify(pname);
                            if hint_categories.contains(&cat) {
                                return Some((pid, entry.clone()));
                            }
                        }
                        None
                    })
                    .collect();
                if !candidates.is_empty() {
                    let outcome = unfreeze_pids_verified_outcome(&candidates);
                    for pid in &outcome.applied_pids {
                        decision_events.push(ActuatorDecisionEvent::local(
                            "unfreeze:markov_prethaw",
                            format!("pid:{pid}"),
                            cycle_count,
                            ActuatorDecisionOutcome::Reverted,
                            "focus-markov",
                            "predicted category pre-thaw applied",
                        ));
                    }
                    for pid in &outcome.stale_pids {
                        decision_events.push(ActuatorDecisionEvent::local(
                            "unfreeze:markov_prethaw",
                            format!("pid:{pid}"),
                            cycle_count,
                            ActuatorDecisionOutcome::Blocked,
                            "focus-markov",
                            "stale process identity",
                        ));
                    }
                    for pid in &outcome.failed_pids {
                        decision_events.push(ActuatorDecisionEvent::local(
                            "unfreeze:markov_prethaw",
                            format!("pid:{pid}"),
                            cycle_count,
                            ActuatorDecisionOutcome::Failed,
                            "focus-markov",
                            "pre-thaw SIGCONT failed",
                        ));
                    }
                    for pid in outcome.forgettable_pids() {
                        frozen_guard.remove(&pid);
                    }
                    for pid in &outcome.applied_pids {
                        let pname = candidates
                            .get(pid)
                            .and_then(|entry| entry.process_name.as_deref())
                            .unwrap_or("");
                        tracing::info!(
                            pid,
                            process = pname,
                            predicted_app = pred.app_name.as_str(),
                            prob = pred.probability,
                            time_to_switch = time_to_switch,
                            "freeze_intelligence: universal pre-thaw — switch imminent"
                        );
                    }
                    write_frozen_state(frozen_state_path, &frozen_guard);
                }
            }
        }
    }

    // ── Temporal predictor ───────────────────────────────────────────────────
    // Shin et al. 2012 — temporal patterns predict app launches with ~80% accuracy.
    // Update hour/weekday unconditionally every cycle for pressure_headroom_for_incoming().
    let now_chrono = Utc::now();
    let mut temporal_hour = now_chrono.hour() as u8;
    let mut temporal_weekday = chrono::Datelike::weekday(&now_chrono).num_days_from_monday() as u8;

    if let Some(fg_name) = foreground_app {
        let now_chrono = Utc::now();
        let hour = now_chrono.hour() as u8;
        let weekday = chrono::Datelike::weekday(&now_chrono).num_days_from_monday() as u8;
        temporal_hour = hour;
        temporal_weekday = weekday;

        let fg_changed = last_fg_name != Some(fg_name);
        if fg_changed {
            temporal_predictor.observe(fg_name, hour, weekday);
        }

        let markov_probs: std::collections::HashMap<String, f64> = focus_markov
            .predict_top_n(fg_name, 5)
            .into_iter()
            .map(|p| (p.app_name, p.probability))
            .collect();
        let temporal_preds = temporal_predictor.predict(hour, weekday, &markov_probs);

        // Temporal-only warming is cache advisory and bounded to one candidate
        // per focus transition. It consumes the same World Model context as
        // Markov but never mutates QoS/jetsam state or competes for a ledger key.
        if cache_warm_allowed && foreground_changed {
            let contextual_bias =
                world_model.contextual_action_bias("markov_prewarm:predicted_app", &workload);
            let negative_bias = (-contextual_bias.score).clamp(0.0, 1.0);
            let probability_floor = 0.15 + 0.05 * negative_bias;
            let temporal_floor = 0.30 + 0.10 * negative_bias;
            if let Some(tpred) = temporal_preds.iter().find(|tpred| {
                !app_names_match(&tpred.app_name, fg_name)
                    && tpred.temporal_score > temporal_floor
                    && tpred.probability > probability_floor
                    && tpred.markov_score < 0.30
            }) {
                let cooldown_open = markov_shadow
                    .temporal_last_at
                    .is_none_or(|last| last.elapsed().as_secs() >= TEMPORAL_PREWARM_COOLDOWN_SECS);
                let candidate_changed =
                    markov_shadow.temporal_last_app.as_deref() != Some(tpred.app_name.as_str());
                if cooldown_open || candidate_changed {
                    if let Some(pid) = find_running_pid(collector, &tpred.app_name) {
                        let forecasts = crate::daemon_dispatch_tick::decision_time_forecasts(
                            world_model,
                            "markov_prewarm:predicted_app",
                            &workload,
                            120,
                        );
                        let bytes = cache_warmer.warm_pid(pid);
                        let mut event = markov_event(
                            tpred.app_name.clone(),
                            cycle_count,
                            if bytes > 0 {
                                ActuatorDecisionOutcome::Applied
                            } else {
                                ActuatorDecisionOutcome::NoOp
                            },
                            format!("temporal_cache_bytes={bytes}"),
                        );
                        for forecast in forecasts {
                            event = event.with_prediction(forecast);
                        }
                        decision_events.push(event);
                        markov_shadow.temporal_last_app = Some(tpred.app_name.clone());
                        markov_shadow.temporal_last_at = Some(Instant::now());
                        let mut metrics = state.metrics.lock_recover();
                        metrics.metrics.temporal_prewarm_attempts =
                            metrics.metrics.temporal_prewarm_attempts.saturating_add(1);
                        metrics.metrics.temporal_prewarm_applied = metrics
                            .metrics
                            .temporal_prewarm_applied
                            .saturating_add(u64::from(bytes > 0));
                        metrics.metrics.temporal_prewarm_cache_bytes = metrics
                            .metrics
                            .temporal_prewarm_cache_bytes
                            .saturating_add(bytes);
                        metrics.metrics.temporal_prewarm_last_app = tpred.app_name.clone();
                    }
                }
            }
        }

        // Suppress unused foreground_pid warning when not passed to jetsam paths above.
        let _ = foreground_pid;
    }

    MarkovTickOutput {
        temporal_hour,
        temporal_weekday,
        decision_events,
    }
}

fn app_names_match(actual: &str, predicted: &str) -> bool {
    actual.eq_ignore_ascii_case(predicted)
}

fn find_running_pid(collector: &SystemCollector, app_name: &str) -> Option<u32> {
    collector
        .system()
        .processes()
        .iter()
        .find(|(_, process)| process.name().eq_ignore_ascii_case(app_name))
        .map(|(pid, _)| pid.as_u32())
}

const MAX_PREWARM_MEMBERS: usize = 6;

/// Select a bounded process-tree family and confirm each child against the
/// root's XNU resource coalition. This keeps selection O(k) in the predicted
/// app's family rather than rescanning/probing every process on the host.
fn coalition_candidates(
    root_pid: u32,
    collector: &SystemCollector,
    process_tree: &ProcessTree,
    coalition_tracker: &CoalitionTracker,
) -> Vec<(u32, String, f32)> {
    let root_coalition = coalition_tracker.get_coalition_id(root_pid);
    let mut pids = process_tree.cascade_pids(root_pid);
    if !pids.contains(&root_pid) {
        pids.push(root_pid);
    }

    let candidates: Vec<(u32, String, f32)> = pids
        .into_iter()
        .filter(|pid| {
            *pid == root_pid
                || root_coalition == 0
                || coalition_tracker.get_coalition_id(*pid) == root_coalition
        })
        .filter_map(|pid| {
            collector
                .system()
                .process(sysinfo::Pid::from_u32(pid))
                .map(|process| (pid, process.name().to_string(), process.cpu_usage()))
        })
        .collect();

    rank_and_cap_candidates(candidates, root_pid)
}

fn rank_and_cap_candidates(
    mut candidates: Vec<(u32, String, f32)>,
    root_pid: u32,
) -> Vec<(u32, String, f32)> {
    // Root first for lifecycle/coordination, then hottest helpers. The fixed
    // cap bounds Mach calls and speculative cache footprint for large browsers.
    candidates.sort_by(|a, b| {
        let a_root = a.0 == root_pid;
        let b_root = b.0 == root_pid;
        b_root
            .cmp(&a_root)
            .then_with(|| b.2.total_cmp(&a.2))
            .then_with(|| a.0.cmp(&b.0))
    });
    candidates.truncate(MAX_PREWARM_MEMBERS);
    candidates
}

#[allow(clippy::too_many_arguments)]
fn acquire_coalition_prewarm(
    root_pid: u32,
    state: &SharedState,
    collector: &SystemCollector,
    process_tree: &ProcessTree,
    coalition_tracker: &CoalitionTracker,
    cache_warmer: &mut CacheWarmer,
    frozen_state_path: &Path,
    allow_kernel_acceleration: bool,
    cycle: u64,
) -> (Vec<PrewarmedMember>, u64, u64, u64, CycleDecisionEvents) {
    let mut decision_events = CycleDecisionEvents::default();
    let candidates = coalition_candidates(root_pid, collector, process_tree, coalition_tracker);
    let eligible: Vec<(u32, String, bool)> = candidates
        .into_iter()
        .filter_map(|(pid, name, _)| {
            let (user_target, kernel_boost_allowed) = prewarm_target_modes(
                pid,
                std::process::id(),
                apollo_engine::engine::process_identity::is_apple_platform_process(pid),
                apollo_engine::engine::safety::is_boost_forbidden(&name),
            );
            user_target.then_some((pid, name, kernel_boost_allowed))
        })
        .collect();

    // Remove every selected member under one lock and persist once. This avoids
    // one lock + JSON rewrite per helper in a multi-process application.
    let thawed = {
        let mut frozen_guard = state.frozen_state.lock_recover();
        let entries: std::collections::HashMap<u32, apollo_engine::engine::types::FrozenEntry> =
            if allow_kernel_acceleration {
                eligible
                    .iter()
                    .filter_map(|(pid, _, _)| {
                        frozen_guard.get(pid).map(|entry| (*pid, entry.clone()))
                    })
                    .collect()
            } else {
                std::collections::HashMap::new()
            };
        let outcome = unfreeze_pids_verified_outcome(&entries);
        for pid in outcome.forgettable_pids() {
            frozen_guard.remove(&pid);
        }
        if !entries.is_empty() {
            write_frozen_state(frozen_state_path, &frozen_guard);
        }
        decision_events.extend_buffer(
            &apollo_engine::engine::daemon_helpers::unfreeze_outcome_events(
                "unfreeze:markov_prewarm",
                "focus-markov",
                cycle,
                &outcome,
            ),
        );
        outcome
            .applied_pids
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
    };

    let mut members = Vec::with_capacity(eligible.len());
    let mut total_cache_bytes = 0u64;
    let mut conflict_skips = 0u64;
    for (pid, name, kernel_boost_allowed) in eligible {
        let kernel_boost_allowed = kernel_boost_allowed && allow_kernel_acceleration;
        let jetsam_effect =
            apollo_engine::engine::effect_ledger::AppliedEffect::JetsamPriority { pid, prior: -1 };
        let tier_effect = apollo_engine::engine::effect_ledger::AppliedEffect::MachTier { pid };
        let task_effect = apollo_engine::engine::effect_ledger::AppliedEffect::TaskQoS { pid };
        let jetsam_available = kernel_boost_allowed
            && !apollo_engine::engine::effect_ledger::is_global_tracked(&jetsam_effect);
        let tier_available = kernel_boost_allowed
            && !apollo_engine::engine::effect_ledger::is_global_tracked(&tier_effect);
        let task_available = kernel_boost_allowed
            && !apollo_engine::engine::effect_ledger::is_global_tracked(&task_effect);
        conflict_skips = conflict_skips
            .saturating_add(u64::from(kernel_boost_allowed && !jetsam_available))
            .saturating_add(u64::from(kernel_boost_allowed && !tier_available))
            .saturating_add(u64::from(kernel_boost_allowed && !task_available));

        if kernel_boost_allowed && !jetsam_available {
            decision_events.push(markov_member_effect_event(
                pid,
                &name,
                MarkovMemberEffect::Jetsam,
                cycle,
                ActuatorDecisionOutcome::Blocked,
                "effect ownership conflict",
            ));
        }
        if kernel_boost_allowed && !tier_available {
            decision_events.push(markov_member_effect_event(
                pid,
                &name,
                MarkovMemberEffect::MachTier,
                cycle,
                ActuatorDecisionOutcome::Blocked,
                "effect ownership conflict",
            ));
        }
        if kernel_boost_allowed && !task_available {
            decision_events.push(markov_member_effect_event(
                pid,
                &name,
                MarkovMemberEffect::TaskQos,
                cycle,
                ActuatorDecisionOutcome::Blocked,
                "effect ownership conflict",
            ));
        }

        let prior_jetsam = jetsam_available
            .then(|| jetsam_control::get_priority(pid).unwrap_or(-1))
            .unwrap_or(-1);
        let jetsam_applied = if jetsam_available {
            match jetsam_control::set_priority(pid, jetsam_control::priority::FOREGROUND) {
                Ok(()) => {
                    decision_events.push(markov_member_effect_event(
                        pid,
                        &name,
                        MarkovMemberEffect::Jetsam,
                        cycle,
                        ActuatorDecisionOutcome::Applied,
                        "foreground jetsam band applied",
                    ));
                    true
                }
                Err(error) => {
                    decision_events.push(markov_member_effect_event(
                        pid,
                        &name,
                        MarkovMemberEffect::Jetsam,
                        cycle,
                        ActuatorDecisionOutcome::Failed,
                        format!("jetsam apply failed: {error}"),
                    ));
                    false
                }
            }
        } else {
            false
        };
        let (tier_applied, task_qos_applied) = if tier_available || task_available {
            let mut qos = state.mach_qos.lock_recover();
            let tier = tier_available.then(|| qos.set_tier(pid, SchedulingTier::Foreground));
            let task_qos = task_available.then(|| {
                qos.set_latency_and_throughput(pid, LatencyTier::Interactive, ThroughputTier::High)
            });
            if let Some(outcome) = tier.as_ref() {
                decision_events.push(markov_member_effect_event(
                    pid,
                    &name,
                    MarkovMemberEffect::MachTier,
                    cycle,
                    if outcome.mutated {
                        ActuatorDecisionOutcome::Applied
                    } else if outcome.success {
                        ActuatorDecisionOutcome::NoOp
                    } else {
                        ActuatorDecisionOutcome::Failed
                    },
                    outcome.error.as_deref().unwrap_or("Mach tier evaluated"),
                ));
            }
            if let Some(outcome) = task_qos.as_ref() {
                decision_events.push(markov_member_effect_event(
                    pid,
                    &name,
                    MarkovMemberEffect::TaskQos,
                    cycle,
                    if outcome.mutated {
                        ActuatorDecisionOutcome::Applied
                    } else if outcome.success {
                        ActuatorDecisionOutcome::NoOp
                    } else {
                        ActuatorDecisionOutcome::Failed
                    },
                    outcome.error.as_deref().unwrap_or("task QoS evaluated"),
                ));
            }
            (
                tier.is_some_and(|outcome| outcome.mutated),
                task_qos.is_some_and(|outcome| outcome.mutated),
            )
        } else {
            (false, false)
        };
        let cache_bytes = cache_warmer.warm_pid(pid);
        decision_events.push(markov_member_effect_event(
            pid,
            &name,
            MarkovMemberEffect::Cache,
            cycle,
            if cache_bytes > 0 {
                ActuatorDecisionOutcome::Applied
            } else {
                ActuatorDecisionOutcome::NoOp
            },
            format!("cache_bytes={cache_bytes}"),
        ));
        total_cache_bytes = total_cache_bytes.saturating_add(cache_bytes);
        let unfrozen = thawed.contains(&pid);

        let (warm_sec, _) = apollo_engine::engine::daemon_helpers::pid_start_time(pid);
        if jetsam_applied {
            apollo_engine::engine::effect_ledger::record_global(
                apollo_engine::engine::effect_ledger::AppliedEffect::JetsamPriority {
                    pid,
                    prior: prior_jetsam,
                },
                apollo_engine::engine::effect_ledger::DEFAULT_TTL,
                warm_sec,
                "markov coalition pre-warm: jetsam FOREGROUND",
            );
        }
        if tier_applied {
            apollo_engine::engine::effect_ledger::record_global(
                apollo_engine::engine::effect_ledger::AppliedEffect::MachTier { pid },
                apollo_engine::engine::effect_ledger::DEFAULT_TTL,
                warm_sec,
                "markov coalition pre-warm: Foreground tier",
            );
        }
        if task_qos_applied {
            apollo_engine::engine::effect_ledger::record_global(
                apollo_engine::engine::effect_ledger::AppliedEffect::TaskQoS { pid },
                apollo_engine::engine::effect_ledger::DEFAULT_TTL,
                warm_sec,
                "markov coalition pre-warm: interactive task QoS",
            );
        }

        if unfrozen || jetsam_applied || tier_applied || task_qos_applied || cache_bytes > 0 {
            members.push(PrewarmedMember {
                pid,
                name,
                prior_jetsam,
                jetsam_applied,
                tier_applied,
                task_qos_applied,
            });
        }
    }

    (
        members,
        total_cache_bytes,
        thawed.len() as u64,
        conflict_skips,
        decision_events,
    )
}

fn prewarm_window_open(
    probability: f64,
    time_to_switch: f64,
    resources_healthy: bool,
    probability_floor: f64,
) -> bool {
    resources_healthy
        && probability >= probability_floor.clamp(0.50, 0.54)
        && time_to_switch.is_finite()
        && (-2.0..=12.0).contains(&time_to_switch)
}

fn prewarm_blocker(
    probability: f64,
    time_to_switch: f64,
    resources_healthy: bool,
    probability_floor: f64,
    admission: PrewarmAdmission,
) -> &'static str {
    if !resources_healthy {
        "resource-gate"
    } else if probability < probability_floor.clamp(0.50, 0.54) {
        "confidence"
    } else if !time_to_switch.is_finite() || !(-2.0..=12.0).contains(&time_to_switch) {
        "timing"
    } else if admission == PrewarmAdmission::Quarantined {
        "quarantine"
    } else {
        "ready"
    }
}

fn markov_blocker_outcome(blocker: &str) -> ActuatorDecisionOutcome {
    if blocker == "quarantine" {
        ActuatorDecisionOutcome::Vetoed
    } else {
        ActuatorDecisionOutcome::Blocked
    }
}

fn contextual_prewarm_probability_floor(bias: ContextualActionBias) -> f64 {
    if !bias.is_informative() {
        return 0.50;
    }
    let score = bias.score.clamp(-1.0, 1.0);
    if score < 0.0 {
        0.50 - 0.04 * score
    } else {
        0.50
    }
}

fn contextual_prewarm_lease_secs(base_secs: f64, bias: ContextualActionBias) -> f64 {
    if !bias.is_informative() {
        return base_secs.clamp(5.0, 20.0);
    }
    (base_secs * (1.0 + 0.15 * bias.score.clamp(-1.0, 1.0))).clamp(5.0, 20.0)
}

fn prediction_window_open(probability: f64, time_to_switch: f64) -> bool {
    probability >= 0.50 && time_to_switch.is_finite() && (-2.0..=12.0).contains(&time_to_switch)
}

fn shadow_lease_resolution(
    lease: &MarkovShadowLease,
    foreground_app: Option<&str>,
    now: Instant,
) -> LeaseResolution {
    if foreground_app
        .map(|app| app_names_match(app, &lease.predicted_app))
        .unwrap_or(false)
    {
        LeaseResolution::Hit
    } else if foreground_app
        .map(|app| !app_names_match(app, &lease.source_app))
        .unwrap_or(false)
        || now >= lease.expires_at
    {
        LeaseResolution::Miss
    } else {
        LeaseResolution::Pending
    }
}

fn maybe_record_settle(
    lease: &mut MarkovPrewarmLease,
    target_is_foreground: bool,
    fluidity_degraded: bool,
    app_launching: bool,
    now: Instant,
) -> bool {
    if !lease.activated
        || !target_is_foreground
        || lease.settle_recorded
        || lease.activated_at == Some(now)
    {
        return false;
    }
    if !fluidity_degraded && !app_launching {
        lease.settle_recorded = true;
        return true;
    }
    false
}

fn prewarm_target_modes(
    pid: u32,
    own_pid: u32,
    apple_platform: bool,
    boost_forbidden: bool,
) -> (bool, bool) {
    let user_target = pid != own_pid && !apple_platform;
    (user_target, user_target && !boost_forbidden)
}

fn lease_resolution(
    lease: &MarkovPrewarmLease,
    foreground_app: Option<&str>,
    now: Instant,
) -> LeaseResolution {
    let target_is_foreground = foreground_app
        .map(|app| app_names_match(app, &lease.predicted_app))
        .unwrap_or(false);
    if lease.activated {
        // A hit does not turn a speculative lease into permanent priority.
        // Keep only the short launch/transition window, then restore every
        // reversible kernel mutation even if the app remains foreground.
        if now >= lease.expires_at {
            LeaseResolution::Completed
        } else if target_is_foreground || foreground_app.is_none() {
            LeaseResolution::Pending
        } else {
            LeaseResolution::Completed
        }
    } else if target_is_foreground {
        LeaseResolution::Hit
    } else if foreground_app
        .map(|app| !app_names_match(app, &lease.source_app))
        .unwrap_or(false)
    {
        // A next-app prediction is disproven as soon as a different app wins
        // focus. Do not hold speculative priority until the timer expires.
        LeaseResolution::Miss
    } else if now >= lease.expires_at {
        LeaseResolution::Miss
    } else {
        LeaseResolution::Pending
    }
}

/// Release all reversible kernel state associated with one prediction.
/// Also used on clean daemon shutdown so a pre-warm cannot ratchet across a
/// service restart.
#[derive(Debug, Default)]
pub struct MarkovReleaseReport {
    pub reverted_effects: u64,
    pub deferred_effects: u64,
    pub decision_events: CycleDecisionEvents,
}

pub fn release_markov_prewarm(
    lease: MarkovPrewarmLease,
    state: &SharedState,
    cycle: u64,
) -> MarkovReleaseReport {
    let mut reverted = false;
    let mut reverted_effects = 0u64;
    let mut deferred_effects = 0u64;
    let mut decision_events = CycleDecisionEvents::default();
    let member_count = lease.members.len();
    let cache_bytes = lease.cache_bytes;
    let calibration_probe = lease.calibration_probe;
    let exploration = lease.exploration.clone();
    let exploration_target = lease.predicted_app.clone();
    for member in lease.members {
        if member.jetsam_applied {
            let effect = apollo_engine::engine::effect_ledger::AppliedEffect::JetsamPriority {
                pid: member.pid,
                prior: member.prior_jetsam,
            };
            const OWNER: &str = "markov coalition pre-warm: jetsam FOREGROUND";
            if apollo_engine::engine::effect_ledger::is_global_owner(&effect, OWNER) {
                match prewarm_jetsam_restore(member.prior_jetsam) {
                    Some(restore) => {
                        if jetsam_control::set_priority(member.pid, restore).is_ok() {
                            apollo_engine::engine::effect_ledger::forget_global_if_justification(
                                &effect, OWNER,
                            );
                            reverted_effects += 1;
                            reverted = true;
                            decision_events.push(markov_member_effect_event(
                                member.pid,
                                &member.name,
                                MarkovMemberEffect::Jetsam,
                                cycle,
                                ActuatorDecisionOutcome::Reverted,
                                format!("restored priority={restore}"),
                            ));
                        } else {
                            deferred_effects += 1;
                            decision_events.push(markov_member_effect_event(
                                member.pid,
                                &member.name,
                                MarkovMemberEffect::Jetsam,
                                cycle,
                                ActuatorDecisionOutcome::Failed,
                                "jetsam restore failed",
                            ));
                        }
                    }
                    None => {
                        apollo_engine::engine::effect_ledger::forget_global_if_justification(
                            &effect, OWNER,
                        );
                        decision_events.push(markov_member_effect_event(
                            member.pid,
                            &member.name,
                            MarkovMemberEffect::Jetsam,
                            cycle,
                            ActuatorDecisionOutcome::NoOp,
                            "prior jetsam priority unavailable",
                        ));
                    }
                }
            } else {
                decision_events.push(markov_member_effect_event(
                    member.pid,
                    &member.name,
                    MarkovMemberEffect::Jetsam,
                    cycle,
                    ActuatorDecisionOutcome::NoOp,
                    "effect ownership no longer held",
                ));
            }
        }
        if member.tier_applied || member.task_qos_applied {
            let mut qos = state.mach_qos.lock_recover();
            if member.tier_applied {
                let effect = apollo_engine::engine::effect_ledger::AppliedEffect::MachTier {
                    pid: member.pid,
                };
                const OWNER: &str = "markov coalition pre-warm: Foreground tier";
                if apollo_engine::engine::effect_ledger::is_global_owner(&effect, OWNER) {
                    let outcome = qos.set_tier(member.pid, SchedulingTier::Normal);
                    if outcome.mutated {
                        apollo_engine::engine::effect_ledger::forget_global_if_justification(
                            &effect, OWNER,
                        );
                        reverted_effects += 1;
                        reverted = true;
                        decision_events.push(markov_member_effect_event(
                            member.pid,
                            &member.name,
                            MarkovMemberEffect::MachTier,
                            cycle,
                            ActuatorDecisionOutcome::Reverted,
                            "Mach tier restored to Normal",
                        ));
                    } else if outcome.success
                        || qos.current_tier(member.pid) == Some(SchedulingTier::Normal)
                    {
                        apollo_engine::engine::effect_ledger::forget_global_if_justification(
                            &effect, OWNER,
                        );
                        decision_events.push(markov_member_effect_event(
                            member.pid,
                            &member.name,
                            MarkovMemberEffect::MachTier,
                            cycle,
                            ActuatorDecisionOutcome::NoOp,
                            "Mach tier already Normal",
                        ));
                    } else {
                        deferred_effects += 1;
                        decision_events.push(markov_member_effect_event(
                            member.pid,
                            &member.name,
                            MarkovMemberEffect::MachTier,
                            cycle,
                            ActuatorDecisionOutcome::Failed,
                            outcome
                                .error
                                .as_deref()
                                .unwrap_or("Mach tier restore failed"),
                        ));
                    }
                } else {
                    decision_events.push(markov_member_effect_event(
                        member.pid,
                        &member.name,
                        MarkovMemberEffect::MachTier,
                        cycle,
                        ActuatorDecisionOutcome::NoOp,
                        "effect ownership no longer held",
                    ));
                }
            }
            if member.task_qos_applied {
                let effect = apollo_engine::engine::effect_ledger::AppliedEffect::TaskQoS {
                    pid: member.pid,
                };
                const OWNER: &str = "markov coalition pre-warm: interactive task QoS";
                if apollo_engine::engine::effect_ledger::is_global_owner(&effect, OWNER) {
                    let outcome = qos.set_latency_and_throughput(
                        member.pid,
                        LatencyTier::Default,
                        ThroughputTier::Default,
                    );
                    if outcome.mutated {
                        apollo_engine::engine::effect_ledger::forget_global_if_justification(
                            &effect, OWNER,
                        );
                        reverted_effects += 1;
                        reverted = true;
                        decision_events.push(markov_member_effect_event(
                            member.pid,
                            &member.name,
                            MarkovMemberEffect::TaskQos,
                            cycle,
                            ActuatorDecisionOutcome::Reverted,
                            "task QoS restored to defaults",
                        ));
                    } else if outcome.success {
                        apollo_engine::engine::effect_ledger::forget_global_if_justification(
                            &effect, OWNER,
                        );
                        decision_events.push(markov_member_effect_event(
                            member.pid,
                            &member.name,
                            MarkovMemberEffect::TaskQos,
                            cycle,
                            ActuatorDecisionOutcome::NoOp,
                            "task QoS already default",
                        ));
                    } else {
                        deferred_effects += 1;
                        decision_events.push(markov_member_effect_event(
                            member.pid,
                            &member.name,
                            MarkovMemberEffect::TaskQos,
                            cycle,
                            ActuatorDecisionOutcome::Failed,
                            outcome
                                .error
                                .as_deref()
                                .unwrap_or("task QoS restore failed"),
                        ));
                    }
                } else {
                    decision_events.push(markov_member_effect_event(
                        member.pid,
                        &member.name,
                        MarkovMemberEffect::TaskQos,
                        cycle,
                        ActuatorDecisionOutcome::NoOp,
                        "effect ownership no longer held",
                    ));
                }
            }
        }
    }
    let mut metrics = state.metrics.lock_recover();
    metrics.metrics.markov_prewarm_active = false;
    metrics.metrics.markov_prewarm_members = 0;
    metrics.metrics.markov_prewarm_reverts += u64::from(reverted);
    metrics.metrics.reverts_applied += reverted_effects;
    drop(metrics);
    if let Some(mut metadata) = exploration {
        if deferred_effects > 0 {
            metadata.cancelled = Some(
                apollo_engine::engine::exploration_scheduler::TerminalDiagnostic::ReleaseFailed,
            );
        }
        decision_events.push(
            markov_event(
                exploration_target,
                cycle,
                if deferred_effects > 0 {
                    ActuatorDecisionOutcome::Failed
                } else if reverted_effects > 0 {
                    ActuatorDecisionOutcome::Reverted
                } else {
                    ActuatorDecisionOutcome::Expired
                },
                "bounded cache-only exploration released",
            )
            .with_exploration(metadata),
        );
    }
    tracing::debug!(
        member_count,
        cache_bytes,
        calibration_probe,
        reverted,
        deferred_effects,
        "markov: released coalition lease"
    );
    MarkovReleaseReport {
        reverted_effects,
        deferred_effects,
        decision_events,
    }
}

/// Anti-ratchet (2026-06-10): what jetsam priority to restore after a
/// missed pre-warm prediction. `-1` is the "prior unreadable" sentinel —
/// in that case we skip the jetsam revert (writing a guessed band could
/// fight runningboard) and rely on the tier drop alone.
fn prewarm_jetsam_restore(prior_jetsam: i32) -> Option<i32> {
    (prior_jetsam >= 0).then_some(prior_jetsam)
}

/// Fight-hunt fix (2026-06-10): speculative cache warming is allowed only
/// below this pressure — above it, the purge paths are evicting the same
/// cache the warmer would fill (self-fight, thrash amplification).
fn cache_warm_allowed_at(pressure: f64) -> bool {
    pressure < 0.60
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome;

    fn lease(expires_at: Instant, activated: bool) -> MarkovPrewarmLease {
        MarkovPrewarmLease {
            source_app: "Finder".to_string(),
            predicted_app: "Terminal".to_string(),
            acquired_at: Instant::now(),
            members: vec![PrewarmedMember {
                pid: 42,
                name: "Terminal Helper".to_string(),
                prior_jetsam: 2,
                jetsam_applied: true,
                tier_applied: true,
                task_qos_applied: true,
            }],
            cache_bytes: 4096,
            expires_at,
            activated,
            activated_at: activated.then(Instant::now),
            settle_recorded: false,
            calibration_probe: false,
            calibration_context: PrewarmContext::new("idle", 0, 0.20, false),
            exploration: None,
        }
    }

    #[test]
    fn cache_warm_gated_above_060() {
        assert!(cache_warm_allowed_at(0.45));
        assert!(cache_warm_allowed_at(0.59));
        assert!(!cache_warm_allowed_at(0.60));
        assert!(!cache_warm_allowed_at(0.75));
    }

    #[test]
    fn markov_member_effect_receipts_keep_pid_and_effect_identity() {
        let jetsam = markov_member_effect_event(
            42,
            "Editor Helper",
            MarkovMemberEffect::Jetsam,
            5,
            ActuatorDecisionOutcome::Applied,
            "foreground band applied",
        );
        let qos = markov_member_effect_event(
            43,
            "Editor GPU",
            MarkovMemberEffect::TaskQos,
            5,
            ActuatorDecisionOutcome::Failed,
            "task QoS failed",
        );

        assert_eq!(jetsam.proposal.action_key, "markov_prewarm:jetsam");
        assert_eq!(jetsam.proposal.target, "Editor Helper:pid:42");
        assert_eq!(qos.proposal.action_key, "markov_prewarm:task_qos");
        assert_eq!(qos.proposal.target, "Editor GPU:pid:43");
        assert_eq!(qos.outcome, ActuatorDecisionOutcome::Failed);
    }

    #[test]
    fn markov_event_carries_prediction_and_cycle() {
        let event = markov_event(
            "Terminal",
            31,
            ActuatorDecisionOutcome::Applied,
            "coalition lease acquired",
        );

        assert_eq!(event.proposal.action_key, "markov_prewarm:predicted_app");
        assert_eq!(event.proposal.target, "Terminal");
        assert_eq!(event.proposal.proposed_cycle, 31);
    }

    #[test]
    fn quarantine_blocker_is_vetoed_while_other_markov_gates_are_blocked() {
        assert_eq!(
            markov_blocker_outcome("quarantine"),
            ActuatorDecisionOutcome::Vetoed
        );
        assert_eq!(
            markov_blocker_outcome("resource-gate"),
            ActuatorDecisionOutcome::Blocked
        );
    }

    #[test]
    fn prewarm_restore_sentinel_semantics() {
        // Unreadable prior → no jetsam write on miss.
        assert_eq!(prewarm_jetsam_restore(-1), None);
        // Captured priors restore verbatim, including IDLE (0).
        assert_eq!(prewarm_jetsam_restore(0), Some(0));
        assert_eq!(prewarm_jetsam_restore(2), Some(2));
        assert_eq!(prewarm_jetsam_restore(9), Some(9));
    }

    #[test]
    fn prewarm_window_is_confident_imminent_and_pressure_safe() {
        assert!(prewarm_window_open(0.70, 8.0, true, 0.50));
        assert!(prewarm_window_open(0.50, -2.0, true, 0.50));
        assert!(!prewarm_window_open(0.49, 2.0, true, 0.50));
        assert!(!prewarm_window_open(0.90, 13.0, true, 0.50));
        assert!(!prewarm_window_open(0.90, 2.0, false, 0.50));
    }

    #[test]
    fn prewarm_blocker_reports_the_first_real_gate() {
        assert_eq!(
            prewarm_blocker(0.90, 2.0, false, 0.50, PrewarmAdmission::Ready),
            "resource-gate"
        );
        assert_eq!(
            prewarm_blocker(0.49, 2.0, true, 0.50, PrewarmAdmission::Ready),
            "confidence"
        );
        assert_eq!(
            prewarm_blocker(0.90, 20.0, true, 0.50, PrewarmAdmission::Ready),
            "timing"
        );
        assert_eq!(
            prewarm_blocker(0.90, 2.0, true, 0.50, PrewarmAdmission::Quarantined),
            "quarantine"
        );
        assert_eq!(
            prewarm_blocker(0.90, 2.0, true, 0.50, PrewarmAdmission::Probe),
            "ready"
        );
    }

    #[test]
    fn scheduler_rejection_blocks_only_markov_probe_not_ready_policy() {
        assert_eq!(
            markov_exploration_admission(PrewarmAdmission::Probe, false),
            (false, false)
        );
        assert_eq!(
            markov_exploration_admission(PrewarmAdmission::Probe, true),
            (true, false)
        );
        assert_eq!(
            markov_exploration_admission(PrewarmAdmission::Ready, false),
            (true, true)
        );
        assert_eq!(
            markov_exploration_admission(PrewarmAdmission::Quarantined, true),
            (false, false)
        );
    }

    #[test]
    fn contextual_prewarm_bias_stays_inside_the_specialist_window() {
        let positive = ContextualActionBias {
            score: 1.0,
            model_observations: 20,
            ..ContextualActionBias::default()
        };
        let negative = ContextualActionBias {
            score: -1.0,
            model_observations: 20,
            authoritative: true,
            ..ContextualActionBias::default()
        };
        assert_eq!(contextual_prewarm_probability_floor(positive), 0.50);
        assert_eq!(contextual_prewarm_probability_floor(negative), 0.54);
        assert_eq!(
            contextual_prewarm_probability_floor(ContextualActionBias::default()),
            0.50
        );
        assert!(!prewarm_window_open(0.49, 2.0, true, 0.50));
        assert!(!prewarm_window_open(0.53, 2.0, true, 0.54));
        assert!(!prewarm_window_open(0.90, 2.0, false, 0.50));
        assert_eq!(contextual_prewarm_lease_secs(10.0, positive), 11.5);
        assert_eq!(contextual_prewarm_lease_secs(10.0, negative), 8.5);
    }

    #[test]
    fn passive_prediction_window_does_not_depend_on_actuator_headroom() {
        assert!(prediction_window_open(0.70, 8.0));
        assert!(prediction_window_open(0.50, -2.0));
        assert!(!prediction_window_open(0.49, 2.0));
        assert!(!prediction_window_open(0.90, 13.0));
    }

    #[test]
    fn shadow_prediction_scores_transition_or_deadline_without_actuation() {
        let now = Instant::now();
        let lease = MarkovShadowLease {
            source_app: "Finder".to_string(),
            predicted_app: "Terminal".to_string(),
            expires_at: now + Duration::from_secs(10),
            calibration_context: PrewarmContext::new("coding", 12, 0.30, false),
        };
        assert_eq!(
            shadow_lease_resolution(&lease, Some("Finder"), now),
            LeaseResolution::Pending
        );
        assert_eq!(
            shadow_lease_resolution(&lease, Some("terminal"), now),
            LeaseResolution::Hit
        );
        assert_eq!(
            shadow_lease_resolution(&lease, Some("Safari"), now),
            LeaseResolution::Miss
        );
        assert_eq!(
            shadow_lease_resolution(&lease, Some("Finder"), lease.expires_at),
            LeaseResolution::Miss
        );
    }

    #[test]
    fn family_roots_get_non_invasive_warming_only() {
        assert_eq!(prewarm_target_modes(42, 7, false, true), (true, false));
        assert_eq!(prewarm_target_modes(42, 7, false, false), (true, true));
        assert_eq!(prewarm_target_modes(7, 7, false, false), (false, false));
        assert_eq!(prewarm_target_modes(42, 7, true, false), (false, false));
    }

    #[test]
    fn coalition_member_selection_is_root_first_hot_and_bounded() {
        let candidates = vec![
            (10, "root".to_string(), 1.0),
            (11, "cold".to_string(), 0.1),
            (12, "hottest".to_string(), 90.0),
            (13, "hot".to_string(), 50.0),
            (14, "warm".to_string(), 30.0),
            (15, "mild".to_string(), 10.0),
            (16, "cool".to_string(), 5.0),
            (17, "colder".to_string(), 0.5),
        ];
        let ranked = rank_and_cap_candidates(candidates, 10);
        assert_eq!(ranked.len(), MAX_PREWARM_MEMBERS);
        assert_eq!(ranked[0].0, 10);
        assert_eq!(ranked[1].0, 12);
        assert_eq!(ranked[2].0, 13);
        assert!(!ranked.iter().any(|member| member.0 == 11));
    }

    #[test]
    fn settle_is_recorded_once_after_hit_when_fluidity_recovers() {
        let now = Instant::now();
        let mut l = lease(now + Duration::from_secs(10), true);
        l.activated_at = Some(now - Duration::from_millis(250));
        assert!(!maybe_record_settle(&mut l, false, false, false, now));
        assert!(!maybe_record_settle(&mut l, true, true, false, now));
        assert!(!maybe_record_settle(&mut l, true, false, true, now));
        assert!(maybe_record_settle(&mut l, true, false, false, now));
        assert!(!maybe_record_settle(&mut l, true, false, false, now));
    }

    #[test]
    fn pending_prediction_is_not_a_false_miss_next_cycle() {
        let now = Instant::now();
        let l = lease(now + Duration::from_secs(10), false);
        assert_eq!(
            lease_resolution(&l, Some("Finder"), now + Duration::from_secs(2)),
            LeaseResolution::Pending
        );
    }

    #[test]
    fn lease_hits_on_target_then_completes_when_target_loses_focus() {
        let now = Instant::now();
        let mut l = lease(now + Duration::from_secs(10), false);
        assert_eq!(
            lease_resolution(&l, Some("terminal"), now),
            LeaseResolution::Hit
        );
        l.activated = true;
        assert_eq!(
            lease_resolution(&l, Some("Terminal"), now),
            LeaseResolution::Pending
        );
        assert_eq!(
            lease_resolution(&l, Some("Finder"), now),
            LeaseResolution::Completed
        );
    }

    #[test]
    fn hit_lease_expires_even_while_target_stays_foreground() {
        let now = Instant::now();
        let l = lease(now, true);
        assert_eq!(
            lease_resolution(&l, Some("Terminal"), now),
            LeaseResolution::Completed
        );
    }

    #[test]
    fn unresolved_lease_misses_only_after_deadline() {
        let now = Instant::now();
        let l = lease(now, false);
        assert_eq!(
            lease_resolution(&l, Some("Finder"), now),
            LeaseResolution::Miss
        );
    }

    #[test]
    fn wrong_foreground_transition_resolves_miss_immediately() {
        let now = Instant::now();
        let l = lease(now + Duration::from_secs(10), false);
        assert_eq!(
            lease_resolution(&l, Some("Safari"), now),
            LeaseResolution::Miss
        );
    }

    #[test]
    fn app_match_does_not_accept_substring_false_positive() {
        assert!(app_names_match("Xcode", "xcode"));
        assert!(!app_names_match("Xcode", "Code"));
    }
}
