//! # Metrics Reporter
//!
//! Per-cycle observability and dispatch work extracted from the daemon main loop.
//!
//! ## What this module does
//!
//! Every cycle, after the learning tick completes, the daemon performs three
//! distinct observability/dispatch operations:
//!
//! 1. **`update_learning_metrics`** — Write predictive agent + signal intelligence
//!    fields into the shared `MetricsState` for `GetStatus` consumption.
//!
//! 2. **`apply_io_shaping`** — Foreground-aware I/O bandwidth allocation
//!    (Iyer & Druschel 2001, every 20 cycles). Updates `sysctl_reactive_writes`.
//!
//! 3. **`apply_qos_routing`** — MachQoS P-Core / E-Core routing for heuristic
//!    decisions, with foreground-family cascade and thermal override.
//!
//! 4. **`merge_cycle_metrics`** — Phase 3: merge `ExecuteOutcomes` into
//!    `MetricsState`, record cycle duration, write to disk.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::time::Instant;

use chrono::Utc;

use apollo_engine::collector::SystemSnapshot;
use apollo_engine::engine::adaptive_governor::ProcessDecision;
use apollo_engine::engine::daemon_helpers::{
    append_timeline, battery_pressure_boost, compute_p95, write_metrics,
};
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::decision_ledger::{DecisionLedger, DecisionLifecycle};
use apollo_engine::engine::execute_actions::ExecuteOutcomes;
use apollo_engine::engine::exploration_scheduler::ExplorationScheduler;
use apollo_engine::engine::intelligence_score::AisScore;
use apollo_engine::engine::io_tiering::IoShaper;
use apollo_engine::engine::learning_pipeline::LearningPipeline;
use apollo_engine::engine::local_consolidation::LocalConsolidator;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::mach_qos::{QoSAuditDisposition, QoSOutcome, SchedulingTier};
use apollo_engine::engine::nars_belief::ArousalState;
use apollo_engine::engine::network_monitor::NetworkMonitor;
use apollo_engine::engine::overflow_guard::OverflowThresholds;
use apollo_engine::engine::pipeline::learning_context::LearningContext;
use apollo_engine::engine::power_management::PowerManager;
use apollo_engine::engine::predictive_agent::Intervention;
use apollo_engine::engine::process_classifier::ProcessTier;
use apollo_engine::engine::process_tree::ProcessTree;
use apollo_engine::engine::profile_governor::GovernorDecision;
use apollo_engine::engine::signal_intelligence::SignalDigest;
use apollo_engine::engine::telemetry_medallion::{EvidenceTier, TelemetryMedallion};
use apollo_engine::engine::thermal_bailout::ThermalAction;
use apollo_engine::engine::types::{BlockerScore, OptimizationProfile, RuntimeMetrics};
use apollo_engine::engine::unified_learning_health::{
    AdviceRecord, ClosureObservation, ClosureOutcome, ExplorationLearningSnapshot,
    HorizonCalibrationInput, LatestResolvedEpisodeSnapshot, UnifiedLearningHealth,
    UnifiedLearningHealthCache, UnifiedLearningInput, UnifiedLearningRevision,
};
use apollo_engine::engine::world_model::WorldModel;

use crate::process_enrichment;

pub fn unified_learning_revision(
    ledger: &DecisionLedger,
    telemetry: &TelemetryMedallion,
    hierarchy: &LocalConsolidator,
    scheduler: &ExplorationScheduler,
) -> UnifiedLearningRevision {
    UnifiedLearningRevision {
        ledger: ledger.revision(),
        ledger_unattributed_applied_total: ledger.unattributed_applied_total(),
        calibration: telemetry.learning_revision(),
        hierarchy: hierarchy.revision(),
        exploration: scheduler.revision(),
        causal: telemetry.causal_dynamics().publication_revision(),
    }
}

pub fn refresh_unified_learning_health(
    cache: &mut UnifiedLearningHealthCache,
    ledger: &DecisionLedger,
    telemetry: &TelemetryMedallion,
    hierarchy: &LocalConsolidator,
    scheduler: &ExplorationScheduler,
) {
    let revision = unified_learning_revision(ledger, telemetry, hierarchy, scheduler);
    cache.refresh(revision, || {
        build_unified_learning_health(ledger, telemetry, hierarchy, scheduler)
    });
}

fn build_unified_learning_health(
    ledger: &DecisionLedger,
    telemetry: &TelemetryMedallion,
    hierarchy: &LocalConsolidator,
    scheduler: &ExplorationScheduler,
) -> UnifiedLearningHealth {
    let calibration = telemetry.model_calibration();
    let authoritative_gold_decision_ids = telemetry.authoritative_gold_decision_ids();
    let mut measured_decision_ids: HashSet<_> =
        authoritative_gold_decision_ids.iter().copied().collect();
    measured_decision_ids.extend(
        telemetry
            .current_machine_recent_evidence()
            .filter_map(|evidence| evidence.decision_id.map(|id| id.0)),
    );
    let mut horizons: BTreeMap<_, (u64, u64, f64, f64, Option<f64>)> = BTreeMap::new();
    let advice_records = calibration
        .records()
        .map(|record| {
            let entry = horizons
                .entry(record.key.horizon)
                .or_insert((0, 0, 0.0, 0.0, None));
            entry.0 = entry.0.saturating_add(record.authority_gold_count);
            entry.1 = entry.1.saturating_add(1);
            entry.2 += record.normalized_mae_ema;
            entry.3 += record.coverage_ema;
            if let Some(brier) = record.brier_ema.filter(|value| value.is_finite()) {
                entry.4 = Some(entry.4.unwrap_or(0.0) + brier);
            }
            AdviceRecord {
                key: record.key.clone(),
                trust: record.trust,
                signed_error: record.signed_error_ema,
                normalized_mae: record.normalized_mae_ema,
                coverage: (record.authority_gold_count > 0).then_some(record.coverage_ema),
                authority_gold_count: record.authority_gold_count,
                model_authority_gold_count: calibration
                    .model_for(&record.key)
                    .map_or(0, |model| model.authority_gold_count),
                current_epoch: true,
            }
        })
        .collect();
    let horizon_calibration = horizons
        .into_iter()
        .map(|(horizon, (count, records, mae, coverage, brier))| {
            let divisor = records.max(1) as f64;
            HorizonCalibrationInput::new(
                horizon,
                count,
                mae / divisor,
                coverage / divisor,
                brier.map(|value| value / divisor),
            )
        })
        .collect();
    let closure_observations = ledger
        .recent()
        .iter()
        .map(|episode| ClosureObservation {
            decision_id: episode.id.0,
            local: episode
                .envelope
                .receipt
                .as_ref()
                .and_then(|receipt| receipt.attribution.as_ref())
                .is_some_and(|attribution| attribution.grants_local_authority())
                || episode
                    .envelope
                    .terminal_attribution
                    .as_ref()
                    .is_some_and(|attribution| attribution.grants_local_authority()),
            outcome: closure_outcome(episode.envelope.lifecycle),
            issued_cycle: episode.envelope.proposed_cycle,
            horizon_cycles: episode
                .envelope
                .predictions
                .iter()
                .map(|prediction| prediction.horizon_cycles)
                .max()
                .unwrap_or(0),
            now_cycle: episode.settled_cycle,
            resolved_evidence: measured_decision_ids.contains(&episode.id.0),
            duplicate: false,
            synthetic_overflow: episode.envelope.action_key == "decision_events:overflow",
        })
        .collect();
    let mut latest_resolved_episodes: Vec<_> = ledger
        .recent()
        .iter()
        .map(|episode| LatestResolvedEpisodeSnapshot {
            present: true,
            id: episode.id.0,
            resolved_cycle: episode.settled_cycle,
            action: episode.envelope.action_key.clone(),
            tier: "ledger".to_string(),
            scope: "local".to_string(),
            authority: episode.authority_eligible,
            treatment: episode
                .envelope
                .exploration
                .as_ref()
                .is_some_and(|meta| meta.treatment),
            control: episode
                .envelope
                .exploration
                .as_ref()
                .is_some_and(|meta| !meta.treatment),
            reverted: episode.envelope.lifecycle == DecisionLifecycle::Reverted,
            expected_utility: episode
                .envelope
                .predictions
                .first()
                .map(|prediction| prediction.expected_utility),
            measured_utility: None,
            quality: None,
            causal_result: String::new(),
            calibration_result: String::new(),
            trust_transition: String::new(),
        })
        .collect();
    latest_resolved_episodes.extend(telemetry.current_machine_recent_evidence().filter_map(
        |evidence| {
            let id = evidence.decision_id?.0;
            let details = evidence.learning_details.as_ref();
            let expected_utility = details.map(|details| details.expected_utility).or_else(|| {
                evidence
                    .calibration_provenance
                    .predictions
                    .iter()
                    .find(|prediction| {
                        prediction.source == evidence.calibration_provenance.proposer
                    })
                    .or_else(|| evidence.calibration_provenance.predictions.first())
                    .map(|prediction| prediction.expected_utility)
            });
            let authority = details.is_some_and(|details| details.is_authoritative());
            Some(LatestResolvedEpisodeSnapshot {
                present: true,
                id,
                resolved_cycle: evidence.resolved_cycle,
                action: evidence.action_key.clone(),
                tier: match evidence.tier {
                    EvidenceTier::Bronze => "bronze",
                    EvidenceTier::Silver => "silver",
                    EvidenceTier::Gold => "gold",
                }
                .to_string(),
                scope: if authority { "authority" } else { "observed" }.to_string(),
                authority,
                treatment: false,
                control: false,
                reverted: false,
                expected_utility,
                measured_utility: Some(evidence.utility.apollo_utility),
                quality: Some(evidence.quality),
                causal_result: String::new(),
                calibration_result: if authority { "accepted" } else { "observed" }.to_string(),
                trust_transition: String::new(),
            })
        },
    ));
    let view = hierarchy.view();
    let (committed, terminal) = scheduler.learning_counts();
    UnifiedLearningHealth::from_input(UnifiedLearningInput {
        decision_ledger_unattributed_applied_total: ledger.unattributed_applied_total(),
        local_gold_decisions: authoritative_gold_decision_ids.len() as u64,
        authoritative_gold_decision_ids,
        trusted_active_models: telemetry.model_calibration_summary().trusted_models as u64,
        active_models: telemetry.model_calibration_metrics().record_count as u64,
        closure_observations,
        horizon_calibration,
        advice_records,
        hierarchy: apollo_engine::engine::unified_learning_health::HierarchyLearningSnapshot {
            prototypes: view.families_with_evidence as u64,
            consolidations: view.total_consolidations,
            duplicates: 0,
        },
        exploration: ExplorationLearningSnapshot {
            committed,
            deduplicated: terminal.saturating_sub(committed),
            ..ExplorationLearningSnapshot::default()
        },
        latest_resolved_episodes,
        ..UnifiedLearningInput::default()
    })
}

fn closure_outcome(lifecycle: DecisionLifecycle) -> ClosureOutcome {
    match lifecycle {
        DecisionLifecycle::Applied => ClosureOutcome::Applied,
        DecisionLifecycle::Rejected => ClosureOutcome::Rejected,
        DecisionLifecycle::Vetoed => ClosureOutcome::Vetoed,
        DecisionLifecycle::Blocked => ClosureOutcome::Blocked,
        DecisionLifecycle::Failed => ClosureOutcome::Failed,
        DecisionLifecycle::NoOp => ClosureOutcome::NoOp,
        DecisionLifecycle::Reverted => ClosureOutcome::Reverted,
        DecisionLifecycle::Expired => ClosureOutcome::Expired,
        DecisionLifecycle::Proposed | DecisionLifecycle::Executing | DecisionLifecycle::Settled => {
            ClosureOutcome::Pending
        }
    }
}

fn publish_world_model_metrics(
    metrics: &mut RuntimeMetrics,
    telemetry_medallion: &TelemetryMedallion,
    world_model: &WorldModel,
) {
    let context = telemetry_medallion.metrics();
    metrics.world_model_context_bronze_total = context.bronze_total;
    metrics.world_model_context_silver_total = context.silver_total;
    metrics.world_model_context_gold_total = context.gold_total;
    metrics.world_model_context_rejected_total = context.rejected_total;
    metrics.world_model_context_non_finite_total = context.non_finite_total;
    metrics.world_model_context_range_total = context.range_total;
    metrics.world_model_context_stale_total = context.stale_total;
    metrics.world_model_context_temporal_total = context.temporal_total;
    metrics.world_model_context_foreign_total = context.foreign_total;
    metrics.world_model_context_coherence_total = context.coherence_total;
    metrics.world_model_context_top_rejected_field = context
        .top_rejected_field
        .map(|field| field.as_str().to_string())
        .unwrap_or_default();
    metrics.world_model_context_top_rejected_field_total = context.top_rejected_field_total;
    if let Some(violation) = context.last_field_violation {
        metrics.world_model_context_last_rejected_field = violation.field.as_str().to_string();
        metrics.world_model_context_last_rejected_reason = violation.reason.as_str().to_string();
        metrics.world_model_context_last_rejected_value =
            violation.value.is_finite().then_some(violation.value);
    }
    metrics.world_model_context_local_gold_total = context.local_gold_total;
    metrics.world_model_context_current_tier = context.current_tier.as_str().to_string();
    metrics.world_model_context_authority_phase =
        world_model.authority_phase().as_str().to_string();
    metrics.world_model_context_quality = context.mean_quality;
    metrics.world_model_actuator_issued_total = context.actuator_issued_total;
    metrics.world_model_actuator_pending_total = context.actuator_pending_total;
    metrics.world_model_actuator_bronze_total = context.actuator_bronze_total;
    metrics.world_model_actuator_silver_total = context.actuator_silver_total;
    metrics.world_model_actuator_gold_total = context.actuator_gold_total;
    metrics.world_model_actuator_effective_total = context.actuator_effective_total;
    metrics.world_model_actuator_rejected_total = context.actuator_rejected_total;
    metrics.world_model_actuator_expired_total = context.actuator_expired_total;
    metrics.world_model_actuator_quality = context.actuator_mean_quality;
    metrics.world_model_actuator_mean_utility = context.actuator_mean_utility;
    metrics.world_model_apollo_utility = context.apollo_utility_ema;
    metrics.world_model_decision_credit_sources = context.decision_credit_sources;
    metrics.world_model_decision_credit_leader = telemetry_medallion
        .decision_credit_leader()
        .map(|(source, _)| source.to_string())
        .unwrap_or_default();
    metrics.world_model_decision_credit_leader_score = context.decision_credit_leader_score;
    metrics.world_model_decision_credit_leader_accuracy = context.decision_credit_leader_accuracy;
    metrics.world_model_decision_credit_leader_observations =
        context.decision_credit_leader_observations;
    if let Some(evidence) = telemetry_medallion.recent_actuator_evidence().back() {
        if evidence.resolved_cycle > metrics.last_episode_resolved_cycle {
            metrics.last_episode_id = evidence.id;
            metrics.last_episode_resolved_cycle = evidence.resolved_cycle;
            metrics.last_episode_action = evidence.action_key.clone();
            metrics.last_episode_target = evidence.target.clone();
            metrics.last_episode_tier = match evidence.tier {
                apollo_engine::engine::telemetry_medallion::EvidenceTier::Bronze => "bronze",
                apollo_engine::engine::telemetry_medallion::EvidenceTier::Silver => "silver",
                apollo_engine::engine::telemetry_medallion::EvidenceTier::Gold => "gold",
            }
            .to_string();
            metrics.last_episode_quality = evidence.quality;
            metrics.last_episode_utility = evidence.net_utility_delta;
            metrics.last_episode_latency_improvement = evidence.perceptual_latency_improvement;
            metrics.last_episode_system_gain = evidence.utility.system_gain;
            metrics.last_episode_human_gain = evidence.utility.human_gain;
            metrics.last_episode_intervention_cost = evidence.utility.intervention_cost;
            metrics.last_episode_apollo_utility = evidence.utility.apollo_utility;
            metrics.last_episode_proposer = evidence.attribution.proposer.clone();
            metrics.last_episode_supporters = evidence.attribution.supporters.clone();
            metrics.last_episode_vetoes = evidence.attribution.vetoes.clone();
            metrics.last_episode_predicted_gain = evidence.attribution.predicted_gain;
        }
    }
    metrics.world_model_gpu_bronze_total = context.gpu_prediction_bronze_total;
    metrics.world_model_gpu_silver_total = context.gpu_prediction_silver_total;
    metrics.world_model_gpu_gold_total = context.gpu_prediction_gold_total;
    metrics.world_model_gpu_rejected_total = context.gpu_prediction_rejected_total;
    metrics.world_model_gpu_evicted_total = context.gpu_prediction_evicted_total;
    metrics.world_model_gpu_unused_total = context.gpu_prediction_unused_total;
    metrics.world_model_gpu_bronze_rejected_total = context.gpu_prediction_bronze_rejected_total;
    metrics.world_model_gpu_unclassified_rejections =
        context.gpu_prediction_unclassified_rejections;
    metrics.world_model_gpu_pending_total = context.gpu_prediction_pending_total;
    metrics.world_model_gpu_calibrated_models = context.gpu_prediction_calibrated_models;
    metrics.world_model_gpu_calibration_mae = context.gpu_prediction_mean_absolute_error;
    metrics.world_model_gpu_calibration_brier = context.gpu_prediction_mean_brier;
    metrics.world_model_gpu_calibration_quality = context.gpu_prediction_mean_quality;
    metrics.world_model_counterfactual_issued_total = context.controlled_holdout_issued_total;
    metrics.world_model_counterfactual_pending_total = context.controlled_holdout_pending_total;
    metrics.world_model_counterfactual_resolved_total = context.controlled_holdout_resolved_total;
    metrics.world_model_counterfactual_rejected_total = context.controlled_holdout_rejected_total;
    metrics.world_model_counterfactual_would_help_total =
        context.controlled_holdout_would_help_total;
    metrics.world_model_counterfactual_control_utility =
        context.controlled_holdout_mean_control_utility;
    metrics.world_model_actuator_known_models = world_model.utility_known_actions() as u64;
    metrics.world_model_actuator_ready_models = world_model.utility_ready_actions() as u64;
    let readiness = world_model.utility_readiness_breakdown();
    metrics.world_model_readiness_no_gold = readiness.no_current_gold;
    metrics.world_model_readiness_immature = readiness.immature;
    metrics.world_model_readiness_dormant = readiness.dormant;
    metrics.world_model_action_model_len = context.action_model_len;
    metrics.world_model_action_model_capacity = context.action_model_capacity;
    metrics.world_model_action_model_evictions_total = context.action_model_evictions_total;
    metrics.world_model_action_model_births_total = context.action_model_births_total;
    metrics.world_model_evidence_updates_total = context.action_model_evidence_updates_total;
    metrics.world_model_last_evidence_cycle = context.action_model_last_evidence_cycle;
    metrics.world_model_readiness_low_quality = readiness.low_quality;
    metrics.world_model_readiness_stale = readiness.stale;
    metrics.world_model_readiness_foreign = readiness.foreign_installation;
    metrics.world_model_readiness_hardware = readiness.hardware_mismatch;
    metrics.world_model_readiness_uncertain = readiness.uncertain_interval;
    metrics.world_model_family_known_models = world_model.utility_known_families() as u64;
    metrics.world_model_family_ready_models = world_model.utility_ready_families() as u64;
    metrics.world_model_actuator_families = telemetry_medallion
        .family_stats()
        .iter()
        .map(
            |(family, stats)| apollo_engine::engine::types::ActuatorEvidenceStatus {
                family: family.as_str().to_string(),
                issued: stats.issued_total,
                resolved: stats.resolved_total,
                gold: stats.gold_total,
                effective: stats.effective_total,
                rejected: stats.rejected_total,
                expired: stats.expired_total,
                mean_quality: if stats.resolved_total == 0 {
                    0.0
                } else {
                    (stats.quality_sum / stats.resolved_total as f64).clamp(0.0, 1.0)
                },
                mean_utility: if stats.resolved_total == 0 {
                    0.0
                } else {
                    (stats.utility_sum / stats.resolved_total as f64).clamp(-1.0, 1.0)
                },
            },
        )
        .collect();
    metrics.world_model_curated_actions = world_model.known_actions() as u64;
    metrics.world_model_ready_actions = world_model.ready_actions() as u64;
    metrics.world_model_gold_evidence = world_model.curated_observations();
    metrics.world_model_contextual_actions = world_model.contextual_actions() as u64;
    metrics.world_model_data_quality = world_model.mean_data_quality();
    let (causal_refreshes, causal_hits, utility_refreshes, utility_hits) =
        world_model.cache_stats();
    metrics.world_model_causal_refreshes = causal_refreshes;
    metrics.world_model_causal_cache_hits = causal_hits;
    metrics.world_model_utility_refreshes = utility_refreshes;
    metrics.world_model_utility_cache_hits = utility_hits;
    metrics.world_model_episodic_memory_samples = world_model.episodic_memory_samples() as u64;
    metrics.world_model_episodic_memory_families = world_model.episodic_memory_families() as u64;
    let dynamics = world_model.causal_dynamics_metrics();
    metrics.world_model_dynamics_phase = dynamics.phase.as_str().to_string();
    metrics.world_model_dynamics_action_models = dynamics.action_models;
    metrics.world_model_dynamics_ready_models = dynamics.ready_models;
    metrics.world_model_dynamics_ranking_models = dynamics.ranking_eligible_models;
    metrics.world_model_dynamics_authoritative_models = dynamics.authoritative_models;
    metrics.world_model_dynamics_baseline_models = dynamics.baseline_models;
    metrics.world_model_dynamics_baseline_ready_models = dynamics.baseline_ready_models;
    metrics.world_model_dynamics_gold_updates = dynamics.gold_action_updates;
    metrics.world_model_dynamics_no_action_updates = dynamics.no_action_updates;
    metrics.world_model_dynamics_validation_samples = dynamics.validation_samples;
    metrics.world_model_dynamics_validation_mae = dynamics.validation_mae;
    metrics.world_model_dynamics_validation_coverage = dynamics.validation_coverage;
    metrics.world_model_dynamics_publication_revision = dynamics.publication_revision;
}

/// Update predictive agent + signal intelligence metrics for status reporting.
///
/// Locks `state.metrics` once, updates all learning-observable fields, then
/// releases the lock. The caller should not hold the lock when calling.
pub fn update_learning_metrics<'a>(
    state: &SharedState,
    lctx: &LearningContext<'a>,
    signal_digest: &SignalDigest,
    agent_intervention: &Intervention,
    arousal_state: &ArousalState,
    learning_pipeline: &LearningPipeline,
    telemetry_medallion: &TelemetryMedallion,
    world_model: &WorldModel,
    unified_learning_health: &UnifiedLearningHealth,
) {
    let mut m = state.metrics.lock_recover();
    m.metrics.predictive_agent_active = lctx.predictive_agent.is_active();
    m.metrics.predictive_agent_cycles = lctx.predictive_agent.total_cycles();
    m.metrics.predictive_agent_arm_pulls = lctx.predictive_agent.arm_pulls();
    m.metrics.predictive_agent_arm_avg_rewards = lctx.predictive_agent.arm_avg_rewards();
    m.metrics.predictive_agent_last_intervention = format!("{:?}", agent_intervention);
    m.metrics.predictive_agent_pending_outcomes = lctx.predictive_agent.pending_outcome_count();
    m.metrics.predictive_agent_resolved_outcomes = lctx.predictive_agent.resolved_outcomes();
    let medallion = learning_pipeline.medallion_metrics();
    m.metrics.learning_bronze_total = medallion.bronze_total;
    m.metrics.learning_silver_total = medallion.silver_total;
    m.metrics.learning_gold_total = medallion.gold_total;
    m.metrics.learning_rejected_total = medallion.rejected_total;
    m.metrics.learning_invalid_total = medallion.invalid_total;
    m.metrics.learning_duplicate_total = medallion.duplicate_total;
    m.metrics.learning_data_quality = medallion.mean_quality;
    m.metrics.learning_gold_rate = medallion.gold_rate;
    publish_world_model_metrics(&mut m.metrics, telemetry_medallion, world_model);
    unified_learning_health.publish_to(&mut m.metrics);
    m.metrics.si_pressure_smooth = signal_digest.pressure_smooth;
    m.metrics.si_pressure_velocity = signal_digest.pressure_velocity;
    m.metrics.si_p_oom_30s = signal_digest.p_oom_30s;
    m.metrics.si_urgency = signal_digest.urgency;
    if signal_digest.regime_shift_up {
        m.metrics.si_regime_shifts += 1;
    }
    m.metrics.si_monopoly_risk = signal_digest.monopoly_risk;
    m.metrics.si_entropy_anomaly = signal_digest.entropy_anomaly;
    m.metrics.si_stability_regime = signal_digest.stability_regime as u8;
    // Cable 4: top_causal_pairs() → expose in metrics for observability.
    m.metrics.causal_pairs = lctx
        .outcome_tracker
        .top_causal_pairs(5)
        .iter()
        .map(|(a, b, c)| format!("{} + {} ({})", a, b, c))
        .collect();
    m.metrics.natural_drift = lctx.outcome_tracker.natural_drift();
    m.metrics.short_drift_velocity = lctx.outcome_tracker.pressure_velocity_short();
    m.metrics.nars_drift_score = lctx.outcome_tracker.nars_drift_score();
    m.metrics.nars_drifted_beliefs = lctx.outcome_tracker.drift_detector.drifted_count;
    m.metrics.nars_beliefs_total = lctx.outcome_tracker.drift_detector.len();
    m.metrics.arousal_level = arousal_state.level;
    m.metrics.arousal_zone = arousal_state.zone().to_string();
    m.metrics.experience_memory_size = lctx.outcome_tracker.experience.len();
    m.metrics.causal_slow_horizon_count = lctx.causal_graph.slow_horizon_count();
    m.metrics.causal_mechanism_count = lctx.causal_graph.mechanism_count();
    // Top mechanism summaries for observability.
    m.metrics.causal_mechanisms = lctx
        .causal_graph
        .solid_edges_by_impact()
        .iter()
        .take(5)
        .filter_map(|e| {
            if e.mechanism.observations >= 3 {
                let m_type = e.mechanism.primary();
                let detail = match m_type {
                    "rss" => format!("−{:.0}MB", e.mechanism.rss_delta_mb),
                    "cpu" => format!("−{:.0}%", e.mechanism.cpu_delta_pct),
                    "swap" => format!("−{:.0}MB", e.mechanism.swap_delta_mb),
                    _ => "?".to_string(),
                };
                Some(format!("{} via {} ({})", e.cause, m_type, detail))
            } else {
                None
            }
        })
        .collect();
    // Causal effect average: mean impact_score across solid causal edges.
    // This is a real signal (confidence × avg_delta from observed pressure drops)
    // instead of the previous synthetic (effectiveness × 0.05) heuristic.
    m.metrics.causal_effect_avg = {
        let solid = lctx.causal_graph.solid_edges();
        if solid.is_empty() {
            0.0
        } else {
            let sum: f64 = solid.iter().map(|e| e.impact_score() as f64).sum();
            sum / solid.len() as f64
        }
    };
    // HRPO / Dr. Zero metrics
    m.metrics.dr_zero_self_challenge = lctx.outcome_tracker.self_challenge_score();
    m.metrics.dr_zero_groups = lctx
        .outcome_tracker
        .hop_group_summary()
        .iter()
        .map(|(hop, eff, count, pred_err)| {
            format!(
                "{:?}(eff={:.0}% n={} err={:.2})",
                hop,
                eff * 100.0,
                count,
                pred_err
            )
        })
        .collect();
    m.metrics.dr_zero_exploration = lctx
        .outcome_tracker
        .exploration_needed()
        .iter()
        .map(|(hop, err)| format!("{:?}(err={:.2})", hop, err))
        .collect();
}

/// I/O Traffic Shaping: foreground-aware disk bandwidth allocation.
///
/// Runs every 20 cycles (~10s). Only runs when root (`is_root` is true).
/// Updates `state.metrics.sysctl_reactive_writes` if changes were made.
///
/// Based on Iyer & Druschel 2001 — anticipatory scheduling + I/O priority classes
/// reduce foreground I/O latency by 50-70% under concurrent background load.
/// MIN_REAPPLY_SECS=60 means nothing actually reapplies within 60s anyway.
fn actionable_io_tiers(
    heuristic_decisions: &[ProcessDecision],
    heuristic_critical_pids: &HashSet<u32>,
) -> Vec<(u32, ProcessTier)> {
    heuristic_decisions
        .iter()
        .filter(|d| !heuristic_critical_pids.contains(&d.pid))
        .map(|d| (d.pid, d.tier))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn apply_io_shaping(
    cycle_count: u64,
    is_root: bool,
    snapshot: &SystemSnapshot,
    foreground_pid: Option<u32>,
    process_tree: &ProcessTree,
    heuristic_decisions: &[ProcessDecision],
    heuristic_critical_pids: &HashSet<u32>,
    power_mgr: &PowerManager,
    thermal_pressure_boost: f64,
    io_shaper: &mut IoShaper,
    state: &SharedState,
) {
    if !cycle_count.is_multiple_of(20) || !is_root {
        return;
    }
    let fg_family_io = process_enrichment::build_foreground_family(foreground_pid, process_tree);
    let fg_pids: Vec<u32> = fg_family_io.iter().copied().collect();
    let process_tiers = actionable_io_tiers(heuristic_decisions, heuristic_critical_pids);
    let under_pressure = snapshot.pressure.memory_pressure
        + battery_pressure_boost(power_mgr)
        + thermal_pressure_boost
        > 0.60;
    let mut qos = state.mach_qos.lock_recover();
    let io_changes = io_shaper.shape(&fg_pids, &process_tiers, under_pressure, Some(&mut qos));
    drop(qos);
    if io_changes > 0 {
        state.metrics.lock_recover().metrics.sysctl_reactive_writes += io_changes as u64;
    }
}

/// Pure mapping: per-process governor decision + foreground family + thermal
/// state → target `SchedulingTier` (or `None` to skip the `task_for_pid`
/// syscall entirely).
///
/// Extracted from `apply_qos_routing` so the QoS routing policy can be unit
/// tested without needing a `SharedState`, `ProcessTree`, or `MachQoSManager`.
/// Precedence is load-bearing and matches the production behavior:
///   1. Thermal `force_ecores` demotes every non-foreground PID to Background.
///   2. Foreground-family membership promotes to Foreground (tree cascade).
///   3. Heuristic governor decision:
///      - `Allow` + `ActiveForeground` → Foreground
///      - `Allow` + other tiers        → None (skip no-op syscall)
///      - `Throttle`                   → None (QoS not the tool for throttling)
///      - `Freeze` | `Kill`            → Background
fn decide_qos_tier(
    decision: &ProcessDecision,
    fg_family: &HashSet<u32>,
    thermal_force_ecores: bool,
) -> Option<SchedulingTier> {
    if thermal_force_ecores && !fg_family.contains(&decision.pid) {
        return Some(SchedulingTier::Background);
    }
    if fg_family.contains(&decision.pid) {
        return Some(SchedulingTier::Foreground);
    }
    use apollo_engine::engine::adaptive_governor::GovernorDecision as GovDecision;
    match decision.decision {
        GovDecision::Allow => {
            if decision.tier == ProcessTier::ActiveForeground {
                Some(SchedulingTier::Foreground)
            } else {
                None
            }
        }
        GovDecision::Throttle => None,
        GovDecision::Freeze | GovDecision::Kill => Some(SchedulingTier::Background),
    }
}

/// MachQoS routing: assign P-Cores / E-Cores based on heuristic decisions.
///
/// Skips SIGSTOP'd processes and forces E-Cores for all non-foreground processes
/// during thermal emergency (`thermal_action.force_ecores`). Cascades Foreground
/// tier to all children of the foreground app via the process tree.
///
/// GCs dead PIDs every 30 cycles. Updates qos_foreground_count, qos_background_count,
/// and qos_errors in `state.metrics`.
#[allow(clippy::too_many_arguments)]
pub fn apply_qos_routing(
    cycle_count: u64,
    state: &SharedState,
    foreground_pid: Option<u32>,
    process_tree: &ProcessTree,
    heuristic_decisions: &[ProcessDecision],
    heuristic_critical_pids: &HashSet<u32>,
    thermal_action: &ThermalAction,
) {
    // F5 — MachQoS: route processes to P-Cores / E-Cores based on heuristic decisions.
    // Skip SIGSTOP'd processes; force E-Cores for all during thermal emergency.
    let frozen_pids: HashSet<u32> = state.frozen_state.lock_recover().keys().copied().collect();

    // Build the foreground family set from the process tree.
    let fg_family = process_enrichment::build_foreground_family(foreground_pid, process_tree);

    let interrupt_frozen = state
        .resource_interrupt
        .interrupt_frozen_pids
        .try_lock()
        .ok()
        .map(|g| g.clone())
        .unwrap_or_default();
    let mut qos_changes: Vec<(u32, SchedulingTier)> = heuristic_decisions
        .iter()
        .filter(|d| {
            !frozen_pids.contains(&d.pid)
                && !heuristic_critical_pids.contains(&d.pid)
                && !interrupt_frozen.contains(&d.pid)
        })
        .filter_map(|decision| {
            decide_qos_tier(decision, &fg_family, thermal_action.force_ecores)
                .map(|tier| (decision.pid, tier))
        })
        .collect();

    // Deduplicate: if a PID appeared in both heuristic decisions and
    // fg_family cascade, the last entry wins (which is fine since both
    // would map to Foreground). The MachQoSManager handles dupes internally.
    let _ = &mut qos_changes; // suppress unused_mut if no further manipulation

    let mut qos = state.mach_qos.lock_recover();
    // GC dead PIDs every 30 cycles to prevent unbounded growth
    // and handle PID recycling (recycled PID must be re-evaluated).
    if cycle_count.is_multiple_of(30) {
        qos.gc_dead_pids();
    }
    let outcomes: Vec<_> = qos_changes
        .iter()
        .map(|(pid, tier)| qos.set_tier_audited(*pid, *tier))
        .collect();
    let accounting = summarize_qos_route_outcomes(&outcomes);
    {
        let mut m = state.metrics.lock_recover();
        m.metrics.qos_requests_count = m
            .metrics
            .qos_requests_count
            .saturating_add(accounting.requests);
        m.metrics.qos_foreground_count = m
            .metrics
            .qos_foreground_count
            .saturating_add(accounting.foreground_applied);
        m.metrics.qos_background_count = m
            .metrics
            .qos_background_count
            .saturating_add(accounting.background_applied);
        m.metrics.qos_noop_count = m.metrics.qos_noop_count.saturating_add(accounting.no_ops);
        m.metrics.qos_blocked_count = m
            .metrics
            .qos_blocked_count
            .saturating_add(accounting.blocked);
        m.metrics.qos_errors = m.metrics.qos_errors.saturating_add(accounting.failed);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct QosRouteAccounting {
    requests: u64,
    foreground_applied: u64,
    background_applied: u64,
    no_ops: u64,
    blocked: u64,
    failed: u64,
}

fn summarize_qos_route_outcomes(
    outcomes: &[(QoSOutcome, QoSAuditDisposition)],
) -> QosRouteAccounting {
    let mut accounting = QosRouteAccounting {
        requests: outcomes.len() as u64,
        ..QosRouteAccounting::default()
    };
    for (outcome, disposition) in outcomes {
        match disposition {
            QoSAuditDisposition::Applied if outcome.mutated => match outcome.tier {
                SchedulingTier::Foreground => {
                    accounting.foreground_applied = accounting.foreground_applied.saturating_add(1)
                }
                SchedulingTier::Background => {
                    accounting.background_applied = accounting.background_applied.saturating_add(1)
                }
                SchedulingTier::Normal => {}
            },
            QoSAuditDisposition::Applied | QoSAuditDisposition::NoOp => {
                accounting.no_ops = accounting.no_ops.saturating_add(1)
            }
            QoSAuditDisposition::Blocked => {
                accounting.blocked = accounting.blocked.saturating_add(1)
            }
            QoSAuditDisposition::Failed => accounting.failed = accounting.failed.saturating_add(1),
        }
    }
    accounting
}

/// How often to flush runtime_metrics.json to disk.
/// At 300ms/cycle the file would otherwise be written ~3/s (73KB/s with
/// atomic-write overhead), hitting macOS's daily 2GB write budget in ~8h.
/// Writing every 25 cycles (~7.5s) reduces disk I/O by 25x.
const METRICS_DISK_WRITE_EVERY_N_CYCLES: u64 = 25;

/// How often to recompute the Apollo Intelligence Score. The computation reads
/// and parses several persisted learning files, so it runs on a dedicated
/// worker instead of extending the control-loop tail.
const AIS_COMPUTE_EVERY_N_CYCLES: u64 = 120;

pub struct AisRuntimeWorker {
    request_tx: SyncSender<()>,
    result_rx: Receiver<Option<AisScore>>,
    request_in_flight: bool,
}

impl AisRuntimeWorker {
    pub fn spawn() -> Self {
        Self::spawn_with(apollo_engine::engine::intelligence_score::compute_runtime_ais)
    }

    fn spawn_with<F>(compute: F) -> Self
    where
        F: Fn() -> Option<AisScore> + Send + 'static,
    {
        let (request_tx, request_rx) = mpsc::sync_channel::<()>(1);
        let (result_tx, result_rx) = mpsc::channel::<Option<AisScore>>();
        std::thread::Builder::new()
            .name("apollo-ais".to_string())
            .spawn(move || {
                while request_rx.recv().is_ok() {
                    let result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compute()))
                            .ok()
                            .flatten();
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn AIS runtime worker");

        Self {
            request_tx,
            result_rx,
            request_in_flight: false,
        }
    }

    /// Poll a completed score without blocking and schedule the next refresh.
    /// Cycle 1 gives the dashboard a fast warm start from persisted evidence.
    fn poll_and_schedule(&mut self, cycle_count: u64) -> Option<AisScore> {
        let mut latest = None;
        loop {
            match self.result_rx.try_recv() {
                Ok(result) => {
                    self.request_in_flight = false;
                    if result.is_some() {
                        latest = result;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.request_in_flight = false;
                    break;
                }
            }
        }

        let due = cycle_count == 1 || cycle_count.is_multiple_of(AIS_COMPUTE_EVERY_N_CYCLES);
        if due && !self.request_in_flight {
            match self.request_tx.try_send(()) {
                Ok(()) | Err(TrySendError::Full(())) => self.request_in_flight = true,
                Err(TrySendError::Disconnected(())) => self.request_in_flight = false,
            }
        }
        latest
    }
}

/// Phase 3: Merge execution outcomes into metrics, update cycle timing, write to disk.
///
/// Acquires `state.metrics` once, merges all `exec_outcomes` counters, records
/// timing, profile transitions, and RL metrics, then writes the snapshot to disk.
///
/// # Mutated locals (via `&mut`)
/// - `override_was_active` — updated to reflect current override state
/// - `critical_failure_timestamps` — rolling 3-minute failure window
#[allow(clippy::too_many_arguments)]
pub fn merge_cycle_metrics<'a>(
    state: &SharedState,
    exec_outcomes: &ExecuteOutcomes,
    network_monitor: &NetworkMonitor,
    decision_reactor_weight: f64,
    decision_blockers: &[BlockerScore],
    current_profile: OptimizationProfile,
    governor_decision: &GovernorDecision,
    lctx: &LearningContext<'a>,
    ais_worker: &mut AisRuntimeWorker,
    overflow_thresholds: &OverflowThresholds,
    cycle_start: &Instant,
    reactor_weight: f64,
    override_was_active: &mut bool,
    critical_failure_timestamps: &mut Vec<Instant>,
    timeline_path: &Path,
    metrics_path: &Path,
    cycle_count: u64,
    in_sleep: bool,
) {
    // File parsing happens on the AIS worker; this poll is non-blocking.
    let ais_snapshot = ais_worker.poll_and_schedule(cycle_count);

    let mut metrics = state.metrics.lock_recover();
    metrics.metrics.boosts_applied += exec_outcomes.boosts_applied;
    metrics.metrics.throttles_applied += exec_outcomes.throttles_applied;
    metrics.metrics.freezes_applied += exec_outcomes.freezes_applied;
    metrics.metrics.unfreezes_applied += exec_outcomes.unfreezes_applied;
    metrics.metrics.paging_hints_applied += exec_outcomes.paging_hints_applied;
    metrics.metrics.sysctl_applied += exec_outcomes.sysctl_applied;
    metrics.metrics.failures += exec_outcomes.failures;
    if let Some(e) = exec_outcomes.last_error.clone() {
        metrics.metrics.last_error = Some(e);
    }
    metrics.metrics.critical_background_skips += exec_outcomes.critical_background_skips;
    metrics.metrics.invalid_sysctl_denied += exec_outcomes.invalid_sysctl_denied;
    for skip in exec_outcomes.top_skipped.iter() {
        let skip = skip.clone();
        if metrics.metrics.top_skipped_processes.len() < 12
            && !metrics.metrics.top_skipped_processes.contains(&skip)
        {
            metrics.metrics.top_skipped_processes.push(skip);
        }
    }
    metrics.metrics.top_skipped_processes.truncate(12);
    metrics.metrics.throttle_reverted += exec_outcomes.throttle_reverted;
    metrics.metrics.thread_qos_applied += exec_outcomes.thread_qos_applied;
    metrics.metrics.thread_qos_hot_routes += exec_outcomes.thread_qos_hot_routes;
    metrics.metrics.thread_qos_cold_routes += exec_outcomes.thread_qos_cold_routes;
    metrics.metrics.journal_rotations += exec_outcomes.journal_rotations;
    metrics.metrics.journal_rotation_failures += exec_outcomes.journal_rotation_failures;

    // SysctlGovernor + NetworkMonitor metrics.
    metrics.metrics.sysctl_reactive_writes += exec_outcomes.sysctl_applied;
    {
        let hw = state.hardware.lock_recover();
        metrics.metrics.sysctl_governor_active_tunings = hw.sysctl_governor_status.active_tunings;
        metrics.metrics.sysctl_governor_total_writes = hw.sysctl_governor_status.total_writes;
    }
    metrics.metrics.network_retransmit_ratio = network_monitor.retransmission_rate();
    metrics.metrics.network_listen_drop_rate = network_monitor.listen_drop_rate();

    let had_new_failures = exec_outcomes.failures > 0;

    metrics.metrics.cycles += 1;
    if decision_reactor_weight > 0.2 {
        apollo_engine::engine::lse_counters::LSE_COUNTERS.increment_reactor_pulses();
    }
    metrics.metrics.last_cycle_at = Some(Utc::now());
    metrics.metrics.last_blockers = decision_blockers.to_vec();
    metrics.metrics.effective_profile = current_profile;
    metrics.throttle_level = governor_decision.throttle_level.clone();
    metrics.metrics.throttle_level = governor_decision.throttle_level.clone();
    // Use MetricsState.thermal_state (set by reactor) — no re-lock needed
    metrics.metrics.thermal_state = metrics.thermal_state.clone();
    metrics.metrics.last_pressure_score = governor_decision.pressure_score;
    if governor_decision.override_expired {
        metrics.metrics.override_expirations += 1;
    }
    if governor_decision.override_active && !*override_was_active {
        metrics.metrics.override_activations += 1;
    }
    if let Some(transition) = governor_decision.transition.clone() {
        metrics.metrics.profile_switches += 1;
        {
            let mut pg = state.policy.lock_recover();
            pg.timeline.push_back(transition.clone());
            if pg.timeline.len() > 200 {
                pg.timeline.pop_front();
            }
        }
        append_timeline(timeline_path, &transition);
    }
    *override_was_active = governor_decision.override_active;

    let elapsed = cycle_start.elapsed().as_millis() as u64;
    metrics.metrics.cycle_durations_ms.push_back(elapsed);
    if metrics.metrics.cycle_durations_ms.len() > 120 {
        metrics.metrics.cycle_durations_ms.pop_front();
    }
    metrics.metrics.p95_cycle_ms =
        compute_p95(metrics.metrics.cycle_durations_ms.make_contiguous());
    metrics.metrics.daemon_cycle_p95_ms = metrics.metrics.p95_cycle_ms;

    // reactor_weight: write back local accumulated value to MetricsState
    apollo_engine::engine::lse_counters::LSE_COUNTERS.set_reactor_event_weight(reactor_weight);

    let nowi = Instant::now();
    critical_failure_timestamps
        .retain(|t| nowi.duration_since(*t) <= std::time::Duration::from_secs(180));
    if had_new_failures {
        critical_failure_timestamps.push(nowi);
    }
    if critical_failure_timestamps.len() > 5 {
        state.policy.lock_recover().governor.force_safe_on_errors();
        critical_failure_timestamps.clear();
    }

    // Actualizar métricas del overflow guard antes de escribir.
    metrics.metrics.overflow_events_total = lctx.overflow_guard.history.total_overflows;
    metrics.metrics.overflow_events_7d = lctx.overflow_guard.recent_overflow_count(7);
    // B6 fix (round-3): report the *applied* compound offset (dynamic + RL +
    // workload + device, capped at -0.15) rather than the dynamic component
    // alone — otherwise the dashboard could show "recovered" while the live
    // threshold was still pinned at the floor.
    metrics.metrics.overflow_threshold_offset_pp = (lctx
        .overflow_guard
        .applied_offset(overflow_thresholds.workload_mode)
        * 100.0)
        .round() as i32;
    metrics.metrics.overflow_workload_mode = overflow_thresholds.workload_mode.as_str().to_string();

    // RL threshold agent metrics (Phase 4).
    if let Some(rl) = &lctx.overflow_guard.rl_agent {
        metrics.metrics.rl_adjustment_pp = (rl.current_adjustment * 100.0).round() as i32;
        metrics.metrics.rl_total_ticks = rl.total_ticks();
        metrics.metrics.rl_total_overflows = rl.total_overflows();
    }

    // Populate AIS fields (computed before the lock was taken).
    if let Some(s) = ais_snapshot {
        metrics.metrics.ais_score = s.total;
        metrics.metrics.ais_capability = s.capability;
        metrics.metrics.ais_optimization_opportunity = s.optimization_opportunity;
        metrics.metrics.ais_grade = s.grade.to_string();
        metrics.metrics.ais_decision = s.decision_precision;
        metrics.metrics.ais_signal = s.signal_quality;
        metrics.metrics.ais_learning = s.learning_velocity;
        metrics.metrics.ais_resource = s.resource_efficiency;
        metrics.metrics.ais_safety = s.safety_compliance;
        metrics.metrics.ais_adaptability = s.adaptability;
        metrics.metrics.ais_wisdom = s.wisdom;
        metrics.metrics.ais_evidence_coverage = s.evidence_coverage;
        metrics.metrics.ais_operational_health = s.operational_health;
        metrics.metrics.ais_pareto_balanced = s.pareto_balanced;
    }

    let should_write = should_snapshot_metrics(cycle_count, in_sleep);
    // Clone only when a write is due. Previously RuntimeMetrics (including
    // vectors and deques) was cloned on 24 cycles out of 25 and discarded.
    let metrics_snapshot = if should_write {
        metrics.metrics.metrics_disk_writes_total =
            metrics.metrics.metrics_disk_writes_total.saturating_add(1);
        Some(metrics.metrics.clone())
    } else {
        metrics.metrics.metrics_snapshot_clone_skips_total = metrics
            .metrics
            .metrics_snapshot_clone_skips_total
            .saturating_add(1);
        None
    };
    drop(metrics);
    // Rate-limit disk writes: atomic write (temp+rename) at 300ms/cycle was
    // writing 11KB × 2 = 22KB every 300ms = 73KB/s. Write every 25 cycles
    // (~7.5s) to stay within macOS's 24.86KB/s daily disk write budget.
    // Also skip writes while the system is sleeping — macOS accounts disk
    // writes against the daemon even during pre-sleep, burning the daily
    // budget while the machine is idle.
    if let Some(metrics_snapshot) = metrics_snapshot {
        write_metrics(metrics_path, &metrics_snapshot);
    }
}

fn should_snapshot_metrics(cycle_count: u64, in_sleep: bool) -> bool {
    !in_sleep && cycle_count.is_multiple_of(METRICS_DISK_WRITE_EVERY_N_CYCLES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::adaptive_governor::GovernorDecision as GovDecision;
    use apollo_engine::engine::intelligence_score::{compute_ais, AisInput};
    use std::time::Duration;

    #[test]
    fn metrics_snapshot_is_cloned_only_on_write_cycles() {
        assert!(!should_snapshot_metrics(24, false));
        assert!(should_snapshot_metrics(25, false));
        assert!(!should_snapshot_metrics(25, true));
    }

    #[test]
    fn unified_learning_revision_is_stable_without_source_changes() {
        let ledger = apollo_engine::engine::decision_ledger::DecisionLedger::new();
        let telemetry = apollo_engine::engine::telemetry_medallion::TelemetryMedallion::new(
            apollo_engine::engine::installation_identity::InstallationId::UNKNOWN,
        );
        let hierarchy = apollo_engine::engine::local_consolidation::LocalConsolidator::default();
        let scheduler =
            apollo_engine::engine::exploration_scheduler::ExplorationScheduler::cold_start(
                apollo_engine::engine::exploration_scheduler::ExplorationOrigin::default(),
            );

        let first = unified_learning_revision(&ledger, &telemetry, &hierarchy, &scheduler);
        let second = unified_learning_revision(&ledger, &telemetry, &hierarchy, &scheduler);

        assert_eq!(first, second);
    }

    #[test]
    fn cached_health_refreshes_after_unattributed_applied_receipt() {
        let mut ledger = apollo_engine::engine::decision_ledger::DecisionLedger::new();
        let decision_id =
            ledger.propose(apollo_engine::engine::decision_ledger::DecisionProposal::default());
        let telemetry = apollo_engine::engine::telemetry_medallion::TelemetryMedallion::new(
            apollo_engine::engine::installation_identity::InstallationId::UNKNOWN,
        );
        let hierarchy = apollo_engine::engine::local_consolidation::LocalConsolidator::default();
        let scheduler =
            apollo_engine::engine::exploration_scheduler::ExplorationScheduler::cold_start(
                apollo_engine::engine::exploration_scheduler::ExplorationOrigin::default(),
            );
        let mut cache = UnifiedLearningHealthCache::default();

        refresh_unified_learning_health(&mut cache, &ledger, &telemetry, &hierarchy, &scheduler);
        assert_eq!(cache.health().decision_ledger_unattributed_applied_total, 0);

        assert!(ledger.record_execution(
            decision_id,
            apollo_engine::engine::decision_ledger::ExecutionReceipt {
                disposition: apollo_engine::engine::decision_ledger::ExecutionDisposition::Applied,
                attribution: None,
                ..Default::default()
            },
        ));
        refresh_unified_learning_health(&mut cache, &ledger, &telemetry, &hierarchy, &scheduler);

        assert_eq!(cache.health().decision_ledger_unattributed_applied_total, 1);
    }

    #[test]
    fn archived_wrapper_does_not_hide_reverted_envelope_closure() {
        use apollo_engine::engine::decision_ledger::{
            ActuatorDecisionEvent, ActuatorDecisionOutcome, CycleDecisionEvents,
        };

        let mut ledger = DecisionLedger::new();
        let mut events = CycleDecisionEvents::default();
        assert!(events.push(ActuatorDecisionEvent::local(
            "interaction_qos:foreground",
            "pid:42",
            10,
            ActuatorDecisionOutcome::Reverted,
            "test",
            "rollback-complete",
        )));
        assert_eq!(ledger.ingest_cycle_events(&mut events).len(), 1);
        let telemetry = TelemetryMedallion::new(
            apollo_engine::engine::installation_identity::InstallationId::UNKNOWN,
        );
        let hierarchy = apollo_engine::engine::local_consolidation::LocalConsolidator::default();
        let scheduler = ExplorationScheduler::cold_start(
            apollo_engine::engine::exploration_scheduler::ExplorationOrigin::default(),
        );

        let health = build_unified_learning_health(&ledger, &telemetry, &hierarchy, &scheduler);
        assert!(health.latest_resolved_episode.reverted);
        assert_eq!(health.ledger_closure.closure_coverage, Some(1.0));
    }

    fn decision(pid: u32, dec: GovDecision, tier: ProcessTier) -> ProcessDecision {
        ProcessDecision {
            pid,
            name: format!("proc-{}", pid),
            decision: dec,
            tier,
            utility_score: 0.5,
            waste_score: 0.1,
            reason: String::new(),
        }
    }

    #[test]
    fn world_model_metrics_publish_phase_without_private_origin() {
        let telemetry = TelemetryMedallion::new(
            apollo_engine::engine::installation_identity::InstallationId(0x1234),
        );
        let world_model = WorldModel::default();
        let mut metrics = RuntimeMetrics::default();

        publish_world_model_metrics(&mut metrics, &telemetry, &world_model);

        assert_eq!(metrics.world_model_context_current_tier, "rejected");
        assert_eq!(metrics.world_model_context_authority_phase, "protected");
        assert_eq!(metrics.world_model_context_local_gold_total, 0);
        let wire = serde_json::to_string(&metrics).unwrap();
        assert!(!wire.contains("installation_id"));
        assert!(!wire.contains("0000000000001234"));
    }

    #[test]
    fn ais_worker_keeps_file_parsing_off_the_control_loop() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let mut worker = AisRuntimeWorker::spawn_with(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Some(compute_ais(&AisInput::default()))
        });

        let watchdog = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            release_tx.send(()).unwrap();
        });
        let started = Instant::now();
        assert!(worker.poll_and_schedule(1).is_none());
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "AIS scheduling blocked on the worker computation"
        );
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        watchdog.join().unwrap();

        let deadline = Instant::now() + Duration::from_secs(1);
        let result = loop {
            if let Some(score) = worker.poll_and_schedule(2) {
                break score;
            }
            assert!(
                Instant::now() < deadline,
                "AIS worker did not publish a result"
            );
            std::thread::sleep(Duration::from_millis(1));
        };
        assert!(result.total.is_finite());
    }

    #[test]
    fn thermal_force_ecores_routes_non_fg_to_background() {
        let fg: HashSet<u32> = HashSet::new();
        let d = decision(42, GovDecision::Allow, ProcessTier::ActiveForeground);
        // Even ActiveForeground gets demoted when thermal force_ecores fires
        // AND the pid is not in the foreground family.
        assert_eq!(
            decide_qos_tier(&d, &fg, true),
            Some(SchedulingTier::Background)
        );
    }

    #[test]
    fn io_tiering_excludes_preprotected_pids() {
        let decisions = vec![
            decision(42, GovDecision::Allow, ProcessTier::SystemEssential),
            decision(43, GovDecision::Throttle, ProcessTier::SilentDaemon),
        ];
        let critical = HashSet::from([42]);

        assert_eq!(
            actionable_io_tiers(&decisions, &critical),
            vec![(43, ProcessTier::SilentDaemon)]
        );
    }

    #[test]
    fn thermal_force_ecores_keeps_fg_family_foreground() {
        let mut fg = HashSet::new();
        fg.insert(42);
        let d = decision(42, GovDecision::Allow, ProcessTier::SilentDaemon);
        // Foreground family survives thermal demotion — UI responsiveness wins.
        assert_eq!(
            decide_qos_tier(&d, &fg, true),
            Some(SchedulingTier::Foreground)
        );
    }

    #[test]
    fn fg_family_overrides_governor_decision() {
        let mut fg = HashSet::new();
        fg.insert(7);
        // Even Throttle (which would otherwise map to None) is ignored for
        // fg-family members; tree cascade promotes to Foreground.
        let d = decision(7, GovDecision::Throttle, ProcessTier::SilentDaemon);
        assert_eq!(
            decide_qos_tier(&d, &fg, false),
            Some(SchedulingTier::Foreground)
        );
    }

    #[test]
    fn allow_active_foreground_maps_to_foreground() {
        let fg: HashSet<u32> = HashSet::new();
        let d = decision(100, GovDecision::Allow, ProcessTier::ActiveForeground);
        assert_eq!(
            decide_qos_tier(&d, &fg, false),
            Some(SchedulingTier::Foreground)
        );
    }

    #[test]
    fn allow_non_active_foreground_is_noop() {
        let fg: HashSet<u32> = HashSet::new();
        // Allow + non-ActiveForeground → None (skip task_for_pid syscall that
        // would otherwise fail SIP-protected and waste ~400 calls/cycle).
        let d = decision(100, GovDecision::Allow, ProcessTier::SilentDaemon);
        assert_eq!(decide_qos_tier(&d, &fg, false), None);
        let d = decision(101, GovDecision::Allow, ProcessTier::SystemEssential);
        assert_eq!(decide_qos_tier(&d, &fg, false), None);
    }

    #[test]
    fn throttle_skips_qos_routing() {
        let fg: HashSet<u32> = HashSet::new();
        // Throttle is handled via renice, not QoS tier — return None so the
        // caller doesn't fight itself by also demoting via MachQoS.
        let d = decision(200, GovDecision::Throttle, ProcessTier::SilentDaemon);
        assert_eq!(decide_qos_tier(&d, &fg, false), None);
    }

    #[test]
    fn freeze_and_kill_route_to_background() {
        let fg: HashSet<u32> = HashSet::new();
        let d = decision(300, GovDecision::Freeze, ProcessTier::SilentDaemon);
        assert_eq!(
            decide_qos_tier(&d, &fg, false),
            Some(SchedulingTier::Background)
        );
        let d = decision(301, GovDecision::Kill, ProcessTier::SilentDaemon);
        assert_eq!(
            decide_qos_tier(&d, &fg, false),
            Some(SchedulingTier::Background)
        );
    }

    #[test]
    fn qos_route_accounting_does_not_report_noops_as_applied() {
        use apollo_engine::engine::mach_qos::{QoSAuditDisposition, QoSOutcome};

        let outcomes = vec![
            (
                QoSOutcome {
                    pid: 10,
                    tier: SchedulingTier::Foreground,
                    success: true,
                    mutated: true,
                    error: None,
                },
                QoSAuditDisposition::Applied,
            ),
            (
                QoSOutcome {
                    pid: 11,
                    tier: SchedulingTier::Foreground,
                    success: true,
                    mutated: false,
                    error: None,
                },
                QoSAuditDisposition::NoOp,
            ),
            (
                QoSOutcome {
                    pid: 12,
                    tier: SchedulingTier::Background,
                    success: true,
                    mutated: false,
                    error: None,
                },
                QoSAuditDisposition::Blocked,
            ),
        ];

        let accounting = summarize_qos_route_outcomes(&outcomes);
        assert_eq!(accounting.requests, 3);
        assert_eq!(accounting.foreground_applied, 1);
        assert_eq!(accounting.background_applied, 0);
        assert_eq!(accounting.no_ops, 1);
        assert_eq!(accounting.blocked, 1);
        assert_eq!(accounting.failed, 0);
    }
}
