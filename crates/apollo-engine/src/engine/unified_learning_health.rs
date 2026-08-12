//! Cached, bounded observability and ranking advice for local learning.
//!
//! This module is deliberately pure: it owns no locks, I/O, process identity,
//! or actuator types. Consumers receive immutable snapshots or scalar support.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::engine::model_calibration::{
    CalibrationActionScope, CalibrationHorizon, CalibrationKey, ProducerId, TrustState,
};
use crate::engine::telemetry_medallion::ActuatorFamily;

pub const UNIFIED_LEARNING_SCHEMA_VERSION: u8 = 2;
pub const MAX_ADVICE_REQUESTS: usize = 48;
const MAX_PRODUCER_BYTES: usize = 48;
const MAX_ACTION_BYTES: usize = 96;
const MAX_STATUS_BYTES: usize = 48;
const VALIDATED_SUPPORT_LIMIT: f64 = 0.00125;
const TRUSTED_SUPPORT_LIMIT: f64 = 0.005;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LearningEvidenceState {
    #[default]
    Collecting,
    Available,
    Inconsistent,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LedgerClosureSnapshot {
    pub local_due: u64,
    pub local_closed: u64,
    pub open_due: u64,
    pub closure_coverage: Option<f64>,
    pub evidence_state: LearningEvidenceState,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TrustInventorySnapshot {
    pub immature: u64,
    pub candidate: u64,
    pub validated: u64,
    pub trusted: u64,
    pub degraded: u64,
    pub local_gold_decisions: u64,
    pub active_trusted: u64,
    pub worst_producer: String,
    pub worst_action: String,
    pub worst_horizon: String,
    pub worst_normalized_mae: Option<f64>,
    pub worst_coverage: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct HorizonCalibrationSnapshot {
    pub horizon: String,
    pub local_count: u64,
    pub normalized_mae: Option<f64>,
    pub coverage: Option<f64>,
    pub brier: Option<f64>,
    pub eligible: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct HierarchyLearningSnapshot {
    pub prototypes: u64,
    pub consolidations: u64,
    pub duplicates: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ExplorationOutcomeSnapshot {
    #[default]
    None,
    Issued,
    Committed,
    Denied,
    NoOp,
    Failed,
    Reverted,
    Deduplicated,
    Cooldown,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ExplorationLearningSnapshot {
    pub issued: u64,
    pub committed: u64,
    pub denied: u64,
    pub no_op: u64,
    pub failed: u64,
    pub reverted: u64,
    pub deduplicated: u64,
    pub cooldown: u64,
    pub last_outcome: ExplorationOutcomeSnapshot,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LatestResolvedEpisodeSnapshot {
    pub present: bool,
    pub id: u64,
    pub resolved_cycle: u64,
    pub action: String,
    pub tier: String,
    pub scope: String,
    pub authority: bool,
    pub treatment: bool,
    pub control: bool,
    pub reverted: bool,
    pub expected_utility: Option<f64>,
    pub measured_utility: Option<f64>,
    pub quality: Option<f64>,
    pub causal_result: String,
    pub calibration_result: String,
    pub trust_transition: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AisLearningSnapshot {
    pub local_learning_maturity: f64,
    pub learning: f64,
    pub wisdom: f64,
    pub unified_learning_evidence: f64,
    pub closure: Option<f64>,
    pub calibrated_accuracy: Option<f64>,
    pub causal_resolution: Option<f64>,
    pub active_breadth: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ClosureOutcome {
    #[default]
    Pending,
    Applied,
    Rejected,
    Vetoed,
    Blocked,
    Failed,
    NoOp,
    Reverted,
    Expired,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClosureObservation {
    pub decision_id: u64,
    pub local: bool,
    pub outcome: ClosureOutcome,
    pub issued_cycle: u64,
    pub horizon_cycles: u64,
    pub now_cycle: u64,
    pub resolved_evidence: bool,
    pub duplicate: bool,
    pub synthetic_overflow: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HorizonCalibrationInput {
    pub horizon: CalibrationHorizon,
    pub local_count: u64,
    pub normalized_mae: f64,
    pub coverage: f64,
    pub brier: Option<f64>,
    pub eligible: bool,
}

impl HorizonCalibrationInput {
    pub fn new(
        horizon: CalibrationHorizon,
        local_count: u64,
        normalized_mae: f64,
        coverage: f64,
        brier: Option<f64>,
    ) -> Self {
        Self {
            horizon,
            local_count,
            normalized_mae,
            coverage,
            brier,
            eligible: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdviceRecord {
    pub key: CalibrationKey,
    pub trust: TrustState,
    pub signed_error: f64,
    pub current_epoch: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdviceConsumer {
    #[default]
    WorldModel,
    Mpc,
    Gpu,
    Markov,
    Planner,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AdviceCandidate {
    pub consumer: AdviceConsumer,
    pub key: CalibrationKey,
}

impl AdviceCandidate {
    pub fn new(key: CalibrationKey) -> Self {
        Self {
            consumer: AdviceConsumer::WorldModel,
            key,
        }
    }

    pub fn for_consumer(consumer: AdviceConsumer, key: CalibrationKey) -> Self {
        Self { consumer, key }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdviceBatch {
    support: BTreeMap<AdviceCandidate, f64>,
}

impl AdviceBatch {
    pub fn build(health: &mut UnifiedLearningHealth, candidates: &[AdviceCandidate]) -> Self {
        let unique: BTreeSet<_> = candidates.iter().cloned().collect();
        if unique.len() > MAX_ADVICE_REQUESTS {
            health.advice_overflow_total = health.advice_overflow_total.saturating_add(1);
        }
        let support = unique
            .into_iter()
            .take(MAX_ADVICE_REQUESTS)
            .map(|candidate| {
                let value = health.support_for_key(&candidate.key);
                (candidate, value)
            })
            .collect();
        Self { support }
    }

    pub fn support(&self, candidate: &AdviceCandidate) -> f64 {
        self.support.get(candidate).copied().unwrap_or(0.0)
    }

    pub fn len(&self) -> usize {
        self.support.len()
    }

    pub fn is_empty(&self) -> bool {
        self.support.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnifiedLearningRevision {
    pub ledger: u64,
    pub ledger_unattributed_applied_total: u64,
    pub calibration: u64,
    pub hierarchy: u64,
    pub exploration: u64,
    pub causal: u64,
}

#[derive(Debug, Clone, Default)]
pub struct UnifiedLearningInput {
    pub decision_ledger_unattributed_applied_total: u64,
    pub local_gold_decisions: u64,
    pub imported_gold_decisions: u64,
    pub raw_action_count: u64,
    pub trusted_models: u64,
    pub trusted_active_models: u64,
    pub active_models: u64,
    /// Distinct current-epoch local Gold decisions with valid forecasts. When
    /// present, this bounded identity set is authoritative over the legacy
    /// scalar count for AIS maturity.
    pub authoritative_gold_decision_ids: Vec<u64>,
    pub closure: Option<(u64, u64)>,
    pub calibrated_accuracy: Option<f64>,
    pub causal_resolution: Option<f64>,
    pub closure_observations: Vec<ClosureObservation>,
    pub horizon_calibration: Vec<HorizonCalibrationInput>,
    pub advice_records: Vec<AdviceRecord>,
    pub hierarchy: HierarchyLearningSnapshot,
    pub exploration: ExplorationLearningSnapshot,
    pub latest_resolved_episode: LatestResolvedEpisodeSnapshot,
    /// Bounded resolved-episode candidates used to choose one deterministic
    /// latest display projection.
    pub latest_resolved_episodes: Vec<LatestResolvedEpisodeSnapshot>,
}

#[derive(Debug, Clone, Default)]
pub struct UnifiedLearningHealth {
    pub schema_version: u8,
    pub decision_ledger_unattributed_applied_total: u64,
    pub ledger_closure: LedgerClosureSnapshot,
    pub trust_inventory: TrustInventorySnapshot,
    pub horizon_calibration: Vec<HorizonCalibrationSnapshot>,
    pub hierarchy: HierarchyLearningSnapshot,
    pub exploration: ExplorationLearningSnapshot,
    pub latest_resolved_episode: LatestResolvedEpisodeSnapshot,
    pub ais: AisLearningSnapshot,
    pub advice_overflow_total: u64,
    advice: BTreeMap<CalibrationKey, AdviceRecord>,
}

impl UnifiedLearningHealth {
    pub fn from_input(input: UnifiedLearningInput) -> Self {
        let local_gold_decisions = distinct_positive_ids(&input.authoritative_gold_decision_ids)
            .map_or(input.local_gold_decisions, |count| count as u64);
        let (ledger_closure, closure_consistent) = if input.closure_observations.is_empty() {
            closure_from_counts(input.closure)
        } else {
            (closure_from_observations(&input.closure_observations), true)
        };
        let horizon_calibration = horizon_projection(&input.horizon_calibration);
        let calibrated_accuracy = input
            .calibrated_accuracy
            .and_then(finite_unit_option)
            .or_else(|| calibrated_accuracy(&input.horizon_calibration));
        let closure = closure_consistent
            .then_some(ledger_closure.closure_coverage)
            .flatten();
        let causal_resolution = input.causal_resolution.and_then(finite_unit_option);
        let active_breadth = (input.active_models > 0)
            .then(|| (input.trusted_active_models.min(5) as f64 / 5.0).clamp(0.0, 1.0));
        let maturity = if local_gold_decisions > 10 {
            ((local_gold_decisions - 10) as f64 / 40.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let learning =
            maturity * (0.55 * closure.unwrap_or(0.0) + 0.45 * calibrated_accuracy.unwrap_or(0.0));
        let wisdom = maturity
            * (0.70 * causal_resolution.unwrap_or(0.0) + 0.30 * active_breadth.unwrap_or(0.0));
        let evidence_count = [
            closure,
            calibrated_accuracy,
            causal_resolution,
            active_breadth,
        ]
        .iter()
        .filter(|value| value.is_some())
        .count() as f64;

        let mut trust_inventory = trust_inventory(
            local_gold_decisions,
            input.trusted_active_models,
            &input.advice_records,
        );
        normalize_latest_worst(&mut trust_inventory);
        let advice = input
            .advice_records
            .into_iter()
            .filter(|record| record.current_epoch)
            .map(|record| (record.key.clone(), record))
            .collect();
        let latest_resolved_episode = input
            .latest_resolved_episodes
            .into_iter()
            .filter(|episode| episode.present && episode.id > 0 && episode.resolved_cycle > 0)
            .max_by(|left, right| {
                left.resolved_cycle
                    .cmp(&right.resolved_cycle)
                    .then_with(|| left.id.cmp(&right.id))
            })
            .or_else(|| {
                input
                    .latest_resolved_episode
                    .present
                    .then_some(input.latest_resolved_episode)
            })
            .map(bounded_latest)
            .unwrap_or_default();

        Self {
            schema_version: UNIFIED_LEARNING_SCHEMA_VERSION,
            decision_ledger_unattributed_applied_total: input
                .decision_ledger_unattributed_applied_total,
            ledger_closure,
            trust_inventory,
            horizon_calibration,
            hierarchy: input.hierarchy,
            exploration: input.exploration,
            latest_resolved_episode,
            ais: AisLearningSnapshot {
                local_learning_maturity: finite_unit(maturity),
                learning: finite_unit(learning),
                wisdom: finite_unit(wisdom),
                unified_learning_evidence: finite_unit(maturity * evidence_count / 4.0),
                closure,
                calibrated_accuracy,
                causal_resolution,
                active_breadth,
            },
            advice_overflow_total: 0,
            advice,
        }
    }

    pub fn support_for_key(&self, key: &CalibrationKey) -> f64 {
        let record = self.advice.get(key).or_else(|| {
            let CalibrationActionScope::Exact(action) = &key.action else {
                return None;
            };
            let mut family_key = key.clone();
            family_key.action = CalibrationActionScope::Family(action_family(action)?);
            self.advice.get(&family_key)
        });
        let Some(record) = record.filter(|record| record.current_epoch) else {
            return 0.0;
        };
        let limit = match record.trust {
            TrustState::Validated => VALIDATED_SUPPORT_LIMIT,
            TrustState::Trusted => TRUSTED_SUPPORT_LIMIT,
            TrustState::Immature | TrustState::Candidate | TrustState::Degraded => return 0.0,
        };
        finite_signed(record.signed_error, limit)
    }

    pub fn publish_to(&self, metrics: &mut crate::engine::types::RuntimeMetrics) {
        metrics.unified_learning_schema_version = self.schema_version;
        metrics.decision_ledger_unattributed_applied_total =
            self.decision_ledger_unattributed_applied_total;
        metrics.ledger_closure = self.ledger_closure.clone();
        metrics.trust_inventory = self.trust_inventory.clone();
        metrics.horizon_calibration = self.horizon_calibration.clone();
        metrics.hierarchy_learning = self.hierarchy.clone();
        metrics.exploration_learning = self.exploration.clone();
        metrics.latest_resolved_episode = self.latest_resolved_episode.clone();
        metrics.unified_learning_advice_overflow_total = self.advice_overflow_total;
        metrics.ais_local_learning_maturity = self.ais.local_learning_maturity;
        metrics.ais_unified_learning_evidence = self.ais.unified_learning_evidence;
        metrics.ais_learning = self.ais.learning;
        metrics.ais_wisdom = self.ais.wisdom;
        metrics.unified_learning_ais = self.ais.clone();
        if self.latest_resolved_episode.present {
            metrics.last_episode_id = self.latest_resolved_episode.id;
            metrics.last_episode_resolved_cycle = self.latest_resolved_episode.resolved_cycle;
            metrics.last_episode_action = self.latest_resolved_episode.action.clone();
            metrics.last_episode_tier = self.latest_resolved_episode.tier.clone();
            if let Some(quality) = self.latest_resolved_episode.quality {
                metrics.last_episode_quality = quality;
            }
            if let Some(measured) = self.latest_resolved_episode.measured_utility {
                metrics.last_episode_utility = measured;
                metrics.last_episode_apollo_utility = measured;
            }
            if let Some(expected) = self.latest_resolved_episode.expected_utility {
                metrics.last_episode_predicted_gain = expected;
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct UnifiedLearningHealthCache {
    revision: Option<UnifiedLearningRevision>,
    health: UnifiedLearningHealth,
    rebuilds: u64,
}

impl UnifiedLearningHealthCache {
    pub fn refresh<F>(&mut self, revision: UnifiedLearningRevision, build: F) -> bool
    where
        F: FnOnce() -> UnifiedLearningHealth,
    {
        if self.revision == Some(revision) {
            return false;
        }
        self.health = build();
        self.revision = Some(revision);
        self.rebuilds = self.rebuilds.saturating_add(1);
        true
    }

    pub fn health(&self) -> &UnifiedLearningHealth {
        &self.health
    }

    pub fn rebuilds(&self) -> u64 {
        self.rebuilds
    }

    pub fn revision(&self) -> Option<UnifiedLearningRevision> {
        self.revision
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuAdjustedCandidate {
    pub expected_gain: f64,
    pub uncertainty: f64,
}

pub fn world_model_rank_contribution(contribution: f64, support: f64) -> f64 {
    let contribution = finite_or_zero(contribution);
    contribution * (1.0 + bounded_advice(support))
}

pub fn mpc_confidence_weight(confidence: f64, support: f64) -> f64 {
    (finite_unit(confidence) * (1.0 + bounded_advice(support))).clamp(0.0, 1.0)
}

pub fn adjust_gpu_candidate(
    expected_gain: f64,
    uncertainty: f64,
    support: f64,
) -> GpuAdjustedCandidate {
    let support = bounded_advice(support);
    GpuAdjustedCandidate {
        expected_gain: (finite_unit_signed(expected_gain) + support).clamp(-1.0, 1.0),
        uncertainty: (finite_unit(uncertainty) - support).clamp(0.0, 1.0),
    }
}

pub fn combine_gpu_support(existing: f64, unified: f64) -> f64 {
    (finite_or_zero(existing) + bounded_advice(unified))
        .clamp(-TRUSTED_SUPPORT_LIMIT, TRUSTED_SUPPORT_LIMIT)
}

pub fn markov_rank(rank: f64, support: f64) -> f64 {
    finite_or_zero(rank) + bounded_advice(support)
}

pub fn planner_rank(rank: f64, support: f64) -> f64 {
    finite_or_zero(rank) + bounded_advice(support)
}

pub fn bound_display_text(value: &str, max_bytes: usize) -> String {
    value
        .bytes()
        .filter(|byte| byte.is_ascii_graphic() || *byte == b' ')
        .take(max_bytes)
        .map(char::from)
        .collect()
}

fn closure_from_counts(counts: Option<(u64, u64)>) -> (LedgerClosureSnapshot, bool) {
    let Some((due, closed_raw)) = counts else {
        return (LedgerClosureSnapshot::default(), true);
    };
    let consistent = closed_raw <= due;
    let closed = closed_raw.min(due);
    let closure_coverage = (due > 0).then(|| closed as f64 / due as f64);
    (
        LedgerClosureSnapshot {
            local_due: due,
            local_closed: closed,
            open_due: due.saturating_sub(closed),
            closure_coverage,
            evidence_state: if !consistent {
                LearningEvidenceState::Inconsistent
            } else if due == 0 {
                LearningEvidenceState::Collecting
            } else {
                LearningEvidenceState::Available
            },
        },
        consistent,
    )
}

fn closure_from_observations(observations: &[ClosureObservation]) -> LedgerClosureSnapshot {
    let mut seen = BTreeSet::new();
    let mut due = 0_u64;
    let mut closed = 0_u64;
    for observation in observations.iter().take(128) {
        if observation.duplicate || (!observation.local && !observation.synthetic_overflow) {
            continue;
        }
        if observation.decision_id > 0 && !seen.insert(observation.decision_id) {
            continue;
        }
        let is_closed_terminal = matches!(
            observation.outcome,
            ClosureOutcome::Rejected
                | ClosureOutcome::Vetoed
                | ClosureOutcome::Blocked
                | ClosureOutcome::Failed
                | ClosureOutcome::NoOp
                | ClosureOutcome::Reverted
                | ClosureOutcome::Expired
        );
        if is_closed_terminal || observation.synthetic_overflow {
            due = due.saturating_add(1);
            closed = closed.saturating_add(1);
        } else if observation.outcome == ClosureOutcome::Applied
            && observation.now_cycle
                >= observation
                    .issued_cycle
                    .saturating_add(observation.horizon_cycles)
        {
            due = due.saturating_add(1);
            closed = closed.saturating_add(u64::from(observation.resolved_evidence));
        }
    }
    closure_from_counts(Some((due, closed))).0
}

fn horizon_projection(input: &[HorizonCalibrationInput]) -> Vec<HorizonCalibrationSnapshot> {
    if input.is_empty() {
        return Vec::new();
    }
    [
        CalibrationHorizon::Sec5,
        CalibrationHorizon::Sec30,
        CalibrationHorizon::Min2,
        CalibrationHorizon::Min10,
    ]
    .into_iter()
    .map(|horizon| {
        let item = input.iter().find(|item| item.horizon == horizon);
        HorizonCalibrationSnapshot {
            horizon: horizon_label(horizon).to_string(),
            local_count: item.map_or(0, |item| item.local_count),
            normalized_mae: item.and_then(|item| finite_nonnegative(item.normalized_mae, 1.0)),
            coverage: item.and_then(|item| finite_unit_option(item.coverage)),
            brier: item.and_then(|item| item.brier.and_then(finite_unit_option)),
            eligible: item.is_some_and(|item| item.eligible && item.local_count > 0),
        }
    })
    .collect()
}

fn calibrated_accuracy(input: &[HorizonCalibrationInput]) -> Option<f64> {
    let values: Vec<_> = input
        .iter()
        .filter(|item| item.eligible && item.local_count > 0)
        .filter_map(|item| {
            let mae = finite_nonnegative(item.normalized_mae, 2.0)?;
            let coverage = finite_unit_option(item.coverage)?;
            let mae_score = 1.0 - mae.clamp(0.0, 1.0);
            match item.brier.and_then(finite_unit_option) {
                Some(brier) => Some(0.60 * mae_score + 0.25 * coverage + 0.15 * (1.0 - brier)),
                None => Some((0.60 * mae_score + 0.25 * coverage) / 0.85),
            }
        })
        .collect();
    (!values.is_empty()).then(|| finite_unit(values.iter().sum::<f64>() / values.len() as f64))
}

fn trust_inventory(
    local_gold_decisions: u64,
    active_trusted: u64,
    records: &[AdviceRecord],
) -> TrustInventorySnapshot {
    let mut inventory = TrustInventorySnapshot {
        local_gold_decisions,
        active_trusted,
        ..TrustInventorySnapshot::default()
    };
    let mut unique = BTreeMap::new();
    for record in records
        .iter()
        .filter(|record| record.current_epoch)
        .take(512)
    {
        unique.insert(record.key.clone(), record);
    }
    let mut worst: Option<(&AdviceRecord, f64)> = None;
    for record in unique.into_values() {
        match record.trust {
            TrustState::Immature => inventory.immature = inventory.immature.saturating_add(1),
            TrustState::Candidate => inventory.candidate = inventory.candidate.saturating_add(1),
            TrustState::Validated => inventory.validated = inventory.validated.saturating_add(1),
            TrustState::Trusted => inventory.trusted = inventory.trusted.saturating_add(1),
            TrustState::Degraded => inventory.degraded = inventory.degraded.saturating_add(1),
        }
        let Some(error) = finite_nonnegative(record.signed_error.abs(), 1.0) else {
            continue;
        };
        let replace = worst.is_none_or(|(current, current_error)| {
            error > current_error || (error == current_error && record.key < current.key)
        });
        if replace {
            worst = Some((record, error));
        }
    }
    if let Some((record, error)) = worst {
        inventory.worst_producer = producer_label(record.key.producer).to_string();
        inventory.worst_action = action_label(&record.key.action);
        inventory.worst_horizon = horizon_label(record.key.horizon).to_string();
        inventory.worst_normalized_mae = Some(error);
    }
    inventory
}

fn normalize_latest_worst(inventory: &mut TrustInventorySnapshot) {
    inventory.worst_producer = bound_display_text(&inventory.worst_producer, MAX_PRODUCER_BYTES);
    inventory.worst_action = bound_display_text(&inventory.worst_action, MAX_ACTION_BYTES);
    inventory.worst_horizon = bound_display_text(&inventory.worst_horizon, MAX_STATUS_BYTES);
    inventory.worst_normalized_mae = inventory
        .worst_normalized_mae
        .and_then(|value| finite_nonnegative(value, 1.0));
    inventory.worst_coverage = inventory.worst_coverage.and_then(finite_unit_option);
}

fn bounded_latest(mut latest: LatestResolvedEpisodeSnapshot) -> LatestResolvedEpisodeSnapshot {
    latest.action = bound_display_text(&latest.action, MAX_ACTION_BYTES);
    latest.tier = bound_display_text(&latest.tier, MAX_STATUS_BYTES);
    latest.scope = bound_display_text(&latest.scope, MAX_STATUS_BYTES);
    latest.causal_result = bound_display_text(&latest.causal_result, MAX_STATUS_BYTES);
    latest.calibration_result = bound_display_text(&latest.calibration_result, MAX_STATUS_BYTES);
    latest.trust_transition = bound_display_text(&latest.trust_transition, MAX_STATUS_BYTES);
    latest.expected_utility = latest.expected_utility.and_then(finite_signed_unit_option);
    latest.measured_utility = latest.measured_utility.and_then(finite_signed_unit_option);
    latest.quality = latest.quality.and_then(finite_unit_option);
    latest
}

fn distinct_positive_ids(ids: &[u64]) -> Option<usize> {
    if ids.is_empty() {
        return None;
    }
    Some(
        ids.iter()
            .copied()
            .filter(|id| *id > 0)
            .collect::<BTreeSet<_>>()
            .len(),
    )
}

fn action_family(action: &str) -> Option<ActuatorFamily> {
    match action.split_once(':').map_or(action, |(family, _)| family) {
        "boost" => Some(ActuatorFamily::Boost),
        "throttle" => Some(ActuatorFamily::Throttle),
        "freeze" => Some(ActuatorFamily::Freeze),
        "unfreeze" => Some(ActuatorFamily::Unfreeze),
        "memorystatus" => Some(ActuatorFamily::Memorystatus),
        "thread_qos" | "interaction_qos" => Some(ActuatorFamily::ThreadQos),
        "markov_prewarm" => Some(ActuatorFamily::MarkovPrewarm),
        "coordinated" => Some(ActuatorFamily::Coordinated),
        _ => None,
    }
}

fn action_label(action: &CalibrationActionScope) -> String {
    match action {
        CalibrationActionScope::Exact(action) => bound_display_text(action, MAX_ACTION_BYTES),
        CalibrationActionScope::Family(family) => family.as_str().to_string(),
    }
}

fn producer_label(producer: ProducerId) -> &'static str {
    match producer {
        ProducerId::Actuator => "actuator",
        ProducerId::WorldModel => "world-model",
        ProducerId::GpuModel => "gpu-model",
        ProducerId::Markov => "markov",
        ProducerId::CausalGraph => "causal-graph",
        ProducerId::Mpc => "mpc",
        ProducerId::Nars => "nars",
        ProducerId::PolicyScorer => "policy-scorer",
        ProducerId::PredictiveAgent => "predictive-agent",
        ProducerId::OutcomeTracker => "outcome-tracker",
        ProducerId::LocalConsolidator => "local-consolidator",
        ProducerId::SurvivalMode => "survival-mode",
        ProducerId::Maintenance => "maintenance",
        ProducerId::Other => "other",
    }
}

fn horizon_label(horizon: CalibrationHorizon) -> &'static str {
    match horizon {
        CalibrationHorizon::Sec5 => "5s",
        CalibrationHorizon::Sec30 => "30s",
        CalibrationHorizon::Min2 => "2m",
        CalibrationHorizon::Min10 => "10m",
    }
}

fn bounded_advice(value: f64) -> f64 {
    finite_signed(value, TRUSTED_SUPPORT_LIMIT)
}

fn finite_signed(value: f64, limit: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-limit, limit)
    } else {
        0.0
    }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn finite_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn finite_unit_signed(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn finite_unit_option(value: f64) -> Option<f64> {
    value.is_finite().then(|| value.clamp(0.0, 1.0))
}

fn finite_signed_unit_option(value: f64) -> Option<f64> {
    value.is_finite().then(|| value.clamp(-1.0, 1.0))
}

fn finite_nonnegative(value: f64, max: f64) -> Option<f64> {
    value.is_finite().then(|| value.clamp(0.0, max))
}
