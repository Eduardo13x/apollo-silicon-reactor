//! Pure, bounded pairing and evidence closure for safe local experiments.
//!
//! The lab never executes an effect and never grants actuator authority. It
//! accepts only the closed exploration catalog, joins one local control with
//! one local treatment, and emits one deduplicated Pair Gold record after all
//! execution, horizon, and rollback facts close independently.

use std::collections::{BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::engine::exploration_scheduler::{
    ActionClass, ExplorationArm, ExplorationContext, ExplorationOrigin,
};
use crate::engine::telemetry_medallion::ActuatorFamily;

pub const MAX_OPEN_PAIRS: usize = 32;
pub const MAX_COMPLETED_PAIRS: usize = 128;
pub const MAX_GOLD_DEDUP: usize = 128;
pub const MAX_SERIALIZED_BYTES: usize = 64 * 1024;
pub const MAX_ACTION_KEY_BYTES: usize = 96;
const LAB_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PairId(pub u128);

impl Serialize for PairId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("{:032x}", self.0))
    }
}

impl<'de> Deserialize<'de> for PairId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value.len() != 32 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(serde::de::Error::custom(
                "pair id must be 32 hexadecimal bytes",
            ));
        }
        u128::from_str_radix(&value, 16)
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArmKind {
    Control,
    Treatment,
}

impl ArmKind {
    fn complement(self) -> Self {
        match self {
            Self::Control => Self::Treatment,
            Self::Treatment => Self::Control,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PairOrder {
    ControlThenTreatment,
    TreatmentThenControl,
}

impl PairOrder {
    fn arms(self) -> (ArmKind, ArmKind) {
        match self {
            Self::ControlThenTreatment => (ArmKind::Control, ArmKind::Treatment),
            Self::TreatmentThenControl => (ArmKind::Treatment, ArmKind::Control),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionClosure {
    Applied,
    NoOp,
    Failed,
    Blocked,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HorizonClosure {
    Complete,
    Incomplete,
    Confounded,
    Expired,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RollbackClosure {
    Succeeded,
    NotRequiredNonKernel,
    Failed,
    IdentityGone,
    Interrupted,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClosure {
    Bronze,
    Silver,
    PairGold,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PairCandidate {
    pub sequence: u64,
    pub origin: ExplorationOrigin,
    pub family: ActuatorFamily,
    pub action_class: ActionClass,
    pub treatment_arm: ExplorationArm,
    pub context: ExplorationContext,
    pub action_key: String,
    pub stratum_hash: u64,
    pub horizon_cycles: u32,
    pub washout_cycles: u32,
    pub minimum_effect_micros: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairGates {
    pub experiments_enabled: bool,
    pub privacy_known: bool,
    pub secure_input: bool,
    pub screen_capture: bool,
    pub camera_active: bool,
    pub sensitive_context: bool,
    pub inherited_safe: bool,
}

impl PairGates {
    pub fn healthy_enabled() -> Self {
        Self {
            experiments_enabled: true,
            privacy_known: true,
            secure_input: false,
            screen_capture: false,
            camera_active: false,
            sensitive_context: false,
            inherited_safe: true,
        }
    }

    fn allows_pair(self) -> bool {
        self.experiments_enabled
            && self.privacy_known
            && !self.secure_input
            && !self.screen_capture
            && !self.camera_active
            && !self.sensitive_context
            && self.inherited_safe
    }
}

impl Default for PairGates {
    fn default() -> Self {
        Self {
            experiments_enabled: false,
            privacy_known: false,
            secure_input: false,
            screen_capture: false,
            camera_active: false,
            sensitive_context: false,
            inherited_safe: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct PairAssignment {
    pub id: PairId,
    pub order: PairOrder,
    pub first: ArmKind,
    pub second: ArmKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PairEndpoint {
    pub arm: ArmKind,
    pub origin: ExplorationOrigin,
    pub family: ActuatorFamily,
    pub action_class: ActionClass,
    pub context: ExplorationContext,
    pub action_key: String,
    pub stratum_hash: u64,
    pub horizon_cycles: u32,
    pub decision_id: u64,
    pub observed_local: bool,
    pub synthetic: bool,
    pub execution: ExecutionClosure,
    pub horizon: HorizonClosure,
    pub rollback: RollbackClosure,
    pub utility_micros: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairProgress {
    AwaitingFirst,
    Washout,
    AwaitingComplement,
    ReadyToClose,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PersistedPairProgress {
    AwaitingFirst,
    Washout,
    AwaitingComplement,
    ReadyToClose,
}

impl From<PersistedPairProgress> for PairProgress {
    fn from(value: PersistedPairProgress) -> Self {
        match value {
            PersistedPairProgress::AwaitingFirst => Self::AwaitingFirst,
            PersistedPairProgress::Washout => Self::Washout,
            PersistedPairProgress::AwaitingComplement => Self::AwaitingComplement,
            PersistedPairProgress::ReadyToClose => Self::ReadyToClose,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PairRecord {
    candidate: PairCandidate,
    assignment: PairAssignment,
    first_endpoint: Option<PairEndpoint>,
    second_endpoint: Option<PairEndpoint>,
    washout_elapsed: u32,
    progress: PersistedPairProgress,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct PairClosure {
    pub id: PairId,
    pub evidence: EvidenceClosure,
    pub effect_micros: i64,
    pub effective: bool,
    pub harmful: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PairGoldRecord {
    pub id: PairId,
    pub origin: ExplorationOrigin,
    pub family: ActuatorFamily,
    pub action_class: ActionClass,
    pub action_key: String,
    pub stratum_hash: u64,
    pub control_decision_id: u64,
    pub treatment_decision_id: u64,
    pub effect_micros: i64,
    pub effective: bool,
    pub harmful: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
struct CompletedPairSummary {
    id: PairId,
    evidence: EvidenceClosure,
    effect_micros: i64,
    effective: bool,
    harmful: bool,
    interrupted: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LabMetrics {
    pub proposed_total: u64,
    pub eligible_total: u64,
    pub randomized_total: u64,
    pub control_endpoints_total: u64,
    pub treatment_endpoints_total: u64,
    pub complete_horizons_total: u64,
    pub rollback_closed_total: u64,
    pub pair_gold_total: u64,
    pub effective_total: u64,
    pub harmful_total: u64,
    pub confounded_total: u64,
    pub interrupted_total: u64,
    pub synthetic_quarantined_total: u64,
    pub shadow_would_open_total: u64,
    pub open_pairs: usize,
    pub completed_pairs: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LabWork {
    pub pairs_examined: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabError {
    Catalog,
    Gate,
    Origin,
    Invalid,
    Capacity,
    DuplicatePair,
    UnknownPair,
    DuplicateArm,
    WrongArm,
    WashoutPending,
    Mismatch,
    NotReady,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreDisposition {
    Restored,
    RestoredInterrupted,
    ResetOrigin,
    ResetHostile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MicroexperimentLabPersisted {
    schema_version: u32,
    origin: ExplorationOrigin,
    next_pair_sequence: u64,
    open: Vec<PairRecord>,
    completed: VecDeque<CompletedPairSummary>,
    gold_dedup: VecDeque<PairId>,
    metrics: PersistedMetrics,
    reserved: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(default)]
struct PersistedMetrics {
    proposed_total: u64,
    eligible_total: u64,
    randomized_total: u64,
    control_endpoints_total: u64,
    treatment_endpoints_total: u64,
    complete_horizons_total: u64,
    rollback_closed_total: u64,
    pair_gold_total: u64,
    effective_total: u64,
    harmful_total: u64,
    confounded_total: u64,
    interrupted_total: u64,
    synthetic_quarantined_total: u64,
    shadow_would_open_total: u64,
}

impl Default for MicroexperimentLabPersisted {
    fn default() -> Self {
        Self {
            schema_version: LAB_SCHEMA_VERSION,
            origin: ExplorationOrigin::default(),
            next_pair_sequence: 0,
            open: Vec::new(),
            completed: VecDeque::new(),
            gold_dedup: VecDeque::new(),
            metrics: PersistedMetrics::default(),
            reserved: String::new(),
        }
    }
}

impl MicroexperimentLabPersisted {
    #[doc(hidden)]
    pub fn oversized_for_test(origin: ExplorationOrigin, bytes: usize) -> Self {
        Self {
            origin,
            reserved: "x".repeat(bytes),
            ..Self::default()
        }
    }

    #[doc(hidden)]
    pub fn with_claimed_gold_for_test(
        mut self,
        pair_gold_total: u64,
        effective_total: u64,
        harmful_total: u64,
    ) -> Self {
        self.metrics.pair_gold_total = pair_gold_total;
        self.metrics.effective_total = effective_total;
        self.metrics.harmful_total = harmful_total;
        self
    }

    #[doc(hidden)]
    pub fn with_forged_gold_for_test(mut self) -> Self {
        self.next_pair_sequence = 1;
        let id = PairId((u128::from(self.origin.installation_id) << 64) | 1);
        self.completed.push_back(CompletedPairSummary {
            id,
            evidence: EvidenceClosure::PairGold,
            effect_micros: 1_000,
            effective: true,
            harmful: false,
            interrupted: false,
        });
        self.gold_dedup.push_back(id);
        self.metrics.pair_gold_total = 1;
        self.metrics.effective_total = 1;
        self
    }
}

#[derive(Debug, Clone)]
pub struct MicroexperimentLab {
    origin: ExplorationOrigin,
    next_pair_sequence: u64,
    open: Vec<PairRecord>,
    completed: VecDeque<CompletedPairSummary>,
    gold_dedup: VecDeque<PairId>,
    pending_gold: VecDeque<PairGoldRecord>,
    metrics: PersistedMetrics,
    last_work: LabWork,
}

impl MicroexperimentLab {
    pub fn cold_start(origin: ExplorationOrigin) -> Self {
        Self {
            origin,
            next_pair_sequence: 0,
            open: Vec::with_capacity(MAX_OPEN_PAIRS),
            completed: VecDeque::with_capacity(MAX_COMPLETED_PAIRS),
            gold_dedup: VecDeque::with_capacity(MAX_GOLD_DEDUP),
            pending_gold: VecDeque::with_capacity(MAX_GOLD_DEDUP),
            metrics: PersistedMetrics::default(),
            last_work: LabWork::default(),
        }
    }

    pub fn propose(
        &mut self,
        candidate: PairCandidate,
        gates: PairGates,
    ) -> Result<PairAssignment, LabError> {
        self.metrics.proposed_total = self.metrics.proposed_total.saturating_add(1);
        if !gates.allows_pair() {
            return Err(LabError::Gate);
        }
        if candidate.origin != self.origin {
            return Err(LabError::Origin);
        }
        validate_candidate(&candidate)?;
        if self.open.len() >= MAX_OPEN_PAIRS {
            return Err(LabError::Capacity);
        }
        let next = self
            .next_pair_sequence
            .checked_add(1)
            .ok_or(LabError::Capacity)?;
        let id = PairId((u128::from(self.origin.installation_id) << 64) | u128::from(next));
        if self.contains_pair(id) {
            return Err(LabError::DuplicatePair);
        }
        let order = if next % 2 == 1 {
            PairOrder::ControlThenTreatment
        } else {
            PairOrder::TreatmentThenControl
        };
        let (first, second) = order.arms();
        let assignment = PairAssignment {
            id,
            order,
            first,
            second,
        };
        self.open.push(PairRecord {
            candidate,
            assignment,
            first_endpoint: None,
            second_endpoint: None,
            washout_elapsed: 0,
            progress: PersistedPairProgress::AwaitingFirst,
        });
        self.next_pair_sequence = next;
        self.metrics.eligible_total = self.metrics.eligible_total.saturating_add(1);
        self.metrics.randomized_total = self.metrics.randomized_total.saturating_add(1);
        Ok(assignment)
    }

    /// Evaluate a real pipeline candidate without opening an experiment or
    /// changing any effect. Privacy opt-in is required only for mutation;
    /// inherited lifecycle/safety health still gates shadow diagnostics.
    pub fn evaluate_shadow(
        &mut self,
        candidate: PairCandidate,
        gates: PairGates,
    ) -> Result<PairAssignment, LabError> {
        self.metrics.proposed_total = self.metrics.proposed_total.saturating_add(1);
        if !gates.inherited_safe {
            return Err(LabError::Gate);
        }
        if candidate.origin != self.origin {
            return Err(LabError::Origin);
        }
        validate_candidate(&candidate)?;
        let next = self
            .next_pair_sequence
            .checked_add(1)
            .ok_or(LabError::Capacity)?;
        let id = PairId((u128::from(self.origin.installation_id) << 64) | u128::from(next));
        let order = if next % 2 == 1 {
            PairOrder::ControlThenTreatment
        } else {
            PairOrder::TreatmentThenControl
        };
        let (first, second) = order.arms();
        self.next_pair_sequence = next;
        self.metrics.eligible_total = self.metrics.eligible_total.saturating_add(1);
        self.metrics.randomized_total = self.metrics.randomized_total.saturating_add(1);
        self.metrics.shadow_would_open_total =
            self.metrics.shadow_would_open_total.saturating_add(1);
        Ok(PairAssignment {
            id,
            order,
            first,
            second,
        })
    }

    pub fn record_endpoint(
        &mut self,
        pair_id: PairId,
        endpoint: PairEndpoint,
    ) -> Result<PairProgress, LabError> {
        let index = self.find_open(pair_id).ok_or(LabError::UnknownPair)?;
        if endpoint.origin != self.origin {
            return Err(LabError::Origin);
        }
        let record = &mut self.open[index];
        if record
            .first_endpoint
            .as_ref()
            .is_some_and(|existing| existing.arm == endpoint.arm)
            || record
                .second_endpoint
                .as_ref()
                .is_some_and(|existing| existing.arm == endpoint.arm)
        {
            return Err(LabError::DuplicateArm);
        }
        if !endpoint_matches(&record.candidate, &endpoint) {
            return Err(LabError::Mismatch);
        }
        if endpoint.synthetic || !endpoint.observed_local || endpoint.decision_id == 0 {
            self.metrics.synthetic_quarantined_total =
                self.metrics.synthetic_quarantined_total.saturating_add(1);
        }
        match record.progress {
            PersistedPairProgress::AwaitingFirst => {
                if endpoint.arm != record.assignment.first {
                    return Err(LabError::WrongArm);
                }
                count_endpoint(&mut self.metrics, &endpoint);
                record.first_endpoint = Some(endpoint);
                record.progress = PersistedPairProgress::Washout;
                Ok(PairProgress::Washout)
            }
            PersistedPairProgress::Washout => Err(LabError::WashoutPending),
            PersistedPairProgress::AwaitingComplement => {
                if endpoint.arm != record.assignment.second {
                    return Err(LabError::WrongArm);
                }
                count_endpoint(&mut self.metrics, &endpoint);
                record.second_endpoint = Some(endpoint);
                record.progress = PersistedPairProgress::ReadyToClose;
                Ok(PairProgress::ReadyToClose)
            }
            PersistedPairProgress::ReadyToClose => Err(LabError::DuplicateArm),
        }
    }

    pub fn advance_washout(
        &mut self,
        pair_id: PairId,
        elapsed_cycles: u32,
    ) -> Result<PairProgress, LabError> {
        let index = self.find_open(pair_id).ok_or(LabError::UnknownPair)?;
        let record = &mut self.open[index];
        if record.progress != PersistedPairProgress::Washout {
            return Ok(record.progress.into());
        }
        record.washout_elapsed = record.washout_elapsed.saturating_add(elapsed_cycles);
        if record.washout_elapsed >= record.candidate.washout_cycles {
            record.progress = PersistedPairProgress::AwaitingComplement;
        }
        Ok(record.progress.into())
    }

    pub fn close_pair(&mut self, pair_id: PairId) -> Result<PairClosure, LabError> {
        if self.gold_dedup.contains(&pair_id)
            || self.completed.iter().any(|summary| summary.id == pair_id)
        {
            return Err(LabError::DuplicatePair);
        }
        let index = self.find_open(pair_id).ok_or(LabError::UnknownPair)?;
        if self.open[index].progress != PersistedPairProgress::ReadyToClose {
            return Err(LabError::NotReady);
        }
        let record = self.open.swap_remove(index);
        let first = record.first_endpoint.ok_or(LabError::NotReady)?;
        let second = record.second_endpoint.ok_or(LabError::NotReady)?;
        let (control, treatment) = if first.arm == ArmKind::Control {
            (&first, &second)
        } else {
            (&second, &first)
        };
        let effect_micros = treatment
            .utility_micros
            .saturating_sub(control.utility_micros);
        let authoritative = endpoints_are_authoritative(&record.candidate, control, treatment);
        let evidence = if authoritative {
            EvidenceClosure::PairGold
        } else {
            EvidenceClosure::Silver
        };
        let minimum = record.candidate.minimum_effect_micros.max(0);
        let effective = authoritative && effect_micros >= minimum && effect_micros > 0;
        let harmful =
            authoritative && effect_micros <= minimum.saturating_neg() && effect_micros < 0;
        let closure = PairClosure {
            id: pair_id,
            evidence,
            effect_micros,
            effective,
            harmful,
        };

        if evidence == EvidenceClosure::PairGold {
            let gold = PairGoldRecord {
                id: pair_id,
                origin: record.candidate.origin,
                family: record.candidate.family,
                action_class: record.candidate.action_class,
                action_key: record.candidate.action_key,
                stratum_hash: record.candidate.stratum_hash,
                control_decision_id: control.decision_id,
                treatment_decision_id: treatment.decision_id,
                effect_micros,
                effective,
                harmful,
            };
            push_bounded(&mut self.gold_dedup, pair_id, MAX_GOLD_DEDUP);
            push_bounded(&mut self.pending_gold, gold, MAX_GOLD_DEDUP);
            self.metrics.pair_gold_total = self.metrics.pair_gold_total.saturating_add(1);
            if effective {
                self.metrics.effective_total = self.metrics.effective_total.saturating_add(1);
            }
            if harmful {
                self.metrics.harmful_total = self.metrics.harmful_total.saturating_add(1);
            }
        }
        push_bounded(
            &mut self.completed,
            CompletedPairSummary {
                id: pair_id,
                evidence,
                effect_micros,
                effective,
                harmful,
                interrupted: false,
            },
            MAX_COMPLETED_PAIRS,
        );
        Ok(closure)
    }

    pub fn drain_pair_gold(&mut self) -> Vec<PairGoldRecord> {
        self.pending_gold.drain(..).collect()
    }

    pub fn persisted(&self) -> MicroexperimentLabPersisted {
        MicroexperimentLabPersisted {
            schema_version: LAB_SCHEMA_VERSION,
            origin: self.origin,
            next_pair_sequence: self.next_pair_sequence,
            open: self.open.clone(),
            completed: self.completed.clone(),
            gold_dedup: self.gold_dedup.clone(),
            metrics: self.metrics,
            reserved: String::new(),
        }
    }

    pub fn restore(
        persisted: MicroexperimentLabPersisted,
        expected_origin: ExplorationOrigin,
    ) -> (Self, RestoreDisposition) {
        if persisted.origin != expected_origin || !expected_origin.is_known() {
            return (
                Self::cold_start(expected_origin),
                RestoreDisposition::ResetOrigin,
            );
        }
        if !valid_persisted(&persisted) {
            return (
                Self::cold_start(expected_origin),
                RestoreDisposition::ResetHostile,
            );
        }
        let mut lab = Self {
            origin: expected_origin,
            next_pair_sequence: persisted.next_pair_sequence,
            open: Vec::with_capacity(MAX_OPEN_PAIRS),
            completed: persisted.completed,
            gold_dedup: persisted.gold_dedup,
            pending_gold: VecDeque::with_capacity(MAX_GOLD_DEDUP),
            metrics: persisted.metrics,
            last_work: LabWork::default(),
        };
        // Completed summaries do not retain both authoritative endpoint
        // receipts. On restart they remain useful history, but they cannot
        // carry Pair Gold/AIS authority until locally reverified.
        for summary in &mut lab.completed {
            if summary.evidence == EvidenceClosure::PairGold {
                summary.evidence = EvidenceClosure::Silver;
                summary.effective = false;
                summary.harmful = false;
            }
        }
        lab.gold_dedup.clear();
        let interrupted = persisted.open.len();
        for record in persisted.open {
            push_bounded(
                &mut lab.completed,
                CompletedPairSummary {
                    id: record.assignment.id,
                    evidence: EvidenceClosure::Bronze,
                    effect_micros: 0,
                    effective: false,
                    harmful: false,
                    interrupted: true,
                },
                MAX_COMPLETED_PAIRS,
            );
        }
        lab.metrics.interrupted_total = lab
            .metrics
            .interrupted_total
            .saturating_add(interrupted as u64);
        lab.reconcile_authoritative_metrics();
        let disposition = if interrupted == 0 {
            RestoreDisposition::Restored
        } else {
            RestoreDisposition::RestoredInterrupted
        };
        (lab, disposition)
    }

    pub fn metrics(&self) -> LabMetrics {
        LabMetrics {
            proposed_total: self.metrics.proposed_total,
            eligible_total: self.metrics.eligible_total,
            randomized_total: self.metrics.randomized_total,
            control_endpoints_total: self.metrics.control_endpoints_total,
            treatment_endpoints_total: self.metrics.treatment_endpoints_total,
            complete_horizons_total: self.metrics.complete_horizons_total,
            rollback_closed_total: self.metrics.rollback_closed_total,
            pair_gold_total: self.metrics.pair_gold_total,
            effective_total: self.metrics.effective_total,
            harmful_total: self.metrics.harmful_total,
            confounded_total: self.metrics.confounded_total,
            interrupted_total: self.metrics.interrupted_total,
            synthetic_quarantined_total: self.metrics.synthetic_quarantined_total,
            shadow_would_open_total: self.metrics.shadow_would_open_total,
            open_pairs: self.open.len(),
            completed_pairs: self.completed.len(),
        }
    }

    pub fn last_work(&self) -> LabWork {
        self.last_work
    }

    fn contains_pair(&self, id: PairId) -> bool {
        self.open.iter().any(|record| record.assignment.id == id)
            || self.completed.iter().any(|record| record.id == id)
            || self.gold_dedup.contains(&id)
    }

    fn reconcile_authoritative_metrics(&mut self) {
        let retained_gold: BTreeSet<_> = self.gold_dedup.iter().copied().collect();
        let mut seen = BTreeSet::new();
        let mut pair_gold_total = 0_u64;
        let mut effective_total = 0_u64;
        let mut harmful_total = 0_u64;
        for summary in &self.completed {
            if summary.evidence != EvidenceClosure::PairGold
                || summary.interrupted
                || !retained_gold.contains(&summary.id)
                || !seen.insert(summary.id)
            {
                continue;
            }
            pair_gold_total = pair_gold_total.saturating_add(1);
            effective_total = effective_total.saturating_add(u64::from(summary.effective));
            harmful_total = harmful_total.saturating_add(u64::from(summary.harmful));
        }
        self.metrics.pair_gold_total = pair_gold_total;
        self.metrics.effective_total = effective_total;
        self.metrics.harmful_total = harmful_total;
    }

    fn find_open(&mut self, id: PairId) -> Option<usize> {
        self.last_work.pairs_examined = 0;
        for (index, record) in self.open.iter().enumerate() {
            self.last_work.pairs_examined += 1;
            if record.assignment.id == id {
                return Some(index);
            }
        }
        None
    }
}

fn validate_candidate(candidate: &PairCandidate) -> Result<(), LabError> {
    if candidate.sequence == 0
        || candidate.stratum_hash == 0
        || candidate.horizon_cycles == 0
        || candidate.washout_cycles == 0
        || candidate.minimum_effect_micros < 0
        || candidate.action_key.is_empty()
        || candidate.action_key.len() > MAX_ACTION_KEY_BYTES
        || !candidate.origin.is_known()
    {
        return Err(LabError::Invalid);
    }
    let catalogued = matches!(
        (
            candidate.family,
            candidate.action_class,
            candidate.treatment_arm
        ),
        (
            ActuatorFamily::InteractionQos,
            ActionClass::InteractionForeground,
            ExplorationArm::InteractionQosShort
                | ExplorationArm::InteractionQosStandard
                | ExplorationArm::InteractionQosLong
        ) | (
            ActuatorFamily::MarkovPrewarm,
            ActionClass::MarkovPredictedApp,
            ExplorationArm::MarkovCacheOnly
        ) | (
            ActuatorFamily::Boost,
            ActionClass::BoostBackground,
            ExplorationArm::BoostOmission
        )
    );
    if !catalogued {
        return Err(LabError::Catalog);
    }
    Ok(())
}

fn endpoint_matches(candidate: &PairCandidate, endpoint: &PairEndpoint) -> bool {
    endpoint.origin == candidate.origin
        && endpoint.family == candidate.family
        && endpoint.action_class == candidate.action_class
        && endpoint.context == candidate.context
        && endpoint.action_key == candidate.action_key
        && endpoint.action_key.len() <= MAX_ACTION_KEY_BYTES
        && endpoint.stratum_hash == candidate.stratum_hash
        && endpoint.horizon_cycles == candidate.horizon_cycles
}

fn endpoints_are_authoritative(
    candidate: &PairCandidate,
    control: &PairEndpoint,
    treatment: &PairEndpoint,
) -> bool {
    let control_execution = matches!(
        control.execution,
        ExecutionClosure::Applied | ExecutionClosure::NoOp
    );
    let rollback_closed = match candidate.family {
        ActuatorFamily::MarkovPrewarm => {
            treatment.rollback == RollbackClosure::NotRequiredNonKernel
                && matches!(
                    control.rollback,
                    RollbackClosure::NotRequiredNonKernel | RollbackClosure::Succeeded
                )
        }
        _ => {
            treatment.rollback == RollbackClosure::Succeeded
                && matches!(
                    control.rollback,
                    RollbackClosure::Succeeded | RollbackClosure::NotRequiredNonKernel
                )
        }
    };
    control_execution
        && treatment.execution == ExecutionClosure::Applied
        && control.horizon == HorizonClosure::Complete
        && treatment.horizon == HorizonClosure::Complete
        && rollback_closed
        && control.observed_local
        && treatment.observed_local
        && !control.synthetic
        && !treatment.synthetic
        && control.decision_id > 0
        && treatment.decision_id > 0
        && control.decision_id != treatment.decision_id
}

fn count_endpoint(metrics: &mut PersistedMetrics, endpoint: &PairEndpoint) {
    match endpoint.arm {
        ArmKind::Control => {
            metrics.control_endpoints_total = metrics.control_endpoints_total.saturating_add(1)
        }
        ArmKind::Treatment => {
            metrics.treatment_endpoints_total = metrics.treatment_endpoints_total.saturating_add(1)
        }
    }
    if endpoint.horizon == HorizonClosure::Complete {
        metrics.complete_horizons_total = metrics.complete_horizons_total.saturating_add(1);
    }
    if endpoint.horizon == HorizonClosure::Confounded {
        metrics.confounded_total = metrics.confounded_total.saturating_add(1);
    }
    if matches!(
        endpoint.rollback,
        RollbackClosure::Succeeded | RollbackClosure::NotRequiredNonKernel
    ) {
        metrics.rollback_closed_total = metrics.rollback_closed_total.saturating_add(1);
    }
}

fn push_bounded<T>(queue: &mut VecDeque<T>, value: T, cap: usize) {
    if queue.len() >= cap {
        queue.pop_front();
    }
    queue.push_back(value);
}

fn valid_persisted(persisted: &MicroexperimentLabPersisted) -> bool {
    if persisted.schema_version != LAB_SCHEMA_VERSION
        || persisted.open.len() > MAX_OPEN_PAIRS
        || persisted.completed.len() > MAX_COMPLETED_PAIRS
        || persisted.gold_dedup.len() > MAX_GOLD_DEDUP
        || !persisted.reserved.is_empty()
        || serde_json::to_vec(persisted).map_or(true, |bytes| bytes.len() > MAX_SERIALIZED_BYTES)
    {
        return false;
    }
    persisted.open.iter().all(|record| {
        validate_candidate(&record.candidate).is_ok()
            && record.candidate.origin == persisted.origin
            && record.assignment.id.0 != 0
            && record.assignment.first.complement() == record.assignment.second
            && record
                .first_endpoint
                .as_ref()
                .is_none_or(|endpoint| endpoint_matches(&record.candidate, endpoint))
            && record
                .second_endpoint
                .as_ref()
                .is_none_or(|endpoint| endpoint_matches(&record.candidate, endpoint))
    })
}
