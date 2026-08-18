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
pub const SHADOW_MIN_OPPORTUNITIES: u64 = 500;
pub const SHADOW_MIN_DURATION_MS: u64 = 15 * 60 * 1_000;
pub const CANARY_MIN_OPPORTUNITIES: u64 = 500;
pub const CANARY_PERCENT: u8 = 10;
pub const ENDPOINT_GRACE_CYCLES: u64 = 12;
const LAB_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LabPhase {
    #[default]
    Shadow,
    Canary,
    Active,
}

impl LabPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shadow => "shadow",
            Self::Canary => "canary",
            Self::Active => "active",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LabRolloutConfig {
    shadow_min_opportunities: u64,
    shadow_min_duration_ms: u64,
    canary_percent: u8,
    canary_min_opportunities: u64,
}

impl Default for LabRolloutConfig {
    fn default() -> Self {
        Self {
            shadow_min_opportunities: SHADOW_MIN_OPPORTUNITIES,
            shadow_min_duration_ms: SHADOW_MIN_DURATION_MS,
            canary_percent: CANARY_PERCENT,
            canary_min_opportunities: CANARY_MIN_OPPORTUNITIES,
        }
    }
}

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

    pub fn allows_pair(self) -> bool {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateDisposition {
    Shadow(PairAssignment),
    CanarySkipped,
    Opened(PairAssignment),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairDirective {
    pub pair_id: PairId,
    pub arm: ArmKind,
    pub treatment_arm: ExplorationArm,
    pub family: ActuatorFamily,
    pub action_class: ActionClass,
    pub context: ExplorationContext,
    pub action_key: String,
    pub stratum_hash: u64,
    pub issued_cycle: u64,
    pub complete_not_before_cycle: u64,
    pub expires_after_cycle: u64,
    pub rollback_required: bool,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedPairEndpoint {
    pub pair_id: PairId,
    pub issued_cycle: u64,
    pub completed_cycle: u64,
    pub endpoint: PairEndpoint,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PairInvalidationReason {
    SafetyGate,
    DeadlineExpired,
    Confounded,
    IncompleteHorizon,
    FailedExecution,
    FailedRollback,
    Unauthoritative,
    ClockRegression,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairInvalidation {
    pub pair_id: PairId,
    pub reason: PairInvalidationReason,
    pub family: ActuatorFamily,
    pub action_key: String,
    pub rollback_required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimedEndpointDisposition {
    Progress(PairProgress),
    Invalidated(PairInvalidation),
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct IssuedArm {
    arm: ArmKind,
    issued_cycle: u64,
    complete_not_before_cycle: u64,
    expires_after_cycle: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PairRecord {
    candidate: PairCandidate,
    assignment: PairAssignment,
    first_endpoint: Option<PairEndpoint>,
    second_endpoint: Option<PairEndpoint>,
    washout_elapsed: u32,
    progress: PersistedPairProgress,
    #[serde(default)]
    issued: Option<IssuedArm>,
    #[serde(default)]
    washout_started_cycle: Option<u64>,
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

#[derive(Debug, Clone, Copy, Default, PartialEq)]
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
    pub invalidated_total: u64,
    pub deadline_expired_total: u64,
    /// Arms that expired without ever reporting, outside Shadow. Counted apart
    /// from `canary_failures`: no data is not adverse data.
    pub unbound_expiries_total: u64,
    pub rollback_failed_total: u64,
    pub open_pairs: usize,
    pub completed_pairs: usize,
    pub mean_effect: f64,
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
    HorizonPending,
    EndpointNotIssued,
    DeadlineExpired,
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
    rollout: PersistedRollout,
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
    invalidated_total: u64,
    deadline_expired_total: u64,
    #[serde(default)]
    unbound_expiries_total: u64,
    rollback_failed_total: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(default)]
struct PersistedRollout {
    phase: LabPhase,
    phase_started_millis: u64,
    last_monotonic_millis: u64,
    shadow_eligible: u64,
    canary_eligible: u64,
    canary_opened: u64,
    canary_completed: u64,
    canary_gold: u64,
    canary_failures: u64,
}

impl PersistedRollout {
    /// Returns the shadow gate progress this reset discarded. Callers that are
    /// not deliberately carrying the value forward must account for it: a
    /// silent drop is indistinguishable on the dashboard from a host that
    /// never produced opportunities in the first place.
    #[must_use]
    fn reset_to_shadow(&mut self, now_millis: u64) -> u64 {
        let dropped = self.shadow_eligible;
        self.phase = LabPhase::Shadow;
        self.phase_started_millis = now_millis;
        self.last_monotonic_millis = now_millis;
        self.shadow_eligible = 0;
        self.canary_eligible = 0;
        self.canary_opened = 0;
        self.canary_completed = 0;
        self.canary_gold = 0;
        self.canary_failures = 0;
        dropped
    }
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
            rollout: PersistedRollout::default(),
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
    rollout: PersistedRollout,
    rollout_config: LabRolloutConfig,
    last_work: LabWork,
    /// Whether the phase clock has been armed against this boot's monotonic
    /// clock. Deliberately not persisted: `phase_started_millis` alone cannot
    /// express "unarmed", because 0 is also a legitimate `now_millis`.
    phase_clock_armed: bool,
    /// Shadow gate progress observed at the end of `restore`, before this boot
    /// ran a single cycle. Not persisted: it describes this boot only.
    ///
    /// Without it a gate counter that fell backwards across a restart is
    /// ambiguous from the dashboard — "the disk held a low value" and "the
    /// runtime discarded a high one" render identically.
    restored_progress: u64,
    /// How many times a runtime reset discarded non-zero gate progress, and
    /// why the last one fired. Not persisted, and deliberately separate from
    /// the deliberate reset inside `restore`, which carries its value forward.
    progress_resets: u64,
    last_progress_reset_reason: &'static str,
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
            rollout: PersistedRollout::default(),
            rollout_config: LabRolloutConfig::default(),
            last_work: LabWork::default(),
            phase_clock_armed: false,
            restored_progress: 0,
            progress_resets: 0,
            last_progress_reset_reason: "",
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
            issued: None,
            washout_started_cycle: None,
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

    pub fn phase(&self) -> LabPhase {
        self.rollout.phase
    }

    /// Opportunities counted toward the current phase gate, and the threshold
    /// they must reach. Without this the remaining wait is invisible: the
    /// published `*_total` counters are cumulative across restarts and do not
    /// track the rollout gate.
    fn note_progress_reset(&mut self, dropped: u64, reason: &'static str) {
        if dropped == 0 {
            return;
        }
        self.progress_resets = self.progress_resets.saturating_add(1);
        self.last_progress_reset_reason = reason;
    }

    /// Boot-scoped provenance for the shadow gate counter: what `restore` left
    /// in place before the first cycle, how many runtime resets have discarded
    /// progress since, and the reason of the most recent one.
    pub fn rollout_provenance(&self) -> (u64, u64, &'static str) {
        (
            self.restored_progress,
            self.progress_resets,
            self.last_progress_reset_reason,
        )
    }

    pub fn rollout_progress(&self) -> (u64, u64) {
        match self.rollout.phase {
            LabPhase::Shadow => (
                self.rollout.shadow_eligible,
                self.rollout_config.shadow_min_opportunities,
            ),
            LabPhase::Canary => (
                self.rollout.canary_eligible,
                self.rollout_config.canary_min_opportunities,
            ),
            LabPhase::Active => (0, 0),
        }
    }

    pub fn readiness_blocker(&self) -> Option<&'static str> {
        if self.open.iter().any(|record| record.issued.is_some()) {
            return Some("awaiting-real-endpoint");
        }
        if self
            .open
            .iter()
            .any(|record| record.progress == PersistedPairProgress::Washout)
        {
            return Some("washout");
        }
        if self
            .open
            .iter()
            .any(|record| record.progress == PersistedPairProgress::ReadyToClose)
        {
            return Some("pair-not-closed");
        }
        (!self.open.is_empty()).then_some("arm-ready")
    }

    pub fn consider_candidate(
        &mut self,
        candidate: PairCandidate,
        gates: PairGates,
        now_millis: u64,
    ) -> Result<CandidateDisposition, LabError> {
        // Arms the phase clock on the first candidate of this boot. A restore
        // carries `shadow_eligible` forward, so the eligibility counters cannot
        // serve as the "unarmed" signal any more, and `phase_started_millis`
        // cannot either because 0 is a legitimate `now_millis`.
        if !self.phase_clock_armed {
            self.rollout.phase_started_millis = now_millis;
            self.phase_clock_armed = true;
        }
        if now_millis < self.rollout.last_monotonic_millis {
            let dropped = self.rollout.reset_to_shadow(now_millis);
            self.note_progress_reset(dropped, "clock-regression-candidate");
        }
        self.rollout.last_monotonic_millis = now_millis;

        let disposition = match self.rollout.phase {
            LabPhase::Shadow => {
                let assignment = self.evaluate_shadow(candidate, gates)?;
                self.rollout.shadow_eligible = self.rollout.shadow_eligible.saturating_add(1);
                CandidateDisposition::Shadow(assignment)
            }
            LabPhase::Canary => {
                let admitted = canary_admitted(&candidate, self.rollout_config.canary_percent);
                if admitted {
                    let assignment = self.propose(candidate, gates)?;
                    self.rollout.canary_eligible = self.rollout.canary_eligible.saturating_add(1);
                    self.rollout.canary_opened = self.rollout.canary_opened.saturating_add(1);
                    CandidateDisposition::Opened(assignment)
                } else {
                    self.metrics.proposed_total = self.metrics.proposed_total.saturating_add(1);
                    if !gates.allows_pair() {
                        return Err(LabError::Gate);
                    }
                    if candidate.origin != self.origin {
                        return Err(LabError::Origin);
                    }
                    validate_candidate(&candidate)?;
                    self.metrics.eligible_total = self.metrics.eligible_total.saturating_add(1);
                    self.metrics.randomized_total = self.metrics.randomized_total.saturating_add(1);
                    self.rollout.canary_eligible = self.rollout.canary_eligible.saturating_add(1);
                    CandidateDisposition::CanarySkipped
                }
            }
            LabPhase::Active => CandidateDisposition::Opened(self.propose(candidate, gates)?),
        };
        self.advance_rollout(now_millis, gates.allows_pair());
        Ok(disposition)
    }

    pub fn advance_cycle(
        &mut self,
        cycle: u64,
        now_millis: u64,
        gates: PairGates,
    ) -> Vec<PairInvalidation> {
        if now_millis < self.rollout.last_monotonic_millis {
            let invalidated = self.invalidate_all(PairInvalidationReason::ClockRegression);
            let dropped = self.rollout.reset_to_shadow(now_millis);
            self.note_progress_reset(dropped, "clock-regression-cycle");
            return invalidated;
        }
        self.rollout.last_monotonic_millis = now_millis;
        if !gates.allows_pair() {
            let invalidated = self.invalidate_all(PairInvalidationReason::SafetyGate);
            if self.rollout.phase != LabPhase::Shadow || !invalidated.is_empty() {
                let dropped = self.rollout.reset_to_shadow(now_millis);
                self.note_progress_reset(dropped, "safety-gate");
            }
            return invalidated;
        }

        for record in &mut self.open {
            if record.progress == PersistedPairProgress::Washout {
                if let Some(started) = record.washout_started_cycle {
                    let elapsed = cycle.saturating_sub(started);
                    record.washout_elapsed = elapsed.min(u64::from(u32::MAX)) as u32;
                    if record.washout_elapsed >= record.candidate.washout_cycles {
                        record.progress = PersistedPairProgress::AwaitingComplement;
                        record.washout_started_cycle = None;
                    }
                }
            }
        }
        let expired: Vec<_> = self
            .open
            .iter()
            .filter_map(|record| {
                record
                    .issued
                    .filter(|issued| cycle > issued.expires_after_cycle)
                    .map(|_| record.assignment.id)
            })
            .collect();
        let mut invalidated = Vec::with_capacity(expired.len());
        for pair_id in expired {
            if let Ok(record) =
                self.invalidate_pair(pair_id, PairInvalidationReason::DeadlineExpired)
            {
                invalidated.push(record);
            }
        }
        if self.rollout.phase != LabPhase::Shadow && self.rollout.canary_failures > 0 {
            invalidated.extend(self.invalidate_all(PairInvalidationReason::Unauthoritative));
            let dropped = self.rollout.reset_to_shadow(now_millis);
            self.note_progress_reset(dropped, "canary-failure");
        }
        self.advance_rollout(now_millis, true);
        invalidated
    }

    pub fn issue_ready_arms(&mut self, cycle: u64, gates: PairGates) -> Vec<PairDirective> {
        if self.rollout.phase == LabPhase::Shadow || !gates.allows_pair() {
            return Vec::new();
        }
        let mut directives = Vec::with_capacity(self.open.len());
        for record in &mut self.open {
            if record.issued.is_some()
                || !matches!(
                    record.progress,
                    PersistedPairProgress::AwaitingFirst
                        | PersistedPairProgress::AwaitingComplement
                )
            {
                continue;
            }
            let arm = match record.progress {
                PersistedPairProgress::AwaitingFirst => record.assignment.first,
                PersistedPairProgress::AwaitingComplement => record.assignment.second,
                _ => continue,
            };
            let complete_not_before_cycle =
                cycle.saturating_add(u64::from(record.candidate.horizon_cycles));
            let expires_after_cycle =
                complete_not_before_cycle.saturating_add(ENDPOINT_GRACE_CYCLES);
            record.issued = Some(IssuedArm {
                arm,
                issued_cycle: cycle,
                complete_not_before_cycle,
                expires_after_cycle,
            });
            directives.push(PairDirective {
                pair_id: record.assignment.id,
                arm,
                treatment_arm: record.candidate.treatment_arm,
                family: record.candidate.family,
                action_class: record.candidate.action_class,
                context: record.candidate.context,
                action_key: record.candidate.action_key.clone(),
                stratum_hash: record.candidate.stratum_hash,
                issued_cycle: cycle,
                complete_not_before_cycle,
                expires_after_cycle,
                rollback_required: arm == ArmKind::Treatment
                    && record.candidate.family != ActuatorFamily::MarkovPrewarm,
            });
        }
        directives
    }

    pub fn record_timed_endpoint(
        &mut self,
        observation: TimedPairEndpoint,
    ) -> Result<TimedEndpointDisposition, LabError> {
        let index = self
            .find_open(observation.pair_id)
            .ok_or(LabError::UnknownPair)?;
        let record = &self.open[index];
        let issued = record.issued.ok_or(LabError::EndpointNotIssued)?;
        if observation.issued_cycle != issued.issued_cycle || observation.endpoint.arm != issued.arm
        {
            return Err(LabError::Mismatch);
        }
        if !endpoint_matches(&record.candidate, &observation.endpoint) {
            return Err(LabError::Mismatch);
        }
        if observation.completed_cycle < issued.complete_not_before_cycle
            && observation.endpoint.horizon == HorizonClosure::Complete
        {
            return Err(LabError::HorizonPending);
        }

        let reason = endpoint_invalidation_reason(
            &record.candidate,
            &observation.endpoint,
            observation.completed_cycle,
            issued,
        );
        if let Some(reason) = reason {
            if observation.endpoint.synthetic
                || !observation.endpoint.observed_local
                || observation.endpoint.decision_id == 0
            {
                self.metrics.synthetic_quarantined_total =
                    self.metrics.synthetic_quarantined_total.saturating_add(1);
            }
            let invalidated = self.invalidate_pair(observation.pair_id, reason)?;
            return Ok(TimedEndpointDisposition::Invalidated(invalidated));
        }

        self.open[index].issued = None;
        let completed_cycle = observation.completed_cycle;
        let progress = self.record_endpoint(observation.pair_id, observation.endpoint)?;
        if progress == PairProgress::Washout {
            let index = self
                .find_open(observation.pair_id)
                .ok_or(LabError::UnknownPair)?;
            self.open[index].washout_started_cycle = Some(completed_cycle);
        }
        Ok(TimedEndpointDisposition::Progress(progress))
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
            record.washout_started_cycle = None;
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
        match self.rollout.phase {
            LabPhase::Canary => {
                self.rollout.canary_completed = self.rollout.canary_completed.saturating_add(1);
                if closure.evidence == EvidenceClosure::PairGold {
                    self.rollout.canary_gold = self.rollout.canary_gold.saturating_add(1);
                }
                if closure.evidence != EvidenceClosure::PairGold || closure.harmful {
                    self.rollout.canary_failures = self.rollout.canary_failures.saturating_add(1);
                }
            }
            LabPhase::Active
                if closure.evidence != EvidenceClosure::PairGold || closure.harmful =>
            {
                self.rollout.canary_failures = self.rollout.canary_failures.saturating_add(1);
            }
            _ => {}
        }
        self.advance_rollout(self.rollout.last_monotonic_millis, true);
        Ok(closure)
    }

    pub fn drain_pair_gold(&mut self) -> Vec<PairGoldRecord> {
        self.pending_gold.drain(..).collect()
    }

    pub fn invalidate_pair(
        &mut self,
        pair_id: PairId,
        reason: PairInvalidationReason,
    ) -> Result<PairInvalidation, LabError> {
        let index = self.find_open(pair_id).ok_or(LabError::UnknownPair)?;
        let record = self.open.swap_remove(index);
        let rollback_required = rollback_still_required(&record);
        self.metrics.invalidated_total = self.metrics.invalidated_total.saturating_add(1);
        self.metrics.interrupted_total = self.metrics.interrupted_total.saturating_add(1);
        match reason {
            PairInvalidationReason::DeadlineExpired => {
                self.metrics.deadline_expired_total =
                    self.metrics.deadline_expired_total.saturating_add(1)
            }
            PairInvalidationReason::FailedRollback => {
                self.metrics.rollback_failed_total =
                    self.metrics.rollback_failed_total.saturating_add(1)
            }
            PairInvalidationReason::Confounded => {
                self.metrics.confounded_total = self.metrics.confounded_total.saturating_add(1)
            }
            _ => {}
        }
        // An arm that never reported produced no evidence about the rollout.
        // Recording an endpoint clears `issued`, so the deadline path is only
        // reachable for an arm nothing ever bound to: treating that as adverse
        // evidence let one evaporated opportunity destroy every banked one.
        // Pair-scoped invalidation retires its own pair; the global reasons
        // (safety gate, clock regression) reset the rollout on their own paths.
        let produced_evidence = record.first_endpoint.is_some() || record.second_endpoint.is_some();
        if self.rollout.phase != LabPhase::Shadow {
            if produced_evidence {
                self.rollout.canary_failures = self.rollout.canary_failures.saturating_add(1);
            } else {
                self.metrics.unbound_expiries_total =
                    self.metrics.unbound_expiries_total.saturating_add(1);
            }
        }
        push_bounded(
            &mut self.completed,
            CompletedPairSummary {
                id: pair_id,
                evidence: EvidenceClosure::Bronze,
                effect_micros: 0,
                effective: false,
                harmful: false,
                interrupted: true,
            },
            MAX_COMPLETED_PAIRS,
        );
        Ok(PairInvalidation {
            pair_id,
            reason,
            family: record.candidate.family,
            action_key: record.candidate.action_key,
            rollback_required,
        })
    }

    #[doc(hidden)]
    pub fn force_phase_for_test(&mut self, phase: LabPhase) {
        self.rollout.phase = phase;
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
            rollout: self.rollout,
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
            rollout: persisted.rollout,
            rollout_config: LabRolloutConfig::default(),
            last_work: LabWork::default(),
            phase_clock_armed: false,
            restored_progress: 0,
            progress_resets: 0,
            last_progress_reset_reason: "",
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
        // A restart never resumes Canary or Active: mutation authority must be
        // re-earned. The count of observed shadow opportunities is durable
        // evidence about the host and is carried forward, but the duration gate
        // re-arms from this boot because `now_millis` is monotonic-since-boot
        // and would otherwise be satisfied instantly.
        let shadow_eligible = lab.rollout.shadow_eligible;
        let _deliberately_carried_forward = lab.rollout.reset_to_shadow(0);
        lab.rollout.shadow_eligible = shadow_eligible;
        // Boot-scoped provenance: this is the value the operator should see
        // published before the new boot has run a cycle.
        lab.restored_progress = shadow_eligible;
        let disposition = if interrupted == 0 {
            RestoreDisposition::Restored
        } else {
            RestoreDisposition::RestoredInterrupted
        };
        (lab, disposition)
    }

    pub fn metrics(&self) -> LabMetrics {
        let (effect_sum, effect_count) = self
            .completed
            .iter()
            .filter(|summary| summary.evidence == EvidenceClosure::PairGold && !summary.interrupted)
            .fold((0_i128, 0_u64), |(sum, count), summary| {
                (
                    sum.saturating_add(i128::from(summary.effect_micros)),
                    count.saturating_add(1),
                )
            });
        let mean_effect = if effect_count == 0 {
            0.0
        } else {
            ((effect_sum as f64 / effect_count as f64) / 1_000_000.0).clamp(-1.0, 1.0)
        };
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
            invalidated_total: self.metrics.invalidated_total,
            deadline_expired_total: self.metrics.deadline_expired_total,
            unbound_expiries_total: self.metrics.unbound_expiries_total,
            rollback_failed_total: self.metrics.rollback_failed_total,
            open_pairs: self.open.len(),
            completed_pairs: self.completed.len(),
            mean_effect,
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

    fn invalidate_all(&mut self, reason: PairInvalidationReason) -> Vec<PairInvalidation> {
        let ids: Vec<_> = self
            .open
            .iter()
            .map(|record| record.assignment.id)
            .collect();
        ids.into_iter()
            .filter_map(|id| self.invalidate_pair(id, reason).ok())
            .collect()
    }

    fn advance_rollout(&mut self, now_millis: u64, gates_enabled: bool) {
        match self.rollout.phase {
            LabPhase::Shadow
                if gates_enabled
                    && self.rollout.shadow_eligible
                        >= self.rollout_config.shadow_min_opportunities
                    && now_millis.saturating_sub(self.rollout.phase_started_millis)
                        >= self.rollout_config.shadow_min_duration_ms =>
            {
                self.rollout.phase = LabPhase::Canary;
                self.rollout.phase_started_millis = now_millis;
                self.rollout.canary_eligible = 0;
                self.rollout.canary_opened = 0;
                self.rollout.canary_completed = 0;
                self.rollout.canary_gold = 0;
                self.rollout.canary_failures = 0;
            }
            LabPhase::Canary
                if gates_enabled
                    && self.rollout.canary_eligible
                        >= self.rollout_config.canary_min_opportunities
                    && self.rollout.canary_opened > 0
                    && self.rollout.canary_completed == self.rollout.canary_opened
                    && self.rollout.canary_gold == self.rollout.canary_opened
                    && self.rollout.canary_failures == 0
                    && self.open.is_empty() =>
            {
                self.rollout.phase = LabPhase::Active;
                self.rollout.phase_started_millis = now_millis;
            }
            _ => {}
        }
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

fn canary_admitted(candidate: &PairCandidate, percent: u8) -> bool {
    let bucket = candidate.sequence.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ candidate.stratum_hash.rotate_left(17);
    bucket % 100 < u64::from(percent.min(100))
}

fn endpoint_invalidation_reason(
    candidate: &PairCandidate,
    endpoint: &PairEndpoint,
    completed_cycle: u64,
    issued: IssuedArm,
) -> Option<PairInvalidationReason> {
    if completed_cycle > issued.expires_after_cycle {
        return Some(PairInvalidationReason::DeadlineExpired);
    }
    if endpoint.synthetic || !endpoint.observed_local || endpoint.decision_id == 0 {
        return Some(PairInvalidationReason::Unauthoritative);
    }
    match endpoint.horizon {
        HorizonClosure::Confounded => return Some(PairInvalidationReason::Confounded),
        HorizonClosure::Incomplete => return Some(PairInvalidationReason::IncompleteHorizon),
        HorizonClosure::Expired => return Some(PairInvalidationReason::DeadlineExpired),
        HorizonClosure::Complete => {}
    }
    let execution_valid = match endpoint.arm {
        ArmKind::Control => endpoint.execution == ExecutionClosure::NoOp,
        ArmKind::Treatment => endpoint.execution == ExecutionClosure::Applied,
    };
    if !execution_valid {
        return Some(PairInvalidationReason::FailedExecution);
    }
    let rollback_valid = match (candidate.family, endpoint.arm) {
        (ActuatorFamily::MarkovPrewarm, _) => matches!(
            endpoint.rollback,
            RollbackClosure::NotRequiredNonKernel | RollbackClosure::Succeeded
        ),
        (_, ArmKind::Control) => matches!(
            endpoint.rollback,
            RollbackClosure::NotRequiredNonKernel | RollbackClosure::Succeeded
        ),
        (_, ArmKind::Treatment) => endpoint.rollback == RollbackClosure::Succeeded,
    };
    (!rollback_valid).then_some(PairInvalidationReason::FailedRollback)
}

fn rollback_still_required(record: &PairRecord) -> bool {
    if record.candidate.family == ActuatorFamily::MarkovPrewarm {
        return false;
    }
    if record
        .issued
        .is_some_and(|issued| issued.arm == ArmKind::Treatment)
    {
        return true;
    }
    record
        .first_endpoint
        .iter()
        .chain(record.second_endpoint.iter())
        .any(|endpoint| {
            endpoint.arm == ArmKind::Treatment
                && endpoint.execution == ExecutionClosure::Applied
                && endpoint.rollback != RollbackClosure::Succeeded
        })
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
    let control_execution = control.execution == ExecutionClosure::NoOp;
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
            && record.issued.is_none_or(|issued| {
                issued.issued_cycle > 0
                    && issued.complete_not_before_cycle
                        >= issued
                            .issued_cycle
                            .saturating_add(u64::from(record.candidate.horizon_cycles))
                    && issued.expires_after_cycle >= issued.complete_not_before_cycle
                    && match record.progress {
                        PersistedPairProgress::AwaitingFirst => {
                            issued.arm == record.assignment.first
                        }
                        PersistedPairProgress::AwaitingComplement => {
                            issued.arm == record.assignment.second
                        }
                        PersistedPairProgress::Washout | PersistedPairProgress::ReadyToClose => {
                            false
                        }
                    }
            })
            && (record.washout_started_cycle.is_none()
                || record.progress == PersistedPairProgress::Washout)
    })
}

#[cfg(test)]
mod rollout_tests {
    use super::*;
    use crate::engine::exploration_scheduler::HardwareIdentity;

    fn origin() -> ExplorationOrigin {
        ExplorationOrigin {
            installation_id: 0xA110,
            hardware: HardwareIdentity {
                p_core_count: 4,
                e_core_count: 6,
                ram_gib: 16,
            },
        }
    }

    fn candidate(sequence: u64) -> PairCandidate {
        PairCandidate {
            sequence,
            origin: origin(),
            family: ActuatorFamily::InteractionQos,
            action_class: ActionClass::InteractionForeground,
            treatment_arm: ExplorationArm::InteractionQosStandard,
            context: ExplorationContext::Interactive,
            action_key: "interaction_qos:foreground@standard".to_string(),
            stratum_hash: sequence.saturating_add(1),
            horizon_cycles: 4,
            washout_cycles: 2,
            minimum_effect_micros: 500,
        }
    }

    fn endpoint(
        candidate: &PairCandidate,
        directive: &PairDirective,
        utility_micros: i64,
    ) -> TimedPairEndpoint {
        TimedPairEndpoint {
            pair_id: directive.pair_id,
            issued_cycle: directive.issued_cycle,
            completed_cycle: directive.complete_not_before_cycle,
            endpoint: PairEndpoint {
                arm: directive.arm,
                origin: candidate.origin,
                family: candidate.family,
                action_class: candidate.action_class,
                context: candidate.context,
                action_key: candidate.action_key.clone(),
                stratum_hash: candidate.stratum_hash,
                horizon_cycles: candidate.horizon_cycles,
                decision_id: directive.issued_cycle.saturating_add(1),
                observed_local: true,
                synthetic: false,
                execution: match directive.arm {
                    ArmKind::Control => ExecutionClosure::NoOp,
                    ArmKind::Treatment => ExecutionClosure::Applied,
                },
                horizon: HorizonClosure::Complete,
                rollback: RollbackClosure::Succeeded,
                utility_micros,
            },
        }
    }

    fn active_lab() -> MicroexperimentLab {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.rollout.phase = LabPhase::Active;
        lab
    }

    #[test]
    fn complete_endpoint_before_real_horizon_is_rejected() {
        let mut lab = active_lab();
        let candidate = candidate(1);
        let assignment = lab
            .propose(candidate.clone(), PairGates::healthy_enabled())
            .unwrap();
        let directive = lab
            .issue_ready_arms(10, PairGates::healthy_enabled())
            .pop()
            .unwrap();
        let mut observation = endpoint(&candidate, &directive, 1_000);
        observation.completed_cycle = directive.complete_not_before_cycle - 1;

        assert_eq!(
            lab.record_timed_endpoint(observation),
            Err(LabError::HorizonPending)
        );
        assert_eq!(lab.metrics().pair_gold_total, 0);
        assert_eq!(lab.metrics().open_pairs, 1);
        assert_eq!(assignment.id, directive.pair_id);
    }

    #[test]
    fn unsafe_cycle_invalidates_inflight_treatment_and_requests_rollback() {
        let mut lab = active_lab();
        lab.propose(candidate(1), PairGates::healthy_enabled())
            .unwrap();
        let treatment_pair = lab
            .propose(candidate(2), PairGates::healthy_enabled())
            .unwrap();
        let directives = lab.issue_ready_arms(20, PairGates::healthy_enabled());
        assert_eq!(
            directives
                .iter()
                .find(|directive| directive.pair_id == treatment_pair.id)
                .unwrap()
                .arm,
            ArmKind::Treatment
        );

        let invalidations = lab.advance_cycle(21, 1_000, PairGates::default());
        let treatment = invalidations
            .iter()
            .find(|record| record.pair_id == treatment_pair.id)
            .unwrap();
        assert_eq!(treatment.reason, PairInvalidationReason::SafetyGate);
        assert!(treatment.rollback_required);
        assert_eq!(lab.metrics().pair_gold_total, 0);
        assert_eq!(lab.phase(), LabPhase::Shadow);
    }

    #[test]
    fn two_real_horizons_and_washout_close_exactly_one_gold_pair() {
        let mut lab = active_lab();
        let candidate = candidate(1);
        let assignment = lab
            .propose(candidate.clone(), PairGates::healthy_enabled())
            .unwrap();
        let first = lab
            .issue_ready_arms(10, PairGates::healthy_enabled())
            .pop()
            .unwrap();
        let first_utility = if first.arm == ArmKind::Control {
            1_000
        } else {
            2_000
        };
        assert_eq!(
            lab.record_timed_endpoint(endpoint(&candidate, &first, first_utility))
                .unwrap(),
            TimedEndpointDisposition::Progress(PairProgress::Washout)
        );

        lab.advance_cycle(
            first.complete_not_before_cycle + u64::from(candidate.washout_cycles),
            2_000,
            PairGates::healthy_enabled(),
        );
        let second = lab
            .issue_ready_arms(
                first.complete_not_before_cycle + u64::from(candidate.washout_cycles),
                PairGates::healthy_enabled(),
            )
            .pop()
            .unwrap();
        let second_utility = if second.arm == ArmKind::Control {
            1_000
        } else {
            2_000
        };
        assert_eq!(
            lab.record_timed_endpoint(endpoint(&candidate, &second, second_utility))
                .unwrap(),
            TimedEndpointDisposition::Progress(PairProgress::ReadyToClose)
        );
        let closure = lab.close_pair(assignment.id).unwrap();

        assert_eq!(closure.evidence, EvidenceClosure::PairGold);
        assert_eq!(lab.drain_pair_gold().len(), 1);
        assert_eq!(lab.metrics().pair_gold_total, 1);
    }

    #[test]
    fn confounded_real_observation_invalidates_without_gold() {
        let mut lab = active_lab();
        let candidate = candidate(1);
        lab.propose(candidate.clone(), PairGates::healthy_enabled())
            .unwrap();
        let directive = lab
            .issue_ready_arms(10, PairGates::healthy_enabled())
            .pop()
            .unwrap();
        let mut observation = endpoint(&candidate, &directive, 1_000);
        observation.endpoint.horizon = HorizonClosure::Confounded;

        let disposition = lab.record_timed_endpoint(observation).unwrap();
        assert!(matches!(
            disposition,
            TimedEndpointDisposition::Invalidated(PairInvalidation {
                reason: PairInvalidationReason::Confounded,
                ..
            })
        ));
        assert_eq!(lab.metrics().pair_gold_total, 0);
        assert_eq!(lab.metrics().open_pairs, 0);
    }

    /// Drives the lab to Canary with `earned` shadow opportunities banked, the
    /// shape production reaches before it issues its first arm.
    fn lab_in_canary_with_progress(earned: u64) -> MicroexperimentLab {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.rollout_config = LabRolloutConfig {
            shadow_min_opportunities: earned,
            shadow_min_duration_ms: 10,
            canary_percent: 100,
            canary_min_opportunities: 500,
        };
        for index in 0..earned {
            let _ = lab.consider_candidate(
                candidate(index + 1),
                PairGates::healthy_enabled(),
                index * 2,
            );
        }
        assert_eq!(lab.phase(), LabPhase::Canary, "setup must reach canary");
        lab
    }

    #[test]
    fn an_arm_that_expires_without_ever_reporting_does_not_erase_banked_progress() {
        // Reproduces production: progress stood at 460, one arm was issued,
        // nothing ever bound to it, its deadline passed, and the whole rollout
        // fell back to shadow with the progress destroyed.
        let mut lab = lab_in_canary_with_progress(460);
        // Banked shadow evidence. `rollout_progress` reports the counter of the
        // current phase, so in Canary it does not surface this value.
        let banked = 460;

        let candidate = candidate(9_001);
        let assignment = match lab
            .consider_candidate(candidate.clone(), PairGates::healthy_enabled(), 1_000)
            .unwrap()
        {
            CandidateDisposition::Opened(assignment) => assignment,
            other => panic!("expected an opened canary pair, got {other:?}"),
        };
        let directive = lab
            .issue_ready_arms(1_001, PairGates::healthy_enabled())
            .pop()
            .expect("canary must issue an arm");
        // Deliberately record no endpoint: this is the unbound case.
        let expiry = directive.expires_after_cycle;
        lab.advance_cycle(expiry + 1, 2_000, PairGates::healthy_enabled());

        let _ = assignment;
        assert_eq!(
            lab.metrics().open_pairs,
            0,
            "the expired pair itself must be retired"
        );
        assert_eq!(
            lab.phase(),
            LabPhase::Canary,
            "one arm that never reported is not evidence that the rollout is unsafe"
        );
        let (_, resets, reason) = lab.rollout_provenance();
        assert_eq!(
            resets, 0,
            "no global reset is warranted; got reason {reason}"
        );
        // `rollout_progress` reports the counter of the *current* phase, so the
        // banked shadow evidence is read back through a restart.
        let (restored, _) = MicroexperimentLab::restore(lab.persisted(), origin());
        assert_eq!(
            restored.rollout_progress().0,
            banked,
            "independent banked progress must survive an unbound expiry"
        );
    }

    #[test]
    fn an_arm_that_reported_can_no_longer_reach_the_deadline_path() {
        // Pins why the unbound case is the only one reachable: recording an
        // endpoint clears `issued`, and the expiry sweep only looks at issued
        // arms. Any future change that lets a reported pair expire would make
        // the reclassification above unsafe, and this test would fail first.
        let mut lab = lab_in_canary_with_progress(120);
        let candidate = candidate(9_100);
        match lab
            .consider_candidate(candidate.clone(), PairGates::healthy_enabled(), 1_000)
            .unwrap()
        {
            CandidateDisposition::Opened(_) => {}
            other => panic!("expected opened, got {other:?}"),
        }
        let directive = lab
            .issue_ready_arms(1_001, PairGates::healthy_enabled())
            .pop()
            .expect("canary must issue an arm");
        lab.record_timed_endpoint(endpoint(&candidate, &directive, 1_000))
            .expect("endpoint recorded");
        lab.advance_cycle(
            directive.expires_after_cycle + 1,
            2_000,
            PairGates::healthy_enabled(),
        );

        assert_eq!(
            lab.metrics().deadline_expired_total,
            0,
            "a pair that reported must not be swept by the deadline path"
        );
        assert_eq!(lab.metrics().unbound_expiries_total, 0);
        assert_eq!(
            lab.metrics().open_pairs,
            1,
            "it stays open, awaiting closure"
        );
    }

    #[test]
    fn a_safety_gate_still_resets_the_whole_rollout() {
        let mut lab = lab_in_canary_with_progress(200);
        assert_eq!(lab.phase(), LabPhase::Canary);
        let unsafe_gates = PairGates {
            secure_input: true,
            ..PairGates::healthy_enabled()
        };
        lab.advance_cycle(1_000, 3_000, unsafe_gates);
        assert_eq!(
            lab.phase(),
            LabPhase::Shadow,
            "an explicit global condition must still reset"
        );
        let (_, resets, reason) = lab.rollout_provenance();
        assert_eq!(resets, 1);
        assert_eq!(reason, "safety-gate");
    }

    #[test]
    fn progress_is_monotonic_while_no_global_invalidation_occurs() {
        let mut lab = lab_in_canary_with_progress(300);
        let mut last = lab.rollout_progress().0;
        for round in 0..3u64 {
            let candidate = candidate(9_200 + round);
            if let Ok(CandidateDisposition::Opened(_)) = lab.consider_candidate(
                candidate,
                PairGates::healthy_enabled(),
                3_000 + round * 1_000,
            ) {
                if let Some(directive) = lab
                    .issue_ready_arms(1_100 + round, PairGates::healthy_enabled())
                    .pop()
                {
                    lab.advance_cycle(
                        directive.expires_after_cycle + 1,
                        3_500 + round * 1_000,
                        PairGates::healthy_enabled(),
                    );
                }
            }
            let now = lab.rollout_progress().0;
            assert!(
                now >= last,
                "round {round}: progress went backwards {last} -> {now}"
            );
            last = now;
        }
        let (_, resets, reason) = lab.rollout_provenance();
        assert_eq!(resets, 0, "unexpected reset, reason={reason}");
    }

    #[test]
    fn an_unbound_expiry_survives_the_persistence_round_trip() {
        let mut lab = lab_in_canary_with_progress(460);
        let candidate = candidate(9_300);
        let _ = lab.consider_candidate(candidate, PairGates::healthy_enabled(), 1_000);
        if let Some(directive) = lab
            .issue_ready_arms(1_001, PairGates::healthy_enabled())
            .pop()
        {
            lab.advance_cycle(
                directive.expires_after_cycle + 1,
                2_000,
                PairGates::healthy_enabled(),
            );
        }
        let (restored, _) = MicroexperimentLab::restore(lab.persisted(), origin());
        assert_eq!(
            restored.rollout_progress().0,
            460,
            "banked shadow evidence must survive restart after an unbound expiry"
        );
        assert_eq!(restored.phase(), LabPhase::Shadow, "authority is re-earned");
    }

    #[test]
    fn rollout_reaches_active_only_after_a_real_canary_pair_closes_gold() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.rollout_config = LabRolloutConfig {
            shadow_min_opportunities: 2,
            shadow_min_duration_ms: 10,
            canary_percent: 100,
            canary_min_opportunities: 1,
        };
        assert!(matches!(
            lab.consider_candidate(candidate(1), PairGates::healthy_enabled(), 0),
            Ok(CandidateDisposition::Shadow(_))
        ));
        assert!(matches!(
            lab.consider_candidate(candidate(2), PairGates::healthy_enabled(), 10),
            Ok(CandidateDisposition::Shadow(_))
        ));
        assert_eq!(lab.phase(), LabPhase::Canary);

        let candidate = candidate(3);
        let assignment = match lab
            .consider_candidate(candidate.clone(), PairGates::healthy_enabled(), 11)
            .unwrap()
        {
            CandidateDisposition::Opened(assignment) => assignment,
            other => panic!("expected opened canary pair, got {other:?}"),
        };
        let first = lab
            .issue_ready_arms(20, PairGates::healthy_enabled())
            .pop()
            .unwrap();
        let first_utility = if first.arm == ArmKind::Control {
            1_000
        } else {
            2_000
        };
        lab.record_timed_endpoint(endpoint(&candidate, &first, first_utility))
            .unwrap();
        let second_cycle = first
            .complete_not_before_cycle
            .saturating_add(u64::from(candidate.washout_cycles));
        lab.advance_cycle(second_cycle, 20, PairGates::healthy_enabled());
        let second = lab
            .issue_ready_arms(second_cycle, PairGates::healthy_enabled())
            .pop()
            .unwrap();
        let second_utility = if second.arm == ArmKind::Control {
            1_000
        } else {
            2_000
        };
        lab.record_timed_endpoint(endpoint(&candidate, &second, second_utility))
            .unwrap();
        lab.close_pair(assignment.id).unwrap();

        assert_eq!(lab.phase(), LabPhase::Active);
        assert_eq!(lab.metrics().pair_gold_total, 1);
    }

    #[test]
    fn a_restart_keeps_shadow_progress_but_re_earns_the_duration_gate() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.rollout_config = LabRolloutConfig {
            shadow_min_opportunities: 2,
            shadow_min_duration_ms: 10_000,
            canary_percent: 100,
            canary_min_opportunities: 1,
        };
        lab.consider_candidate(candidate(1), PairGates::healthy_enabled(), 1_000)
            .unwrap();
        lab.consider_candidate(candidate(2), PairGates::healthy_enabled(), 2_000)
            .unwrap();
        assert_eq!(lab.phase(), LabPhase::Shadow);

        let (mut restored, disposition) = MicroexperimentLab::restore(lab.persisted(), origin());
        restored.rollout_config = lab.rollout_config;
        assert_eq!(disposition, RestoreDisposition::Restored);
        assert_eq!(restored.phase(), LabPhase::Shadow);
        // The opportunity count survives; it is durable evidence about the host.
        assert_eq!(restored.rollout.shadow_eligible, 2);

        // The duration gate must NOT be instantly satisfiable after restore:
        // the phase clock re-arms on the first candidate of the new boot.
        restored
            .consider_candidate(candidate(3), PairGates::healthy_enabled(), 500_000)
            .unwrap();
        assert_eq!(
            restored.phase(),
            LabPhase::Shadow,
            "restore must not let a monotonic clock satisfy the duration gate instantly"
        );

        // Once the minimum duration genuinely elapses on this boot, it promotes.
        restored
            .consider_candidate(candidate(4), PairGates::healthy_enabled(), 511_000)
            .unwrap();
        assert_eq!(restored.phase(), LabPhase::Canary);
    }

    /// The in-memory restore path is covered above. Production does not use
    /// it: the struct is written into `learned_state.json` and read back, and
    /// the value the operator actually sees is `rollout_progress()`, published
    /// after several cycles of the new boot have already run. This pins the
    /// whole chain — persisted -> serialized -> restored -> published — at the
    /// production gate size, because a defect anywhere in it looks identical
    /// from the dashboard: a gate counter that fell backwards after a restart.
    #[test]
    fn shadow_progress_survives_the_json_round_trip_and_the_first_cycles() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        for sequence in 1..=413u64 {
            lab.consider_candidate(
                candidate(sequence),
                PairGates::healthy_enabled(),
                1_000 + sequence,
            )
            .unwrap();
        }
        assert_eq!(lab.rollout_progress(), (413, SHADOW_MIN_OPPORTUNITIES));

        let encoded = serde_json::to_string(&lab.persisted()).expect("encode");
        let decoded: MicroexperimentLabPersisted = serde_json::from_str(&encoded).expect("decode");
        let (mut restored, disposition) = MicroexperimentLab::restore(decoded, origin());

        assert_eq!(disposition, RestoreDisposition::Restored);
        assert_eq!(
            restored.rollout_progress(),
            (413, SHADOW_MIN_OPPORTUNITIES),
            "the gate counter must survive the disk round trip"
        );

        // The new boot's clock restarts near zero while the persisted rollout
        // carries millisecond stamps from the previous boot. The early cycles
        // are exactly where a monotonic-regression reset would fire.
        for cycle in 1..=5u64 {
            restored.advance_cycle(cycle, cycle * 300, PairGates::healthy_enabled());
        }
        assert_eq!(
            restored.rollout_progress().0,
            413,
            "the first cycles of the new boot must not reset the gate counter"
        );
    }

    #[test]
    fn rollout_progress_reports_the_gate_counter_not_the_cumulative_total() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.rollout_config = LabRolloutConfig {
            shadow_min_opportunities: 4,
            shadow_min_duration_ms: 10_000,
            canary_percent: 100,
            canary_min_opportunities: 1,
        };
        assert_eq!(lab.rollout_progress(), (0, 4));

        lab.consider_candidate(candidate(1), PairGates::healthy_enabled(), 1_000)
            .unwrap();
        lab.consider_candidate(candidate(2), PairGates::healthy_enabled(), 2_000)
            .unwrap();

        assert_eq!(lab.rollout_progress(), (2, 4));
        // The cumulative metric counts the same events but is not the gate.
        assert_eq!(lab.metrics().eligible_total, 2);

        // After a restart the cumulative total keeps climbing from persisted
        // state while the gate counter is what actually governs promotion.
        let (mut restored, _) = MicroexperimentLab::restore(lab.persisted(), origin());
        restored.rollout_config = lab.rollout_config;
        assert_eq!(restored.rollout_progress(), (2, 4));
        assert_eq!(restored.metrics().eligible_total, 2);
    }

    #[test]
    fn a_restart_never_resumes_mutation_authority() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.rollout_config = LabRolloutConfig {
            shadow_min_opportunities: 1,
            shadow_min_duration_ms: 0,
            canary_percent: 100,
            canary_min_opportunities: 1,
        };
        lab.consider_candidate(candidate(1), PairGates::healthy_enabled(), 10)
            .unwrap();
        assert_eq!(lab.phase(), LabPhase::Canary);

        let (mut restored, _) = MicroexperimentLab::restore(lab.persisted(), origin());

        assert_eq!(restored.phase(), LabPhase::Shadow);
        assert!(restored
            .issue_ready_arms(1, PairGates::healthy_enabled())
            .is_empty());
    }

    #[test]
    fn legacy_checkpoint_without_rollout_fields_restores_in_shadow() {
        let persisted = MicroexperimentLab::cold_start(origin()).persisted();
        let mut value = serde_json::to_value(persisted).unwrap();
        value.as_object_mut().unwrap().remove("rollout");
        let legacy: MicroexperimentLabPersisted = serde_json::from_value(value).unwrap();
        let (restored, disposition) = MicroexperimentLab::restore(legacy, origin());

        assert_eq!(disposition, RestoreDisposition::Restored);
        assert_eq!(restored.phase(), LabPhase::Shadow);
    }
}
