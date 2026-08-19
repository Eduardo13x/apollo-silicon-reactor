//! Pure, bounded pairing and evidence closure for safe local experiments.
//!
//! The lab never executes an effect and never grants actuator authority. It
//! accepts only the closed exploration catalog, joins one local control with
//! one local treatment, and emits one deduplicated Pair Gold record after all
//! execution, horizon, and rollback facts close independently.

use crate::engine::exploration_pair::{CompletedExplorationPair, ExperimentId};
use std::collections::{BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::engine::exploration_scheduler::{
    ActionClass, ExplorationArm, ExplorationContext, ExplorationOrigin,
};
use crate::engine::telemetry_medallion::ActuatorFamily;

pub const MAX_OPEN_PAIRS: usize = 32;
/// Terminalised shadow pair ids kept so a late endpoint can be named as late.
pub const MAX_SHADOW_TERMINAL_MEMORY: usize = 64;
/// Experiment ids remembered as already counted. Bounded, and large enough
/// that a replay cannot outlive the memory before the gate is satisfied.
pub const MAX_CONSUMED_EXPERIMENTS: usize = 512;
/// Shadow arms issued per cycle.
///
/// The emit→expire→emit loop is already rate-limited upstream: a shadow pair
/// issues its arm exactly once and is reaped on expiry, so new arms need new
/// candidates, and the tick produces at most two per cycle. This cap is the
/// belt to that braces — it makes the bound a property of this module rather
/// than of a caller that could change.
pub const MAX_SHADOW_ARMS_PER_CYCLE: usize = 4;
pub const MAX_COMPLETED_PAIRS: usize = 128;
pub const MAX_GOLD_DEDUP: usize = 128;
pub const MAX_SERIALIZED_BYTES: usize = 64 * 1024;
pub const MAX_ACTION_KEY_BYTES: usize = 96;
pub const SHADOW_MIN_OPPORTUNITIES: u64 = 500;
/// Valid Shadow measurements required before Canary. This is *evidence*, not
/// exposure: each unit is a control endpoint that was actually observed,
/// bound, non-synthetic, and closed at a complete horizon. Seeing candidates
/// or waiting out a duration cannot substitute for it.
///
/// Far smaller than `SHADOW_MIN_OPPORTUNITIES` on purpose. The question Shadow
/// answers is binary — "can this daemon measure at all?" — and a handful of
/// clean closures answers it. The old 500 was large precisely because seeing a
/// candidate proved nothing, so the gate leaned on volume instead.
pub const SHADOW_MIN_MEASUREMENTS: u64 = 8;
pub const SHADOW_MIN_DURATION_MS: u64 = 15 * 60 * 1_000;
pub const CANARY_MIN_OPPORTUNITIES: u64 = 500;
pub const CANARY_PERCENT: u8 = 10;
pub const ENDPOINT_GRACE_CYCLES: u64 = 12;
/// Bumped to 2 on 2026-08-19. Version 1 state contains pair counters produced
/// by a micro-canary that opened both arms' utility windows on a single cycle.
/// The medallion scores a system-wide objective, so those two windows shared a
/// `before` and an `after`: every such pair had zero effect by construction,
/// and its treatment arm described an action that never ran. Four of them
/// reached `lab_pairs_accepted` in production. They are not measurements, so
/// the counter that would carry them forward is discarded rather than
/// explained in a footnote nobody will read next to the number.
const LAB_SCHEMA_VERSION: u32 = 2;

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
    shadow_min_measurements: u64,
    shadow_min_duration_ms: u64,
    canary_percent: u8,
    canary_min_opportunities: u64,
}

impl Default for LabRolloutConfig {
    fn default() -> Self {
        Self {
            shadow_min_opportunities: SHADOW_MIN_OPPORTUNITIES,
            shadow_min_measurements: SHADOW_MIN_MEASUREMENTS,
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
    /// Registered only so an episode can bind to it. A Shadow directive is
    /// observe-only: it never reaches `outstanding_control_actions`, so it can
    /// never withhold an action the machine was going to take.
    pub observe_only: bool,
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
    /// A shadow pair consumed the endpoint. Carries no causal meaning: it only
    /// records that the measurement pipeline delivered something bindable.
    ShadowMeasured,
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

/// What the lab did with a certified pair. `Accepted` names the experiment it
/// came from: a gate observation that cannot say which experiment produced it
/// is a counter, not evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairConsumption {
    Accepted { experiment_id: ExperimentId },
    Duplicate,
    Rejected(PairRejection),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairRejection {
    /// A control was issued and never confirmed honoured, so some pair in this
    /// run has a control arm that is not a control.
    ControlNotHonoured,
    /// From a family the gate does not trust for causal evidence.
    UnexpectedFamily,
}

/// A pair the lab tracks **without acting**.
///
/// Deliberately a different type in a different collection from `PairRecord`.
/// `issue_ready_arms` iterates `self.open` only, so there is no code path from
/// a shadow pair to a real directive — the guarantee is structural, not a flag
/// someone can invert later. Its arms are issued separately and always carry
/// `observe_only: true`, so they cannot reach `outstanding_control_actions`
/// and cannot withhold anything either.
///
/// Only the control arm is ever tracked. A treatment arm would require the
/// treatment to have been applied, and applying is exactly what Shadow must
/// never do — which is also why a shadow pair can never yield `PairGold`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShadowPairRecord {
    candidate: PairCandidate,
    assignment: PairAssignment,
    #[serde(default)]
    issued: Option<IssuedArm>,
    #[serde(default)]
    control_endpoint: Option<PairEndpoint>,
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
    /// EVIDENCE. Shadow pairs closed on a valid observed control endpoint.
    pub shadow_measurements_proven_total: u64,
    /// Shadow closures refused for an endpoint that was not a clean observed
    /// non-synthetic control no-op.
    pub shadow_measurements_refused_total: u64,
    /// Certified pairs offered, and what became of each.
    pub lab_pairs_seen: u64,
    pub lab_pairs_accepted: u64,
    pub lab_pairs_duplicate: u64,
    pub lab_pairs_rejected: u64,
    /// Distinct causal experiments counted — what the gate reads. Markov only.
    pub causal_pairs_consumed: u64,
    /// Complete causal chains observed that grant no authority. Derived from
    /// the two counters above, not stored.
    pub protocol_pairs_validated: u64,
    /// Arms that passed their deadline (a detection).
    pub shadow_pairs_expired_total: u64,
    /// Shadow pairs actually removed from the collection (a removal).
    pub shadow_pairs_reaped_total: u64,
    /// Current size of the shadow collection. The quantity a baseline needs to
    /// assert `shadow_open < MAX_OPEN_PAIRS` directly instead of inferring it.
    pub shadow_open_pairs: usize,
    /// Largest size seen this boot.
    pub shadow_open_high_watermark: u32,
    /// Endpoints that arrived after their pair was terminalised.
    pub shadow_endpoints_late_total: u64,
    pub invalidated_total: u64,
    pub deadline_expired_total: u64,
    /// Arms that expired without ever reporting, outside Shadow. Counted apart
    /// from `canary_failures`: no data is not adverse data.
    pub unbound_expiries_total: u64,
    pub rollback_failed_total: u64,
    pub open_pairs: usize,
    /// Pairs no longer open, **whatever ended them** — a clean closure and an
    /// interrupted one both land here. Named for what it counts: production
    /// read 9 "completed" while all 9 were interrupted.
    pub terminal_pairs: usize,
    /// Pairs that reached a closure without being interrupted. This is the one
    /// to read when asking whether the lab finished anything.
    pub completed_pairs_valid: usize,
    /// Pairs that ended by interruption, invalidation or deadline.
    pub interrupted_pairs: usize,
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
    /// Shadow-phase bookkeeping. Never read by `issue_ready_arms`, so nothing
    /// here can become a real directive. See `ShadowPairRecord`.
    #[serde(default)]
    shadow_open: Vec<ShadowPairRecord>,
    /// Experiments already counted. **Persisted**: if this lived only in RAM, a
    /// crash between accepting a pair and the next write would let a replay
    /// count the same experiment twice, and the gate would advance on evidence
    /// that existed once.
    #[serde(default)]
    consumed_experiments: VecDeque<ExperimentId>,
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
    /// EVIDENCE. Lifetime count of shadow pairs closed on a valid control
    /// endpoint. Distinct from `shadow_would_open_total`, which is exposure.
    #[serde(default)]
    shadow_measurements_proven_total: u64,
    /// Shadow closures refused because the endpoint was not a clean, observed,
    /// non-synthetic control no-op. Published so a host that never proves a
    /// measurement can be told apart from one that never tried.
    #[serde(default)]
    shadow_measurements_refused_total: u64,
    #[serde(default)]
    lab_pairs_seen: u64,
    #[serde(default)]
    lab_pairs_accepted: u64,
    #[serde(default)]
    lab_pairs_duplicate: u64,
    #[serde(default)]
    lab_pairs_rejected: u64,
    #[serde(default)]
    causal_pairs_consumed: u64,
    /// Arms that passed their deadline. A *detection* event, distinct from the
    /// removal below: contrasting the two is how a pair that expires without
    /// being removed becomes visible instead of quietly holding a slot.
    #[serde(default)]
    shadow_pairs_expired_total: u64,
    /// Shadow pairs actually removed from the collection, whatever removed
    /// them — deadline, endpoint, or a restart discarding a stale arm.
    #[serde(default)]
    shadow_pairs_reaped_total: u64,
    /// Largest `shadow_open` ever observed this boot. A gauge read once a
    /// cycle can sit at zero through a burst that touched the ceiling; the
    /// high-water mark cannot.
    #[serde(default)]
    shadow_open_high_watermark: u32,
    /// Endpoints that arrived after their pair was terminalised. Named rather
    /// than silently refused, so a slow observation route is distinguishable
    /// from a broken one.
    #[serde(default)]
    shadow_endpoints_late_total: u64,
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
    /// EXPOSURE. Candidates seen in Shadow. Kept because it describes the host,
    /// but it no longer gates anything — seeing a candidate is not measuring.
    shadow_eligible: u64,
    /// EVIDENCE the gate actually reads: distinct certified causal pairs
    /// consumed. New field, so state written under the previous rule carries
    /// nothing into it.
    causal_pairs_consumed: u64,
    /// Retained, and currently unreachable. A Shadow measurement was a real
    /// form of evidence, but Shadow cannot obtain a control endpoint — that
    /// needs a deliberate holdout, and Shadow's contract is to change nothing.
    /// Kept rather than deleted because the shape is right and a future phase
    /// that *can* withhold could produce it.
    #[allow(dead_code)]
    ///
    /// A new field on purpose: state persisted under the old semantics carries
    /// no value for it and deserialises to 0, so the exposure a previous boot
    /// accumulated cannot be laundered into evidence by the upgrade itself.
    shadow_measurements_proven: u64,
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
        // The quantity that actually gated. It used to report `shadow_eligible`
        // — exposure — which after the gate moved to causal evidence would have
        // made a reset that discarded real evidence render as discarding
        // nothing.
        let dropped = self.causal_pairs_consumed;
        self.phase = LabPhase::Shadow;
        self.phase_started_millis = now_millis;
        self.last_monotonic_millis = now_millis;
        self.shadow_eligible = 0;
        self.shadow_measurements_proven = 0;
        self.causal_pairs_consumed = 0;
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
            shadow_open: Vec::new(),
            consumed_experiments: VecDeque::new(),
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
    /// Shadow-phase bookkeeping. Never read by `issue_ready_arms`, so nothing
    /// here can become a real directive. See `ShadowPairRecord`.
    shadow_open: Vec<ShadowPairRecord>,
    /// Experiments already counted, persisted so idempotency survives a crash.
    consumed_experiments: VecDeque<ExperimentId>,
    /// Ids of shadow pairs already terminalised. An endpoint arriving after its
    /// pair was reaped is late, not evidence: without this it would find no
    /// record, fall through as an unknown pair, and the reason it was refused
    /// would be indistinguishable from a genuine identity failure.
    ///
    /// Bounded and FIFO — remembering every id forever is how a leak comes back
    /// wearing a different hat.
    shadow_terminal: VecDeque<PairId>,
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
            shadow_open: Vec::new(),
            consumed_experiments: VecDeque::new(),
            shadow_terminal: VecDeque::new(),
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
        let assignment = PairAssignment {
            id,
            order,
            first,
            second,
        };
        // Track it so an episode can bind and prove the measurement pipeline.
        // Bounded by the same ceiling as real pairs: the point is to prove
        // capability, and a handful of tracked pairs proves it as well as a
        // thousand would.
        if self.shadow_open.len() < MAX_OPEN_PAIRS {
            self.shadow_open.push(ShadowPairRecord {
                candidate,
                assignment,
                issued: None,
                control_endpoint: None,
            });
            let len = self.shadow_open.len() as u32;
            if len > self.metrics.shadow_open_high_watermark {
                self.metrics.shadow_open_high_watermark = len;
            }
        }
        Ok(assignment)
    }

    /// Arms for shadow pairs. Always `observe_only`, so the adapter will bind
    /// episodes to them but never surface them as control withholds.
    ///
    /// Separate from `issue_ready_arms` on purpose: that function reads
    /// `self.open`, this one reads `self.shadow_open`, and neither can see the
    /// other's collection. A shadow pair therefore has no reachable path to a
    /// real directive.
    pub fn issue_shadow_arms(&mut self, cycle: u64, gates: PairGates) -> Vec<PairDirective> {
        if self.rollout.phase != LabPhase::Shadow || !gates.inherited_safe {
            return Vec::new();
        }
        let mut directives = Vec::with_capacity(MAX_SHADOW_ARMS_PER_CYCLE);
        for record in &mut self.shadow_open {
            if directives.len() >= MAX_SHADOW_ARMS_PER_CYCLE {
                break;
            }
            if record.issued.is_some() || record.control_endpoint.is_some() {
                continue;
            }
            let horizon = u64::from(record.candidate.horizon_cycles);
            let issued = IssuedArm {
                arm: ArmKind::Control,
                issued_cycle: cycle,
                complete_not_before_cycle: cycle.saturating_add(horizon),
                expires_after_cycle: cycle
                    .saturating_add(horizon)
                    .saturating_mul(2)
                    .max(cycle + 1),
            };
            record.issued = Some(issued);
            directives.push(PairDirective {
                observe_only: true,
                pair_id: record.assignment.id,
                arm: ArmKind::Control,
                treatment_arm: record.candidate.treatment_arm,
                family: record.candidate.family,
                action_class: record.candidate.action_class,
                context: record.candidate.context,
                action_key: record.candidate.action_key.clone(),
                stratum_hash: record.candidate.stratum_hash,
                issued_cycle: issued.issued_cycle,
                complete_not_before_cycle: issued.complete_not_before_cycle,
                expires_after_cycle: issued.expires_after_cycle,
                rollback_required: false,
            });
        }
        directives
    }

    /// Consume one certified pair. Read-only with respect to everything else:
    /// it touches no producer, no arm, no budget and no daemon decision.
    ///
    /// Idempotent on `experiment_id`, never on an arm or a timestamp. The same
    /// pair delivered once or a hundred times advances the gate exactly once,
    /// and the memory is persisted so a crash between accepting and writing
    /// cannot turn a replay into fresh evidence.
    pub fn consume_exploration_pair(
        &mut self,
        pair: &CompletedExplorationPair,
        control_issued: u64,
        control_honoured: u64,
    ) -> PairConsumption {
        self.metrics.lab_pairs_seen = self.metrics.lab_pairs_seen.saturating_add(1);

        if self.consumed_experiments.contains(&pair.experiment_id) {
            self.metrics.lab_pairs_duplicate = self.metrics.lab_pairs_duplicate.saturating_add(1);
            return PairConsumption::Duplicate;
        }
        // A control that was issued and not honoured means some pair in this
        // run has a control arm that is not a control. Which one is unknowable
        // from here, so no pair from that run is counted.
        if control_honoured < control_issued {
            self.metrics.lab_pairs_rejected = self.metrics.lab_pairs_rejected.saturating_add(1);
            return PairConsumption::Rejected(PairRejection::ControlNotHonoured);
        }
        // Two kinds of acceptance, and the difference is authority.
        //
        // A pair proves the experimental protocol works end to end whatever
        // family produced it — that is a statement about the machinery. It is
        // not a statement about the family the *gate* governs. `LabPhase` is
        // global, so letting any family's evidence advance it would hand every
        // other family mutation authority it never earned.
        //
        // So only MarkovPrewarm touches `causal_pairs_consumed`, the quantity
        // `advance_rollout` reads. Everything else is recorded, deduplicated
        // and counted, and moves no phase.
        // Boost is not an authorised family for the micro-canary, so a pair
        // claiming to be one did not come from the producer this lab trusts.
        // Refused rather than merely un-counted: an unexpected origin is a fact
        // worth surfacing, not silence.
        if !matches!(
            pair.family,
            ActuatorFamily::MarkovPrewarm | ActuatorFamily::InteractionQos
        ) {
            self.metrics.lab_pairs_rejected = self.metrics.lab_pairs_rejected.saturating_add(1);
            return PairConsumption::Rejected(PairRejection::UnexpectedFamily);
        }
        let governs_rollout = pair.family == ActuatorFamily::MarkovPrewarm;

        if self.consumed_experiments.len() >= MAX_CONSUMED_EXPERIMENTS {
            self.consumed_experiments.pop_front();
        }
        self.consumed_experiments.push_back(pair.experiment_id);
        self.metrics.lab_pairs_accepted = self.metrics.lab_pairs_accepted.saturating_add(1);
        if governs_rollout {
            self.metrics.causal_pairs_consumed =
                self.metrics.causal_pairs_consumed.saturating_add(1);
            self.rollout.causal_pairs_consumed =
                self.rollout.causal_pairs_consumed.saturating_add(1);
        }
        PairConsumption::Accepted {
            experiment_id: pair.experiment_id,
        }
    }

    /// Complete causal chains observed that grant no authority to anyone.
    ///
    /// **Derived, not stored.** Accepted pairs minus the ones that govern the
    /// rollout is exactly the set that proved the protocol without promoting a
    /// family. A separate persisted counter would be a second source of truth
    /// for a fact these two already contain.
    pub fn protocol_pairs_validated(&self) -> u64 {
        self.metrics
            .lab_pairs_accepted
            .saturating_sub(self.metrics.causal_pairs_consumed)
    }

    /// Terminalise one shadow pair: drop it and remember the id so a late
    /// endpoint can be named. Idempotent — reaping an id twice is a no-op, so
    /// expiry, endpoint arrival, a safety cancellation and shutdown can all
    /// call it without coordinating.
    fn terminalise_shadow_pair(&mut self, pair_id: PairId) -> bool {
        let existed = if let Some(index) = self
            .shadow_open
            .iter()
            .position(|record| record.assignment.id == pair_id)
        {
            self.shadow_open.swap_remove(index);
            true
        } else {
            false
        };
        if existed {
            self.metrics.shadow_pairs_reaped_total =
                self.metrics.shadow_pairs_reaped_total.saturating_add(1);
        }
        if !self.shadow_terminal.contains(&pair_id) {
            if self.shadow_terminal.len() >= MAX_SHADOW_TERMINAL_MEMORY {
                self.shadow_terminal.pop_front();
            }
            self.shadow_terminal.push_back(pair_id);
        }
        existed
    }

    /// Reap shadow pairs whose arm passed its deadline.
    ///
    /// Without this the collection filled with pairs that had already issued,
    /// `issue_shadow_arms` skipped every one of them, and the whole route went
    /// quiet — production reached exactly `MAX_OPEN_PAIRS` arms registered and
    /// the same number expired, then stopped forever.
    fn reap_expired_shadow_pairs(&mut self, cycle: u64) -> u64 {
        let expired: Vec<PairId> = self
            .shadow_open
            .iter()
            .filter(|record| {
                record
                    .issued
                    .is_some_and(|issued| cycle > issued.expires_after_cycle)
            })
            .map(|record| record.assignment.id)
            .collect();
        let mut reaped = 0_u64;
        for pair_id in expired {
            if self.terminalise_shadow_pair(pair_id) {
                reaped += 1;
                self.metrics.shadow_pairs_expired_total =
                    self.metrics.shadow_pairs_expired_total.saturating_add(1);
            }
        }
        reaped
    }

    /// Bind an observed endpoint to a shadow pair.
    ///
    /// Accepts only a control arm that genuinely ran as a no-op and completed
    /// its horizon locally. Nothing here is inferred or filled in: an endpoint
    /// that does not already say `NoOp`/`Complete`/observed/non-synthetic is
    /// refused rather than coerced, because the whole value of this evidence is
    /// that it was measured and not assumed.
    fn record_shadow_endpoint(&mut self, observation: &TimedPairEndpoint) -> bool {
        let Some(index) = self
            .shadow_open
            .iter()
            .position(|record| record.assignment.id == observation.pair_id)
        else {
            // Already terminalised. Name it late rather than letting it read as
            // an unknown pair, and grant nothing: the arm it answers no longer
            // exists, so the measurement it claims was never bounded by one.
            if self.shadow_terminal.contains(&observation.pair_id) {
                self.metrics.shadow_endpoints_late_total =
                    self.metrics.shadow_endpoints_late_total.saturating_add(1);
                return true;
            }
            return false;
        };
        let endpoint = &observation.endpoint;
        let valid = endpoint.arm == ArmKind::Control
            && endpoint.execution == ExecutionClosure::NoOp
            && endpoint.horizon == HorizonClosure::Complete
            && endpoint.observed_local
            && !endpoint.synthetic
            && endpoint_matches(&self.shadow_open[index].candidate, endpoint);
        let pair_id = self.shadow_open[index].assignment.id;
        let record = self.shadow_open.swap_remove(index);
        self.terminalise_shadow_pair(pair_id);
        if valid {
            self.rollout.shadow_measurements_proven =
                self.rollout.shadow_measurements_proven.saturating_add(1);
            self.metrics.shadow_measurements_proven_total = self
                .metrics
                .shadow_measurements_proven_total
                .saturating_add(1);
        } else {
            self.metrics.shadow_measurements_refused_total = self
                .metrics
                .shadow_measurements_refused_total
                .saturating_add(1);
        }
        let _ = record;
        true
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
                self.rollout.causal_pairs_consumed,
                self.rollout_config.shadow_min_measurements,
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
        // Shadow pairs expire on the same clock as real ones, and must be reaped
        // on the same tick. Leaving them behind is what silently closed the
        // route in production.
        self.reap_expired_shadow_pairs(cycle);
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
                // Real pair: this arm may legitimately request a control
                // withhold. Shadow arms are issued elsewhere with `true`.
                observe_only: false,
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
        let index = match self.find_open(observation.pair_id) {
            Some(index) => index,
            None => {
                // Not a real pair. It may still be a shadow pair proving the
                // measurement path; that route validates the endpoint itself
                // and never produces causal evidence.
                if self.record_shadow_endpoint(&observation) {
                    return Ok(TimedEndpointDisposition::ShadowMeasured);
                }
                return Err(LabError::UnknownPair);
            }
        };
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
            shadow_open: self.shadow_open.clone(),
            consumed_experiments: self.consumed_experiments.clone(),
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
            // Carried so a pair that had not yet issued can re-issue on this
            // boot. The ones that had are dropped below, with their arms.
            shadow_open: persisted.shadow_open,
            // Carried across the restart on purpose: this is the memory that
            // makes a replay after a crash a duplicate instead of new evidence.
            consumed_experiments: persisted.consumed_experiments,
            shadow_terminal: VecDeque::new(),
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
        // The adapter is new, so any arm registered before the restart is gone
        // with it. A pair still carrying `issued` can never be answered and
        // would occupy a slot forever; un-issued pairs are still legitimate and
        // simply re-issue on this boot.
        let stale_shadow = lab
            .shadow_open
            .iter()
            .filter(|record| record.issued.is_some())
            .count();
        lab.shadow_open.retain(|record| record.issued.is_none());
        // A restart discarding a stale arm is a removal, not a deadline. Folding
        // it into `expired` would break the `expired == reaped` contrast that
        // makes an un-reaped expiry visible.
        lab.metrics.shadow_pairs_reaped_total = lab
            .metrics
            .shadow_pairs_reaped_total
            .saturating_add(stale_shadow as u64);
        lab.shadow_terminal.clear();
        let shadow_eligible = lab.rollout.shadow_eligible;
        // Evidence earned is durable: a measurement the host actually produced
        // stays produced across a restart. Exposure is carried too, but only
        // because it describes the host — it no longer gates anything.
        let shadow_measurements_proven = lab.rollout.shadow_measurements_proven;
        let causal_pairs_consumed = lab.rollout.causal_pairs_consumed;
        let _deliberately_carried_forward = lab.rollout.reset_to_shadow(0);
        lab.rollout.shadow_eligible = shadow_eligible;
        lab.rollout.shadow_measurements_proven = shadow_measurements_proven;
        lab.rollout.causal_pairs_consumed = causal_pairs_consumed;
        // Boot-scoped provenance: the value the operator should see published
        // before the new boot has run a cycle. It must mirror what
        // `rollout_progress` reports, which is evidence — publishing restored
        // *exposure* beside an evidence gate reads as a counter that fell from
        // 304 to 0 across the restart, the exact confusion this whole change
        // set out to remove.
        lab.restored_progress = causal_pairs_consumed;
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
            shadow_measurements_proven_total: self.metrics.shadow_measurements_proven_total,
            shadow_measurements_refused_total: self.metrics.shadow_measurements_refused_total,
            lab_pairs_seen: self.metrics.lab_pairs_seen,
            lab_pairs_accepted: self.metrics.lab_pairs_accepted,
            lab_pairs_duplicate: self.metrics.lab_pairs_duplicate,
            lab_pairs_rejected: self.metrics.lab_pairs_rejected,
            causal_pairs_consumed: self.metrics.causal_pairs_consumed,
            protocol_pairs_validated: self.protocol_pairs_validated(),
            shadow_pairs_expired_total: self.metrics.shadow_pairs_expired_total,
            shadow_pairs_reaped_total: self.metrics.shadow_pairs_reaped_total,
            shadow_open_pairs: self.shadow_open.len(),
            shadow_open_high_watermark: self.metrics.shadow_open_high_watermark,
            shadow_endpoints_late_total: self.metrics.shadow_endpoints_late_total,
            invalidated_total: self.metrics.invalidated_total,
            deadline_expired_total: self.metrics.deadline_expired_total,
            unbound_expiries_total: self.metrics.unbound_expiries_total,
            rollback_failed_total: self.metrics.rollback_failed_total,
            open_pairs: self.open.len(),
            terminal_pairs: self.completed.len(),
            completed_pairs_valid: self
                .completed
                .iter()
                .filter(|summary| !summary.interrupted)
                .count(),
            interrupted_pairs: self
                .completed
                .iter()
                .filter(|summary| summary.interrupted)
                .count(),
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
            // Evidence, not exposure. `shadow_eligible` deliberately does not
            // appear here: a host that shows a million candidates and never
            // closes one measurement stays in Shadow forever, which is the
            // correct outcome. The duration remains as a floor — it can only
            // delay a graduation, never cause one.
            // The gate's input is a certified causal pair and nothing else. It
            // does not learn how the experiment was obtained, which is what
            // stops a phase from being asked for evidence it cannot produce —
            // the failure this whole line of work started from.
            LabPhase::Shadow
                if gates_enabled
                    && self.rollout.causal_pairs_consumed
                        >= self.rollout_config.shadow_min_measurements
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

    /// Drive one full Shadow measurement: candidate → tracked shadow pair →
    /// observe-only arm → observed control endpoint → proven.
    ///
    /// Nothing is applied and nothing is forged. The endpoint says `NoOp`
    /// because in Shadow the action genuinely did not run, which is exactly
    /// what makes this evidence honest rather than a stand-in for PairGold.
    fn prove_shadow_measurement(
        lab: &mut MicroexperimentLab,
        sequence: u64,
        cycle: u64,
        now_millis: u64,
    ) {
        let cand = candidate(sequence);
        lab.consider_candidate(cand.clone(), PairGates::healthy_enabled(), now_millis)
            .expect("shadow candidate accepted");
        // Arms are rate-limited per cycle, so the pair we just created may sit
        // behind others. Drain in bounded rounds until ours is issued.
        let mut directive = None;
        for round in 0..64 {
            let issued = lab.issue_shadow_arms(cycle + round, PairGates::healthy_enabled());
            // Candidates share an action_key; the stratum hash is what makes
            // this one distinct, so bind by that or another pair steals it.
            if let Some(found) = issued
                .into_iter()
                .find(|d| d.stratum_hash == cand.stratum_hash)
            {
                directive = Some(found);
                break;
            }
        }
        let directive = &directive.expect("shadow arm issued");
        assert!(directive.observe_only, "a shadow arm must be observe-only");
        let observation = endpoint(&cand, directive, 0);
        let disposition = lab
            .record_timed_endpoint(observation)
            .expect("shadow endpoint accepted");
        assert_eq!(disposition, TimedEndpointDisposition::ShadowMeasured);
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

    /// Advance the gate the only way it can now be advanced: by consuming
    /// certified causal pairs. Shadow measurements no longer count, because
    /// Shadow cannot obtain a control endpoint at all.
    fn consume_pairs(lab: &mut MicroexperimentLab, n: u64, seq_base: u64) {
        for i in 0..n {
            assert!(matches!(
                lab.consume_exploration_pair(&markov_pair(seq_base + i), 0, 0),
                PairConsumption::Accepted { .. }
            ));
        }
    }

    /// Drives the lab to Canary with `earned` **measurements** banked — the
    /// shape production reaches before it issues its first real arm.
    ///
    /// Previously this walked candidates, because exposure was what the gate
    /// read. It now has to actually measure, which is the whole point of the
    /// change: a test cannot reach Canary by looking at candidates either.
    fn lab_in_canary_with_progress(earned: u64) -> MicroexperimentLab {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.rollout_config = LabRolloutConfig {
            shadow_min_opportunities: earned,
            shadow_min_measurements: earned,
            shadow_min_duration_ms: 10,
            canary_percent: 100,
            canary_min_opportunities: 500,
        };
        consume_pairs(&mut lab, earned, 1);
        lab.advance_cycle(earned + 1, earned * 2, PairGates::healthy_enabled());
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
            shadow_min_measurements: 1,
            shadow_min_duration_ms: 10,
            canary_percent: 100,
            canary_min_opportunities: 1,
        };
        // Canary is now reached by a certified causal pair.
        consume_pairs(&mut lab, 1, 1);
        lab.advance_cycle(2, 11, PairGates::healthy_enabled());
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
            shadow_min_measurements: 1,
            shadow_min_duration_ms: 10_000,
            canary_percent: 100,
            canary_min_opportunities: 1,
        };
        consume_pairs(&mut lab, 1, 1);
        assert_eq!(lab.phase(), LabPhase::Shadow);

        let (mut restored, disposition) = MicroexperimentLab::restore(lab.persisted(), origin());
        restored.rollout_config = lab.rollout_config;
        assert_eq!(disposition, RestoreDisposition::Restored);
        assert_eq!(restored.phase(), LabPhase::Shadow);
        // Evidence the host actually produced is durable across the restart.
        assert_eq!(restored.rollout.causal_pairs_consumed, 1);

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

        // Once the minimum duration genuinely elapses on this boot, it promotes
        // — the evidence requirement was already met before the restart.
        restored.advance_cycle(9, 511_000, PairGates::healthy_enabled());
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
        // Earn the evidence first: `shadow_open` is bounded, so a pair created
        // after the flood would never be tracked and could never be measured.
        consume_pairs(&mut lab, 1, 900);
        for sequence in 1..=413u64 {
            lab.consider_candidate(
                candidate(sequence),
                PairGates::healthy_enabled(),
                1_000 + sequence,
            )
            .unwrap();
        }
        // Exposure accumulated; the gate did not move, because none of those
        // 413 candidates was ever measured.
        // One measurement earned, 413 candidates merely seen: the gate reads
        // the first number and ignores the second.
        assert_eq!(lab.rollout_progress(), (1, SHADOW_MIN_MEASUREMENTS));
        assert_eq!(lab.metrics().shadow_would_open_total, 413);

        let encoded = serde_json::to_string(&lab.persisted()).expect("encode");
        let decoded: MicroexperimentLabPersisted = serde_json::from_str(&encoded).expect("decode");
        let (mut restored, disposition) = MicroexperimentLab::restore(decoded, origin());

        assert_eq!(disposition, RestoreDisposition::Restored);
        assert_eq!(
            restored.rollout_progress(),
            (1, SHADOW_MIN_MEASUREMENTS),
            "earned evidence must survive the disk round trip"
        );

        // The new boot's clock restarts near zero while the persisted rollout
        // carries millisecond stamps from the previous boot. The early cycles
        // are exactly where a monotonic-regression reset would fire.
        for cycle in 1..=5u64 {
            restored.advance_cycle(cycle, cycle * 300, PairGates::healthy_enabled());
        }
        assert_eq!(
            restored.rollout_progress().0,
            1,
            "the first cycles of the new boot must not reset earned evidence"
        );
    }

    #[test]
    fn rollout_progress_reports_the_gate_counter_not_the_cumulative_total() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.rollout_config = LabRolloutConfig {
            shadow_min_opportunities: 4,
            shadow_min_measurements: 1,
            shadow_min_duration_ms: 10_000,
            canary_percent: 100,
            canary_min_opportunities: 1,
        };
        assert_eq!(lab.rollout_progress(), (0, 1));

        lab.consider_candidate(candidate(1), PairGates::healthy_enabled(), 1_000)
            .unwrap();
        lab.consider_candidate(candidate(2), PairGates::healthy_enabled(), 2_000)
            .unwrap();

        // Two candidates seen, nothing measured. Exposure is recorded but the
        // gate does not move: seeing a candidate is not measuring one.
        assert_eq!(lab.rollout_progress(), (0, 1));
        assert_eq!(lab.metrics().eligible_total, 2);
        assert_eq!(lab.metrics().shadow_would_open_total, 2);

        // A restart carries the same distinction: exposure survives, evidence
        // is still zero because none was ever earned.
        let (mut restored, _) = MicroexperimentLab::restore(lab.persisted(), origin());
        restored.rollout_config = lab.rollout_config;
        assert_eq!(restored.rollout_progress(), (0, 1));
        assert_eq!(restored.metrics().eligible_total, 2);
    }

    // ── Causal pair consumption ─────────────────────────────────────────────

    fn markov_pair(seq: u64) -> CompletedExplorationPair {
        use crate::engine::exploration_pair::{ArmTerminalState, ExplorationArmOutcome};
        use crate::engine::exploration_scheduler::ProbeCorrelation;
        let outcome = |c: u64, st, u| ExplorationArmOutcome {
            correlation_id: ProbeCorrelation(c),
            arm: ExplorationArm::MarkovCacheOnly,
            issued_cycle: 10,
            settled_cycle: 14,
            terminal_state: st,
            utility_micros: u,
        };
        CompletedExplorationPair::assemble(
            ExperimentId::new(7, seq).expect("id"),
            ActuatorFamily::MarkovPrewarm,
            ActionClass::MarkovPredictedApp,
            "markov_prewarm:predicted_app@cache_only".to_string(),
            3,
            7,
            outcome(seq * 2, ArmTerminalState::AppliedNoRevertNeeded, 100),
            outcome(seq * 2 + 1, ArmTerminalState::WithheldByHoldout, 40),
            20,
            20,
        )
        .expect("valid pair")
    }

    #[test]
    fn the_same_experiment_advances_the_gate_exactly_once() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        let pair = markov_pair(1);
        assert_eq!(
            lab.consume_exploration_pair(&pair, 1, 1),
            PairConsumption::Accepted {
                experiment_id: pair.experiment_id
            }
        );
        for _ in 0..100 {
            assert_eq!(
                lab.consume_exploration_pair(&pair, 1, 1),
                PairConsumption::Duplicate
            );
        }
        assert_eq!(
            lab.rollout_progress().0,
            1,
            "one hundred deliveries, one advance"
        );
        let m = lab.metrics();
        assert_eq!(m.lab_pairs_seen, 101);
        assert_eq!(m.lab_pairs_accepted, 1);
        assert_eq!(m.lab_pairs_duplicate, 100);
        assert_eq!(m.lab_pairs_rejected, 0);
    }

    #[test]
    fn an_accepted_pair_names_the_experiment_it_came_from() {
        // A gate observation with no provenance is a counter, not evidence.
        let mut lab = MicroexperimentLab::cold_start(origin());
        let pair = markov_pair(9);
        match lab.consume_exploration_pair(&pair, 0, 0) {
            PairConsumption::Accepted { experiment_id } => {
                assert_eq!(experiment_id, pair.experiment_id)
            }
            other => panic!("expected acceptance, got {other:?}"),
        }
    }

    #[test]
    fn a_control_that_was_never_honoured_blocks_the_gate() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        let pair = markov_pair(2);
        assert_eq!(
            lab.consume_exploration_pair(&pair, 3, 2),
            PairConsumption::Rejected(PairRejection::ControlNotHonoured)
        );
        assert_eq!(lab.rollout_progress().0, 0);
        assert_eq!(lab.metrics().lab_pairs_rejected, 1);
    }

    #[test]
    fn a_pair_from_an_untrusted_family_is_refused() {
        use crate::engine::exploration_pair::{ArmTerminalState, ExplorationArmOutcome};
        use crate::engine::exploration_scheduler::ProbeCorrelation;
        let mut lab = MicroexperimentLab::cold_start(origin());
        let outcome = |c: u64, st| ExplorationArmOutcome {
            correlation_id: ProbeCorrelation(c),
            arm: ExplorationArm::BoostOmission,
            issued_cycle: 10,
            settled_cycle: 14,
            terminal_state: st,
            utility_micros: 0,
        };
        let pair = CompletedExplorationPair::assemble(
            ExperimentId::new(7, 3).expect("id"),
            ActuatorFamily::Boost,
            ActionClass::BoostBackground,
            "boost:background@omission".to_string(),
            3,
            7,
            outcome(1, ArmTerminalState::AppliedAndReverted),
            outcome(2, ArmTerminalState::WithheldByHoldout),
            20,
            20,
        )
        .expect("assembles");
        assert_eq!(
            lab.consume_exploration_pair(&pair, 0, 0),
            PairConsumption::Rejected(PairRejection::UnexpectedFamily)
        );
        assert_eq!(lab.rollout_progress().0, 0);
    }

    #[test]
    fn a_crash_after_accepting_cannot_turn_a_replay_into_new_evidence() {
        // The ugliest boundary: pair accepted, process dies, the ledger replays
        // it on the next boot. If "already consumed" lived only in RAM, the
        // gate would advance twice on evidence that existed once.
        let mut lab = MicroexperimentLab::cold_start(origin());
        let pair = markov_pair(4);
        assert!(matches!(
            lab.consume_exploration_pair(&pair, 1, 1),
            PairConsumption::Accepted { .. }
        ));
        assert_eq!(lab.rollout_progress().0, 1);

        // Crash: only what reached disk survives.
        let encoded = serde_json::to_string(&lab.persisted()).expect("encode");
        let decoded: MicroexperimentLabPersisted = serde_json::from_str(&encoded).expect("decode");
        let (mut restarted, _) = MicroexperimentLab::restore(decoded, origin());

        assert_eq!(
            restarted.consume_exploration_pair(&pair, 1, 1),
            PairConsumption::Duplicate,
            "the replay must be recognised across the restart"
        );
        assert_eq!(
            restarted.rollout_progress().0,
            1,
            "and the gate must not move again"
        );
    }

    #[test]
    fn distinct_experiments_each_advance_the_gate_once() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        for seq in 1..=5 {
            assert!(matches!(
                lab.consume_exploration_pair(&markov_pair(seq), 0, 0),
                PairConsumption::Accepted { .. }
            ));
        }
        assert_eq!(lab.rollout_progress().0, 5);
        assert_eq!(lab.metrics().causal_pairs_consumed, 5);
    }

    fn qos_pair(seq: u64) -> CompletedExplorationPair {
        use crate::engine::exploration_pair::{ArmTerminalState, ExplorationArmOutcome};
        use crate::engine::exploration_scheduler::ProbeCorrelation;
        let outcome = |c: u64, st, u| ExplorationArmOutcome {
            correlation_id: ProbeCorrelation(c),
            arm: ExplorationArm::InteractionQosStandard,
            issued_cycle: 10,
            settled_cycle: 14,
            terminal_state: st,
            utility_micros: u,
        };
        CompletedExplorationPair::assemble(
            ExperimentId::new(7, 5_000 + seq).expect("id"),
            ActuatorFamily::InteractionQos,
            ActionClass::InteractionForeground,
            "interaction_qos:foreground@standard".to_string(),
            3,
            7,
            outcome(9_000 + seq * 2, ArmTerminalState::AppliedAndReverted, 100),
            outcome(9_001 + seq * 2, ArmTerminalState::WithheldByHoldout, 40),
            20,
            20,
        )
        .expect("valid pair")
    }

    #[test]
    fn eight_qos_pairs_cannot_promote_the_phase_or_let_markov_open_a_real_pair() {
        // The test that decides whether the bootstrap is safe at all. `LabPhase`
        // is global, so if QoS evidence could advance it, MarkovPrewarm would
        // gain the authority to open real pairs on evidence it never produced.
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.rollout_config = LabRolloutConfig {
            shadow_min_opportunities: 1,
            shadow_min_measurements: 8,
            shadow_min_duration_ms: 0,
            canary_percent: 100,
            canary_min_opportunities: 1,
        };

        for seq in 1..=8 {
            assert!(
                matches!(
                    lab.consume_exploration_pair(&qos_pair(seq), 0, 0),
                    PairConsumption::Accepted { .. }
                ),
                "a QoS pair is valid evidence of the protocol"
            );
        }

        // The protocol was demonstrated eight times.
        assert_eq!(lab.protocol_pairs_validated(), 8);
        assert_eq!(lab.metrics().lab_pairs_accepted, 8);
        // And the gate has not moved a single step.
        assert_eq!(
            lab.rollout_progress().0,
            0,
            "QoS evidence must not advance the Markov gate"
        );
        assert_eq!(lab.metrics().causal_pairs_consumed, 0);

        lab.advance_cycle(100, 100_000, PairGates::healthy_enabled());
        assert_eq!(
            lab.phase(),
            LabPhase::Shadow,
            "eight QoS pairs must leave the phase exactly where it was"
        );

        // And with the phase still Shadow, a Markov candidate cannot open a
        // real pair — the authority was never transferred.
        let disposition = lab
            .consider_candidate(candidate(1), PairGates::healthy_enabled(), 200_000)
            .expect("candidate considered");
        assert!(
            matches!(disposition, CandidateDisposition::Shadow(_)),
            "Markov must still be simulated, not opened: {disposition:?}"
        );
        assert!(lab.open.is_empty(), "no real pair was opened");
    }

    #[test]
    fn markov_evidence_still_moves_the_gate_and_qos_evidence_does_not() {
        // The two are recorded side by side and only one carries authority.
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.consume_exploration_pair(&qos_pair(1), 0, 0);
        assert_eq!(lab.rollout_progress().0, 0);
        assert_eq!(lab.protocol_pairs_validated(), 1);

        lab.consume_exploration_pair(&markov_pair(1), 0, 0);
        assert_eq!(lab.rollout_progress().0, 1, "Markov evidence gates");
        assert_eq!(
            lab.protocol_pairs_validated(),
            1,
            "and is not double counted as protocol-only validation"
        );
    }

    #[test]
    fn a_qos_pair_is_deduplicated_like_any_other() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        let pair = qos_pair(3);
        assert!(matches!(
            lab.consume_exploration_pair(&pair, 0, 0),
            PairConsumption::Accepted { .. }
        ));
        for _ in 0..20 {
            assert_eq!(
                lab.consume_exploration_pair(&pair, 0, 0),
                PairConsumption::Duplicate
            );
        }
        assert_eq!(lab.protocol_pairs_validated(), 1);
    }

    // ── Crash boundaries around the durable commit ──────────────────────────

    /// Simulate a crash: only what reached disk survives.
    fn crash_and_restart(snapshot: &MicroexperimentLabPersisted) -> MicroexperimentLab {
        let encoded = serde_json::to_string(snapshot).expect("encode");
        let decoded: MicroexperimentLabPersisted = serde_json::from_str(&encoded).expect("decode");
        MicroexperimentLab::restore(decoded, origin()).0
    }

    #[test]
    fn a_crash_before_the_durable_commit_loses_the_advance_with_the_dedupe() {
        // The dangerous asymmetry would be losing the dedupe while keeping the
        // gate advance. They live in one serialised object written by one
        // atomic rename, so a snapshot taken before the acceptance carries
        // neither, and the replay is legitimately new work.
        let mut lab = MicroexperimentLab::cold_start(origin());
        let before = lab.persisted();
        let pair = markov_pair(11);
        assert!(matches!(
            lab.consume_exploration_pair(&pair, 0, 0),
            PairConsumption::Accepted { .. }
        ));
        assert_eq!(lab.rollout_progress().0, 1);

        // Crash before that state was written: roll back to the older snapshot.
        let mut restarted = crash_and_restart(&before);
        assert_eq!(
            restarted.rollout_progress().0,
            0,
            "the advance was lost too"
        );
        assert!(
            matches!(
                restarted.consume_exploration_pair(&pair, 0, 0),
                PairConsumption::Accepted { .. }
            ),
            "with neither durable, the replay is new work and counts once"
        );
        assert_eq!(restarted.rollout_progress().0, 1);
    }

    #[test]
    fn a_crash_after_the_durable_commit_makes_the_replay_a_duplicate() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        let pair = markov_pair(12);
        assert!(matches!(
            lab.consume_exploration_pair(&pair, 0, 0),
            PairConsumption::Accepted { .. }
        ));
        let after = lab.persisted();
        let mut restarted = crash_and_restart(&after);
        assert_eq!(
            restarted.consume_exploration_pair(&pair, 0, 0),
            PairConsumption::Duplicate
        );
        assert_eq!(restarted.rollout_progress().0, 1, "exactly +1, still");
    }

    #[test]
    fn a_replayed_experiment_leaves_the_gate_at_exactly_plus_one() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        let pair = markov_pair(13);
        lab.consume_exploration_pair(&pair, 0, 0);
        let mut restarted = crash_and_restart(&lab.persisted());
        for _ in 0..50 {
            assert_eq!(
                restarted.consume_exploration_pair(&pair, 0, 0),
                PairConsumption::Duplicate
            );
        }
        assert_eq!(restarted.rollout_progress().0, 1);
        assert_eq!(restarted.metrics().lab_pairs_duplicate, 50);
    }

    #[test]
    fn no_snapshot_can_hold_a_gate_advance_without_its_dedupe_entry() {
        // The property the three boundaries above depend on, asserted directly
        // rather than left as an accident of which struct a field landed in.
        let mut lab = MicroexperimentLab::cold_start(origin());
        for seq in 1..=6 {
            lab.consume_exploration_pair(&markov_pair(seq), 0, 0);
            let snapshot = lab.persisted();
            let encoded = serde_json::to_string(&snapshot).expect("encode");
            let decoded: MicroexperimentLabPersisted =
                serde_json::from_str(&encoded).expect("decode");
            let (restored, _) = MicroexperimentLab::restore(decoded, origin());
            assert_eq!(
                restored.rollout_progress().0 as usize,
                restored.consumed_experiments.len(),
                "gate advances and dedupe entries must survive together"
            );
        }
    }

    // ── End to end, synthetic ───────────────────────────────────────────────

    #[test]
    fn producer_to_pair_to_lab_to_gate_end_to_end() {
        use crate::engine::exploration_pair::ArmTerminalState;
        use crate::engine::micro_canary::{ArmDecision, MicroCanary};

        let mut canary = MicroCanary::new(0x5EED);
        let mut lab = MicroexperimentLab::cold_start(origin());
        let mut produced = 0_u64;
        let mut cycle = 0_u64;

        while produced < 3 && cycle < 200_000 {
            cycle += 1;
            let decision = canary.offer(
                ActuatorFamily::MarkovPrewarm,
                ActionClass::MarkovPredictedApp,
                ExplorationArm::MarkovCacheOnly,
                "markov_prewarm:predicted_app@cache_only",
                3,
                cycle,
            );
            let ArmDecision::Proceed = decision else {
                // Sampled. Run both arms and honour a control if asked.
                let (id, corr, state) = match &decision {
                    ArmDecision::Treatment {
                        experiment_id,
                        correlation,
                    } => (
                        *experiment_id,
                        *correlation,
                        ArmTerminalState::AppliedNoRevertNeeded,
                    ),
                    ArmDecision::WithholdAsControl {
                        experiment_id,
                        correlation,
                    } => {
                        canary.confirm_control_honoured();
                        (
                            *experiment_id,
                            *correlation,
                            ArmTerminalState::WithheldByHoldout,
                        )
                    }
                    ArmDecision::Proceed => unreachable!(),
                };
                let utility = if matches!(state, ArmTerminalState::WithheldByHoldout) {
                    40
                } else {
                    100
                };
                assert!(canary
                    .record_arm(id, corr, cycle + 1, state, utility)
                    .is_none());

                let other = canary.complementary_arm(id).expect("complement");
                let (corr2, state2) = match &other {
                    ArmDecision::Treatment { correlation, .. } => {
                        (*correlation, ArmTerminalState::AppliedNoRevertNeeded)
                    }
                    ArmDecision::WithholdAsControl { correlation, .. } => {
                        canary.confirm_control_honoured();
                        (*correlation, ArmTerminalState::WithheldByHoldout)
                    }
                    ArmDecision::Proceed => unreachable!(),
                };
                let utility2 = if matches!(state2, ArmTerminalState::WithheldByHoldout) {
                    40
                } else {
                    100
                };
                let pair = canary
                    .record_arm(id, corr2, cycle + 2, state2, utility2)
                    .expect("the pair completes");
                assert_eq!(pair.effect_micros(), 60, "treatment 100 - control 40");

                let m = canary.metrics();
                assert_eq!(
                    m.control_issued, m.control_honoured,
                    "every withheld control was honoured"
                );
                assert!(matches!(
                    lab.consume_exploration_pair(&pair, m.control_issued, m.control_honoured),
                    PairConsumption::Accepted { .. }
                ));
                produced += 1;
                continue;
            };
            canary.expire(cycle + 1_000);
        }

        assert_eq!(produced, 3, "the producer reached three pairs");
        assert_eq!(
            lab.rollout_progress().0,
            3,
            "and the gate counted each once"
        );
        assert_eq!(lab.metrics().lab_pairs_accepted, 3);
        assert_eq!(lab.metrics().lab_pairs_rejected, 0);
        assert_eq!(canary.metrics().assembly_refused, 0);
    }

    #[test]
    fn a_run_with_an_unhonoured_control_moves_the_gate_not_at_all() {
        // The negative path end to end: a certified pair arrives, but the run
        // it came from left a control unhonoured, so no pair from that run can
        // be trusted and the gate stays where it was.
        let mut lab = MicroexperimentLab::cold_start(origin());
        let pair = markov_pair(21);
        assert_eq!(
            lab.consume_exploration_pair(&pair, 2, 1),
            PairConsumption::Rejected(PairRejection::ControlNotHonoured)
        );
        assert_eq!(lab.rollout_progress().0, 0);
        assert_eq!(lab.metrics().lab_pairs_rejected, 1);
        assert_eq!(lab.metrics().lab_pairs_accepted, 0);
        // And once the gap closes, the same pair is still allowed through.
        assert!(matches!(
            lab.consume_exploration_pair(&pair, 2, 2),
            PairConsumption::Accepted { .. }
        ));
        assert_eq!(lab.rollout_progress().0, 1);
    }

    // ── Shadow evidence invariants ──────────────────────────────────────────

    #[test]
    fn exposure_without_measurement_never_graduates() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        for seq in 0..600 {
            let _ = lab.consider_candidate(
                candidate(seq),
                PairGates::healthy_enabled(),
                (seq + 1) * 1_000,
            );
            lab.advance_cycle(seq, (seq + 1) * 1_000, PairGates::healthy_enabled());
        }
        assert_eq!(lab.phase(), LabPhase::Shadow);
        assert_eq!(lab.rollout_progress().0, 0);
        assert!(
            lab.metrics().shadow_would_open_total >= 500,
            "exposure did accumulate: {}",
            lab.metrics().shadow_would_open_total
        );
    }

    #[test]
    fn a_shadow_arm_is_never_a_control_withhold() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.consider_candidate(candidate(1), PairGates::healthy_enabled(), 1_000)
            .unwrap();
        let directives = lab.issue_shadow_arms(1, PairGates::healthy_enabled());
        assert!(!directives.is_empty(), "a shadow pair must issue its arm");
        assert!(
            directives.iter().all(|d| d.observe_only),
            "an observe-only arm cannot withhold an action the machine would take"
        );
    }

    #[test]
    fn issue_ready_arms_cannot_see_a_shadow_pair() {
        // The guarantee is structural: real directives come from `open`, shadow
        // bookkeeping lives in `shadow_open`, and no code path bridges them.
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.consider_candidate(candidate(1), PairGates::healthy_enabled(), 1_000)
            .unwrap();
        assert!(lab.open.is_empty(), "Shadow must never open a real pair");
        assert_eq!(lab.shadow_open.len(), 1);
        assert!(
            lab.issue_ready_arms(1, PairGates::healthy_enabled())
                .is_empty(),
            "no real directive may originate in Shadow"
        );
    }

    #[test]
    fn an_endpoint_that_was_applied_proves_nothing_in_shadow() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        let cand = candidate(1);
        lab.consider_candidate(cand.clone(), PairGates::healthy_enabled(), 1_000)
            .unwrap();
        let directives = lab.issue_shadow_arms(1, PairGates::healthy_enabled());
        let mut observation = endpoint(&cand, &directives[0], 0);
        // Claiming the treatment ran is exactly the fabrication this design
        // refuses: in Shadow nothing was applied.
        observation.endpoint.execution = ExecutionClosure::Applied;
        let _ = lab.record_timed_endpoint(observation);
        assert_eq!(lab.rollout_progress().0, 0);
        assert_eq!(lab.metrics().shadow_measurements_refused_total, 1);
    }

    #[test]
    fn a_synthetic_endpoint_proves_nothing_in_shadow() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        let cand = candidate(1);
        lab.consider_candidate(cand.clone(), PairGates::healthy_enabled(), 1_000)
            .unwrap();
        let directives = lab.issue_shadow_arms(1, PairGates::healthy_enabled());
        let mut observation = endpoint(&cand, &directives[0], 0);
        observation.endpoint.synthetic = true;
        let _ = lab.record_timed_endpoint(observation);
        assert_eq!(lab.rollout_progress().0, 0);
    }

    #[test]
    fn a_shadow_measurement_never_becomes_causal_evidence() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        prove_shadow_measurement(&mut lab, 1, 1, 1_000);
        // A shadow measurement proves the pipeline can measure. It is not a
        // controlled comparison, so it moves no causal quantity at all — the
        // gate included.
        assert_eq!(
            lab.rollout_progress().0,
            0,
            "shadow proves capability, not cause"
        );
        assert_eq!(lab.metrics().shadow_measurements_proven_total, 1);
        assert_eq!(lab.metrics().pair_gold_total, 0);
        assert_eq!(lab.metrics().effective_total, 0);
    }

    #[test]
    fn the_shadow_gate_is_exact_at_its_boundary() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.rollout_config = LabRolloutConfig {
            shadow_min_opportunities: 1,
            shadow_min_measurements: 3,
            shadow_min_duration_ms: 0,
            canary_percent: 100,
            canary_min_opportunities: 1,
        };
        consume_pairs(&mut lab, 2, 1);
        // Monotonic throughout: a clock that goes backwards is itself a reset
        // trigger, and this test is about the threshold, not about regression.
        lab.advance_cycle(3, 3_000, PairGates::healthy_enabled());
        assert_eq!(lab.phase(), LabPhase::Shadow, "threshold - 1 stays Shadow");

        consume_pairs(&mut lab, 1, 3);
        lab.advance_cycle(5, 5_000, PairGates::healthy_enabled());
        assert_eq!(lab.phase(), LabPhase::Canary, "threshold promotes");
    }

    #[test]
    fn a_restart_keeps_earned_evidence_and_invents_none() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        consume_pairs(&mut lab, 2, 1);
        assert_eq!(lab.rollout_progress().0, 2);

        let (restored, _) = MicroexperimentLab::restore(lab.persisted(), origin());
        assert_eq!(
            restored.rollout_progress().0,
            2,
            "evidence actually earned survives a restart"
        );
        assert_eq!(restored.phase(), LabPhase::Shadow);
    }

    // ── Shadow lifecycle / soak, on a controlled clock ──────────────────────

    /// Create a tracked shadow pair and issue its arm, returning the deadline.
    fn open_shadow_arm(lab: &mut MicroexperimentLab, seq: u64, cycle: u64) -> u64 {
        let cand = candidate(seq);
        lab.consider_candidate(cand.clone(), PairGates::healthy_enabled(), cycle * 10)
            .expect("candidate accepted");
        let directives = lab.issue_shadow_arms(cycle, PairGates::healthy_enabled());
        directives
            .iter()
            .find(|d| d.stratum_hash == cand.stratum_hash)
            .map(|d| d.expires_after_cycle)
            .unwrap_or(0)
    }

    #[test]
    fn a_full_shadow_collection_drains_on_expiry_and_accepts_new_work() {
        // Production shape: exactly MAX_OPEN_PAIRS arms registered, exactly that
        // many expired, and then silence. Nothing reaped the pairs, so every
        // slot stayed occupied by a pair that had already issued.
        let mut lab = MicroexperimentLab::cold_start(origin());
        let mut deadline = 0;
        let mut cycle = 1;
        while lab.shadow_open.len() < MAX_OPEN_PAIRS {
            // Arms are rate-limited per cycle, so filling takes several ticks.
            let d = open_shadow_arm(&mut lab, cycle, cycle);
            deadline = deadline.max(d);
            cycle += 1;
            assert!(cycle < 500, "filling must terminate");
        }
        assert_eq!(lab.shadow_open.len(), MAX_OPEN_PAIRS);

        // Push the clock past every deadline and tick once.
        let after = deadline + 1;
        lab.advance_cycle(after, after * 10, PairGates::healthy_enabled());
        assert_eq!(
            lab.shadow_open.len(),
            0,
            "every expired shadow pair must be reaped, not left holding a slot"
        );
        assert_eq!(
            lab.metrics().shadow_pairs_expired_total,
            MAX_OPEN_PAIRS as u64
        );

        // Slot 33 is now reachable.
        let d = open_shadow_arm(&mut lab, 9_001, after + 1);
        assert!(d > 0, "a new arm must issue once the collection drained");
        assert_eq!(lab.shadow_open.len(), 1);
    }

    #[test]
    fn the_shadow_collection_is_observable_and_stays_under_its_ceiling() {
        // The previous baseline asserted this through `open_pairs`, which is
        // `self.open` — the *real* pair collection, always empty in Shadow. It
        // passed by vacuity. These are the shadow collection's own numbers.
        let mut lab = MicroexperimentLab::cold_start(origin());
        assert_eq!(lab.metrics().shadow_open_pairs, 0);
        assert_eq!(lab.metrics().shadow_open_high_watermark, 0);

        let mut deadline = 0;
        let mut cycle = 1;
        while lab.shadow_open.len() < MAX_OPEN_PAIRS {
            deadline = deadline.max(open_shadow_arm(&mut lab, cycle, cycle));
            cycle += 1;
            assert!(cycle < 500, "filling must terminate");
            assert!(
                lab.metrics().shadow_open_pairs <= MAX_OPEN_PAIRS,
                "the ceiling is never exceeded"
            );
        }
        assert_eq!(lab.metrics().shadow_open_pairs, MAX_OPEN_PAIRS);
        assert_eq!(
            lab.metrics().shadow_open_high_watermark,
            MAX_OPEN_PAIRS as u32,
            "a burst that touched the ceiling must remain visible after it drains"
        );

        let after = deadline + 1;
        lab.advance_cycle(after, after * 10, PairGates::healthy_enabled());
        assert_eq!(
            lab.metrics().shadow_open_pairs,
            0,
            "the gauge follows the drain"
        );
        assert_eq!(
            lab.metrics().shadow_open_high_watermark,
            MAX_OPEN_PAIRS as u32,
            "the high-water mark does not fall back with the gauge"
        );
    }

    #[test]
    fn an_expiry_that_is_not_reaped_would_be_visible() {
        // The two counters answer different questions on purpose: one counts
        // deadlines detected, the other slots actually freed. They agreed here,
        // and a future divergence is precisely the silent-slot bug returning.
        let mut lab = MicroexperimentLab::cold_start(origin());
        let mut deadline = 0;
        for seq in 1..=5 {
            deadline = deadline.max(open_shadow_arm(&mut lab, seq, seq));
        }
        let after = deadline + 1;
        lab.advance_cycle(after, after * 10, PairGates::healthy_enabled());
        let m = lab.metrics();
        assert_eq!(m.shadow_pairs_expired_total, 5);
        assert_eq!(
            m.shadow_pairs_reaped_total, m.shadow_pairs_expired_total,
            "every deadline detected must have freed its slot"
        );
    }

    #[test]
    fn repeated_fill_and_expiry_stays_bounded() {
        // Soak: the emit→expire→emit loop must not grow anything without bound.
        let mut lab = MicroexperimentLab::cold_start(origin());
        let mut cycle = 1_u64;
        let mut seq = 1_u64;
        for _round in 0..6 {
            let mut deadline = 0;
            for _ in 0..MAX_OPEN_PAIRS {
                deadline = deadline.max(open_shadow_arm(&mut lab, seq, cycle));
                seq += 1;
                cycle += 1;
            }
            cycle = cycle.max(deadline + 1);
            lab.advance_cycle(cycle, cycle * 10, PairGates::healthy_enabled());
            assert!(
                lab.shadow_open.len() <= MAX_OPEN_PAIRS,
                "the collection must never exceed its ceiling"
            );
            assert!(
                lab.shadow_terminal.len() <= MAX_SHADOW_TERMINAL_MEMORY,
                "terminal memory must stay bounded: {}",
                lab.shadow_terminal.len()
            );
            cycle += 1;
        }
        assert_eq!(lab.shadow_open.len(), 0);
        assert_eq!(lab.rollout_progress().0, 0, "churn is not evidence");
    }

    #[test]
    fn a_late_endpoint_neither_resurrects_a_pair_nor_pays_twice() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        let cand = candidate(1);
        lab.consider_candidate(cand.clone(), PairGates::healthy_enabled(), 10)
            .unwrap();
        let directives = lab.issue_shadow_arms(1, PairGates::healthy_enabled());
        let directive = directives[0].clone();
        let deadline = directive.expires_after_cycle;

        lab.advance_cycle(
            deadline + 1,
            (deadline + 1) * 10,
            PairGates::healthy_enabled(),
        );
        assert_eq!(lab.shadow_open.len(), 0);

        // The endpoint arrives after the reaping.
        let observation = endpoint(&cand, &directive, 0);
        assert!(lab.record_shadow_endpoint(&observation));
        assert_eq!(
            lab.rollout_progress().0,
            0,
            "a late endpoint answers an arm that no longer exists"
        );
        assert_eq!(lab.metrics().shadow_endpoints_late_total, 1);
        assert_eq!(lab.shadow_open.len(), 0, "and must not resurrect the pair");

        // A duplicate of the same late endpoint pays nothing either.
        assert!(lab.record_shadow_endpoint(&observation));
        assert_eq!(lab.metrics().shadow_endpoints_late_total, 2);
        assert_eq!(lab.rollout_progress().0, 0);
    }

    #[test]
    fn terminalising_the_same_pair_twice_is_a_no_op() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.consider_candidate(candidate(1), PairGates::healthy_enabled(), 10)
            .unwrap();
        let id = lab.shadow_open[0].assignment.id;
        assert!(lab.terminalise_shadow_pair(id), "first call removes it");
        assert!(
            !lab.terminalise_shadow_pair(id),
            "second call finds nothing"
        );
        assert_eq!(lab.shadow_terminal.iter().filter(|x| **x == id).count(), 1);
    }

    #[test]
    fn a_restart_drops_shadow_pairs_whose_arm_no_longer_exists() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        open_shadow_arm(&mut lab, 1, 1);
        lab.consider_candidate(candidate(2), PairGates::healthy_enabled(), 20)
            .unwrap();
        assert_eq!(lab.shadow_open.len(), 2);

        let (restored, _) = MicroexperimentLab::restore(lab.persisted(), origin());
        assert_eq!(
            restored.shadow_open.len(),
            1,
            "the issued pair answers an adapter that died with the old boot"
        );
        assert!(restored.shadow_open[0].issued.is_none());
        assert!(restored.shadow_terminal.is_empty());
    }

    #[test]
    fn an_interrupted_pair_is_never_counted_as_a_completed_one() {
        // Production shape: 9 terminal pairs, every one of them interrupted,
        // published as `completed_pairs = 9`. A reader could not tell that the
        // lab had finished exactly nothing.
        let mut lab = active_lab();
        for seq in 1..=9u64 {
            let assignment = match lab
                .consider_candidate(candidate(seq), PairGates::healthy_enabled(), seq * 10)
                .expect("pair opens in Active")
            {
                CandidateDisposition::Opened(assignment) => assignment,
                other => panic!("expected an opened pair, got {other:?}"),
            };
            lab.invalidate_pair(assignment.id, PairInvalidationReason::DeadlineExpired)
                .expect("pair invalidated");
        }
        let metrics = lab.metrics();
        assert_eq!(metrics.terminal_pairs, 9, "nine pairs did end");
        assert_eq!(
            metrics.interrupted_pairs, 9,
            "and all nine ended by interruption"
        );
        assert_eq!(
            metrics.completed_pairs_valid, 0,
            "so the lab finished nothing, and the number must say so"
        );
    }

    #[test]
    fn boot_provenance_reports_the_same_quantity_as_the_gate() {
        // Publishing restored *exposure* beside an evidence gate renders as a
        // counter that collapsed across the restart. The two must agree.
        let mut lab = MicroexperimentLab::cold_start(origin());
        for seq in 1..=20 {
            let _ = lab.consider_candidate(candidate(seq), PairGates::healthy_enabled(), seq * 100);
        }
        consume_pairs(&mut lab, 1, 900);
        let (restored, _) = MicroexperimentLab::restore(lab.persisted(), origin());
        assert_eq!(
            restored.restored_progress,
            restored.rollout_progress().0,
            "boot provenance and the gate must publish the same quantity"
        );
        assert_eq!(restored.restored_progress, 1);
    }

    #[test]
    fn state_written_before_the_fix_carries_no_evidence_forward() {
        // The old semantics banked exposure in `shadow_eligible`. Evidence
        // lives in a field that state did not have, so it deserialises to 0
        // and yesterday's 213 opportunities cannot be laundered into progress.
        let mut lab = MicroexperimentLab::cold_start(origin());
        for seq in 1..=40 {
            let _ =
                lab.consider_candidate(candidate(seq), PairGates::healthy_enabled(), seq * 1_000);
        }
        let mut json = serde_json::to_value(lab.persisted()).expect("serialise");
        json.get_mut("rollout")
            .and_then(|r| r.as_object_mut())
            .map(|r| r.remove("shadow_measurements_proven"));
        let legacy: MicroexperimentLabPersisted =
            serde_json::from_value(json).expect("legacy state must still load");
        let (restored, _) = MicroexperimentLab::restore(legacy, origin());
        assert_eq!(
            restored.rollout_progress().0,
            0,
            "exposure banked under the old rule is not evidence under the new one"
        );
        assert_eq!(restored.phase(), LabPhase::Shadow);
    }

    #[test]
    fn a_restart_never_resumes_mutation_authority() {
        let mut lab = MicroexperimentLab::cold_start(origin());
        lab.rollout_config = LabRolloutConfig {
            shadow_min_opportunities: 1,
            shadow_min_measurements: 1,
            shadow_min_duration_ms: 0,
            canary_percent: 100,
            canary_min_opportunities: 1,
        };
        consume_pairs(&mut lab, 1, 1);
        lab.advance_cycle(2, 20, PairGates::healthy_enabled());
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
