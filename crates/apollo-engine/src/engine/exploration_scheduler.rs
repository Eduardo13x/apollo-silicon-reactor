//! Pure, bounded admission for local exploration proposals.
//!
//! This module owns no clock, I/O, process discovery, synchronization, or
//! effect implementation. The daemon loop supplies immutable current inputs;
//! existing actuator owners remain responsible for every mutation and revert.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::marker::PhantomData;

use serde::de::{Error as DeError, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::engine::telemetry_medallion::ActuatorFamily;

pub const MAX_CANDIDATES_PER_FAMILY: usize = 4;
pub const MAX_CANDIDATES_PER_CYCLE: usize = 12;
pub const MAX_COOLDOWNS: usize = 256;
pub const MAX_TERMINAL_DEDUP: usize = 128;
pub const MAX_SERIALIZED_BYTES: usize = 64 * 1024;
pub const GLOBAL_INTERVAL_SECS: i64 = 900;
pub const KEY_COOLDOWN_SECS: i64 = 86_400;
const MAX_COMMITS_PER_DAY: usize = 96;
const SCHEDULER_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(transparent)]
pub struct ProbeCorrelation(pub u64);

impl ProbeCorrelation {
    const LEDGER_NAMESPACE: u64 = 1_u64 << 63;

    pub fn ledger_correlation_id(self) -> u64 {
        Self::LEDGER_NAMESPACE | self.0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ExplorationMode {
    Natural,
    Treatment,
    Control,
}

impl ExplorationMode {
    fn rank(self) -> u8 {
        match self {
            Self::Natural => 0,
            Self::Treatment => 1,
            Self::Control => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ExplorationArm {
    NaturalObservation,
    MarkovCacheOnly,
    InteractionQosShort,
    InteractionQosStandard,
    InteractionQosLong,
    BoostOmission,
}

impl ExplorationArm {
    fn rank(self) -> u8 {
        match self {
            Self::NaturalObservation => 0,
            Self::MarkovCacheOnly => 1,
            Self::InteractionQosShort => 2,
            Self::InteractionQosStandard => 3,
            Self::InteractionQosLong => 4,
            Self::BoostOmission => 5,
        }
    }

    pub fn allows_kernel_acceleration(self) -> bool {
        false
    }

    pub fn ttl_millis(self) -> Option<u64> {
        match self {
            Self::InteractionQosShort => Some(1_080),
            Self::InteractionQosStandard => Some(1_200),
            Self::InteractionQosLong => Some(1_320),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ActionClass {
    Natural,
    BoostBackground,
    InteractionForeground,
    MarkovPredictedApp,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ExplorationContext {
    General,
    Interactive,
    Background,
    Build,
    Workload(u8),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(default)]
pub struct HardwareIdentity {
    pub p_core_count: u32,
    pub e_core_count: u32,
    pub ram_gib: u32,
}

impl HardwareIdentity {
    pub fn is_known(self) -> bool {
        self.p_core_count.saturating_add(self.e_core_count) > 0 && self.ram_gib > 0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(default)]
pub struct ExplorationOrigin {
    pub installation_id: u64,
    pub hardware: HardwareIdentity,
}

impl ExplorationOrigin {
    pub fn is_known(self) -> bool {
        self.installation_id != 0 && self.hardware.is_known()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimePoint {
    pub wall_unix_secs: i64,
    pub monotonic_secs: u64,
    pub boot_id: u64,
}

impl TimePoint {
    pub fn valid(self) -> bool {
        self.wall_unix_secs > 0 && self.boot_id != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExplorationGates {
    pub daemon_shutdown: bool,
    pub kill_switch: bool,
    pub cognitive_paused: bool,
    pub audio_output_active: bool,
    pub audio_input_active: bool,
    pub call_active: bool,
    pub sleep_assertion: bool,
    pub media_available: bool,
    pub app_launching: bool,
    pub window_operation: bool,
    pub fluidity_degraded: bool,
    pub predicted_fluidity_degraded: bool,
    pub memory_pressure: f64,
    pub thermal_available: bool,
    pub thermal_nominal: bool,
    pub hazard_available: bool,
    pub p_oom_30s: f64,
    pub circuit_closed: bool,
    pub speculation_allowed: bool,
    pub build_workload: bool,
    pub build_phase_idle: bool,
    pub compiler_protection_active: bool,
    pub identity_present: bool,
    pub identity_start_nonzero: bool,
    pub identity_stale: bool,
    pub identity_recycled: bool,
    pub target_protected: bool,
    pub target_apple_owned: bool,
    pub identity_recheck_ok: bool,
    pub markov_quarantined: bool,
    pub effect_owner_conflict: bool,
    pub coalition_conflict: bool,
    pub target_foreground: bool,
    pub target_launching: bool,
    pub interactive_lease_active: bool,
    pub recovery_required: bool,
}

impl ExplorationGates {
    pub fn healthy() -> Self {
        Self {
            daemon_shutdown: false,
            kill_switch: false,
            cognitive_paused: false,
            audio_output_active: false,
            audio_input_active: false,
            call_active: false,
            sleep_assertion: false,
            media_available: true,
            app_launching: false,
            window_operation: false,
            fluidity_degraded: false,
            predicted_fluidity_degraded: false,
            memory_pressure: 0.20,
            thermal_available: true,
            thermal_nominal: true,
            hazard_available: true,
            p_oom_30s: 0.01,
            circuit_closed: true,
            speculation_allowed: true,
            build_workload: false,
            build_phase_idle: true,
            compiler_protection_active: false,
            identity_present: true,
            identity_start_nonzero: true,
            identity_stale: false,
            identity_recycled: false,
            target_protected: false,
            target_apple_owned: false,
            identity_recheck_ok: true,
            markov_quarantined: false,
            effect_owner_conflict: false,
            coalition_conflict: false,
            target_foreground: false,
            target_launching: false,
            interactive_lease_active: false,
            recovery_required: false,
        }
    }
}

impl Default for ExplorationGates {
    fn default() -> Self {
        let mut gates = Self::healthy();
        gates.media_available = false;
        gates.thermal_available = false;
        gates.hazard_available = false;
        gates.identity_present = false;
        gates.identity_start_nonzero = false;
        gates.identity_recheck_ok = false;
        gates
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExplorationGateBlocker {
    Lifecycle,
    Media,
    UserInteraction,
    Fluidity,
    Pressure,
    Thermal,
    Hazard,
    Circuit,
    Build,
    Identity,
    Ownership,
    Policy,
    Capacity,
    GlobalBudget,
    KeyCooldown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalDiagnostic {
    Cancelled,
    Reverted,
    Expired,
    Failed,
    Wake,
    KillSwitch,
    Shutdown,
    Deadline,
    Hit,
    Miss,
    ReleaseFailed,
    DroppedOnRestart,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(default)]
pub struct ExplorationKey {
    pub family: ActuatorFamily,
    pub mode: ExplorationMode,
    pub arm: ExplorationArm,
    pub action_class: ActionClass,
    pub context: ExplorationContext,
}

impl Default for ExplorationKey {
    fn default() -> Self {
        Self {
            family: ActuatorFamily::Boost,
            mode: ExplorationMode::Natural,
            arm: ExplorationArm::NaturalObservation,
            action_class: ActionClass::Natural,
            context: ExplorationContext::General,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ExplorationMetadata {
    pub correlation: ProbeCorrelation,
    pub family: ActuatorFamily,
    pub key: ExplorationKey,
    pub arm: ExplorationArm,
    pub treatment: bool,
    pub committed: bool,
    pub cancelled: Option<TerminalDiagnostic>,
}

impl Default for ExplorationMetadata {
    fn default() -> Self {
        Self {
            correlation: ProbeCorrelation(0),
            family: ActuatorFamily::Boost,
            key: ExplorationKey::default(),
            arm: ExplorationArm::NaturalObservation,
            treatment: false,
            committed: false,
            cancelled: None,
        }
    }
}

impl ExplorationMetadata {
    pub fn valid(&self) -> bool {
        self.correlation.0 > 0
            && self.correlation.0 < ProbeCorrelation::LEDGER_NAMESPACE
            && self.family == self.key.family
            && self.treatment == (self.key.mode == ExplorationMode::Treatment)
            && valid_key(&self.key)
    }

    pub fn clean_treatment(&self) -> bool {
        self.valid() && self.treatment && self.committed && self.cancelled.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplorationCandidate {
    key: ExplorationKey,
    origin: ExplorationOrigin,
}

impl ExplorationCandidate {
    pub fn new(
        family: ActuatorFamily,
        mode: ExplorationMode,
        arm: ExplorationArm,
        action_class: ActionClass,
        context: ExplorationContext,
        origin: ExplorationOrigin,
    ) -> Result<Self, ExplorationGateBlocker> {
        let valid = match (family, mode, arm, action_class) {
            (
                _,
                ExplorationMode::Natural,
                ExplorationArm::NaturalObservation,
                ActionClass::Natural,
            ) => ExplorationScheduler::family_allowed(family),
            (
                ActuatorFamily::Boost,
                ExplorationMode::Control,
                ExplorationArm::BoostOmission,
                ActionClass::BoostBackground,
            ) => true,
            (
                ActuatorFamily::MarkovPrewarm,
                ExplorationMode::Treatment,
                ExplorationArm::MarkovCacheOnly,
                ActionClass::MarkovPredictedApp,
            ) => true,
            (
                ActuatorFamily::InteractionQos,
                ExplorationMode::Treatment,
                ExplorationArm::InteractionQosShort
                | ExplorationArm::InteractionQosStandard
                | ExplorationArm::InteractionQosLong,
                ActionClass::InteractionForeground,
            ) => true,
            _ => false,
        };
        if !valid || !origin.is_known() {
            return Err(ExplorationGateBlocker::Policy);
        }
        Ok(Self {
            key: ExplorationKey {
                family,
                mode,
                arm,
                action_class,
                context,
            },
            origin,
        })
    }

    pub fn key(&self) -> &ExplorationKey {
        &self.key
    }

    pub fn is_mutable(&self) -> bool {
        self.key.mode != ExplorationMode::Natural
    }

    pub fn natural_observation(&self) -> Self {
        Self {
            key: ExplorationKey {
                family: self.key.family,
                mode: ExplorationMode::Natural,
                arm: ExplorationArm::NaturalObservation,
                action_class: ActionClass::Natural,
                context: self.key.context,
            },
            origin: self.origin,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplorationApproval {
    pub metadata: ExplorationMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitEvidence {
    MutationApplied,
    OmissionEndpointOpened,
    NoOp,
    Failed,
    DryRun,
    StaleIdentity,
    OwnershipConflict,
    ZeroMembers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitResult {
    Committed,
    ReleasedWithoutCommit,
    Duplicate,
    UnknownCorrelation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct PersistedCooldown {
    key: ExplorationKey,
    expires_wall_unix_secs: i64,
    committed_monotonic_secs: u64,
    boot_id: u64,
}

impl Default for PersistedCooldown {
    fn default() -> Self {
        Self {
            key: ExplorationKey::default(),
            expires_wall_unix_secs: 0,
            committed_monotonic_secs: 0,
            boot_id: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(default)]
struct PersistedCommit {
    wall_unix_secs: i64,
    monotonic_secs: u64,
    boot_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExplorationSchedulerPersisted {
    pub schema_version: u32,
    origin: ExplorationOrigin,
    #[serde(default, deserialize_with = "deserialize_cooldowns")]
    cooldowns: Vec<PersistedCooldown>,
    #[serde(default, deserialize_with = "deserialize_commits")]
    commits: VecDeque<PersistedCommit>,
    #[serde(default, deserialize_with = "deserialize_terminal_dedup")]
    terminal_dedup: VecDeque<ProbeCorrelation>,
    arm_sequence: u64,
    next_correlation: u64,
    #[serde(default, deserialize_with = "deserialize_reserved")]
    reserved: String,
}

impl Default for ExplorationSchedulerPersisted {
    fn default() -> Self {
        Self::default_for(ExplorationOrigin::default())
    }
}

impl ExplorationSchedulerPersisted {
    pub fn default_for(origin: ExplorationOrigin) -> Self {
        Self {
            schema_version: SCHEDULER_SCHEMA_VERSION,
            origin,
            cooldowns: Vec::new(),
            commits: VecDeque::new(),
            terminal_dedup: VecDeque::new(),
            arm_sequence: 0,
            next_correlation: 0,
            reserved: String::new(),
        }
    }

    #[doc(hidden)]
    pub fn inject_hostile_cooldowns_for_test(&mut self, count: usize) {
        self.cooldowns = fixture_cooldowns(count, i64::MAX / 2);
    }

    #[doc(hidden)]
    pub fn inject_live_cooldowns_for_test(&mut self, count: usize, expiry: i64) {
        self.cooldowns = fixture_cooldowns(count, expiry);
    }

    #[doc(hidden)]
    pub fn oversized_for_test(origin: ExplorationOrigin, bytes: usize) -> Self {
        let mut state = Self::default_for(origin);
        state.reserved = "x".repeat(bytes);
        state
    }
}

fn fixture_cooldowns(count: usize, expiry: i64) -> Vec<PersistedCooldown> {
    (0..count)
        .map(|index| PersistedCooldown {
            key: ExplorationKey {
                family: ActuatorFamily::MarkovPrewarm,
                mode: ExplorationMode::Treatment,
                arm: ExplorationArm::MarkovCacheOnly,
                action_class: ActionClass::MarkovPredictedApp,
                context: ExplorationContext::Workload(index.min(u8::MAX as usize) as u8),
            },
            expires_wall_unix_secs: expiry,
            committed_monotonic_secs: 1,
            boot_id: 10,
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreSource {
    Local,
    Unknown,
    ImportedM1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreContext {
    origin: ExplorationOrigin,
    now: TimePoint,
    source: RestoreSource,
}

impl RestoreContext {
    pub fn local(origin: ExplorationOrigin, now: TimePoint) -> Self {
        Self {
            origin,
            now,
            source: RestoreSource::Local,
        }
    }

    pub fn unknown(origin: ExplorationOrigin, now: TimePoint) -> Self {
        Self {
            origin,
            now,
            source: RestoreSource::Unknown,
        }
    }

    pub fn imported_m1(origin: ExplorationOrigin, now: TimePoint) -> Self {
        Self {
            origin,
            now,
            source: RestoreSource::ImportedM1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreDisposition {
    Restored,
    ResetOrigin,
    ResetHostile,
}

#[derive(Debug, Clone)]
struct ActiveReservation {
    metadata: ExplorationMetadata,
}

#[derive(Debug, Clone, Copy)]
struct CooldownDeadline {
    expires_wall_unix_secs: i64,
    committed_monotonic_secs: u64,
    boot_id: u64,
}

#[derive(Debug, Clone)]
pub struct ExplorationScheduler {
    origin: ExplorationOrigin,
    cooldowns: HashMap<ExplorationKey, CooldownDeadline>,
    commits: VecDeque<PersistedCommit>,
    terminal_dedup: VecDeque<ProbeCorrelation>,
    active: Option<ActiveReservation>,
    arm_sequence: u64,
    next_correlation: u64,
    cycle_id: Option<u64>,
    cycle_candidates_examined: usize,
    last_work: SchedulerWork,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedulerWork {
    pub candidates_examined: usize,
    pub cooldowns_examined: usize,
}

impl ExplorationScheduler {
    pub fn cold_start(origin: ExplorationOrigin) -> Self {
        Self {
            origin,
            cooldowns: HashMap::with_capacity(MAX_COOLDOWNS),
            commits: VecDeque::with_capacity(MAX_COMMITS_PER_DAY),
            terminal_dedup: VecDeque::with_capacity(MAX_TERMINAL_DEDUP),
            active: None,
            arm_sequence: 0,
            next_correlation: 0,
            cycle_id: None,
            cycle_candidates_examined: 0,
            last_work: SchedulerWork::default(),
        }
    }

    pub fn begin_exploration_cycle(&mut self, cycle: u64) {
        if self.cycle_id != Some(cycle) {
            self.cycle_id = Some(cycle);
            self.cycle_candidates_examined = 0;
            self.last_work = SchedulerWork::default();
        }
    }

    pub fn family_allowed(family: ActuatorFamily) -> bool {
        matches!(
            family,
            ActuatorFamily::Boost | ActuatorFamily::InteractionQos | ActuatorFamily::MarkovPrewarm
        )
    }

    pub fn origin(&self) -> ExplorationOrigin {
        self.origin
    }

    pub fn interaction_arm(&self) -> ExplorationArm {
        match self.arm_sequence % 3 {
            0 => ExplorationArm::InteractionQosShort,
            1 => ExplorationArm::InteractionQosStandard,
            _ => ExplorationArm::InteractionQosLong,
        }
    }

    pub fn request(
        &mut self,
        candidate: &ExplorationCandidate,
        gates: &ExplorationGates,
        now: TimePoint,
    ) -> Result<ExplorationApproval, ExplorationGateBlocker> {
        self.select(std::slice::from_ref(candidate), gates, now)
    }

    pub fn select(
        &mut self,
        candidates: &[ExplorationCandidate],
        gates: &ExplorationGates,
        now: TimePoint,
    ) -> Result<ExplorationApproval, ExplorationGateBlocker> {
        first_safety_gate_blocker(gates)?;
        if self.cycle_id.is_none() {
            self.cycle_candidates_examined = 0;
        }
        self.last_work.cooldowns_examined = 0;
        if candidates.is_empty() || candidates.len() > MAX_CANDIDATES_PER_CYCLE {
            return Err(ExplorationGateBlocker::Capacity);
        }
        if self
            .cycle_candidates_examined
            .saturating_add(candidates.len())
            > MAX_CANDIDATES_PER_CYCLE
        {
            return Err(ExplorationGateBlocker::Capacity);
        }
        let mut family_counts = [0_u8; 3];
        for candidate in candidates {
            self.cycle_candidates_examined += 1;
            self.last_work.candidates_examined = self.cycle_candidates_examined;
            if candidate.origin != self.origin || !candidate.origin.is_known() {
                return Err(ExplorationGateBlocker::Policy);
            }
            let Some(index) = family_index(candidate.key.family) else {
                return Err(ExplorationGateBlocker::Policy);
            };
            family_counts[index] = family_counts[index].saturating_add(1);
            if usize::from(family_counts[index]) > MAX_CANDIDATES_PER_FAMILY {
                return Err(ExplorationGateBlocker::Capacity);
            }
        }
        let selected = candidates
            .iter()
            .min_by_key(|candidate| selection_key(&candidate.key))
            .ok_or(ExplorationGateBlocker::Capacity)?;
        let correlation = self.peek_next_correlation();
        let mut metadata = ExplorationMetadata {
            correlation,
            family: selected.key.family,
            key: selected.key.clone(),
            arm: selected.key.arm,
            treatment: selected.key.mode == ExplorationMode::Treatment,
            committed: false,
            cancelled: None,
        };
        if selected.key.mode == ExplorationMode::Natural {
            self.next_correlation = correlation.0;
            metadata.treatment = false;
            return Ok(ExplorationApproval { metadata });
        }
        selected_key_gate_blocker(gates, &selected.key)?;
        if self.active.is_some() {
            return Err(ExplorationGateBlocker::Ownership);
        }
        if !now.valid() {
            return Err(ExplorationGateBlocker::Policy);
        }
        self.prune_expired(now);
        self.check_global_budget(now)?;
        if self
            .cooldowns
            .get(&selected.key)
            .is_some_and(|deadline| !cooldown_satisfied(*deadline, now))
        {
            return Err(ExplorationGateBlocker::KeyCooldown);
        }
        if self.cooldowns.len() >= MAX_COOLDOWNS && !self.cooldowns.contains_key(&selected.key) {
            return Err(ExplorationGateBlocker::Capacity);
        }
        self.next_correlation = correlation.0;
        self.active = Some(ActiveReservation {
            metadata: metadata.clone(),
        });
        Ok(ExplorationApproval { metadata })
    }

    pub fn commit(
        &mut self,
        correlation: ProbeCorrelation,
        now: TimePoint,
        evidence: CommitEvidence,
    ) -> CommitResult {
        let matched_active = self
            .active
            .as_ref()
            .is_some_and(|active| active.metadata.correlation == correlation);
        if self.commit_metadata(correlation, now, evidence).is_some() {
            CommitResult::Committed
        } else if matched_active {
            CommitResult::ReleasedWithoutCommit
        } else if self.terminal_dedup.contains(&correlation) {
            CommitResult::Duplicate
        } else {
            CommitResult::UnknownCorrelation
        }
    }

    pub fn commit_metadata(
        &mut self,
        correlation: ProbeCorrelation,
        now: TimePoint,
        evidence: CommitEvidence,
    ) -> Option<ExplorationMetadata> {
        let reservation = match self.active.take() {
            Some(reservation) if reservation.metadata.correlation == correlation => reservation,
            Some(reservation) => {
                self.active = Some(reservation);
                return None;
            }
            None => return None,
        };
        let valid = match reservation.metadata.key.mode {
            ExplorationMode::Treatment => evidence == CommitEvidence::MutationApplied,
            ExplorationMode::Control => evidence == CommitEvidence::OmissionEndpointOpened,
            ExplorationMode::Natural => false,
        };
        if !valid || !now.valid() {
            return None;
        }
        if self.cooldowns.len() >= MAX_COOLDOWNS
            && !self.cooldowns.contains_key(&reservation.metadata.key)
        {
            return None;
        }
        self.cooldowns.insert(
            reservation.metadata.key.clone(),
            CooldownDeadline {
                expires_wall_unix_secs: now.wall_unix_secs.saturating_add(KEY_COOLDOWN_SECS),
                committed_monotonic_secs: now.monotonic_secs,
                boot_id: now.boot_id,
            },
        );
        self.commits.push_back(PersistedCommit {
            wall_unix_secs: now.wall_unix_secs,
            monotonic_secs: now.monotonic_secs,
            boot_id: now.boot_id,
        });
        while self.commits.len() > MAX_COMMITS_PER_DAY {
            self.commits.pop_front();
        }
        if reservation.metadata.family == ActuatorFamily::InteractionQos {
            self.arm_sequence = self.arm_sequence.wrapping_add(1);
        }
        self.push_terminal(correlation);
        let mut metadata = reservation.metadata;
        metadata.committed = true;
        Some(metadata)
    }

    pub fn cancel(
        &mut self,
        correlation: ProbeCorrelation,
        _diagnostic: TerminalDiagnostic,
    ) -> bool {
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.metadata.correlation == correlation)
        {
            self.active = None;
            self.push_terminal(correlation);
            true
        } else {
            false
        }
    }

    pub fn cancel_active(&mut self, diagnostic: TerminalDiagnostic) -> Option<ExplorationMetadata> {
        let mut metadata = self.active.take()?.metadata;
        metadata.cancelled = Some(diagnostic);
        self.push_terminal(metadata.correlation);
        Some(metadata)
    }

    pub fn record_terminal(
        &mut self,
        correlation: ProbeCorrelation,
        _diagnostic: TerminalDiagnostic,
    ) -> bool {
        if correlation.0 == 0 || self.terminal_dedup.contains(&correlation) {
            return false;
        }
        self.push_terminal(correlation);
        true
    }

    pub fn persisted(&self) -> ExplorationSchedulerPersisted {
        let cooldowns: Vec<_> = self
            .cooldowns
            .iter()
            .map(|(key, deadline)| PersistedCooldown {
                key: key.clone(),
                expires_wall_unix_secs: deadline.expires_wall_unix_secs,
                committed_monotonic_secs: deadline.committed_monotonic_secs,
                boot_id: deadline.boot_id,
            })
            .collect();
        ExplorationSchedulerPersisted {
            schema_version: SCHEDULER_SCHEMA_VERSION,
            origin: self.origin,
            cooldowns,
            commits: self.commits.clone(),
            terminal_dedup: self.terminal_dedup.clone(),
            arm_sequence: self.arm_sequence,
            next_correlation: self.next_correlation,
            reserved: String::new(),
        }
    }

    pub fn restore(
        persisted: ExplorationSchedulerPersisted,
        context: RestoreContext,
    ) -> (Self, RestoreDisposition) {
        if context.source != RestoreSource::Local
            || !context.origin.is_known()
            || persisted.origin != context.origin
        {
            return (
                Self::cold_start(context.origin),
                RestoreDisposition::ResetOrigin,
            );
        }
        if !valid_persisted(&persisted) {
            return (
                Self::cold_start(context.origin),
                RestoreDisposition::ResetHostile,
            );
        }
        let mut scheduler = Self::cold_start(context.origin);
        scheduler.cooldowns = persisted
            .cooldowns
            .into_iter()
            .map(|cooldown| {
                (
                    cooldown.key,
                    CooldownDeadline {
                        expires_wall_unix_secs: cooldown.expires_wall_unix_secs,
                        committed_monotonic_secs: cooldown.committed_monotonic_secs,
                        boot_id: cooldown.boot_id,
                    },
                )
            })
            .collect();
        scheduler.commits = persisted.commits;
        scheduler.terminal_dedup = persisted.terminal_dedup;
        scheduler.arm_sequence = persisted.arm_sequence;
        scheduler.next_correlation = persisted.next_correlation;
        if context.now.valid() {
            scheduler.prune_expired(context.now);
            scheduler.prune_commit_window(context.now);
        }
        (scheduler, RestoreDisposition::Restored)
    }

    pub fn has_active_reservation(&self) -> bool {
        self.active.is_some()
    }

    pub fn active_correlation(&self) -> Option<ProbeCorrelation> {
        self.active
            .as_ref()
            .map(|active| active.metadata.correlation)
    }

    pub fn committed_count(&self) -> usize {
        self.commits.len()
    }

    pub fn cooldown_count(&self) -> usize {
        self.cooldowns.len()
    }

    pub fn terminal_dedup_count(&self) -> usize {
        self.terminal_dedup.len()
    }

    pub fn last_work(&self) -> SchedulerWork {
        self.last_work
    }

    pub fn recheck(
        &self,
        approval: &ExplorationApproval,
        gates: &ExplorationGates,
    ) -> Result<(), ExplorationGateBlocker> {
        first_safety_gate_blocker(gates)?;
        selected_key_gate_blocker(gates, &approval.metadata.key)
    }

    fn peek_next_correlation(&self) -> ProbeCorrelation {
        let next = self.next_correlation.wrapping_add(1) & !ProbeCorrelation::LEDGER_NAMESPACE;
        ProbeCorrelation(if next == 0 { 1 } else { next })
    }

    fn prune_expired(&mut self, now: TimePoint) {
        if self.cooldowns.len() <= MAX_COOLDOWNS {
            self.last_work.cooldowns_examined = self.cooldowns.len();
            self.cooldowns
                .retain(|_, deadline| !cooldown_satisfied(*deadline, now));
        }
    }

    fn prune_commit_window(&mut self, now: TimePoint) {
        while self
            .commits
            .front()
            .copied()
            .is_some_and(|commit| deadline_satisfied(commit, now, KEY_COOLDOWN_SECS as u64))
        {
            self.commits.pop_front();
        }
    }

    fn check_global_budget(&mut self, now: TimePoint) -> Result<(), ExplorationGateBlocker> {
        self.prune_commit_window(now);
        if self.commits.len() >= MAX_COMMITS_PER_DAY {
            return Err(ExplorationGateBlocker::GlobalBudget);
        }
        if self
            .commits
            .back()
            .copied()
            .is_some_and(|last| !deadline_satisfied(last, now, GLOBAL_INTERVAL_SECS as u64))
        {
            return Err(ExplorationGateBlocker::GlobalBudget);
        }
        Ok(())
    }

    fn push_terminal(&mut self, correlation: ProbeCorrelation) {
        if correlation.0 == 0 || self.terminal_dedup.contains(&correlation) {
            return;
        }
        if self.terminal_dedup.len() >= MAX_TERMINAL_DEDUP {
            self.terminal_dedup.pop_front();
        }
        self.terminal_dedup.push_back(correlation);
    }
}

fn family_index(family: ActuatorFamily) -> Option<usize> {
    match family {
        ActuatorFamily::Boost => Some(0),
        ActuatorFamily::InteractionQos => Some(1),
        ActuatorFamily::MarkovPrewarm => Some(2),
        _ => None,
    }
}

fn family_rank(family: ActuatorFamily) -> u8 {
    family_index(family).unwrap_or(3) as u8
}

fn selection_key(key: &ExplorationKey) -> (u8, u8, ActionClass, ExplorationContext, u8) {
    (
        key.arm.rank(),
        family_rank(key.family),
        key.action_class,
        key.context,
        key.mode.rank(),
    )
}

fn first_safety_gate_blocker(gates: &ExplorationGates) -> Result<(), ExplorationGateBlocker> {
    if gates.daemon_shutdown || gates.kill_switch || gates.cognitive_paused {
        return Err(ExplorationGateBlocker::Lifecycle);
    }
    if gates.audio_output_active
        || gates.audio_input_active
        || gates.call_active
        || gates.sleep_assertion
        || !gates.media_available
    {
        return Err(ExplorationGateBlocker::Media);
    }
    if gates.app_launching || gates.window_operation {
        return Err(ExplorationGateBlocker::UserInteraction);
    }
    if gates.fluidity_degraded || gates.predicted_fluidity_degraded {
        return Err(ExplorationGateBlocker::Fluidity);
    }
    if !gates.memory_pressure.is_finite() || gates.memory_pressure >= 0.55 {
        return Err(ExplorationGateBlocker::Pressure);
    }
    if !gates.thermal_available || !gates.thermal_nominal {
        return Err(ExplorationGateBlocker::Thermal);
    }
    if !gates.hazard_available || !gates.p_oom_30s.is_finite() || gates.p_oom_30s >= 0.30 {
        return Err(ExplorationGateBlocker::Hazard);
    }
    if !gates.circuit_closed || !gates.speculation_allowed {
        return Err(ExplorationGateBlocker::Circuit);
    }
    if gates.build_workload || !gates.build_phase_idle || gates.compiler_protection_active {
        return Err(ExplorationGateBlocker::Build);
    }
    if !gates.identity_present
        || !gates.identity_start_nonzero
        || gates.identity_stale
        || gates.identity_recycled
        || gates.target_protected
        || gates.target_apple_owned
        || !gates.identity_recheck_ok
    {
        return Err(ExplorationGateBlocker::Identity);
    }
    Ok(())
}

fn selected_key_gate_blocker(
    gates: &ExplorationGates,
    key: &ExplorationKey,
) -> Result<(), ExplorationGateBlocker> {
    let boost_omission_conflict = key.family == ActuatorFamily::Boost
        && (gates.coalition_conflict
            || gates.target_foreground
            || gates.target_launching
            || gates.interactive_lease_active
            || gates.recovery_required);
    if gates.markov_quarantined || gates.effect_owner_conflict || boost_omission_conflict {
        return Err(ExplorationGateBlocker::Ownership);
    }
    if !ExplorationScheduler::family_allowed(key.family)
        || (key.family == ActuatorFamily::MarkovPrewarm
            && key.arm != ExplorationArm::MarkovCacheOnly)
    {
        return Err(ExplorationGateBlocker::Policy);
    }
    Ok(())
}

fn valid_persisted(persisted: &ExplorationSchedulerPersisted) -> bool {
    if persisted.schema_version != SCHEDULER_SCHEMA_VERSION
        || !persisted.origin.is_known()
        || persisted.cooldowns.len() > MAX_COOLDOWNS
        || persisted.commits.len() > MAX_COMMITS_PER_DAY
        || persisted.terminal_dedup.len() > MAX_TERMINAL_DEDUP
        || persisted.next_correlation >= ProbeCorrelation::LEDGER_NAMESPACE
        || !persisted.reserved.is_empty()
        || serde_json::to_vec(persisted).map_or(true, |bytes| bytes.len() > MAX_SERIALIZED_BYTES)
    {
        return false;
    }
    let mut keys = HashSet::with_capacity(persisted.cooldowns.len());
    if persisted.cooldowns.iter().any(|cooldown| {
        cooldown.expires_wall_unix_secs <= 0
            || cooldown.boot_id == 0
            || !valid_key(&cooldown.key)
            || !keys.insert(cooldown.key.clone())
    }) {
        return false;
    }
    let mut correlations = HashSet::with_capacity(persisted.terminal_dedup.len());
    if persisted.terminal_dedup.iter().any(|correlation| {
        correlation.0 == 0
            || correlation.0 >= ProbeCorrelation::LEDGER_NAMESPACE
            || !correlations.insert(*correlation)
    }) {
        return false;
    }
    persisted
        .commits
        .iter()
        .all(|commit| commit.wall_unix_secs > 0 && commit.boot_id > 0)
        && persisted
            .commits
            .iter()
            .zip(persisted.commits.iter().skip(1))
            .all(|(left, right)| left.wall_unix_secs <= right.wall_unix_secs)
}

fn deadline_satisfied(commit: PersistedCommit, now: TimePoint, duration_secs: u64) -> bool {
    if now.wall_unix_secs < commit.wall_unix_secs
        || now.wall_unix_secs.saturating_sub(commit.wall_unix_secs) < duration_secs as i64
    {
        return false;
    }
    now.boot_id != commit.boot_id
        || (now.monotonic_secs >= commit.monotonic_secs
            && now.monotonic_secs.saturating_sub(commit.monotonic_secs) >= duration_secs)
}

fn cooldown_satisfied(deadline: CooldownDeadline, now: TimePoint) -> bool {
    if now.wall_unix_secs < deadline.expires_wall_unix_secs {
        return false;
    }
    now.boot_id != deadline.boot_id
        || (now.monotonic_secs >= deadline.committed_monotonic_secs
            && now
                .monotonic_secs
                .saturating_sub(deadline.committed_monotonic_secs)
                >= KEY_COOLDOWN_SECS as u64)
}

struct BoundedVecVisitor<T, const MAX: usize>(PhantomData<T>);

impl<'de, T, const MAX: usize> Visitor<'de> for BoundedVecVisitor<T, MAX>
where
    T: Deserialize<'de>,
{
    type Value = Vec<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "at most {MAX} scheduler entries")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(MAX.min(sequence.size_hint().unwrap_or(MAX)));
        while let Some(value) = sequence.next_element()? {
            if values.len() == MAX {
                return Err(A::Error::custom("scheduler field exceeds bounded capacity"));
            }
            values.push(value);
        }
        Ok(values)
    }
}

fn deserialize_cooldowns<'de, D>(deserializer: D) -> Result<Vec<PersistedCooldown>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(BoundedVecVisitor::<PersistedCooldown, MAX_COOLDOWNS>(
        PhantomData,
    ))
}

fn deserialize_commits<'de, D>(deserializer: D) -> Result<VecDeque<PersistedCommit>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, PersistedCommit, MAX_COMMITS_PER_DAY>(deserializer)
        .map(VecDeque::from)
}

fn deserialize_terminal_dedup<'de, D>(
    deserializer: D,
) -> Result<VecDeque<ProbeCorrelation>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, ProbeCorrelation, MAX_TERMINAL_DEDUP>(deserializer)
        .map(VecDeque::from)
}

fn deserialize_bounded_vec<'de, D, T, const MAX: usize>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    deserializer.deserialize_seq(BoundedVecVisitor::<T, MAX>(PhantomData))
}

fn deserialize_reserved<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    struct BoundedStringVisitor;

    impl Visitor<'_> for BoundedStringVisitor {
        type Value = String;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "a scheduler string no larger than {MAX_SERIALIZED_BYTES} bytes"
            )
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: DeError,
        {
            if value.len() > MAX_SERIALIZED_BYTES {
                return Err(E::custom("scheduler string exceeds serialized payload cap"));
            }
            Ok(value.to_string())
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: DeError,
        {
            if value.len() > MAX_SERIALIZED_BYTES {
                return Err(E::custom("scheduler string exceeds serialized payload cap"));
            }
            Ok(value)
        }
    }

    deserializer.deserialize_string(BoundedStringVisitor)
}

fn valid_key(key: &ExplorationKey) -> bool {
    matches!(
        (key.family, key.mode, key.arm, key.action_class),
        (
            ActuatorFamily::Boost | ActuatorFamily::InteractionQos | ActuatorFamily::MarkovPrewarm,
            ExplorationMode::Natural,
            ExplorationArm::NaturalObservation,
            ActionClass::Natural,
        ) | (
            ActuatorFamily::Boost,
            ExplorationMode::Control,
            ExplorationArm::BoostOmission,
            ActionClass::BoostBackground,
        ) | (
            ActuatorFamily::MarkovPrewarm,
            ExplorationMode::Treatment,
            ExplorationArm::MarkovCacheOnly,
            ActionClass::MarkovPredictedApp,
        ) | (
            ActuatorFamily::InteractionQos,
            ExplorationMode::Treatment,
            ExplorationArm::InteractionQosShort
                | ExplorationArm::InteractionQosStandard
                | ExplorationArm::InteractionQosLong,
            ActionClass::InteractionForeground,
        )
    )
}
