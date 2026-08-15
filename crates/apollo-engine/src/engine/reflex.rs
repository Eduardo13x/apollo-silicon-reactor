//! Bounded admission and rollout state for Apollo's reversible reflex lane.
//!
//! This module never performs a syscall. It decides whether a typed intent may
//! reach an existing actuator, leaving identity rechecks, effect ownership and
//! rollback with those established paths.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, TryLockError};
use std::thread::JoinHandle;
use std::time::Instant;

use serde::{Deserialize, Serialize};

const REFLEX_SCHEMA_VERSION: u32 = 2;
const MIN_TTL_MS: u64 = 100;
const MAX_TTL_MS: u64 = 12_000;
const BASELINE_CYCLES: u64 = 100;
const MAX_P95_MS: f64 = 75.0;
const MAX_REGRESSION: f64 = 1.10;
const MIN_CHURN_ALLOWANCE: f64 = 0.05;
const DECISIVE_CONFIDENCE: f64 = 0.80;
const DECISIVE_NEGATIVE_BOUND: f64 = -0.002;
const DEDUP_CAPACITY: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReflexActionKind {
    InteractionQos,
    Nice,
    InteractiveIoRelease,
    TemporalBoost,
    MarkovPrewarm,
}

impl ReflexActionKind {
    pub fn requires_process(self) -> bool {
        true
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::InteractionQos => "interaction-qos",
            Self::Nice => "nice",
            Self::InteractiveIoRelease => "interactive-io-release",
            Self::TemporalBoost => "temporal-boost",
            Self::MarkovPrewarm => "markov-prewarm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReflexTrigger {
    Input,
    WindowOperation,
    BuildStart,
    AppLaunch,
    WebNavigation,
    NetworkActivity,
    Prediction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReflexSource {
    Deterministic,
    WorldModel,
    Nars,
    Markov,
    Mpc,
    Causal,
    Gpu,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReflexTarget {
    pub pid: u32,
    pub start_sec: u64,
    pub start_usec: u64,
    pub name: String,
}

impl ReflexTarget {
    fn valid(&self) -> bool {
        self.pid > 1 && (self.start_sec > 0 || self.start_usec > 0) && !self.name.trim().is_empty()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReflexAdvice {
    pub score: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence: f64,
    pub authoritative: bool,
    #[serde(default)]
    pub sources: Vec<ReflexSource>,
}

impl ReflexAdvice {
    fn decisive_negative(&self) -> bool {
        self.authoritative
            && self.confidence.is_finite()
            && self.confidence >= DECISIVE_CONFIDENCE
            && self.upper_bound.is_finite()
            && self.upper_bound < DECISIVE_NEGATIVE_BOUND
    }

    fn sanitized(mut self) -> Self {
        if !self.score.is_finite()
            || !self.lower_bound.is_finite()
            || !self.upper_bound.is_finite()
            || !self.confidence.is_finite()
            || self.lower_bound > self.upper_bound
        {
            return Self::default();
        }
        self.score = self.score.clamp(-1.0, 1.0);
        self.lower_bound = self.lower_bound.clamp(-1.0, 1.0);
        self.upper_bound = self.upper_bound.clamp(-1.0, 1.0);
        self.confidence = self.confidence.clamp(0.0, 1.0);
        self.sources.sort_by_key(|source| *source as u8);
        self.sources.dedup();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReflexIntent {
    pub action: ReflexActionKind,
    pub target: Option<ReflexTarget>,
    pub trigger: ReflexTrigger,
    pub primary_source: ReflexSource,
    pub cycle: u64,
    pub ttl_ms: u64,
    pub reversible: bool,
    #[serde(default)]
    pub advice: ReflexAdvice,
}

impl ReflexIntent {
    pub fn new(
        action: ReflexActionKind,
        target: Option<ReflexTarget>,
        trigger: ReflexTrigger,
        primary_source: ReflexSource,
        cycle: u64,
        ttl_ms: u64,
    ) -> Result<Self, ReflexIntentError> {
        if action.requires_process() && target.as_ref().is_none_or(|target| !target.valid()) {
            return Err(ReflexIntentError::InvalidTarget);
        }
        if !(MIN_TTL_MS..=MAX_TTL_MS).contains(&ttl_ms) {
            return Err(ReflexIntentError::InvalidTtl);
        }
        Ok(Self {
            action,
            target,
            trigger,
            primary_source,
            cycle,
            ttl_ms,
            reversible: true,
            advice: ReflexAdvice::default(),
        })
    }

    pub fn with_advice(mut self, advice: ReflexAdvice) -> Self {
        self.advice = advice.sanitized();
        self
    }

    fn dedup_key(&self) -> ReflexIntentKey {
        ReflexIntentKey {
            action: self.action,
            pid: self.target.as_ref().map(|target| target.pid).unwrap_or(0),
            start_sec: self
                .target
                .as_ref()
                .map(|target| target.start_sec)
                .unwrap_or(0),
            start_usec: self
                .target
                .as_ref()
                .map(|target| target.start_usec)
                .unwrap_or(0),
            cycle: self.cycle,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflexIntentError {
    InvalidTarget,
    InvalidTtl,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReflexSafetyContext {
    pub identity_present: bool,
    pub identity_start_nonzero: bool,
    pub identity_recheck_ok: bool,
    pub target_protected: bool,
    pub target_apple_owned: bool,
    pub capability_available: bool,
    pub kill_switch: bool,
    pub thermal_force_ecores: bool,
    pub existing_conflict: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReflexBlocker {
    Disabled,
    KillSwitch,
    ThermalConstraint,
    MissingIdentity,
    IdentityMismatch,
    ProtectedTarget,
    AppleOwnedTarget,
    MissingCapability,
    Conflict,
    Duplicate,
    DecisiveModelVeto,
}

impl ReflexBlocker {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::KillSwitch => "kill-switch",
            Self::ThermalConstraint => "thermal-constraint",
            Self::MissingIdentity => "missing-identity",
            Self::IdentityMismatch => "identity-mismatch",
            Self::ProtectedTarget => "protected-target",
            Self::AppleOwnedTarget => "apple-owned-target",
            Self::MissingCapability => "missing-capability",
            Self::Conflict => "conflict",
            Self::Duplicate => "duplicate",
            Self::DecisiveModelVeto => "decisive-model-veto",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflexDecision {
    Shadow,
    Admit,
    Veto(ReflexBlocker),
    Skipped(ReflexBlocker),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReflexRolloutPhase {
    #[default]
    Shadow,
    Active,
    Disabled,
}

impl ReflexRolloutPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Shadow => "shadow",
            Self::Active => "active",
            Self::Disabled => "disabled",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReflexRolloutState {
    #[serde(default = "schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_shadow_cycles")]
    pub shadow_cycles: u64,
    #[serde(default)]
    pub build_profile: String,
    #[serde(default)]
    pub phase: ReflexRolloutPhase,
    #[serde(default)]
    pub blocker: String,
    #[serde(default)]
    pub valid_cycles: u64,
    #[serde(default)]
    pub invalid_samples: u64,
    #[serde(default)]
    baseline_samples: u64,
    #[serde(default)]
    baseline_p95_sum: f64,
    #[serde(default)]
    baseline_churn_sum: f64,
    #[serde(default)]
    candidate_samples: u64,
    #[serde(default)]
    candidate_p95_sum: f64,
    #[serde(default)]
    candidate_churn_sum: f64,
    #[serde(default)]
    previous_applied: u64,
    #[serde(default)]
    previous_reverted: u64,
    #[serde(default)]
    protected_admissions_seen: u64,
    #[serde(default)]
    failures_seen: u64,
    #[serde(default)]
    rollback_failures_seen: u64,
}

fn schema_version() -> u32 {
    REFLEX_SCHEMA_VERSION
}

fn default_shadow_cycles() -> u64 {
    500
}

impl ReflexRolloutState {
    fn new(enabled: bool, shadow_cycles: u64, build_profile: &str) -> Self {
        Self {
            schema_version: REFLEX_SCHEMA_VERSION,
            enabled,
            shadow_cycles: shadow_cycles.max(default_shadow_cycles()),
            build_profile: build_profile.to_string(),
            phase: if enabled {
                ReflexRolloutPhase::Shadow
            } else {
                ReflexRolloutPhase::Disabled
            },
            blocker: if enabled { "warming-up" } else { "disabled" }.to_string(),
            ..Self::default()
        }
    }

    pub fn baseline_p95_ms(&self) -> f64 {
        mean(self.baseline_p95_sum, self.baseline_samples)
    }

    pub fn candidate_p95_ms(&self) -> f64 {
        mean(self.candidate_p95_sum, self.candidate_samples)
    }

    pub fn baseline_churn(&self) -> f64 {
        mean(self.baseline_churn_sum, self.baseline_samples)
    }

    pub fn candidate_churn(&self) -> f64 {
        mean(self.candidate_churn_sum, self.candidate_samples)
    }
}

fn mean(sum: f64, samples: u64) -> f64 {
    if samples == 0 {
        0.0
    } else {
        sum / samples as f64
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReflexHealthSample {
    pub cycle: u64,
    pub p95_cycle_ms: f64,
    pub applied_total: u64,
    pub reverted_total: u64,
    pub protected_admissions_total: u64,
    pub failures_total: u64,
    pub rollback_failures_total: u64,
    pub expected_profile: String,
    pub compiled_profile: String,
    pub effective_profile: String,
    pub paused: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReflexCounters {
    pub proposed: u64,
    pub admitted: u64,
    pub applied: u64,
    pub shadowed: u64,
    pub omitted: u64,
    pub no_op: u64,
    pub vetoed: u64,
    pub reverted: u64,
    /// Unique process members with at least one applied effect. Rollout
    /// health uses members so a multi-effect lease is one action.
    pub members_applied: u64,
    /// Unique process members with at least one reverted effect.
    pub members_reverted: u64,
    pub failed: u64,
    /// Failures that were not recovered by another admitted effect for the
    /// same process member in the same decision batch.
    pub health_failures: u64,
    pub protected_blocked: u64,
    /// Safety invariant audit: this must remain zero because protected targets
    /// are rejected before model vetoes, deduplication, or admission.
    pub protected_admitted: u64,
    pub last_decision_latency_us: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReflexIntentKey {
    action: ReflexActionKind,
    pid: u32,
    start_sec: u64,
    start_usec: u64,
    cycle: u64,
}

#[derive(Debug, Clone)]
pub struct ReflexBroker {
    rollout: ReflexRolloutState,
    recent: VecDeque<ReflexIntentKey>,
    counters: ReflexCounters,
}

impl ReflexBroker {
    pub fn new(enabled: bool, shadow_cycles: u64, build_profile: &str) -> Self {
        Self {
            rollout: ReflexRolloutState::new(enabled, shadow_cycles, build_profile),
            recent: VecDeque::with_capacity(DEDUP_CAPACITY),
            counters: ReflexCounters::default(),
        }
    }

    pub fn active_for_test() -> Self {
        let mut broker = Self::new(true, 500, "test");
        broker.rollout.phase = ReflexRolloutPhase::Active;
        broker.rollout.blocker = "ready".to_string();
        broker
    }

    pub fn restore_json(json: &str, build_profile: &str) -> Self {
        if json.len() > 64 * 1024 {
            return Self::corrupt(build_profile, "state-oversized");
        }
        let Ok(mut rollout) = serde_json::from_str::<ReflexRolloutState>(json) else {
            return Self::corrupt(build_profile, "state-corrupt");
        };
        if rollout.schema_version == 1 {
            // Schema 1 treated a failed optional lane as a broker failure even
            // when another lane recovered the same member. Preserve rollout
            // samples, but reset only those semantically incompatible totals.
            rollout.schema_version = REFLEX_SCHEMA_VERSION;
            rollout.failures_seen = 0;
            rollout.rollback_failures_seen = 0;
            rollout.phase = ReflexRolloutPhase::Shadow;
            rollout.blocker = "warming-up".to_string();
        } else if rollout.schema_version != REFLEX_SCHEMA_VERSION {
            return Self::corrupt(build_profile, "state-schema-mismatch");
        }
        if rollout.build_profile != build_profile {
            return Self::corrupt(build_profile, "state-profile-mismatch");
        }
        rollout.shadow_cycles = rollout.shadow_cycles.max(default_shadow_cycles());
        // Action counters belong to the current daemon process. Persisted
        // baselines remain valid, but deltas must restart from this process.
        rollout.previous_applied = 0;
        rollout.previous_reverted = 0;
        if !rollout.enabled {
            rollout.phase = ReflexRolloutPhase::Disabled;
            rollout.blocker = "disabled".to_string();
        } else if rollout.valid_cycles < rollout.shadow_cycles {
            rollout.phase = ReflexRolloutPhase::Shadow;
            rollout.blocker = "warming-up".to_string();
        }
        Self {
            rollout,
            recent: VecDeque::with_capacity(DEDUP_CAPACITY),
            counters: ReflexCounters::default(),
        }
    }

    fn corrupt(build_profile: &str, blocker: &str) -> Self {
        let mut broker = Self::new(true, 500, build_profile);
        broker.rollout.blocker = blocker.to_string();
        broker
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.rollout)
    }

    pub fn rollout(&self) -> &ReflexRolloutState {
        &self.rollout
    }

    pub fn counters(&self) -> &ReflexCounters {
        &self.counters
    }

    pub fn record_applied(&mut self, count: u64) {
        self.counters.applied = self.counters.applied.saturating_add(count);
    }

    pub fn record_reverted(&mut self, count: u64) {
        self.counters.reverted = self.counters.reverted.saturating_add(count);
    }

    pub fn record_members_applied(&mut self, count: u64) {
        self.counters.members_applied = self.counters.members_applied.saturating_add(count);
    }

    pub fn record_members_reverted(&mut self, count: u64) {
        self.counters.members_reverted = self.counters.members_reverted.saturating_add(count);
    }

    pub fn record_failed(&mut self, count: u64) {
        self.counters.failed = self.counters.failed.saturating_add(count);
    }

    pub fn record_health_failed(&mut self, count: u64) {
        self.counters.health_failures = self.counters.health_failures.saturating_add(count);
    }

    pub fn record_noop(&mut self, count: u64) {
        self.counters.no_op = self.counters.no_op.saturating_add(count);
    }

    pub fn decide(&mut self, intent: &ReflexIntent, safety: ReflexSafetyContext) -> ReflexDecision {
        let started = Instant::now();
        let decision = self.decide_inner(intent, safety);
        self.counters.last_decision_latency_us =
            started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        decision
    }

    fn decide_inner(
        &mut self,
        intent: &ReflexIntent,
        safety: ReflexSafetyContext,
    ) -> ReflexDecision {
        self.counters.proposed = self.counters.proposed.saturating_add(1);
        let skipped = if !self.rollout.enabled {
            Some(ReflexBlocker::Disabled)
        } else if safety.kill_switch {
            Some(ReflexBlocker::KillSwitch)
        } else if safety.thermal_force_ecores {
            Some(ReflexBlocker::ThermalConstraint)
        } else if intent.action.requires_process() && !safety.identity_present {
            Some(ReflexBlocker::MissingIdentity)
        } else if intent.action.requires_process()
            && (!safety.identity_start_nonzero || !safety.identity_recheck_ok)
        {
            Some(ReflexBlocker::IdentityMismatch)
        } else if safety.target_protected {
            self.counters.protected_blocked = self.counters.protected_blocked.saturating_add(1);
            Some(ReflexBlocker::ProtectedTarget)
        } else if safety.target_apple_owned {
            Some(ReflexBlocker::AppleOwnedTarget)
        } else if !safety.capability_available {
            Some(ReflexBlocker::MissingCapability)
        } else if safety.existing_conflict {
            Some(ReflexBlocker::Conflict)
        } else {
            None
        };
        if let Some(blocker) = skipped {
            self.counters.omitted = self.counters.omitted.saturating_add(1);
            return ReflexDecision::Skipped(blocker);
        }
        if intent.advice.decisive_negative() {
            self.counters.vetoed = self.counters.vetoed.saturating_add(1);
            return ReflexDecision::Veto(ReflexBlocker::DecisiveModelVeto);
        }
        let key = intent.dedup_key();
        if self.recent.contains(&key) {
            self.counters.omitted = self.counters.omitted.saturating_add(1);
            return ReflexDecision::Skipped(ReflexBlocker::Duplicate);
        }
        if self.recent.len() == DEDUP_CAPACITY {
            self.recent.pop_front();
        }
        self.recent.push_back(key);
        match self.rollout.phase {
            ReflexRolloutPhase::Active => {
                self.counters.admitted = self.counters.admitted.saturating_add(1);
                ReflexDecision::Admit
            }
            ReflexRolloutPhase::Shadow => {
                self.counters.shadowed = self.counters.shadowed.saturating_add(1);
                ReflexDecision::Shadow
            }
            ReflexRolloutPhase::Disabled => {
                self.counters.omitted = self.counters.omitted.saturating_add(1);
                ReflexDecision::Skipped(ReflexBlocker::Disabled)
            }
        }
    }

    pub fn observe_health(&mut self, sample: ReflexHealthSample) {
        if !self.rollout.enabled || sample.paused {
            return;
        }
        if let Some(blocker) = profile_blocker(&self.rollout, &sample) {
            self.rollout.phase = ReflexRolloutPhase::Shadow;
            self.rollout.blocker = blocker.to_string();
            return;
        }
        if self.rollout.phase != ReflexRolloutPhase::Shadow {
            return;
        }
        if !sample.p95_cycle_ms.is_finite() || sample.p95_cycle_ms <= 0.0 {
            self.rollout.invalid_samples = self.rollout.invalid_samples.saturating_add(1);
            return;
        }
        let applied_delta = sample
            .applied_total
            .saturating_sub(self.rollout.previous_applied);
        let reverted_delta = sample
            .reverted_total
            .saturating_sub(self.rollout.previous_reverted)
            .min(applied_delta);
        self.rollout.previous_applied = sample.applied_total;
        self.rollout.previous_reverted = sample.reverted_total;
        let churn = if applied_delta == 0 {
            0.0
        } else {
            reverted_delta as f64 / applied_delta as f64
        };
        self.rollout.valid_cycles = self.rollout.valid_cycles.saturating_add(1);
        if self.rollout.baseline_samples < BASELINE_CYCLES {
            self.rollout.baseline_samples = self.rollout.baseline_samples.saturating_add(1);
            self.rollout.baseline_p95_sum += sample.p95_cycle_ms;
            self.rollout.baseline_churn_sum += churn;
        } else {
            self.rollout.candidate_samples = self.rollout.candidate_samples.saturating_add(1);
            self.rollout.candidate_p95_sum += sample.p95_cycle_ms;
            self.rollout.candidate_churn_sum += churn;
        }
        self.rollout.protected_admissions_seen = self
            .rollout
            .protected_admissions_seen
            .max(sample.protected_admissions_total);
        self.rollout.failures_seen = self.rollout.failures_seen.max(sample.failures_total);
        self.rollout.rollback_failures_seen = self
            .rollout
            .rollback_failures_seen
            .max(sample.rollback_failures_total);

        if self.rollout.valid_cycles < self.rollout.shadow_cycles {
            self.rollout.blocker = "warming-up".to_string();
            return;
        }
        self.rollout.blocker = activation_blocker(&self.rollout, &sample).to_string();
        if self.rollout.blocker == "ready" {
            self.rollout.phase = ReflexRolloutPhase::Active;
        }
    }
}

fn activation_blocker(state: &ReflexRolloutState, _sample: &ReflexHealthSample) -> &'static str {
    if state.protected_admissions_seen > 0 {
        "protected-admission"
    } else if state.failures_seen > 0 {
        "reflex-failure"
    } else if state.rollback_failures_seen > 0 {
        "rollback-failure"
    } else if state.candidate_samples == 0 {
        "candidate-window-empty"
    } else if state.candidate_p95_ms() > MAX_P95_MS {
        "p95-limit"
    } else if state.candidate_p95_ms() > state.baseline_p95_ms().max(1.0) * MAX_REGRESSION {
        "p95-regression"
    } else if state.candidate_churn()
        > (state.baseline_churn() * MAX_REGRESSION).max(MIN_CHURN_ALLOWANCE)
    {
        "churn-regression"
    } else {
        "ready"
    }
}

fn profile_blocker(
    state: &ReflexRolloutState,
    sample: &ReflexHealthSample,
) -> Option<&'static str> {
    if sample.expected_profile != state.build_profile {
        Some("expected-profile-mismatch")
    } else if sample.compiled_profile != sample.expected_profile {
        Some("compiled-profile-mismatch")
    } else if sample.effective_profile != sample.compiled_profile {
        Some("effective-profile-mismatch")
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningIdentity {
    pub pid: u32,
    pub start_sec: u64,
    pub start_usec: u64,
}

impl From<&ReflexTarget> for ReasoningIdentity {
    fn from(target: &ReflexTarget) -> Self {
        Self {
            pid: target.pid,
            start_sec: target.start_sec,
            start_usec: target.start_usec,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReasoningSnapshot<P> {
    pub cycle: u64,
    pub identity: ReasoningIdentity,
    pub payload: P,
}

impl<P> ReasoningSnapshot<P> {
    pub fn new(cycle: u64, identity: ReasoningIdentity, payload: P) -> Self {
        Self {
            cycle,
            identity,
            payload,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReasoningResult<R> {
    pub cycle: u64,
    pub identity: ReasoningIdentity,
    pub payload: R,
    pub latency_us: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReasoningLookup<R> {
    Fresh(ReasoningResult<R>),
    Pending,
    Stale { age_cycles: u64 },
    IdentityMismatch,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReasoningWorkerStats {
    pub submitted: u64,
    pub dropped: u64,
    pub completed: u64,
    pub failed: u64,
    pub deadline_misses: u64,
    pub identity_mismatches: u64,
    pub last_latency_us: u64,
    pub last_result_cycle: u64,
}

#[derive(Debug, Default)]
struct ReasoningAtomicStats {
    submitted: AtomicU64,
    dropped: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    deadline_misses: AtomicU64,
    identity_mismatches: AtomicU64,
    last_latency_us: AtomicU64,
    last_result_cycle: AtomicU64,
}

impl ReasoningAtomicStats {
    fn snapshot(&self) -> ReasoningWorkerStats {
        ReasoningWorkerStats {
            submitted: self.submitted.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            deadline_misses: self.deadline_misses.load(Ordering::Relaxed),
            identity_mismatches: self.identity_mismatches.load(Ordering::Relaxed),
            last_latency_us: self.last_latency_us.load(Ordering::Relaxed),
            last_result_cycle: self.last_result_cycle.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
struct ReasoningMailbox<P, R> {
    pending: Option<ReasoningSnapshot<P>>,
    latest: Option<ReasoningResult<R>>,
    shutdown: bool,
}

impl<P, R> Default for ReasoningMailbox<P, R> {
    fn default() -> Self {
        Self {
            pending: None,
            latest: None,
            shutdown: false,
        }
    }
}

/// Single-worker, capacity-one mailbox. Producers overwrite pending work and
/// consumers only read a completed immutable result; neither path waits for
/// model execution.
pub struct LatestReasoningWorker<P, R> {
    shared: Arc<(Mutex<ReasoningMailbox<P, R>>, Condvar)>,
    stats: Arc<ReasoningAtomicStats>,
    handle: Option<JoinHandle<()>>,
}

impl<P, R> std::fmt::Debug for LatestReasoningWorker<P, R>
where
    P: Send + 'static,
    R: Clone + Send + 'static,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LatestReasoningWorker")
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl<P, R> LatestReasoningWorker<P, R>
where
    P: Send + 'static,
    R: Clone + Send + 'static,
{
    pub fn spawn(name: &str, process: impl Fn(P) -> R + Send + 'static) -> std::io::Result<Self> {
        let shared = Arc::new((Mutex::new(ReasoningMailbox::default()), Condvar::new()));
        let stats = Arc::new(ReasoningAtomicStats::default());
        let worker_shared = Arc::clone(&shared);
        let worker_stats = Arc::clone(&stats);
        let handle = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || loop {
                let snapshot = {
                    let (lock, condvar) = &*worker_shared;
                    let mut mailbox = lock.lock().unwrap_or_else(|error| error.into_inner());
                    while mailbox.pending.is_none() && !mailbox.shutdown {
                        mailbox = condvar
                            .wait(mailbox)
                            .unwrap_or_else(|error| error.into_inner());
                    }
                    if mailbox.shutdown {
                        return;
                    }
                    mailbox.pending.take()
                };
                let Some(snapshot) = snapshot else {
                    continue;
                };
                let started = Instant::now();
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    process(snapshot.payload)
                }));
                let Ok(payload) = outcome else {
                    worker_stats.failed.fetch_add(1, Ordering::Relaxed);
                    continue;
                };
                let latency_us = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
                let result = ReasoningResult {
                    cycle: snapshot.cycle,
                    identity: snapshot.identity,
                    payload,
                    latency_us,
                };
                let (lock, _) = &*worker_shared;
                let mut mailbox = lock.lock().unwrap_or_else(|error| error.into_inner());
                if !mailbox.shutdown {
                    mailbox.latest = Some(result);
                    worker_stats.completed.fetch_add(1, Ordering::Relaxed);
                    worker_stats
                        .last_latency_us
                        .store(latency_us, Ordering::Relaxed);
                    worker_stats
                        .last_result_cycle
                        .store(snapshot.cycle, Ordering::Relaxed);
                }
            })?;
        Ok(Self {
            shared,
            stats,
            handle: Some(handle),
        })
    }

    /// Returns false when the mailbox lock itself is briefly busy. This is a
    /// deliberate no-wait fallback: the daemon keeps its deterministic path.
    pub fn submit(&self, snapshot: ReasoningSnapshot<P>) -> bool {
        let (lock, condvar) = &*self.shared;
        let mut mailbox = match lock.try_lock() {
            Ok(mailbox) => mailbox,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => {
                self.stats.dropped.fetch_add(1, Ordering::Relaxed);
                return false;
            }
        };
        if mailbox.shutdown {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if mailbox.pending.replace(snapshot).is_some() {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        }
        self.stats.submitted.fetch_add(1, Ordering::Relaxed);
        condvar.notify_one();
        true
    }

    pub fn latest_for(
        &self,
        current_cycle: u64,
        identity: ReasoningIdentity,
    ) -> ReasoningLookup<R> {
        let (lock, _) = &*self.shared;
        let mut mailbox = match lock.try_lock() {
            Ok(mailbox) => mailbox,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return ReasoningLookup::Pending,
        };
        let Some(result) = mailbox.latest.as_ref() else {
            return ReasoningLookup::Pending;
        };
        if result.identity != identity {
            self.stats
                .identity_mismatches
                .fetch_add(1, Ordering::Relaxed);
            mailbox.latest.take();
            return ReasoningLookup::IdentityMismatch;
        }
        let age_cycles = current_cycle.saturating_sub(result.cycle);
        if current_cycle < result.cycle || age_cycles > 2 {
            self.stats.deadline_misses.fetch_add(1, Ordering::Relaxed);
            mailbox.latest.take();
            return ReasoningLookup::Stale { age_cycles };
        }
        ReasoningLookup::Fresh(result.clone())
    }

    pub fn stats(&self) -> ReasoningWorkerStats {
        self.stats.snapshot()
    }
}

impl<P, R> Drop for LatestReasoningWorker<P, R> {
    fn drop(&mut self) {
        let (lock, condvar) = &*self.shared;
        let mut mailbox = lock.lock().unwrap_or_else(|error| error.into_inner());
        mailbox.shutdown = true;
        mailbox.pending = None;
        condvar.notify_all();
        drop(mailbox);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
