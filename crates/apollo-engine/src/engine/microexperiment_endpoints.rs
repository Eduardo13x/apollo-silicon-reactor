//! Bounded adapter between real `DecisionLedger` resolutions and the endpoints
//! the Microexperiment Lab needs to close a pair.
//!
//! The lab never actuates. It issues `PairDirective`s that declare *what to
//! observe*; production keeps executing through the broker, the arbiter, the
//! leases and the ledger exactly as before. This adapter is the missing wire:
//! it binds an already-authorized `ResolvedDecisionEpisode` to one outstanding
//! arm, waits for that decision's measured utility, and only then emits a
//! `TimedPairEndpoint`. It fabricates no identity, no outcome and no timestamp.
//!
//! Ordering is a deliberate one-cycle latency. `microexperiment_runtime.tick`
//! runs near the head of a daemon cycle while `ingest_cycle_events` closes at
//! the tail, so a directive issued in cycle N can only be answered from cycle
//! N+1 onwards:
//!
//! ```text
//! cycle N   : PairDirective -> normal pipeline -> DecisionLedger
//! cycle N+1 : ResolvedDecisionEpisode -> bind(arm, decision_id)
//! cycle N+h : measured utility -> TimedPairEndpoint -> lab
//! ```
//!
//! Every queue is fixed size, every entry has a deadline, and every rejection
//! has its own counter so a stalled circuit is diagnosable without a debugger.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::decision_ledger::{
    DecisionLifecycle, ReceiptAttribution, ResolvedDecisionEpisode,
};
use crate::engine::exploration_scheduler::{
    ActionClass, ExplorationContext, ExplorationMode, ExplorationOrigin,
};
use crate::engine::microexperiment_actions::{parse_action_key, ActionVariant, CanonicalAction};
use crate::engine::microexperiment_lab::{
    ArmKind, ExecutionClosure, HorizonClosure, PairDirective, PairEndpoint, PairId,
    RollbackClosure, TimedPairEndpoint, MAX_OPEN_PAIRS,
};
use crate::engine::telemetry_medallion::ActuatorFamily;

/// One arm per open pair, plus headroom for the complement issued after
/// washout. Never grows with traffic.
pub const MAX_OUTSTANDING_ARMS: usize = MAX_OPEN_PAIRS * 2;
pub const MAX_BOUND_DECISIONS: usize = MAX_OPEN_PAIRS * 2;
pub const MAX_READY_ENDPOINTS: usize = MAX_OPEN_PAIRS * 2;
pub const MAX_CONSUMED_DEDUP: usize = 128;
/// Cycles a bound decision may wait for its utility sample before the arm is
/// abandoned. Bounded well above the longest catalogued horizon.
pub const BOUND_TTL_CYCLES: u64 = 512;
/// The observation path counts as live only if the ledger delivered a batch
/// this recently. A daemon that stopped resolving decisions must not keep
/// claiming its endpoint contract is ready.
pub const OBSERVATION_LIVENESS_CYCLES: u64 = 8;

/// Why one observation did not become an endpoint. Each variant has its own
/// counter; nothing collapses into a generic "rejected" bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointReject {
    /// Action key is uncatalogued, legacy, malformed, or names another family.
    ActionKeyMismatch,
    /// Nothing is waiting on this action at all. This is the daemon's ordinary
    /// traffic passing by, not a defect: most decisions in a cycle are simply
    /// outside any open experiment. Production read 7,809 of these under the
    /// name `unknown_arm`, which made routine work look broken.
    RoutineUnclaimed,
    /// An experiment is waiting on this action, but the episode does not fit
    /// the arm role it would have to fill. This one is a real anomaly.
    InvalidExperimental,
    /// Decision id was already bound to an arm.
    Duplicate,
    /// Settled after the arm's grace deadline, or before it was issued.
    Expired,
    /// Generation or clock inconsistency: a directive dated after the current
    /// cycle. Cross-restart separation is structural — a new daemon generation
    /// builds a new adapter with an empty outstanding set.
    Epoch,
    /// Not locally attributed, not committed, cancelled, or the wrong
    /// disposition for the arm role.
    Authority,
    /// Terminal record is missing the metadata the endpoint contract needs.
    IncompleteMetadata,
    /// Bounded queue is full; the observation is dropped, never queued.
    Capacity,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EndpointAdapterCounters {
    pub registered_arms: u64,
    /// Episodes actually examined, i.e. seen while at least one arm was open.
    pub observed_episodes: u64,
    /// Episodes skipped wholesale because no arm was outstanding.
    pub episodes_skipped_idle: u64,
    /// Examined episodes whose family is outside the experiment catalog. This
    /// is routine traffic, not a failure.
    pub uncatalogued_episodes: u64,
    pub bound_decisions: u64,
    pub emitted_endpoints: u64,
    pub pending_utility: u64,
    pub rejected_action_mismatch: u64,
    /// Ordinary daemon traffic no open experiment was waiting on. Expected to
    /// be large and to mean nothing is wrong.
    pub routine_unclaimed: u64,
    /// An episode for an action an experiment *was* waiting on, that could not
    /// fill the arm role. Small and worth reading.
    pub invalid_experimental: u64,
    pub rejected_duplicate: u64,
    pub rejected_expired: u64,
    pub rejected_epoch: u64,
    pub rejected_authority: u64,
    pub rejected_incomplete_metadata: u64,
    pub dropped_capacity: u64,
    pub expired_arms: u64,
    pub rollback_observed: u64,
    pub rollback_failed: u64,
}

/// One arm that just acquired a real decision identity. The caller opens an
/// outcome-measurement window for it; the adapter itself measures nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NewArmBinding {
    pub decision_id: u64,
    pub family: ActuatorFamily,
    pub horizon_cycles: u32,
}

/// Measured outcome for one decision identity. The daemon derives these from
/// `TelemetryMedallion::drain_lab_utility`, which measures each arm's window
/// without admitting it as evidence, and is fresh-only so a restart cannot
/// replay a stale measurement as new experimental evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointUtilitySample {
    pub decision_id: u64,
    pub utility_micros: i64,
    pub resolved_cycle: u64,
    pub confounded: bool,
}

#[derive(Debug, Clone)]
struct OutstandingArm {
    pair_id: PairId,
    arm: ArmKind,
    /// Shadow arms are registered purely so an episode can bind to them. They
    /// must never surface through `outstanding_control_actions`, because a
    /// withheld control window changes what the machine does — and Shadow's
    /// whole contract is that it changes nothing.
    observe_only: bool,
    canonical: CanonicalAction,
    action_key: String,
    family: ActuatorFamily,
    action_class: ActionClass,
    context: ExplorationContext,
    stratum_hash: u64,
    horizon_cycles: u32,
    issued_cycle: u64,
    complete_not_before_cycle: u64,
    expires_after_cycle: u64,
}

#[derive(Debug, Clone)]
struct BoundDecision {
    arm: OutstandingArm,
    decision_id: u64,
    settled_cycle: u64,
    execution: ExecutionClosure,
    rollback: RollbackClosure,
    bound_cycle: u64,
}

/// Bounded, deterministic ledger-to-lab endpoint adapter.
#[derive(Debug, Clone)]
pub struct MicroexperimentEndpointAdapter {
    origin: ExplorationOrigin,
    epoch: u64,
    outstanding: VecDeque<OutstandingArm>,
    bound: VecDeque<BoundDecision>,
    ready: VecDeque<TimedPairEndpoint>,
    new_bindings: VecDeque<NewArmBinding>,
    consumed: VecDeque<u64>,
    last_episode_cycle: Option<u64>,
    counters: EndpointAdapterCounters,
}

impl MicroexperimentEndpointAdapter {
    /// `epoch` is the daemon generation token. Observations and directives from
    /// another generation are never mixed.
    pub fn new(origin: ExplorationOrigin, epoch: u64) -> Self {
        Self {
            origin,
            epoch,
            outstanding: VecDeque::with_capacity(MAX_OUTSTANDING_ARMS),
            bound: VecDeque::with_capacity(MAX_BOUND_DECISIONS),
            ready: VecDeque::with_capacity(MAX_READY_ENDPOINTS),
            new_bindings: VecDeque::with_capacity(MAX_BOUND_DECISIONS),
            consumed: VecDeque::with_capacity(MAX_CONSUMED_DEDUP),
            last_episode_cycle: None,
            counters: EndpointAdapterCounters::default(),
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn counters(&self) -> EndpointAdapterCounters {
        self.counters
    }

    pub fn outstanding_len(&self) -> usize {
        self.outstanding.len()
    }

    pub fn bound_len(&self) -> usize {
        self.bound.len()
    }

    pub fn ready_len(&self) -> usize {
        self.ready.len()
    }

    /// The observation path is live only while the ledger keeps delivering
    /// resolved episodes. A daemon that has never ingested a batch, or stopped
    /// doing so, reports `false` and the lab abstains.
    pub fn observation_path_live(&self, cycle: u64) -> bool {
        self.last_episode_cycle.is_some_and(|last| {
            cycle >= last && cycle.saturating_sub(last) <= OBSERVATION_LIVENESS_CYCLES
        })
    }

    /// Real endpoint-contract condition. This is what `endpoint_contract_ready`
    /// must be derived from: a known origin, a matching generation, and a
    /// demonstrably live observation path.
    pub fn contract_ready(&self, cycle: u64, daemon_epoch: u64) -> bool {
        self.origin.is_known() && self.epoch == daemon_epoch && self.observation_path_live(cycle)
    }

    /// Record the arms the lab wants observed. Re-registering the same
    /// `(pair, arm, issued_cycle)` is a no-op, so a repeated directive cannot
    /// double-bind a decision.
    pub fn register_directives(&mut self, directives: &[PairDirective], cycle: u64) {
        for directive in directives {
            if directive.issued_cycle > cycle {
                self.counters.rejected_epoch = self.counters.rejected_epoch.saturating_add(1);
                continue;
            }
            let Ok(canonical) = parse_action_key(&directive.action_key) else {
                self.counters.rejected_action_mismatch =
                    self.counters.rejected_action_mismatch.saturating_add(1);
                continue;
            };
            if canonical.family != directive.family
                || canonical.action_class != directive.action_class
            {
                self.counters.rejected_action_mismatch =
                    self.counters.rejected_action_mismatch.saturating_add(1);
                continue;
            }
            if self.outstanding.iter().any(|arm| {
                arm.pair_id == directive.pair_id
                    && arm.arm == directive.arm
                    && arm.issued_cycle == directive.issued_cycle
            }) {
                continue;
            }
            if self.outstanding.len() >= MAX_OUTSTANDING_ARMS {
                self.counters.dropped_capacity = self.counters.dropped_capacity.saturating_add(1);
                continue;
            }
            self.outstanding.push_back(OutstandingArm {
                pair_id: directive.pair_id,
                arm: directive.arm,
                observe_only: directive.observe_only,
                canonical,
                action_key: directive.action_key.clone(),
                family: directive.family,
                action_class: directive.action_class,
                context: directive.context,
                stratum_hash: directive.stratum_hash,
                horizon_cycles: directive
                    .complete_not_before_cycle
                    .saturating_sub(directive.issued_cycle)
                    .min(u64::from(u32::MAX)) as u32,
                issued_cycle: directive.issued_cycle,
                complete_not_before_cycle: directive.complete_not_before_cycle,
                expires_after_cycle: directive.expires_after_cycle,
            });
            self.counters.registered_arms = self.counters.registered_arms.saturating_add(1);
        }
    }

    /// Catalogued actions with an outstanding control arm: the lab is waiting
    /// to observe a window in which this exact action was deliberately
    /// withheld. Actuator owners consult this to decide whether declining a
    /// lease is currently useful; an empty result means never withhold.
    ///
    /// This grants no authority. It cannot make an actuator *do* anything, only
    /// skip something it was already free to skip.
    pub fn outstanding_control_actions(&self) -> Vec<CanonicalAction> {
        let mut actions = Vec::with_capacity(self.outstanding.len());
        for arm in &self.outstanding {
            if arm.observe_only {
                continue;
            }
            if arm.arm == ArmKind::Control && !actions.contains(&arm.canonical) {
                actions.push(arm.canonical);
            }
        }
        actions
    }

    /// Drop every arm belonging to a pair the lab already closed or invalidated.
    pub fn forget_pair(&mut self, pair_id: PairId) {
        let dropped: Vec<u64> = self
            .bound
            .iter()
            .filter(|bound| bound.arm.pair_id == pair_id)
            .map(|bound| bound.decision_id)
            .collect();
        self.outstanding.retain(|arm| arm.pair_id != pair_id);
        self.bound.retain(|bound| bound.arm.pair_id != pair_id);
        self.ready.retain(|ready| ready.pair_id != pair_id);
        self.new_bindings
            .retain(|binding| !dropped.contains(&binding.decision_id));
    }

    /// Bind resolved decisions from the previous cycle to outstanding arms.
    /// This never emits an endpoint: the measured utility has not resolved yet.
    ///
    /// A rejection counter is only meaningful for a decision the lab was
    /// actually waiting on. With no outstanding arm — the normal state in
    /// Shadow — the whole batch is skipped rather than classified, so
    /// `rejected_action_mismatch` keeps meaning "a real key mismatch" instead
    /// of drowning in the routine `boost:`/`throttle:` traffic every cycle
    /// carries. It also keeps the hot path free of per-episode parsing when no
    /// experiment is open.
    pub fn observe_episodes(&mut self, episodes: &[ResolvedDecisionEpisode], cycle: u64) {
        // Liveness means the ledger delivered something, not that this function
        // was reached. The previous unconditional stamp made
        // `observation_path_live` — and through it `endpoint_contract_ready` —
        // report a healthy path on a daemon that had ingested nothing but empty
        // batches, which is precisely the claim the doc comment above denies.
        if episodes.is_empty() {
            return;
        }
        self.last_episode_cycle = Some(cycle);
        if self.outstanding.is_empty() {
            self.counters.episodes_skipped_idle = self
                .counters
                .episodes_skipped_idle
                .saturating_add(episodes.len() as u64);
            return;
        }
        for episode in episodes {
            self.counters.observed_episodes = self.counters.observed_episodes.saturating_add(1);
            if let Err(reject) = self.bind_episode(episode, cycle) {
                self.count_reject(reject);
            }
        }
    }

    fn bind_episode(
        &mut self,
        episode: &ResolvedDecisionEpisode,
        cycle: u64,
    ) -> Result<(), EndpointReject> {
        if episode.id.0 == 0 {
            return Err(EndpointReject::IncompleteMetadata);
        }
        // An uncatalogued family is not a mismatch: most decisions in a cycle
        // are simply outside the experiment catalog. Only a key that names a
        // catalogued family can be a genuine identity failure.
        let canonical = match parse_action_key(&episode.envelope.action_key) {
            Ok(canonical) => canonical,
            Err(_) => {
                self.counters.uncatalogued_episodes =
                    self.counters.uncatalogued_episodes.saturating_add(1);
                return Ok(());
            }
        };
        // Nothing is waiting on this action, so there is no experiment to
        // contaminate and nothing to diagnose.
        if !self
            .outstanding
            .iter()
            .any(|arm| arm.canonical.matches(canonical))
        {
            return Err(EndpointReject::RoutineUnclaimed);
        }
        if self.consumed.contains(&episode.id.0)
            || self
                .bound
                .iter()
                .any(|bound| bound.decision_id == episode.id.0)
        {
            return Err(EndpointReject::Duplicate);
        }
        let closures = endpoint_closures(episode, canonical)?;
        let index = self
            .outstanding
            .iter()
            .position(|arm| {
                arm.canonical.matches(canonical)
                    && arm.arm == closures.arm
                    && episode.settled_cycle >= arm.issued_cycle
                    && episode.settled_cycle <= arm.expires_after_cycle
            })
            .ok_or_else(|| {
                if self
                    .outstanding
                    .iter()
                    .any(|arm| arm.canonical.matches(canonical) && arm.arm == closures.arm)
                {
                    // Right action, right role, wrong window: late.
                    EndpointReject::Expired
                } else {
                    // Right action, wrong role: an experiment was waiting and
                    // this episode cannot fill the arm it would have to fill.
                    EndpointReject::InvalidExperimental
                }
            })?;
        if self.bound.len() >= MAX_BOUND_DECISIONS {
            return Err(EndpointReject::Capacity);
        }
        let arm = self.outstanding.remove(index).expect("index from position");
        push_bounded(&mut self.consumed, episode.id.0, MAX_CONSUMED_DEDUP);
        match closures.rollback {
            RollbackClosure::Failed => {
                self.counters.rollback_failed = self.counters.rollback_failed.saturating_add(1)
            }
            RollbackClosure::Succeeded | RollbackClosure::NotRequiredNonKernel => {
                self.counters.rollback_observed = self.counters.rollback_observed.saturating_add(1)
            }
            _ => {}
        }
        push_bounded(
            &mut self.new_bindings,
            NewArmBinding {
                decision_id: episode.id.0,
                family: arm.family,
                horizon_cycles: arm.horizon_cycles,
            },
            MAX_BOUND_DECISIONS,
        );
        self.bound.push_back(BoundDecision {
            arm,
            decision_id: episode.id.0,
            settled_cycle: episode.settled_cycle,
            execution: closures.execution,
            rollback: closures.rollback,
            bound_cycle: cycle,
        });
        self.counters.bound_decisions = self.counters.bound_decisions.saturating_add(1);
        self.counters.pending_utility = self.bound.len() as u64;
        Ok(())
    }

    /// Turn measured outcomes into endpoints. A bound arm without a sample stays
    /// pending; it is never closed with an invented utility.
    pub fn observe_utilities(&mut self, samples: &[EndpointUtilitySample], cycle: u64) {
        for sample in samples {
            let Some(index) = self
                .bound
                .iter()
                .position(|bound| bound.decision_id == sample.decision_id)
            else {
                continue;
            };
            let bound = self.bound.remove(index).expect("index from position");
            // The endpoint closes no earlier than its own settle and no later
            // than now: forward-dated evidence cannot buy horizon credit.
            let completed_cycle = sample
                .resolved_cycle
                .max(bound.settled_cycle)
                .min(cycle.max(bound.settled_cycle));
            let horizon = if completed_cycle > bound.arm.expires_after_cycle {
                HorizonClosure::Expired
            } else if sample.confounded {
                HorizonClosure::Confounded
            } else if completed_cycle >= bound.arm.complete_not_before_cycle {
                HorizonClosure::Complete
            } else {
                HorizonClosure::Incomplete
            };
            if self.ready.len() >= MAX_READY_ENDPOINTS {
                self.counters.dropped_capacity = self.counters.dropped_capacity.saturating_add(1);
                continue;
            }
            self.ready.push_back(TimedPairEndpoint {
                pair_id: bound.arm.pair_id,
                issued_cycle: bound.arm.issued_cycle,
                completed_cycle,
                endpoint: PairEndpoint {
                    arm: bound.arm.arm,
                    origin: self.origin,
                    family: bound.arm.family,
                    action_class: bound.arm.action_class,
                    context: bound.arm.context,
                    action_key: bound.arm.action_key.clone(),
                    stratum_hash: bound.arm.stratum_hash,
                    horizon_cycles: bound.arm.horizon_cycles,
                    decision_id: bound.decision_id,
                    observed_local: true,
                    synthetic: false,
                    execution: bound.execution,
                    horizon,
                    rollback: bound.rollback,
                    utility_micros: sample.utility_micros,
                },
            });
            self.counters.emitted_endpoints = self.counters.emitted_endpoints.saturating_add(1);
        }
        self.counters.pending_utility = self.bound.len() as u64;
    }

    /// Expire arms and bindings that outlived their deadline. Bounded and
    /// deterministic; keeps insertion order for everything retained.
    pub fn prune(&mut self, cycle: u64) {
        let before = self.outstanding.len() + self.bound.len();
        self.outstanding
            .retain(|arm| cycle <= arm.expires_after_cycle);
        self.bound
            .retain(|bound| cycle.saturating_sub(bound.bound_cycle) <= BOUND_TTL_CYCLES);
        let dropped = before.saturating_sub(self.outstanding.len() + self.bound.len()) as u64;
        self.counters.expired_arms = self.counters.expired_arms.saturating_add(dropped);
        self.counters.pending_utility = self.bound.len() as u64;
    }

    /// Hand the lab every endpoint that closed since the previous drain.
    pub fn drain_ready(&mut self) -> Vec<TimedPairEndpoint> {
        self.ready.drain(..).collect()
    }

    /// Arms bound since the previous drain, so the caller can start measuring
    /// their outcome window. Draining twice yields each binding once.
    pub fn drain_new_bindings(&mut self) -> Vec<NewArmBinding> {
        self.new_bindings.drain(..).collect()
    }

    fn count_reject(&mut self, reject: EndpointReject) {
        let counter = match reject {
            EndpointReject::ActionKeyMismatch => &mut self.counters.rejected_action_mismatch,
            EndpointReject::RoutineUnclaimed => &mut self.counters.routine_unclaimed,
            EndpointReject::InvalidExperimental => &mut self.counters.invalid_experimental,
            EndpointReject::Duplicate => &mut self.counters.rejected_duplicate,
            EndpointReject::Expired => &mut self.counters.rejected_expired,
            EndpointReject::Epoch => &mut self.counters.rejected_epoch,
            EndpointReject::Authority => &mut self.counters.rejected_authority,
            EndpointReject::IncompleteMetadata => &mut self.counters.rejected_incomplete_metadata,
            EndpointReject::Capacity => &mut self.counters.dropped_capacity,
        };
        *counter = counter.saturating_add(1);
    }
}

/// Lock-free handoff of the lab's open control arms to the actuator owners.
///
/// A set bit means "the lab currently has an open control arm for this exact
/// catalogued action". It is advisory in one direction only: an owner may use
/// it to *decline* work it was already free to decline, never to perform work
/// it was not already authorized to perform. The lab therefore gains no
/// actuation authority, and a lab crash simply leaves the mask at zero, which
/// reads as "never withhold".
///
/// Follows the `shadow_signals` idiom: one global static, no local copies.
pub static CONTROL_WITHHOLD_REQUESTS: AtomicU64 = AtomicU64::new(0);
/// Times an actuator owner actually withheld an action for a control arm.
pub static CONTROL_WITHHOLDS_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Stable bit index for one catalogued action. Uncatalogued combinations have
/// no bit and therefore can never be requested.
fn withhold_bit(action: CanonicalAction) -> Option<u32> {
    match (action.family, action.action_class, action.variant) {
        (
            ActuatorFamily::InteractionQos,
            ActionClass::InteractionForeground,
            ActionVariant::Short,
        ) => Some(0),
        (
            ActuatorFamily::InteractionQos,
            ActionClass::InteractionForeground,
            ActionVariant::Standard,
        ) => Some(1),
        (
            ActuatorFamily::InteractionQos,
            ActionClass::InteractionForeground,
            ActionVariant::Long,
        ) => Some(2),
        (ActuatorFamily::MarkovPrewarm, ActionClass::MarkovPredictedApp, ActionVariant::None) => {
            Some(3)
        }
        _ => None,
    }
}

/// Replace the published mask. Called once per cycle by the daemon loop, so a
/// closed or expired pair stops requesting a withhold on the very next cycle.
pub fn publish_control_withholds(actions: &[CanonicalAction]) {
    let mask = actions
        .iter()
        .filter_map(|action| withhold_bit(*action))
        .fold(0_u64, |mask, bit| mask | (1 << bit));
    CONTROL_WITHHOLD_REQUESTS.store(mask, Ordering::Relaxed);
}

/// True when the lab is waiting to observe this action deliberately withheld.
pub fn control_withhold_requested(action: CanonicalAction) -> bool {
    withhold_bit(action)
        .is_some_and(|bit| CONTROL_WITHHOLD_REQUESTS.load(Ordering::Relaxed) & (1 << bit) != 0)
}

/// Record that an owner withheld one action for a control arm.
pub fn record_control_withhold() {
    CONTROL_WITHHOLDS_TOTAL.fetch_add(1, Ordering::Relaxed);
}

pub fn control_withholds_total() -> u64 {
    CONTROL_WITHHOLDS_TOTAL.load(Ordering::Relaxed)
}

struct EndpointClosures {
    arm: ArmKind,
    execution: ExecutionClosure,
    rollback: RollbackClosure,
}

/// Decide whether one resolved episode may act as a control or treatment
/// endpoint, and with which closures.
///
/// `ResolvedDecisionEpisode::authority_eligible` is the ledger's own strict
/// filter, but it is `true` only for `Applied` receipts with a clean treatment.
/// A bounded exploration lease resolves its single correlated episode with the
/// *release* disposition (`Reverted`/`Failed`), and a control arm resolves as
/// `NoOp`/`Rejected`, so neither is ever `authority_eligible`. This function is
/// the equivalent condition for those arms: it still demands local attribution,
/// valid and committed exploration metadata, no cancellation, and the exact
/// disposition the arm role implies. `authority_eligible` remains mandatory for
/// the one case it does cover — a plain `Applied` treatment with no exploration
/// metadata.
///
/// Control arms are checked against an explicit predicate rather than
/// `ExplorationMetadata::valid`, because `valid` encodes the *actuation*
/// catalog, whose only control entry today is `Boost`/`BoostOmission`. The
/// observation contract is a separate concern: it asks for a correlated,
/// uncancelled, control-mode decision that names this exact catalogued action.
fn endpoint_closures(
    episode: &ResolvedDecisionEpisode,
    canonical: CanonicalAction,
) -> Result<EndpointClosures, EndpointReject> {
    let family = canonical.family;
    let attribution = episode
        .envelope
        .receipt
        .as_ref()
        .and_then(|receipt| receipt.attribution.as_ref())
        .or(episode.envelope.terminal_attribution.as_ref())
        .ok_or(EndpointReject::IncompleteMetadata)?;
    if !ReceiptAttribution::grants_local_authority(attribution) {
        return Err(EndpointReject::Authority);
    }
    let lifecycle = episode.envelope.lifecycle;
    let exploration = episode.envelope.exploration.as_ref();
    if let Some(metadata) = exploration {
        if metadata.cancelled.is_some() {
            return Err(EndpointReject::Authority);
        }
        if metadata.key.mode == ExplorationMode::Treatment {
            if !metadata.valid() || !metadata.treatment || !metadata.committed {
                return Err(EndpointReject::Authority);
            }
            let rollback = match (family, lifecycle) {
                // A cache-only pre-warm holds no kernel state; its release is
                // the non-kernel closure the lab expects.
                (ActuatorFamily::MarkovPrewarm, DecisionLifecycle::Reverted)
                | (ActuatorFamily::MarkovPrewarm, DecisionLifecycle::Expired) => {
                    RollbackClosure::NotRequiredNonKernel
                }
                (ActuatorFamily::MarkovPrewarm, DecisionLifecycle::Failed) => {
                    RollbackClosure::Failed
                }
                (_, DecisionLifecycle::Reverted) => RollbackClosure::Succeeded,
                (_, DecisionLifecycle::Failed) => RollbackClosure::Failed,
                // The lease is still held: the rollback fact has not closed, so
                // there is nothing authoritative to observe yet.
                _ => return Err(EndpointReject::Authority),
            };
            return Ok(EndpointClosures {
                arm: ArmKind::Treatment,
                execution: ExecutionClosure::Applied,
                rollback,
            });
        }
        // Control arm: the action was deliberately withheld. A natural
        // observation is not a control and never closes a pair.
        if metadata.key.mode != ExplorationMode::Control
            || metadata.treatment
            || metadata.correlation.0 == 0
            || metadata.key.family != family
            || metadata.key.action_class != canonical.action_class
        {
            return Err(EndpointReject::Authority);
        }
        return match lifecycle {
            DecisionLifecycle::NoOp | DecisionLifecycle::Rejected => Ok(EndpointClosures {
                arm: ArmKind::Control,
                execution: ExecutionClosure::NoOp,
                rollback: RollbackClosure::NotRequiredNonKernel,
            }),
            _ => Err(EndpointReject::Authority),
        };
    }
    // No exploration metadata: only the ledger's own authority filter can
    // admit it, and only for a family whose treatment needs no kernel rollback.
    if !episode.authority_eligible || lifecycle != DecisionLifecycle::Applied {
        return Err(EndpointReject::Authority);
    }
    if family != ActuatorFamily::MarkovPrewarm {
        return Err(EndpointReject::Authority);
    }
    Ok(EndpointClosures {
        arm: ArmKind::Treatment,
        execution: ExecutionClosure::Applied,
        rollback: RollbackClosure::NotRequiredNonKernel,
    })
}

fn push_bounded<T>(queue: &mut VecDeque<T>, value: T, cap: usize) {
    if queue.len() >= cap {
        queue.pop_front();
    }
    queue.push_back(value);
}
