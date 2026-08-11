use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};

use serde::{Deserialize, Deserializer, Serialize};

pub const MAX_PENDING_DECISIONS: usize = 192;
pub const MAX_RECENT_DECISIONS: usize = 64;
pub const MAX_EPISODIC_DECISIONS: usize = 128;
pub const MAX_CANDIDATE_ALTERNATIVES: usize = 8;
pub const MAX_ADVISER_CONTRIBUTIONS: usize = 8;
/// One daemon cycle can contain the broker's bounded 512-action batch plus
/// bounded side-channel and coordination receipts.
pub const MAX_CYCLE_DECISION_EVENTS: usize = 640;
const MAX_RETAINED_DECISION_IDS: usize =
    MAX_PENDING_DECISIONS + MAX_RECENT_DECISIONS + MAX_EPISODIC_DECISIONS;

const MAX_PREDICTIONS: usize = 8;
const MAX_ACTION_KEY_CHARS: usize = 320;
const MAX_TARGET_CHARS: usize = 256;
const MAX_SOURCE_CHARS: usize = 48;
const MAX_REASON_CHARS: usize = 160;

/// Stable, local identity for one proposed Apollo decision.
#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, Hash, PartialOrd, Ord,
)]
#[serde(transparent)]
pub struct DecisionId(pub u64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DecisionLifecycle {
    #[default]
    Proposed,
    Rejected,
    Vetoed,
    Blocked,
    Executing,
    Applied,
    Failed,
    NoOp,
    Reverted,
    Expired,
    Settled,
}

impl DecisionLifecycle {
    fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Rejected
                | Self::Vetoed
                | Self::Blocked
                | Self::Applied
                | Self::Failed
                | Self::NoOp
                | Self::Reverted
                | Self::Expired
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct CandidateAlternative {
    pub action_key: String,
    pub target: String,
    pub expected_utility: f64,
    pub uncertainty: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(default)]
pub struct HierarchyCoordinates {
    pub level: u8,
    pub parent: Option<DecisionId>,
    pub cohort: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct PredictionRecord {
    pub source: String,
    pub expected_utility: f64,
    pub uncertainty: f64,
    pub horizon_cycles: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct AdviserContribution {
    pub adviser: String,
    pub support: f64,
    pub uncertainty: f64,
}

/// Explicit provenance for an execution receipt. Imported or omitted
/// provenance is descriptive only and cannot create local learning authority.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ReceiptAttribution {
    Local {
        source: String,
    },
    #[default]
    Imported,
}

impl ReceiptAttribution {
    pub fn local(source: impl Into<String>) -> Self {
        Self::Local {
            source: bounded_text(&source.into(), MAX_SOURCE_CHARS),
        }
    }

    fn grants_local_authority(&self) -> bool {
        matches!(self, Self::Local { source } if !source.is_empty())
    }

    fn bounded(self) -> Self {
        match self {
            Self::Local { source } => Self::local(source),
            Self::Imported => Self::Imported,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionDisposition {
    #[default]
    Applied,
    Blocked,
    Failed,
    NoOp,
    Reverted,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct ExecutionReceipt {
    pub receipt_id: u64,
    pub disposition: ExecutionDisposition,
    pub observed_cycle: u64,
    pub attribution: Option<ReceiptAttribution>,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct DecisionProposal {
    pub action_key: String,
    pub target: String,
    pub proposed_cycle: u64,
    pub expires_cycle: u64,
    pub alternatives: Vec<CandidateAlternative>,
    pub hierarchy: HierarchyCoordinates,
    pub predictions: Vec<PredictionRecord>,
    pub adviser_contributions: Vec<AdviserContribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct DecisionEnvelope {
    pub id: DecisionId,
    pub action_key: String,
    pub target: String,
    pub proposed_cycle: u64,
    pub expires_cycle: u64,
    pub alternatives: Vec<CandidateAlternative>,
    pub hierarchy: HierarchyCoordinates,
    pub predictions: Vec<PredictionRecord>,
    pub adviser_contributions: Vec<AdviserContribution>,
    pub lifecycle: DecisionLifecycle,
    pub terminal_reason: String,
    /// Provenance for terminal outcomes that do not carry an execution
    /// receipt, including rejection, veto, and expiry.
    pub terminal_attribution: Option<ReceiptAttribution>,
    pub receipt: Option<ExecutionReceipt>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct ResolvedDecisionEpisode {
    pub id: DecisionId,
    pub lifecycle: DecisionLifecycle,
    pub settled_cycle: u64,
    pub authority_eligible: bool,
    pub envelope: DecisionEnvelope,
}

/// Terminal result emitted by an existing actuator owner. The event is a
/// transport record only; it grants no actuation authority.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ActuatorDecisionOutcome {
    #[default]
    Applied,
    Pending,
    Blocked,
    Failed,
    NoOp,
    Reverted,
    Rejected,
    Vetoed,
    Expired,
}

/// Cycle-local proposal and exact terminal outcome produced at an existing
/// actuator boundary. `DecisionLedger` assigns the stable `DecisionId` while
/// ingesting the single bounded batch owned by the daemon loop.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct ActuatorDecisionEvent {
    pub proposal: DecisionProposal,
    /// Bounded asynchronous command identity. A pending launch and its later
    /// completion carry the same correlation so the ledger reuses one
    /// `DecisionId` across control-loop cycles.
    pub correlation_id: Option<u64>,
    /// Cycle in which the terminal result became observable. This differs
    /// from `proposal.proposed_cycle` for asynchronous completions.
    pub observed_cycle: u64,
    pub outcome: ActuatorDecisionOutcome,
    pub attribution: ReceiptAttribution,
    pub detail: String,
}

impl ActuatorDecisionEvent {
    pub fn local(
        action_key: impl Into<String>,
        target: impl Into<String>,
        cycle: u64,
        outcome: ActuatorDecisionOutcome,
        source: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            proposal: DecisionProposal {
                action_key: bounded_text(&action_key.into(), MAX_ACTION_KEY_CHARS),
                target: bounded_text(&target.into(), MAX_TARGET_CHARS),
                proposed_cycle: cycle,
                ..DecisionProposal::default()
            },
            correlation_id: None,
            observed_cycle: cycle,
            outcome,
            attribution: ReceiptAttribution::local(source),
            detail: bounded_text(&detail.into(), MAX_REASON_CHARS),
        }
    }

    pub fn imported(
        action_key: impl Into<String>,
        target: impl Into<String>,
        cycle: u64,
        outcome: ActuatorDecisionOutcome,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            proposal: DecisionProposal {
                action_key: bounded_text(&action_key.into(), MAX_ACTION_KEY_CHARS),
                target: bounded_text(&target.into(), MAX_TARGET_CHARS),
                proposed_cycle: cycle,
                ..DecisionProposal::default()
            },
            correlation_id: None,
            observed_cycle: cycle,
            outcome,
            attribution: ReceiptAttribution::Imported,
            detail: bounded_text(&detail.into(), MAX_REASON_CHARS),
        }
    }

    pub fn with_expiry(mut self, expires_cycle: u64) -> Self {
        self.proposal.expires_cycle = expires_cycle;
        self
    }

    pub fn with_hierarchy(mut self, hierarchy: HierarchyCoordinates) -> Self {
        self.proposal.hierarchy = hierarchy;
        self
    }

    pub fn with_correlation(mut self, correlation_id: u64) -> Self {
        self.correlation_id = (correlation_id > 0).then_some(correlation_id);
        self
    }

    pub fn with_prediction(mut self, prediction: PredictionRecord) -> Self {
        if self.proposal.predictions.len() < MAX_PREDICTIONS {
            self.proposal.predictions.push(prediction);
        }
        self
    }

    pub fn set_cycle(&mut self, cycle: u64) {
        self.proposal.proposed_cycle = cycle;
        self.observed_cycle = cycle;
    }
}

/// Bounded, cycle-local handoff from actuator owners to the daemon's single
/// ledger ingestion point. It contains no lock and performs no I/O.
#[derive(Debug, Clone)]
pub struct CycleDecisionEvents {
    events: Vec<ActuatorDecisionEvent>,
    dropped_total: u64,
}

impl Default for CycleDecisionEvents {
    fn default() -> Self {
        Self {
            events: Vec::with_capacity(32),
            dropped_total: 0,
        }
    }
}

impl CycleDecisionEvents {
    pub fn push(&mut self, event: ActuatorDecisionEvent) -> bool {
        if self.events.len() >= MAX_CYCLE_DECISION_EVENTS {
            self.dropped_total = self.dropped_total.saturating_add(1);
            return false;
        }
        self.events.push(event);
        true
    }

    pub fn extend(&mut self, events: impl IntoIterator<Item = ActuatorDecisionEvent>) {
        for event in events {
            self.push(event);
        }
    }

    pub fn extend_at_cycle(&mut self, events: &[ActuatorDecisionEvent], cycle: u64) {
        for event in events {
            let mut event = event.clone();
            event.set_cycle(cycle);
            self.push(event);
        }
    }

    /// Copy a producer's retained events while preserving its honest nested
    /// drop count. The outer buffer may add further drops of its own.
    pub fn extend_buffer(&mut self, source: &CycleDecisionEvents) {
        self.dropped_total = self.dropped_total.saturating_add(source.dropped_total);
        self.extend(source.events.iter().cloned());
    }

    pub fn extend_buffer_at_cycle(&mut self, source: &CycleDecisionEvents, cycle: u64) {
        self.dropped_total = self.dropped_total.saturating_add(source.dropped_total);
        self.extend_at_cycle(source.as_slice(), cycle);
    }

    /// Ensure overflow itself receives a DecisionId. When the buffer is full,
    /// one retained event is summarized as an additional drop so the summary
    /// always fits without allocating beyond the fixed cycle capacity.
    pub fn seal_overflow_summary(&mut self, cycle: u64) -> bool {
        if self.dropped_total == 0 {
            return false;
        }
        if self.events.len() >= MAX_CYCLE_DECISION_EVENTS {
            self.events.pop();
            self.dropped_total = self.dropped_total.saturating_add(1);
        }
        self.events.push(ActuatorDecisionEvent::local(
            "decision_events:overflow",
            "cycle-buffer",
            cycle,
            ActuatorDecisionOutcome::Failed,
            "decision-event-buffer",
            format!("dropped={}", self.dropped_total),
        ));
        true
    }

    pub fn as_slice(&self) -> &[ActuatorDecisionEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn dropped_total(&self) -> u64 {
        self.dropped_total
    }

    fn drain(&mut self) -> impl Iterator<Item = ActuatorDecisionEvent> + '_ {
        self.events.drain(..)
    }
}

/// Build one cohort-level receipt when multiple actuator families applied in
/// the same bounded cycle batch. Deterministic bounded sets avoid pairwise
/// action comparisons.
pub fn coordinated_action_event(
    events: &CycleDecisionEvents,
    cycle: u64,
) -> Option<ActuatorDecisionEvent> {
    let mut families = BTreeSet::new();
    let mut action_keys = BTreeSet::new();
    for event in events.as_slice() {
        if event.outcome != ActuatorDecisionOutcome::Applied {
            continue;
        }
        let family = event
            .proposal
            .action_key
            .split_once(':')
            .map_or(event.proposal.action_key.as_str(), |(family, _)| family);
        if family == "coordinated" {
            continue;
        }
        families.insert(family.to_string());
        action_keys.insert(event.proposal.action_key.clone());
    }
    if families.len() < 2 {
        return None;
    }
    let action_key = format!(
        "coordinated:{}",
        families.into_iter().collect::<Vec<_>>().join("+")
    );
    let mut target = String::new();
    for key in action_keys {
        let separator_len = usize::from(!target.is_empty());
        if target
            .len()
            .saturating_add(separator_len)
            .saturating_add(key.len())
            > MAX_TARGET_CHARS
        {
            break;
        }
        if !target.is_empty() {
            target.push('|');
        }
        target.push_str(&key);
    }
    Some(
        ActuatorDecisionEvent::local(
            action_key,
            target,
            cycle,
            ActuatorDecisionOutcome::Applied,
            "coordinated-audit",
            "multiple actuator families applied in one cycle",
        )
        .with_hierarchy(HierarchyCoordinates {
            level: 1,
            cohort: cycle,
            ..HierarchyCoordinates::default()
        }),
    )
}

/// Bounded, per-owner ledger. It is intentionally not global: callers keep a
/// cycle-local instance or their existing owner lock, avoiding new hot-path
/// synchronization and kernel authority.
#[derive(Debug, Clone, Serialize)]
pub struct DecisionLedger {
    next_id: u64,
    pending_correlations: HashMap<u64, DecisionId>,
    pending: HashMap<DecisionId, DecisionEnvelope>,
    pending_order: VecDeque<DecisionId>,
    recent: VecDeque<ResolvedDecisionEpisode>,
    episodic: VecDeque<ResolvedDecisionEpisode>,
    duplicate_receipts_total: u64,
    expired_total: u64,
    unattributed_applied_total: u64,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct DecisionLedgerPersisted {
    next_id: u64,
    pending_correlations: HashMap<u64, DecisionId>,
    pending: HashMap<DecisionId, DecisionEnvelope>,
    pending_order: VecDeque<DecisionId>,
    recent: VecDeque<ResolvedDecisionEpisode>,
    episodic: VecDeque<ResolvedDecisionEpisode>,
    duplicate_receipts_total: u64,
    expired_total: u64,
    unattributed_applied_total: u64,
}

impl Default for DecisionLedger {
    fn default() -> Self {
        Self {
            next_id: 0,
            pending_correlations: HashMap::with_capacity(MAX_PENDING_DECISIONS),
            pending: HashMap::with_capacity(MAX_PENDING_DECISIONS),
            pending_order: VecDeque::with_capacity(MAX_PENDING_DECISIONS),
            recent: VecDeque::with_capacity(MAX_RECENT_DECISIONS),
            episodic: VecDeque::with_capacity(MAX_EPISODIC_DECISIONS),
            duplicate_receipts_total: 0,
            expired_total: 0,
            unattributed_applied_total: 0,
        }
    }
}

impl<'de> Deserialize<'de> for DecisionLedger {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let persisted = DecisionLedgerPersisted::deserialize(deserializer)?;
        let mut ledger = Self {
            next_id: persisted.next_id,
            pending_correlations: persisted.pending_correlations,
            pending: persisted.pending,
            pending_order: persisted.pending_order,
            recent: persisted.recent,
            episodic: persisted.episodic,
            duplicate_receipts_total: persisted.duplicate_receipts_total,
            expired_total: persisted.expired_total,
            unattributed_applied_total: persisted.unattributed_applied_total,
        };
        ledger.normalize_restored_state();
        Ok(ledger)
    }
}

impl DecisionLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign IDs and close one bounded cycle batch. Every returned episode
    /// contains the `DecisionId` shared by its proposal and receipt.
    pub fn ingest_cycle_events(
        &mut self,
        events: &mut CycleDecisionEvents,
    ) -> Vec<ResolvedDecisionEpisode> {
        let mut episodes = Vec::with_capacity(events.len());
        for event in events.drain() {
            if let Some(episode) = self.ingest_event(event) {
                episodes.push(episode);
            }
        }
        episodes
    }

    fn ingest_event(&mut self, event: ActuatorDecisionEvent) -> Option<ResolvedDecisionEpisode> {
        let settled_cycle = if event.observed_cycle > 0 {
            event.observed_cycle
        } else {
            event.proposal.proposed_cycle
        };
        if event.outcome == ActuatorDecisionOutcome::Pending {
            let id = self.propose(event.proposal);
            if let Some(correlation_id) = event.correlation_id {
                self.pending_correlations.insert(correlation_id, id);
                self.begin_execution(id);
                return None;
            }
            if let Some(envelope) = self.pending.get_mut(&id) {
                envelope.terminal_attribution = Some(event.attribution.clone().bounded());
            }
            self.record_execution(
                id,
                ExecutionReceipt {
                    receipt_id: id.0,
                    disposition: ExecutionDisposition::Failed,
                    observed_cycle: settled_cycle,
                    attribution: Some(event.attribution),
                    detail: "pending event missing correlation".to_string(),
                },
            );
            return self.settle(id, settled_cycle);
        }
        let id = event
            .correlation_id
            .and_then(|correlation_id| self.pending_correlations.get(&correlation_id).copied())
            .unwrap_or_else(|| self.propose(event.proposal));
        if let Some(envelope) = self.pending.get_mut(&id) {
            envelope.terminal_attribution = Some(event.attribution.clone().bounded());
        }
        let closed = match event.outcome {
            ActuatorDecisionOutcome::Rejected => self.reject(id, &event.detail),
            ActuatorDecisionOutcome::Vetoed => self.veto(id, &event.detail),
            ActuatorDecisionOutcome::Expired => {
                self.close_without_execution(id, DecisionLifecycle::Expired, &event.detail)
            }
            outcome => {
                let disposition = match outcome {
                    ActuatorDecisionOutcome::Applied => ExecutionDisposition::Applied,
                    ActuatorDecisionOutcome::Blocked => ExecutionDisposition::Blocked,
                    ActuatorDecisionOutcome::Failed => ExecutionDisposition::Failed,
                    ActuatorDecisionOutcome::NoOp => ExecutionDisposition::NoOp,
                    ActuatorDecisionOutcome::Reverted => ExecutionDisposition::Reverted,
                    ActuatorDecisionOutcome::Rejected
                    | ActuatorDecisionOutcome::Vetoed
                    | ActuatorDecisionOutcome::Expired
                    | ActuatorDecisionOutcome::Pending => unreachable!(),
                };
                self.record_execution(
                    id,
                    ExecutionReceipt {
                        receipt_id: id.0,
                        disposition,
                        observed_cycle: settled_cycle,
                        attribution: Some(event.attribution),
                        detail: event.detail,
                    },
                )
            }
        };
        let settled = closed.then(|| self.settle(id, settled_cycle)).flatten();
        if settled.is_some() {
            if let Some(correlation_id) = event.correlation_id {
                self.pending_correlations.remove(&correlation_id);
            }
        }
        settled
    }

    pub fn propose(&mut self, proposal: DecisionProposal) -> DecisionId {
        if self.pending.len() >= MAX_PENDING_DECISIONS {
            if let Some(evicted_id) = self.pending_order.pop_front() {
                if let Some(mut evicted) = self.pending.remove(&evicted_id) {
                    self.pending_correlations
                        .retain(|_, pending_id| *pending_id != evicted_id);
                    evicted.lifecycle = DecisionLifecycle::Expired;
                    evicted.terminal_reason = "pending-capacity".to_string();
                    self.expired_total = self.expired_total.saturating_add(1);
                    self.archive(evicted, proposal.proposed_cycle);
                }
            }
        }
        let id = self.allocate_id();
        let envelope = DecisionEnvelope {
            id,
            action_key: bounded_text(&proposal.action_key, MAX_ACTION_KEY_CHARS),
            target: bounded_text(&proposal.target, MAX_TARGET_CHARS),
            proposed_cycle: proposal.proposed_cycle,
            expires_cycle: proposal.expires_cycle,
            alternatives: proposal
                .alternatives
                .into_iter()
                .map(CandidateAlternative::bounded)
                .take(MAX_CANDIDATE_ALTERNATIVES)
                .collect(),
            hierarchy: proposal.hierarchy,
            predictions: proposal
                .predictions
                .into_iter()
                .map(PredictionRecord::bounded)
                .take(MAX_PREDICTIONS)
                .collect(),
            adviser_contributions: proposal
                .adviser_contributions
                .into_iter()
                .map(AdviserContribution::bounded)
                .take(MAX_ADVISER_CONTRIBUTIONS)
                .collect(),
            lifecycle: DecisionLifecycle::Proposed,
            terminal_attribution: None,
            terminal_reason: String::new(),
            receipt: None,
        };
        self.pending.insert(id, envelope);
        self.pending_order.push_back(id);
        id
    }

    pub fn record_execution(&mut self, id: DecisionId, receipt: ExecutionReceipt) -> bool {
        let Some(envelope) = self.pending.get_mut(&id) else {
            return false;
        };
        if envelope.receipt.is_some() {
            self.duplicate_receipts_total = self.duplicate_receipts_total.saturating_add(1);
            return true;
        }
        if envelope.lifecycle.is_terminal() {
            return false;
        }
        let receipt = receipt.bounded();
        if receipt.disposition == ExecutionDisposition::Applied && receipt.attribution.is_none() {
            self.unattributed_applied_total = self.unattributed_applied_total.saturating_add(1);
        }
        envelope.lifecycle = match receipt.disposition {
            ExecutionDisposition::Applied => DecisionLifecycle::Applied,
            ExecutionDisposition::Blocked => DecisionLifecycle::Blocked,
            ExecutionDisposition::Failed => DecisionLifecycle::Failed,
            ExecutionDisposition::NoOp => DecisionLifecycle::NoOp,
            ExecutionDisposition::Reverted => DecisionLifecycle::Reverted,
        };
        envelope.receipt = Some(receipt);
        true
    }

    pub fn begin_execution(&mut self, id: DecisionId) -> bool {
        let Some(envelope) = self.pending.get_mut(&id) else {
            return false;
        };
        if envelope.lifecycle != DecisionLifecycle::Proposed {
            return false;
        }
        envelope.lifecycle = DecisionLifecycle::Executing;
        true
    }

    pub fn reject(&mut self, id: DecisionId, reason: &str) -> bool {
        self.close_without_execution(id, DecisionLifecycle::Rejected, reason)
    }

    pub fn veto(&mut self, id: DecisionId, reason: &str) -> bool {
        self.close_without_execution(id, DecisionLifecycle::Vetoed, reason)
    }

    pub fn block(&mut self, id: DecisionId, reason: &str) -> bool {
        self.close_without_execution(id, DecisionLifecycle::Blocked, reason)
    }

    pub fn pending(&self, id: DecisionId) -> Option<&DecisionEnvelope> {
        self.pending.get(&id)
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn pending_for_correlation(&self, correlation_id: u64) -> Option<DecisionId> {
        self.pending_correlations.get(&correlation_id).copied()
    }

    pub fn high_water(&self) -> u64 {
        self.next_id
    }

    pub fn seed_high_water(&mut self, high_water: u64) {
        self.next_id = self.next_id.max(high_water);
    }

    pub fn recent(&self) -> &VecDeque<ResolvedDecisionEpisode> {
        &self.recent
    }

    pub fn episodic(&self) -> &VecDeque<ResolvedDecisionEpisode> {
        &self.episodic
    }

    pub fn duplicate_receipts_total(&self) -> u64 {
        self.duplicate_receipts_total
    }

    pub fn unattributed_applied_total(&self) -> u64 {
        self.unattributed_applied_total
    }

    /// Expire decisions that reached their declared cycle. The pass is bounded
    /// by `MAX_PENDING_DECISIONS` and performs no ordering or global work.
    pub fn expire(&mut self, cycle: u64) -> usize {
        let mut retained = VecDeque::with_capacity(self.pending_order.len());
        let mut expired = 0;
        while let Some(id) = self.pending_order.pop_front() {
            let due = self.pending.get(&id).is_some_and(|envelope| {
                !envelope.lifecycle.is_terminal()
                    && envelope.expires_cycle > 0
                    && envelope.expires_cycle <= cycle
            });
            if !due {
                retained.push_back(id);
                continue;
            }
            if let Some(mut envelope) = self.pending.remove(&id) {
                self.pending_correlations
                    .retain(|_, pending_id| *pending_id != id);
                envelope.lifecycle = DecisionLifecycle::Expired;
                envelope.terminal_reason = "expired".to_string();
                self.expired_total = self.expired_total.saturating_add(1);
                self.archive(envelope, cycle);
                expired += 1;
            }
        }
        self.pending_order = retained;
        expired
    }

    pub fn settle(
        &mut self,
        id: DecisionId,
        settled_cycle: u64,
    ) -> Option<ResolvedDecisionEpisode> {
        let envelope = self.pending.get(&id)?;
        if !envelope.lifecycle.is_terminal() {
            return None;
        }
        let envelope = self.pending.remove(&id)?;
        remove_pending_id(&mut self.pending_order, id);
        self.pending_correlations
            .retain(|_, pending_id| *pending_id != id);
        Some(self.archive(envelope, settled_cycle))
    }

    fn archive(
        &mut self,
        envelope: DecisionEnvelope,
        settled_cycle: u64,
    ) -> ResolvedDecisionEpisode {
        let id = envelope.id;
        let authority_eligible = envelope.receipt.as_ref().is_some_and(|receipt| {
            receipt.disposition == ExecutionDisposition::Applied
                && receipt
                    .attribution
                    .as_ref()
                    .is_some_and(ReceiptAttribution::grants_local_authority)
        });
        let episode = ResolvedDecisionEpisode {
            id,
            lifecycle: DecisionLifecycle::Settled,
            settled_cycle,
            authority_eligible,
            envelope,
        };
        push_bounded(&mut self.recent, episode.clone(), MAX_RECENT_DECISIONS);
        if episode.authority_eligible {
            push_bounded(&mut self.episodic, episode.clone(), MAX_EPISODIC_DECISIONS);
        }
        episode
    }

    fn allocate_id(&mut self) -> DecisionId {
        let mut occupied = HashSet::with_capacity(MAX_RETAINED_DECISION_IDS);
        occupied.extend(self.pending.keys().copied());
        occupied.extend(self.recent.iter().map(|episode| episode.id));
        occupied.extend(self.episodic.iter().map(|episode| episode.id));
        for _ in 0..=MAX_RETAINED_DECISION_IDS {
            self.next_id = self.next_id.wrapping_add(1);
            if self.next_id == 0 {
                continue;
            }
            let id = DecisionId(self.next_id);
            if !occupied.contains(&id) {
                return id;
            }
        }
        unreachable!("the bounded ledger must leave an unused decision id");
    }

    fn normalize_restored_state(&mut self) {
        let mut restored = std::mem::take(&mut self.pending);
        let mut pending = HashMap::with_capacity(MAX_PENDING_DECISIONS);
        let mut order = VecDeque::with_capacity(MAX_PENDING_DECISIONS);
        for id in restored.keys() {
            self.next_id = self.next_id.max(id.0);
        }
        if pending_order_is_valid(&self.pending_order, &restored) {
            for id in std::mem::take(&mut self.pending_order) {
                let envelope = restored
                    .remove(&id)
                    .expect("validated pending order must reference a pending envelope");
                self.insert_restored_pending(id, envelope, &mut pending, &mut order);
            }
        } else {
            self.pending_order.clear();
            for (id, envelope) in select_reconstructed_pending(restored) {
                self.insert_restored_pending(id, envelope, &mut pending, &mut order);
            }
        }
        self.pending = pending;
        self.pending_order = order;
        self.pending_correlations
            .retain(|correlation, id| *correlation > 0 && self.pending.contains_key(id));

        self.recent = normalize_episodes(std::mem::take(&mut self.recent), MAX_RECENT_DECISIONS);
        self.episodic = normalize_episodes(
            std::mem::take(&mut self.episodic)
                .into_iter()
                .filter(|episode| episode_authority_eligible(&episode.envelope))
                .collect(),
            MAX_EPISODIC_DECISIONS,
        );
    }

    fn insert_restored_pending(
        &mut self,
        id: DecisionId,
        envelope: DecisionEnvelope,
        pending: &mut HashMap<DecisionId, DecisionEnvelope>,
        order: &mut VecDeque<DecisionId>,
    ) {
        pending.insert(id, envelope.bounded(id));
        order.push_back(id);
    }

    fn close_without_execution(
        &mut self,
        id: DecisionId,
        lifecycle: DecisionLifecycle,
        reason: &str,
    ) -> bool {
        let Some(envelope) = self.pending.get_mut(&id) else {
            return false;
        };
        if envelope.lifecycle.is_terminal() || envelope.receipt.is_some() {
            return false;
        }
        envelope.lifecycle = lifecycle;
        envelope.terminal_reason = bounded_text(reason, MAX_REASON_CHARS);
        true
    }
}

impl CandidateAlternative {
    fn bounded(mut self) -> Self {
        self.action_key = bounded_text(&self.action_key, MAX_ACTION_KEY_CHARS);
        self.target = bounded_text(&self.target, MAX_TARGET_CHARS);
        self.expected_utility = finite_unit(self.expected_utility);
        self.uncertainty = finite_unit(self.uncertainty).max(0.0);
        self
    }
}

impl PredictionRecord {
    fn bounded(mut self) -> Self {
        self.source = bounded_text(&self.source, MAX_SOURCE_CHARS);
        self.expected_utility = finite_unit(self.expected_utility);
        self.uncertainty = finite_unit(self.uncertainty).max(0.0);
        self
    }
}

impl AdviserContribution {
    fn bounded(mut self) -> Self {
        self.adviser = bounded_text(&self.adviser, MAX_SOURCE_CHARS);
        self.support = finite_unit(self.support);
        self.uncertainty = finite_unit(self.uncertainty).max(0.0);
        self
    }
}

impl ExecutionReceipt {
    fn bounded(mut self) -> Self {
        self.attribution = self
            .attribution
            .and_then(|attribution| match attribution.bounded() {
                ReceiptAttribution::Local { source } if source.is_empty() => None,
                attribution => Some(attribution),
            });
        self.detail = bounded_text(&self.detail, MAX_REASON_CHARS);
        self
    }
}

impl DecisionEnvelope {
    fn bounded(mut self, id: DecisionId) -> Self {
        self.id = id;
        self.action_key = bounded_text(&self.action_key, MAX_ACTION_KEY_CHARS);
        self.target = bounded_text(&self.target, MAX_TARGET_CHARS);
        self.alternatives = self
            .alternatives
            .into_iter()
            .map(CandidateAlternative::bounded)
            .take(MAX_CANDIDATE_ALTERNATIVES)
            .collect();
        self.predictions = self
            .predictions
            .into_iter()
            .map(PredictionRecord::bounded)
            .take(MAX_PREDICTIONS)
            .collect();
        self.adviser_contributions = self
            .adviser_contributions
            .into_iter()
            .map(AdviserContribution::bounded)
            .take(MAX_ADVISER_CONTRIBUTIONS)
            .collect();
        self.terminal_reason = bounded_text(&self.terminal_reason, MAX_REASON_CHARS);
        self.terminal_attribution =
            self.terminal_attribution
                .and_then(|attribution| match attribution.bounded() {
                    ReceiptAttribution::Local { source } if source.is_empty() => None,
                    attribution => Some(attribution),
                });
        self.receipt = self.receipt.map(ExecutionReceipt::bounded);
        self
    }
}

fn bounded_text(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn finite_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn remove_pending_id(order: &mut VecDeque<DecisionId>, id: DecisionId) {
    if let Some(index) = order.iter().position(|candidate| *candidate == id) {
        order.remove(index);
    }
}

fn push_bounded<T>(queue: &mut VecDeque<T>, value: T, limit: usize) {
    if queue.len() >= limit {
        queue.pop_front();
    }
    queue.push_back(value);
}

fn pending_order_is_valid(
    order: &VecDeque<DecisionId>,
    pending: &HashMap<DecisionId, DecisionEnvelope>,
) -> bool {
    if order.len() != pending.len() || order.len() > MAX_PENDING_DECISIONS {
        return false;
    }
    let mut seen = HashSet::with_capacity(order.len());
    order
        .iter()
        .all(|id| pending.contains_key(id) && seen.insert(*id))
}

struct RestoredPending {
    id: DecisionId,
    envelope: DecisionEnvelope,
}

impl PartialEq for RestoredPending {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.envelope.proposed_cycle == other.envelope.proposed_cycle
    }
}

impl Eq for RestoredPending {}

impl PartialOrd for RestoredPending {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RestoredPending {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.envelope
            .proposed_cycle
            .cmp(&other.envelope.proposed_cycle)
            .then_with(|| self.id.cmp(&other.id))
    }
}

fn select_reconstructed_pending(
    pending: HashMap<DecisionId, DecisionEnvelope>,
) -> Vec<(DecisionId, DecisionEnvelope)> {
    let mut selected = BinaryHeap::with_capacity(MAX_PENDING_DECISIONS + 1);
    for (id, envelope) in pending {
        selected.push(Reverse(RestoredPending { id, envelope }));
        if selected.len() > MAX_PENDING_DECISIONS {
            selected.pop();
        }
    }
    let mut reconstructed: Vec<_> = selected
        .into_iter()
        .map(|Reverse(candidate)| (candidate.id, candidate.envelope))
        .collect();
    reconstructed.sort_by(|(left_id, left), (right_id, right)| {
        left.proposed_cycle
            .cmp(&right.proposed_cycle)
            .then_with(|| left_id.cmp(right_id))
    });
    reconstructed
}

fn normalize_episodes(
    episodes: VecDeque<ResolvedDecisionEpisode>,
    limit: usize,
) -> VecDeque<ResolvedDecisionEpisode> {
    let mut normalized = VecDeque::with_capacity(limit);
    for mut episode in episodes {
        episode.envelope = episode.envelope.bounded(episode.id);
        episode.authority_eligible = episode_authority_eligible(&episode.envelope);
        if normalized.len() >= limit {
            normalized.pop_front();
        }
        normalized.push_back(episode);
    }
    normalized
}

fn episode_authority_eligible(envelope: &DecisionEnvelope) -> bool {
    envelope.receipt.as_ref().is_some_and(|receipt| {
        receipt.disposition == ExecutionDisposition::Applied
            && receipt
                .attribution
                .as_ref()
                .is_some_and(ReceiptAttribution::grants_local_authority)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        coordinated_action_event, select_reconstructed_pending, ActuatorDecisionEvent,
        ActuatorDecisionOutcome, AdviserContribution, CandidateAlternative, CycleDecisionEvents,
        DecisionEnvelope, DecisionId, DecisionLedger, DecisionLifecycle, DecisionProposal,
        ExecutionDisposition, ExecutionReceipt, ReceiptAttribution, ResolvedDecisionEpisode,
        MAX_ADVISER_CONTRIBUTIONS, MAX_CANDIDATE_ALTERNATIVES, MAX_CYCLE_DECISION_EVENTS,
        MAX_PENDING_DECISIONS,
    };

    #[test]
    fn applied_decision_completes_its_lifecycle() {
        let mut ledger = DecisionLedger::new();
        let id = ledger.propose(DecisionProposal {
            action_key: "freeze:background".to_string(),
            proposed_cycle: 10,
            expires_cycle: 20,
            ..DecisionProposal::default()
        });

        assert_eq!(
            ledger.record_execution(
                id,
                ExecutionReceipt {
                    receipt_id: 1,
                    disposition: ExecutionDisposition::Applied,
                    observed_cycle: 11,
                    attribution: Some(ReceiptAttribution::local("actuation-broker")),
                    ..ExecutionReceipt::default()
                },
            ),
            true
        );
        assert_eq!(
            ledger.pending(id).unwrap().lifecycle,
            DecisionLifecycle::Applied
        );

        let episode = ledger.settle(id, 12).unwrap();
        assert_eq!(episode.id, id);
        assert_eq!(episode.lifecycle, DecisionLifecycle::Settled);
        assert!(episode.authority_eligible);
        assert!(ledger.pending(id).is_none());
        assert_eq!(ledger.recent().len(), 1);
        assert_eq!(ledger.episodic().len(), 1);
    }

    #[test]
    fn duplicate_receipts_are_idempotent_and_counted() {
        let mut ledger = DecisionLedger::new();
        let id = ledger.propose(DecisionProposal::default());
        let receipt = ExecutionReceipt {
            receipt_id: 9,
            disposition: ExecutionDisposition::Failed,
            observed_cycle: 3,
            detail: "first result".to_string(),
            ..ExecutionReceipt::default()
        };

        assert!(ledger.record_execution(id, receipt));
        assert!(ledger.record_execution(
            id,
            ExecutionReceipt {
                receipt_id: 10,
                disposition: ExecutionDisposition::Applied,
                observed_cycle: 4,
                detail: "replayed result".to_string(),
                ..ExecutionReceipt::default()
            },
        ));

        let envelope = ledger.pending(id).unwrap();
        assert_eq!(envelope.lifecycle, DecisionLifecycle::Failed);
        assert_eq!(envelope.receipt.as_ref().unwrap().receipt_id, 9);
        assert_eq!(ledger.duplicate_receipts_total(), 1);
    }

    #[test]
    fn proposal_eviction_expires_the_oldest_pending_decision() {
        let mut ledger = DecisionLedger::new();
        let first = ledger.propose(DecisionProposal::default());
        for _ in 1..MAX_PENDING_DECISIONS {
            ledger.propose(DecisionProposal::default());
        }
        ledger.propose(DecisionProposal::default());

        assert!(ledger.pending(first).is_none());
        assert_eq!(ledger.recent().len(), 1);
        assert_eq!(ledger.recent().front().unwrap().envelope.id, first);
        assert_eq!(
            ledger.recent().front().unwrap().envelope.lifecycle,
            DecisionLifecycle::Expired
        );
    }

    #[test]
    fn round_trip_preserves_valid_serialized_fifo_order() {
        let mut ledger = DecisionLedger::new();
        let first = ledger.propose(DecisionProposal {
            action_key: "first".to_string(),
            proposed_cycle: 100,
            ..DecisionProposal::default()
        });
        let second = ledger.propose(DecisionProposal {
            action_key: "second".to_string(),
            proposed_cycle: 0,
            ..DecisionProposal::default()
        });
        for _ in 2..MAX_PENDING_DECISIONS {
            ledger.propose(DecisionProposal::default());
        }
        let serialized = serde_json::to_string(&ledger).unwrap();
        let mut restored: DecisionLedger = serde_json::from_str(&serialized).unwrap();

        restored.propose(DecisionProposal::default());

        assert!(restored.pending(first).is_none());
        assert!(restored.pending(second).is_some());
    }

    #[test]
    fn expiry_closes_only_due_pending_decisions() {
        let mut ledger = DecisionLedger::new();
        let due = ledger.propose(DecisionProposal {
            expires_cycle: 5,
            ..DecisionProposal::default()
        });
        let future = ledger.propose(DecisionProposal {
            expires_cycle: 6,
            ..DecisionProposal::default()
        });

        assert_eq!(ledger.expire(5), 1);
        assert!(ledger.pending(due).is_none());
        assert!(ledger.pending(future).is_some());
        assert_eq!(
            ledger.recent().front().unwrap().envelope.lifecycle,
            DecisionLifecycle::Expired
        );
    }

    #[test]
    fn unattributed_applied_receipts_are_excluded_from_authority_learning() {
        let mut ledger = DecisionLedger::new();
        let id = ledger.propose(DecisionProposal::default());

        assert!(ledger.record_execution(
            id,
            ExecutionReceipt {
                receipt_id: 42,
                disposition: ExecutionDisposition::Applied,
                ..ExecutionReceipt::default()
            },
        ));
        let episode = ledger.settle(id, 1).unwrap();

        assert!(!episode.authority_eligible);
        assert_eq!(ledger.unattributed_applied_total(), 1);
        assert_eq!(ledger.recent().len(), 1);
        assert!(ledger.episodic().is_empty());
    }

    #[test]
    fn empty_local_receipt_attribution_is_counted_as_unattributed() {
        let mut ledger = DecisionLedger::new();
        let id = ledger.propose(DecisionProposal::default());

        assert!(ledger.record_execution(
            id,
            ExecutionReceipt {
                receipt_id: 43,
                attribution: Some(ReceiptAttribution::local("")),
                ..ExecutionReceipt::default()
            },
        ));

        assert_eq!(ledger.unattributed_applied_total(), 1);
        assert!(ledger.settle(id, 1).is_some());
        assert!(ledger.episodic().is_empty());
    }

    #[test]
    fn reject_veto_and_block_close_decisions_without_execution() {
        let mut ledger = DecisionLedger::new();
        let rejected = ledger.propose(DecisionProposal::default());
        let vetoed = ledger.propose(DecisionProposal::default());
        let blocked = ledger.propose(DecisionProposal::default());

        assert!(ledger.reject(rejected, "policy"));
        assert!(ledger.veto(vetoed, "safety"));
        assert!(ledger.block(blocked, "actuation-broker"));

        assert_eq!(
            ledger.pending(rejected).unwrap().lifecycle,
            DecisionLifecycle::Rejected
        );
        assert_eq!(
            ledger.pending(vetoed).unwrap().lifecycle,
            DecisionLifecycle::Vetoed
        );
        assert_eq!(
            ledger.pending(blocked).unwrap().lifecycle,
            DecisionLifecycle::Blocked
        );
        assert!(ledger.settle(rejected, 1).is_some());
        assert!(ledger.settle(vetoed, 1).is_some());
        assert!(ledger.settle(blocked, 1).is_some());
        assert!(ledger.episodic().is_empty());
    }

    #[test]
    fn concurrent_actions_keep_independent_receipts() {
        let mut ledger = DecisionLedger::new();
        let first = ledger.propose(DecisionProposal {
            action_key: "nice:101".to_string(),
            ..DecisionProposal::default()
        });
        let second = ledger.propose(DecisionProposal {
            action_key: "jetsam:202".to_string(),
            ..DecisionProposal::default()
        });

        assert!(ledger.record_execution(
            first,
            ExecutionReceipt {
                receipt_id: 1,
                disposition: ExecutionDisposition::Applied,
                attribution: Some(ReceiptAttribution::local("actuation-broker")),
                ..ExecutionReceipt::default()
            },
        ));
        assert!(ledger.record_execution(
            second,
            ExecutionReceipt {
                receipt_id: 2,
                disposition: ExecutionDisposition::Failed,
                ..ExecutionReceipt::default()
            },
        ));
        assert!(ledger.settle(first, 3).is_some());
        assert!(ledger.settle(second, 3).is_some());

        assert_eq!(ledger.pending_len(), 0);
        assert_eq!(ledger.recent().len(), 2);
        assert_eq!(ledger.episodic().len(), 1);
        assert_eq!(ledger.recent()[0].envelope.id, first);
        assert_eq!(ledger.recent()[1].envelope.id, second);
    }

    #[test]
    fn serde_defaults_restore_a_bounded_pending_order() {
        let serialized = serde_json::json!({
            "next_id": 7,
            "pending": {
                "7": {
                    "id": 7,
                    "expires_cycle": 1,
                }
            }
        })
        .to_string();

        let mut restored: DecisionLedger = serde_json::from_str(&serialized).unwrap();
        let empty: DecisionLedger = serde_json::from_str("{}").unwrap();

        assert_eq!(restored.pending_len(), 1);
        assert_eq!(restored.expire(1), 1);
        assert_eq!(empty.pending_len(), 0);
        assert!(empty.recent().is_empty());
        assert!(empty.episodic().is_empty());
    }

    #[test]
    fn execution_can_be_marked_before_its_receipt_arrives() {
        let mut ledger = DecisionLedger::new();
        let id = ledger.propose(DecisionProposal::default());

        assert!(ledger.begin_execution(id));
        assert_eq!(
            ledger.pending(id).unwrap().lifecycle,
            DecisionLifecycle::Executing
        );
        assert!(ledger.record_execution(id, ExecutionReceipt::default()));
        assert_eq!(
            ledger.pending(id).unwrap().lifecycle,
            DecisionLifecycle::Applied
        );
    }

    #[test]
    fn imported_receipts_remain_descriptive_only() {
        let mut ledger = DecisionLedger::new();
        let id = ledger.propose(DecisionProposal::default());

        assert!(ledger.record_execution(
            id,
            ExecutionReceipt {
                receipt_id: 8,
                attribution: Some(ReceiptAttribution::Imported),
                ..ExecutionReceipt::default()
            },
        ));
        let episode = ledger.settle(id, 2).unwrap();

        assert!(!episode.authority_eligible);
        assert_eq!(ledger.unattributed_applied_total(), 0);
        assert!(ledger.episodic().is_empty());
    }

    #[test]
    fn receipt_after_a_gate_closure_is_not_treated_as_a_duplicate() {
        let mut ledger = DecisionLedger::new();
        let id = ledger.propose(DecisionProposal::default());
        assert!(ledger.block(id, "safety"));

        assert!(!ledger.record_execution(id, ExecutionReceipt::default()));
        assert_eq!(ledger.duplicate_receipts_total(), 0);
        assert_eq!(
            ledger.pending(id).unwrap().lifecycle,
            DecisionLifecycle::Blocked
        );
    }

    #[test]
    fn proposal_bounds_alternatives_and_adviser_contributions() {
        let mut ledger = DecisionLedger::new();
        let id = ledger.propose(DecisionProposal {
            alternatives: (0..=MAX_CANDIDATE_ALTERNATIVES)
                .map(|index| CandidateAlternative {
                    action_key: format!("candidate-{index}"),
                    ..CandidateAlternative::default()
                })
                .collect(),
            adviser_contributions: (0..=MAX_ADVISER_CONTRIBUTIONS)
                .map(|index| AdviserContribution {
                    adviser: format!("adviser-{index}"),
                    ..AdviserContribution::default()
                })
                .collect(),
            ..DecisionProposal::default()
        });

        let envelope = ledger.pending(id).unwrap();
        assert_eq!(envelope.alternatives.len(), MAX_CANDIDATE_ALTERNATIVES);
        assert_eq!(
            envelope.adviser_contributions.len(),
            MAX_ADVISER_CONTRIBUTIONS
        );
    }

    #[test]
    fn expiry_preserves_an_existing_terminal_gate_outcome() {
        let mut ledger = DecisionLedger::new();
        let id = ledger.propose(DecisionProposal {
            expires_cycle: 2,
            ..DecisionProposal::default()
        });
        assert!(ledger.veto(id, "safety"));

        assert_eq!(ledger.expire(2), 0);
        assert_eq!(
            ledger.pending(id).unwrap().lifecycle,
            DecisionLifecycle::Vetoed
        );
    }

    #[test]
    fn restore_without_order_evicts_the_oldest_proposal_first() {
        let mut pending = serde_json::Map::new();
        for raw_id in 1..=MAX_PENDING_DECISIONS as u64 {
            pending.insert(
                raw_id.to_string(),
                serde_json::json!({
                    "id": raw_id,
                    "proposed_cycle": (raw_id - 1) / 2,
                }),
            );
        }
        let serialized = serde_json::json!({
            "next_id": MAX_PENDING_DECISIONS,
            "pending": pending,
        })
        .to_string();
        let mut restored: DecisionLedger = serde_json::from_str(&serialized).unwrap();

        restored.propose(DecisionProposal::default());

        assert!(restored.pending(DecisionId(1)).is_none());
        assert!(restored.pending(DecisionId(2)).is_some());
    }

    #[test]
    fn oversized_reconstruction_selects_a_bounded_recent_window_before_ordering() {
        let mut pending = std::collections::HashMap::new();
        for raw_id in 1..=(MAX_PENDING_DECISIONS as u64 + 1) {
            pending.insert(
                DecisionId(raw_id),
                DecisionEnvelope {
                    id: DecisionId(raw_id),
                    proposed_cycle: raw_id,
                    ..DecisionEnvelope::default()
                },
            );
        }

        let reconstructed = select_reconstructed_pending(pending);

        assert_eq!(reconstructed.len(), MAX_PENDING_DECISIONS);
        assert_eq!(reconstructed.first().unwrap().0, DecisionId(2));
        assert_eq!(
            reconstructed.last().unwrap().0,
            DecisionId(MAX_PENDING_DECISIONS as u64 + 1)
        );
    }

    #[test]
    fn oversized_restore_retains_only_the_newest_bounded_pending_window() {
        let mut pending = serde_json::Map::new();
        for raw_id in 1..=(MAX_PENDING_DECISIONS as u64 + 1) {
            pending.insert(
                raw_id.to_string(),
                serde_json::json!({
                    "id": raw_id,
                    "proposed_cycle": raw_id,
                }),
            );
        }
        let serialized = serde_json::json!({ "pending": pending }).to_string();

        let restored: DecisionLedger = serde_json::from_str(&serialized).unwrap();

        assert_eq!(restored.pending_len(), MAX_PENDING_DECISIONS);
        assert!(restored.pending(DecisionId(1)).is_none());
        assert!(restored
            .pending(DecisionId(MAX_PENDING_DECISIONS as u64 + 1))
            .is_some());
    }

    #[test]
    fn exhausted_id_high_water_mark_does_not_overwrite_a_live_decision() {
        let max_id = u64::MAX;
        let serialized = serde_json::json!({
            "next_id": max_id,
            "pending": {
                max_id.to_string(): {
                    "id": max_id,
                    "action_key": "existing",
                }
            }
        })
        .to_string();
        let mut restored: DecisionLedger = serde_json::from_str(&serialized).unwrap();

        let new_id = restored.propose(DecisionProposal {
            action_key: "new".to_string(),
            ..DecisionProposal::default()
        });

        assert_ne!(new_id, DecisionId(max_id));
        assert_eq!(
            restored.pending(DecisionId(max_id)).unwrap().action_key,
            "existing"
        );
        assert_eq!(restored.pending(new_id).unwrap().action_key, "new");
    }

    #[test]
    fn wrapped_allocation_skips_recent_and_episodic_decision_ids() {
        let mut ledger = DecisionLedger::new();
        ledger.next_id = u64::MAX;
        ledger.recent.push_back(ResolvedDecisionEpisode {
            id: DecisionId(1),
            ..ResolvedDecisionEpisode::default()
        });
        ledger.episodic.push_back(ResolvedDecisionEpisode {
            id: DecisionId(2),
            ..ResolvedDecisionEpisode::default()
        });

        let new_id = ledger.propose(DecisionProposal::default());

        assert_eq!(new_id, DecisionId(3));
    }

    #[test]
    fn cycle_events_close_every_supported_lifecycle_with_decision_ids() {
        let mut events = CycleDecisionEvents::default();
        for outcome in [
            ActuatorDecisionOutcome::Applied,
            ActuatorDecisionOutcome::Blocked,
            ActuatorDecisionOutcome::Failed,
            ActuatorDecisionOutcome::NoOp,
            ActuatorDecisionOutcome::Reverted,
            ActuatorDecisionOutcome::Rejected,
            ActuatorDecisionOutcome::Vetoed,
            ActuatorDecisionOutcome::Expired,
        ] {
            assert!(events.push(ActuatorDecisionEvent::local(
                format!("test:{outcome:?}"),
                "pid:42",
                7,
                outcome,
                "test-actuator",
                "focused lifecycle test",
            )));
        }
        let mut ledger = DecisionLedger::new();

        let episodes = ledger.ingest_cycle_events(&mut events);

        assert!(events.is_empty());
        assert_eq!(episodes.len(), 8);
        assert!(episodes.iter().all(|episode| episode.id.0 > 0));
        assert_eq!(
            episodes
                .iter()
                .map(|episode| episode.envelope.lifecycle)
                .collect::<Vec<_>>(),
            vec![
                DecisionLifecycle::Applied,
                DecisionLifecycle::Blocked,
                DecisionLifecycle::Failed,
                DecisionLifecycle::NoOp,
                DecisionLifecycle::Reverted,
                DecisionLifecycle::Rejected,
                DecisionLifecycle::Vetoed,
                DecisionLifecycle::Expired,
            ]
        );
        assert!(episodes[0].authority_eligible);
        assert!(episodes[0].envelope.receipt.is_some());
        assert!(episodes[1].envelope.receipt.is_some());
        assert!(episodes[5].envelope.receipt.is_none());
    }

    #[test]
    fn cycle_event_buffer_is_bounded_and_reports_drops() {
        let mut events = CycleDecisionEvents::default();
        for index in 0..MAX_CYCLE_DECISION_EVENTS {
            assert!(events.push(ActuatorDecisionEvent::local(
                format!("bounded:{index}"),
                "host",
                1,
                ActuatorDecisionOutcome::NoOp,
                "test",
                "bounded",
            )));
        }

        assert!(!events.push(ActuatorDecisionEvent::local(
            "bounded:overflow",
            "host",
            1,
            ActuatorDecisionOutcome::NoOp,
            "test",
            "bounded",
        )));
        assert_eq!(events.len(), MAX_CYCLE_DECISION_EVENTS);
        assert_eq!(events.dropped_total(), 1);
    }

    #[test]
    fn imported_cycle_event_cannot_gain_local_learning_authority() {
        let mut events = CycleDecisionEvents::default();
        assert!(events.push(ActuatorDecisionEvent::imported(
            "chromium_purge:purgeable_renderer",
            "pid:77",
            4,
            ActuatorDecisionOutcome::Applied,
            "imported receipt",
        )));
        let mut ledger = DecisionLedger::new();

        let episode = ledger.ingest_cycle_events(&mut events).remove(0);

        assert!(!episode.authority_eligible);
        assert!(ledger.episodic().is_empty());
    }

    #[test]
    fn rejected_cycle_event_retains_local_terminal_attribution() {
        let mut events = CycleDecisionEvents::default();
        events.push(ActuatorDecisionEvent::local(
            "boost:Editor",
            "Editor:pid:42",
            9,
            ActuatorDecisionOutcome::Rejected,
            "dispatch-filter",
            "controlled holdout",
        ));

        let episode = DecisionLedger::new()
            .ingest_cycle_events(&mut events)
            .pop()
            .expect("settled rejection");

        assert_eq!(
            episode.envelope.terminal_attribution,
            Some(ReceiptAttribution::local("dispatch-filter"))
        );
        assert!(!episode.authority_eligible);
    }

    #[test]
    fn coordinated_event_uses_distinct_applied_families_without_quadratic_pairing() {
        let mut events = CycleDecisionEvents::default();
        events.push(ActuatorDecisionEvent::local(
            "freeze:background",
            "worker:pid:1",
            12,
            ActuatorDecisionOutcome::Applied,
            "broker",
            "applied",
        ));
        events.push(ActuatorDecisionEvent::local(
            "predictive_purge:maintenance",
            "host",
            12,
            ActuatorDecisionOutcome::Applied,
            "maintenance",
            "applied",
        ));
        events.push(ActuatorDecisionEvent::local(
            "freeze:other",
            "worker:pid:2",
            12,
            ActuatorDecisionOutcome::Applied,
            "broker",
            "applied",
        ));

        let event = coordinated_action_event(&events, 12).expect("multi-family cohort");

        assert_eq!(
            event.proposal.action_key,
            "coordinated:freeze+predictive_purge"
        );
        assert_eq!(event.outcome, ActuatorDecisionOutcome::Applied);
        assert_eq!(event.proposal.hierarchy.cohort, 12);
    }

    #[test]
    fn pending_command_completion_reuses_one_decision_id_across_cycles() {
        let mut ledger = DecisionLedger::new();
        let mut launched = CycleDecisionEvents::default();
        launched.push(
            ActuatorDecisionEvent::local(
                "predictive_purge:maintenance",
                "host",
                7,
                ActuatorDecisionOutcome::Pending,
                "maintenance-purge",
                "queued for asynchronous completion",
            )
            .with_correlation(41),
        );

        assert!(ledger.ingest_cycle_events(&mut launched).is_empty());
        let pending_id = ledger
            .pending_for_correlation(41)
            .expect("queued command must retain one pending decision");

        let mut completed = CycleDecisionEvents::default();
        completed.push(
            ActuatorDecisionEvent::local(
                "predictive_purge:maintenance",
                "host",
                9,
                ActuatorDecisionOutcome::Applied,
                "async-command-completion",
                "exit status 0",
            )
            .with_correlation(41),
        );
        let episodes = ledger.ingest_cycle_events(&mut completed);

        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].id, pending_id);
        assert_eq!(episodes[0].envelope.proposed_cycle, 7);
        assert_eq!(episodes[0].settled_cycle, 9);
        assert!(episodes[0].authority_eligible);
        assert!(ledger.pending_for_correlation(41).is_none());
    }

    #[test]
    fn restored_high_water_seeds_ids_above_prior_process_run() {
        let mut prior = DecisionLedger::new();
        prior.seed_high_water(8_400);
        let persisted = serde_json::to_string(&prior).expect("serialize ledger high water");
        let restored: DecisionLedger = serde_json::from_str(&persisted).expect("restore ledger");
        let mut restart = DecisionLedger::new();
        restart.seed_high_water(restored.high_water());

        let id = restart.propose(DecisionProposal::default());

        assert_eq!(id, DecisionId(8_401));
    }

    #[test]
    fn nested_drops_propagate_and_seal_one_auditable_overflow_summary() {
        let mut producer = CycleDecisionEvents::default();
        for index in 0..=MAX_CYCLE_DECISION_EVENTS {
            producer.push(ActuatorDecisionEvent::local(
                format!("freeze:producer-{index}"),
                format!("pid:{index}"),
                11,
                ActuatorDecisionOutcome::Blocked,
                "producer",
                "bounded producer",
            ));
        }
        assert_eq!(producer.dropped_total(), 1);

        let mut cycle = CycleDecisionEvents::default();
        cycle.extend_buffer(&producer);
        assert_eq!(cycle.dropped_total(), 1);
        assert!(cycle.seal_overflow_summary(11));
        assert_eq!(cycle.dropped_total(), 2);
        let summary = cycle
            .as_slice()
            .iter()
            .find(|event| event.proposal.action_key == "decision_events:overflow")
            .expect("overflow must retain one bounded audit receipt");
        assert_eq!(summary.outcome, ActuatorDecisionOutcome::Failed);
        assert!(summary.detail.contains("dropped=2"));

        let episodes = DecisionLedger::new().ingest_cycle_events(&mut cycle);
        assert!(episodes
            .iter()
            .any(|episode| episode.envelope.action_key == "decision_events:overflow"));
    }
}
