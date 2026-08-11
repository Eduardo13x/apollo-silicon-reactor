use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Deserializer, Serialize};

pub const MAX_PENDING_DECISIONS: usize = 192;
pub const MAX_RECENT_DECISIONS: usize = 64;
pub const MAX_EPISODIC_DECISIONS: usize = 128;
pub const MAX_CANDIDATE_ALTERNATIVES: usize = 8;
pub const MAX_ADVISER_CONTRIBUTIONS: usize = 8;

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

/// Bounded, per-owner ledger. It is intentionally not global: callers keep a
/// cycle-local instance or their existing owner lock, avoiding new hot-path
/// synchronization and kernel authority.
#[derive(Debug, Clone, Serialize)]
pub struct DecisionLedger {
    next_id: u64,
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

    pub fn propose(&mut self, proposal: DecisionProposal) -> DecisionId {
        if self.pending.len() >= MAX_PENDING_DECISIONS {
            if let Some(evicted_id) = self.pending_order.pop_front() {
                if let Some(mut evicted) = self.pending.remove(&evicted_id) {
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
        for _ in 0..=MAX_PENDING_DECISIONS {
            self.next_id = self.next_id.wrapping_add(1);
            if self.next_id == 0 {
                continue;
            }
            let id = DecisionId(self.next_id);
            if !self.pending.contains_key(&id) {
                return id;
            }
        }
        unreachable!("the bounded pending ledger must leave an unused decision id");
    }

    fn normalize_restored_state(&mut self) {
        let mut restored: Vec<_> = std::mem::take(&mut self.pending).into_iter().collect();
        restored.sort_by(|(left_id, left), (right_id, right)| {
            left.proposed_cycle
                .cmp(&right.proposed_cycle)
                .then_with(|| left_id.cmp(right_id))
        });
        let mut pending = HashMap::with_capacity(MAX_PENDING_DECISIONS);
        let mut order = VecDeque::with_capacity(MAX_PENDING_DECISIONS);
        self.pending_order.clear();
        for (id, _) in &restored {
            self.next_id = self.next_id.max(id.0);
        }
        let retained_start = restored.len().saturating_sub(MAX_PENDING_DECISIONS);
        for (id, envelope) in restored.into_iter().skip(retained_start) {
            self.insert_restored_pending(id, envelope, &mut pending, &mut order);
        }
        self.pending = pending;
        self.pending_order = order;

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
        AdviserContribution, CandidateAlternative, DecisionId, DecisionLedger, DecisionLifecycle,
        DecisionProposal, ExecutionDisposition, ExecutionReceipt, ReceiptAttribution,
        MAX_ADVISER_CONTRIBUTIONS, MAX_CANDIDATE_ALTERNATIVES, MAX_PENDING_DECISIONS,
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
}
