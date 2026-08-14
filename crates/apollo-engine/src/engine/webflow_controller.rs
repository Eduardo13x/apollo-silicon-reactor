//! Deterministic navigation state and bounded WebFlow acceleration intents.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::engine::webflow_types::{
    OpaqueBucket, OpaqueId, ReceivedWebFlowEvent, WebFlowEvent, WebFlowPhase, WebFlowSource,
};

pub const INITIAL_WEBFLOW_LEASE_MS: u64 = 2_000;
pub const MAX_CONTINUOUS_WEBFLOW_LEASE_MS: u64 = 15_000;
pub const LIFECYCLE_LOADED_GRACE_MS: u64 = 500;
pub const MAX_WEBFLOW_EVENT_AGE_MS: u64 = 2_000;
pub const MAX_ACTIVE_WEBFLOW_NAVIGATIONS: usize = 64;
const MAX_BROWSER_SEQUENCES: usize = 64;
const MAX_CLOSED_PER_TICK: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebFlowRolloutPhase {
    Observe,
    Shadow,
    Canary,
    Active,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebFlowReason {
    NavigationLifecycle,
    NavigationVitals,
    InferredNavigation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebFlowIntent {
    pub navigation_id: Option<OpaqueId>,
    pub confidence_q: u16,
    pub intensity_q: u16,
    pub ttl_ms: u32,
    pub reason: WebFlowReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebFlowClosure {
    Settled,
    Failed,
    Abandoned,
    Expired,
    Invalidated,
    Capacity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebFlowEpisodeOutcome {
    pub navigation_id: OpaqueId,
    pub site_bucket: Option<OpaqueBucket>,
    pub closure: WebFlowClosure,
    pub elapsed_ms: u64,
    pub source: WebFlowSource,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebFlowCounters {
    pub accepted: u64,
    pub invalid: u64,
    pub stale: u64,
    pub duplicate: u64,
    pub out_of_order: u64,
    pub dropped: u64,
    pub proposed: u64,
    pub admitted: u64,
    pub skipped: u64,
    pub closed: u64,
    pub settled: u64,
    pub failed: u64,
    pub abandoned: u64,
    pub expired: u64,
    pub invalidated: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebWorldObservation {
    pub accepted_events: u16,
    pub active_navigations: u16,
    pub last_phase: Option<WebFlowPhase>,
    pub source: Option<WebFlowSource>,
    pub confidence_q: u16,
    pub last_event_age_ms: Option<u32>,
    pub vitals_available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebFlowTickInput {
    pub now_ms: u64,
    pub foreground_browser: bool,
    pub identity_available: bool,
    pub pressure_constrained: bool,
    pub thermal_constrained: bool,
    pub low_power: bool,
    pub sleeping: bool,
    pub kill_switch: bool,
    pub session_revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebFlowTickOutput {
    pub observation: WebWorldObservation,
    pub intent: Option<WebFlowIntent>,
    pub admitted: bool,
    pub closed: Vec<WebFlowEpisodeOutcome>,
    pub counters: WebFlowCounters,
}

#[derive(Debug, Clone)]
struct ActiveEpisode {
    navigation_id: OpaqueId,
    site_bucket: Option<OpaqueBucket>,
    source: WebFlowSource,
    phase: WebFlowPhase,
    started_at_ms: u64,
    last_update_ms: u64,
    expires_at_ms: u64,
    hard_deadline_ms: u64,
}

#[derive(Debug, Clone, Copy)]
struct BrowserSequence {
    sequence: u64,
    seen_at_ms: u64,
}

pub struct WebFlowController {
    rollout: WebFlowRolloutPhase,
    active: BTreeMap<OpaqueId, ActiveEpisode>,
    browser_sequences: BTreeMap<OpaqueId, BrowserSequence>,
    session_revision: u64,
    last_phase: Option<WebFlowPhase>,
    last_source: Option<WebFlowSource>,
    last_event_at_ms: Option<u64>,
    counters: WebFlowCounters,
}

impl WebFlowController {
    pub fn new(rollout: WebFlowRolloutPhase) -> Self {
        Self {
            rollout,
            active: BTreeMap::new(),
            browser_sequences: BTreeMap::new(),
            session_revision: 0,
            last_phase: None,
            last_source: None,
            last_event_at_ms: None,
            counters: WebFlowCounters::default(),
        }
    }

    pub fn tick(
        &mut self,
        input: WebFlowTickInput,
        events: impl IntoIterator<Item = ReceivedWebFlowEvent>,
    ) -> WebFlowTickOutput {
        let mut closed = Vec::new();
        if self.session_revision != 0 && self.session_revision != input.session_revision {
            self.close_all(WebFlowClosure::Invalidated, input.now_ms, &mut closed);
            self.browser_sequences.clear();
            self.last_phase = None;
            self.last_source = None;
            self.last_event_at_ms = None;
        }
        self.session_revision = input.session_revision;
        self.expire(input.now_ms, &mut closed);

        let mut accepted_this_tick = 0u16;
        for received in events {
            if received.event.validate().is_err() || received.received_at_ms == 0 {
                self.counters.invalid = self.counters.invalid.saturating_add(1);
                continue;
            }
            if received.received_at_ms > input.now_ms
                || input.now_ms.saturating_sub(received.received_at_ms) > MAX_WEBFLOW_EVENT_AGE_MS
            {
                self.counters.stale = self.counters.stale.saturating_add(1);
                continue;
            }
            if !self.accept_sequence(&received.event, input.now_ms) {
                continue;
            }
            accepted_this_tick = accepted_this_tick.saturating_add(1);
            self.counters.accepted = self.counters.accepted.saturating_add(1);
            self.last_phase = Some(received.event.phase);
            self.last_source = Some(match self.last_source {
                Some(current)
                    if current.precision_rank() > received.event.source.precision_rank() =>
                {
                    current
                }
                _ => received.event.source,
            });
            self.last_event_at_ms = Some(received.received_at_ms);
            self.apply_event(received.event, input.now_ms, &mut closed);
        }

        let constrained = !input.foreground_browser
            || !input.identity_available
            || input.pressure_constrained
            || input.thermal_constrained
            || input.low_power
            || input.sleeping
            || input.kill_switch;
        let intent = if constrained {
            if !self.active.is_empty() {
                self.counters.skipped = self.counters.skipped.saturating_add(1);
            }
            None
        } else {
            self.current_intent(input.now_ms)
        };
        let admitted = intent
            .as_ref()
            .is_some_and(|intent| self.admits(intent.navigation_id));
        if intent.is_some() {
            self.counters.proposed = self.counters.proposed.saturating_add(1);
            self.counters.admitted = self.counters.admitted.saturating_add(u64::from(admitted));
        }

        WebFlowTickOutput {
            observation: WebWorldObservation {
                accepted_events: accepted_this_tick,
                active_navigations: self.active.len().min(u16::MAX as usize) as u16,
                last_phase: self.last_phase,
                source: self.last_source,
                confidence_q: self.last_source.map_or(0, WebFlowSource::confidence_q),
                last_event_age_ms: self.last_event_at_ms.map(|at| {
                    input
                        .now_ms
                        .saturating_sub(at)
                        .min(u32::MAX as u64) as u32
                }),
                vitals_available: self.last_source == Some(WebFlowSource::ExtensionVitals),
            },
            intent,
            admitted,
            closed,
            counters: self.counters,
        }
    }

    fn accept_sequence(&mut self, event: &WebFlowEvent, now_ms: u64) -> bool {
        if let Some(previous) = self.browser_sequences.get(&event.browser_session_id) {
            if event.sequence == previous.sequence {
                self.counters.duplicate = self.counters.duplicate.saturating_add(1);
                return false;
            }
            if event.sequence < previous.sequence {
                self.counters.out_of_order = self.counters.out_of_order.saturating_add(1);
                return false;
            }
        }
        if !self.browser_sequences.contains_key(&event.browser_session_id)
            && self.browser_sequences.len() == MAX_BROWSER_SEQUENCES
        {
            if let Some(oldest) = self
                .browser_sequences
                .iter()
                .min_by_key(|(_, sequence)| sequence.seen_at_ms)
                .map(|(id, _)| *id)
            {
                self.browser_sequences.remove(&oldest);
            }
        }
        self.browser_sequences.insert(
            event.browser_session_id,
            BrowserSequence {
                sequence: event.sequence,
                seen_at_ms: now_ms,
            },
        );
        true
    }

    fn apply_event(
        &mut self,
        event: WebFlowEvent,
        now_ms: u64,
        closed: &mut Vec<WebFlowEpisodeOutcome>,
    ) {
        if event.phase.terminal() {
            if let Some(episode) = self.active.remove(&event.tab_session_id) {
                let closure = match event.phase {
                    WebFlowPhase::Settled => WebFlowClosure::Settled,
                    WebFlowPhase::Failed => WebFlowClosure::Failed,
                    WebFlowPhase::Abandoned => WebFlowClosure::Abandoned,
                    _ => unreachable!(),
                };
                self.push_closed(episode, closure, now_ms, closed);
            }
            return;
        }

        if self
            .active
            .get(&event.tab_session_id)
            .is_some_and(|episode| episode.navigation_id != event.navigation_id)
        {
            if let Some(previous) = self.active.remove(&event.tab_session_id) {
                self.push_closed(previous, WebFlowClosure::Abandoned, now_ms, closed);
            }
        }

        if !self.active.contains_key(&event.tab_session_id)
            && self.active.len() == MAX_ACTIVE_WEBFLOW_NAVIGATIONS
        {
            if let Some(oldest_tab) = self
                .active
                .iter()
                .min_by_key(|(_, episode)| episode.last_update_ms)
                .map(|(tab, _)| *tab)
            {
                if let Some(oldest) = self.active.remove(&oldest_tab) {
                    self.push_closed(oldest, WebFlowClosure::Capacity, now_ms, closed);
                    self.counters.dropped = self.counters.dropped.saturating_add(1);
                }
            }
        }

        let episode = self.active.entry(event.tab_session_id).or_insert_with(|| {
            ActiveEpisode {
                navigation_id: event.navigation_id,
                site_bucket: event.site_bucket,
                source: event.source,
                phase: event.phase,
                started_at_ms: now_ms,
                last_update_ms: now_ms,
                expires_at_ms: now_ms.saturating_add(INITIAL_WEBFLOW_LEASE_MS),
                hard_deadline_ms: now_ms.saturating_add(MAX_CONTINUOUS_WEBFLOW_LEASE_MS),
            }
        });
        if event.source.precision_rank() >= episode.source.precision_rank() {
            episode.source = event.source;
            episode.site_bucket = event.site_bucket.or(episode.site_bucket);
        }
        episode.phase = event.phase;
        episode.last_update_ms = now_ms;
        let requested_expiry = if event.phase == WebFlowPhase::Loaded
            && event.source == WebFlowSource::ExtensionLifecycle
        {
            now_ms.saturating_add(LIFECYCLE_LOADED_GRACE_MS)
        } else {
            now_ms.saturating_add(INITIAL_WEBFLOW_LEASE_MS)
        };
        episode.expires_at_ms = requested_expiry.min(episode.hard_deadline_ms);
    }

    fn expire(&mut self, now_ms: u64, closed: &mut Vec<WebFlowEpisodeOutcome>) {
        let expired: Vec<OpaqueId> = self
            .active
            .iter()
            .filter(|(_, episode)| {
                now_ms > episode.expires_at_ms || now_ms > episode.hard_deadline_ms
            })
            .map(|(tab, _)| *tab)
            .collect();
        for tab in expired {
            if let Some(episode) = self.active.remove(&tab) {
                self.push_closed(episode, WebFlowClosure::Expired, now_ms, closed);
            }
        }
    }

    fn close_all(
        &mut self,
        closure: WebFlowClosure,
        now_ms: u64,
        closed: &mut Vec<WebFlowEpisodeOutcome>,
    ) {
        let episodes: VecDeque<_> = std::mem::take(&mut self.active).into_values().collect();
        for episode in episodes {
            self.push_closed(episode, closure, now_ms, closed);
        }
    }

    fn push_closed(
        &mut self,
        episode: ActiveEpisode,
        closure: WebFlowClosure,
        now_ms: u64,
        closed: &mut Vec<WebFlowEpisodeOutcome>,
    ) {
        self.counters.closed = self.counters.closed.saturating_add(1);
        match closure {
            WebFlowClosure::Settled => {
                self.counters.settled = self.counters.settled.saturating_add(1)
            }
            WebFlowClosure::Failed => {
                self.counters.failed = self.counters.failed.saturating_add(1)
            }
            WebFlowClosure::Abandoned => {
                self.counters.abandoned = self.counters.abandoned.saturating_add(1)
            }
            WebFlowClosure::Expired | WebFlowClosure::Capacity => {
                self.counters.expired = self.counters.expired.saturating_add(1)
            }
            WebFlowClosure::Invalidated => {
                self.counters.invalidated = self.counters.invalidated.saturating_add(1)
            }
        }
        if closed.len() < MAX_CLOSED_PER_TICK {
            closed.push(WebFlowEpisodeOutcome {
                navigation_id: episode.navigation_id,
                site_bucket: episode.site_bucket,
                closure,
                elapsed_ms: now_ms.saturating_sub(episode.started_at_ms),
                source: episode.source,
            });
        }
    }

    fn current_intent(&self, now_ms: u64) -> Option<WebFlowIntent> {
        let episode = self
            .active
            .values()
            .max_by_key(|episode| episode.last_update_ms)?;
        let ttl_ms = episode
            .expires_at_ms
            .saturating_sub(now_ms)
            .min(INITIAL_WEBFLOW_LEASE_MS);
        if ttl_ms == 0 {
            return None;
        }
        Some(WebFlowIntent {
            navigation_id: Some(episode.navigation_id),
            confidence_q: episode.source.confidence_q(),
            intensity_q: match episode.source {
                WebFlowSource::ExtensionVitals => 8_500,
                WebFlowSource::ExtensionLifecycle => 7_500,
                WebFlowSource::DaemonInference => 5_000,
            },
            ttl_ms: ttl_ms as u32,
            reason: match episode.source {
                WebFlowSource::ExtensionVitals => WebFlowReason::NavigationVitals,
                WebFlowSource::ExtensionLifecycle => WebFlowReason::NavigationLifecycle,
                WebFlowSource::DaemonInference => WebFlowReason::InferredNavigation,
            },
        })
    }

    fn admits(&self, navigation_id: Option<OpaqueId>) -> bool {
        match self.rollout {
            WebFlowRolloutPhase::Observe | WebFlowRolloutPhase::Shadow => false,
            WebFlowRolloutPhase::Active => true,
            WebFlowRolloutPhase::Canary => navigation_id.is_some_and(|id| id.bytes()[0] % 10 == 0),
        }
    }
}
