//! Bounded, ordered event ingress shared by platform sources.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::engine::webflow_types::{WebFlowPhase, WebFlowSource};

pub const EVENT_MESH_CAPACITY: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EventSource {
    Lifecycle,
    Pressure,
    Thermal,
    Power,
    Process,
    Session,
    AudioActivity,
    VisualActivity,
    Network,
    Filesystem,
    WebFlow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LifecycleEvent {
    Sleep,
    Wake,
    SessionChanged,
    SourceDisconnected,
    SourceRestarted,
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum EventPayload {
    Scalar {
        value: f32,
    },
    Lifecycle {
        event: LifecycleEvent,
    },
    WebFlow {
        phase: WebFlowPhase,
        source: WebFlowSource,
        active_navigations: u16,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub daemon_epoch: u64,
    pub source: EventSource,
    pub source_generation: u64,
    pub source_sequence: u64,
    pub monotonic_time_us: u64,
    pub confidence_q: u16,
    pub ingest_sequence: u64,
    pub payload: EventPayload,
}

impl EventEnvelope {
    pub fn scalar(
        daemon_epoch: u64,
        source: EventSource,
        source_generation: u64,
        source_sequence: u64,
        monotonic_time_us: u64,
        confidence_q: u16,
        value: f32,
    ) -> Result<Self, &'static str> {
        if daemon_epoch == 0 || !value.is_finite() || confidence_q > 10_000 {
            return Err("invalid scalar event");
        }
        Ok(Self {
            daemon_epoch,
            source,
            source_generation,
            source_sequence,
            monotonic_time_us,
            confidence_q,
            ingest_sequence: 0,
            payload: EventPayload::Scalar { value },
        })
    }

    pub fn lifecycle(
        daemon_epoch: u64,
        source_generation: u64,
        source_sequence: u64,
        monotonic_time_us: u64,
        event: LifecycleEvent,
    ) -> Self {
        Self {
            daemon_epoch,
            source: EventSource::Lifecycle,
            source_generation,
            source_sequence,
            monotonic_time_us,
            confidence_q: 10_000,
            ingest_sequence: 0,
            payload: EventPayload::Lifecycle { event },
        }
    }

    pub fn webflow(
        daemon_epoch: u64,
        source_generation: u64,
        source_sequence: u64,
        monotonic_time_us: u64,
        phase: WebFlowPhase,
        source: WebFlowSource,
        active_navigations: u16,
    ) -> Result<Self, &'static str> {
        if daemon_epoch == 0 || source_generation == 0 || active_navigations > 64 {
            return Err("invalid WebFlow event");
        }
        Ok(Self {
            daemon_epoch,
            source: EventSource::WebFlow,
            source_generation,
            source_sequence,
            monotonic_time_us,
            confidence_q: source.confidence_q(),
            ingest_sequence: 0,
            payload: EventPayload::WebFlow {
                phase,
                source,
                active_navigations,
            },
        })
    }

    fn replaceable(&self) -> bool {
        matches!(
            self.source,
            EventSource::Pressure
                | EventSource::Thermal
                | EventSource::Power
                | EventSource::AudioActivity
                | EventSource::VisualActivity
                | EventSource::Network
        ) && matches!(self.payload, EventPayload::Scalar { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventIngestOutcome {
    Accepted,
    Coalesced,
    Duplicate,
    OutOfOrder,
    Dropped,
    WrongEpoch,
    Closed,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventMeshMetrics {
    pub accepted_total: u64,
    pub coalesced_total: u64,
    pub duplicate_total: u64,
    pub out_of_order_total: u64,
    pub dropped_total: u64,
    pub wrong_epoch_total: u64,
}

pub struct EventMesh {
    daemon_epoch: u64,
    next_ingest_sequence: u64,
    queue: VecDeque<EventEnvelope>,
    last_source: BTreeMap<EventSource, (u64, u64)>,
    degraded: BTreeSet<EventSource>,
    metrics: EventMeshMetrics,
    closed: bool,
}

impl EventMesh {
    pub fn new(daemon_epoch: u64) -> Self {
        Self {
            daemon_epoch: daemon_epoch.max(1),
            next_ingest_sequence: 1,
            queue: VecDeque::with_capacity(EVENT_MESH_CAPACITY),
            last_source: BTreeMap::new(),
            degraded: BTreeSet::new(),
            metrics: EventMeshMetrics::default(),
            closed: false,
        }
    }

    pub const fn capacity(&self) -> usize {
        EVENT_MESH_CAPACITY
    }
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    pub const fn metrics(&self) -> EventMeshMetrics {
        self.metrics
    }
    pub fn source_degraded(&self, source: EventSource) -> bool {
        self.degraded.contains(&source)
    }

    pub fn ingest(&mut self, mut event: EventEnvelope) -> EventIngestOutcome {
        if self.closed || self.next_ingest_sequence == u64::MAX {
            self.closed = true;
            return EventIngestOutcome::Closed;
        }
        if event.daemon_epoch != self.daemon_epoch {
            self.metrics.wrong_epoch_total = self.metrics.wrong_epoch_total.saturating_add(1);
            return EventIngestOutcome::WrongEpoch;
        }
        if let Some((generation, sequence)) = self.last_source.get(&event.source).copied() {
            if event.source_generation < generation
                || (event.source_generation == generation && event.source_sequence < sequence)
            {
                self.metrics.out_of_order_total = self.metrics.out_of_order_total.saturating_add(1);
                return EventIngestOutcome::OutOfOrder;
            }
            if event.source_generation == generation && event.source_sequence == sequence {
                self.metrics.duplicate_total = self.metrics.duplicate_total.saturating_add(1);
                return EventIngestOutcome::Duplicate;
            }
        }

        event.ingest_sequence = self.next_ingest_sequence;
        self.next_ingest_sequence = self.next_ingest_sequence.saturating_add(1);
        self.last_source.insert(
            event.source,
            (event.source_generation, event.source_sequence),
        );

        if self.queue.len() == EVENT_MESH_CAPACITY {
            if event.replaceable() {
                if let Some(index) = self
                    .queue
                    .iter()
                    .position(|queued| queued.source == event.source && queued.replaceable())
                {
                    self.queue.remove(index);
                    self.queue.push_back(event);
                    self.metrics.coalesced_total = self.metrics.coalesced_total.saturating_add(1);
                    return EventIngestOutcome::Coalesced;
                }
            }
            self.degraded.insert(event.source);
            self.metrics.dropped_total = self.metrics.dropped_total.saturating_add(1);
            return EventIngestOutcome::Dropped;
        }

        self.queue.push_back(event);
        self.metrics.accepted_total = self.metrics.accepted_total.saturating_add(1);
        EventIngestOutcome::Accepted
    }

    pub fn drain(&mut self, maximum: usize) -> Vec<EventEnvelope> {
        let count = maximum.min(self.queue.len());
        self.queue.drain(..count).collect()
    }

    pub fn close(&mut self) {
        self.closed = true;
    }
}
