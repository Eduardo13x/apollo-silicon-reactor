//! Bounded, source-agnostic store for perceptual observations.
//!
//! One collection for every source. A per-application collection would make the
//! store grow with the machine's app list and would let two sources' evidence
//! be compared without anyone noticing the modality differed.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::types::{
    CorrelationState, MeasurementMode, PerceptualObservation, PerceptualSourceKind, QUALITY_SCALE,
};

pub const MAX_OBSERVATIONS: usize = 512;
pub const OBSERVATION_TTL_MS: u64 = 15 * 60 * 1000;
/// Producers tracked at once. Beyond this the machine is not running more
/// instrumented apps — something is spoofing producer identities.
pub const MAX_ACTIVE_PRODUCERS: usize = 16;
pub const PERCEPTUAL_STORE_SCHEMA_VERSION: u16 = 1;

/// Counts that reconcile: every ingested observation lands in exactly one of
/// `stored`, `refused_correlation` or `refused_capacity`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptualStoreMetrics {
    pub ingested_total: u64,
    pub stored_total: u64,
    pub refused_correlation: u64,
    pub refused_capacity: u64,
    pub evicted_capacity: u64,
    pub evicted_ttl: u64,
    pub instrumented_total: u64,
    pub inferred_total: u64,
    pub window_total: u64,
}

impl PerceptualStoreMetrics {
    pub fn reconciles(&self) -> bool {
        self.stored_total
            .saturating_add(self.refused_correlation)
            .saturating_add(self.refused_capacity)
            == self.ingested_total
    }
}

#[derive(Debug, Default)]
pub struct PerceptualObservationStore {
    observations: VecDeque<PerceptualObservation>,
    /// Bounded index by source kind — a closed enum, never an arbitrary string.
    per_source: [u64; 9],
    active_producers: Vec<[u8; 16]>,
    pub metrics: PerceptualStoreMetrics,
}

fn source_slot(kind: PerceptualSourceKind) -> usize {
    match kind {
        PerceptualSourceKind::Unknown => 0,
        PerceptualSourceKind::BrowserChromium => 1,
        PerceptualSourceKind::Editor => 2,
        PerceptualSourceKind::Terminal => 3,
        PerceptualSourceKind::NativeApplication => 4,
        PerceptualSourceKind::ElectronApplication => 5,
        PerceptualSourceKind::WindowSystem => 6,
        PerceptualSourceKind::GenericForegroundApplication => 7,
        PerceptualSourceKind::Synthetic => 8,
    }
}

impl PerceptualObservationStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one observation. Refused observations are still counted: a rising
    /// ambiguous rate is evidence about the producer, not silence.
    pub fn record(&mut self, observation: PerceptualObservation) -> bool {
        self.metrics.ingested_total = self.metrics.ingested_total.saturating_add(1);
        let header = observation.header();

        if !header.correlation.admits_to_aggregate() {
            self.metrics.refused_correlation = self.metrics.refused_correlation.saturating_add(1);
            return false;
        }
        let producer = header.scope.producer_session_id.bytes();
        if !self.active_producers.contains(&producer) {
            if self.active_producers.len() >= MAX_ACTIVE_PRODUCERS {
                self.metrics.refused_capacity = self.metrics.refused_capacity.saturating_add(1);
                return false;
            }
            self.active_producers.push(producer);
        }

        if self.observations.len() >= MAX_OBSERVATIONS {
            if let Some(dropped) = self.observations.pop_front() {
                self.per_source[source_slot(dropped.header().source_kind)] =
                    self.per_source[source_slot(dropped.header().source_kind)].saturating_sub(1);
            }
            self.metrics.evicted_capacity = self.metrics.evicted_capacity.saturating_add(1);
        }
        match &observation {
            PerceptualObservation::InstrumentedInteraction(_) => {
                self.metrics.instrumented_total = self.metrics.instrumented_total.saturating_add(1);
            }
            PerceptualObservation::InferredInteraction(_) => {
                self.metrics.inferred_total = self.metrics.inferred_total.saturating_add(1);
            }
            PerceptualObservation::PerceptualWindow(_) => {
                self.metrics.window_total = self.metrics.window_total.saturating_add(1);
            }
        }
        self.per_source[source_slot(header.source_kind)] =
            self.per_source[source_slot(header.source_kind)].saturating_add(1);
        self.metrics.stored_total = self.metrics.stored_total.saturating_add(1);
        self.observations.push_back(observation);
        true
    }

    pub fn expire(&mut self, now_ms: u64) {
        while let Some(front) = self.observations.front() {
            if now_ms.saturating_sub(front.header().observed_at_ms) > OBSERVATION_TTL_MS {
                let kind = front.header().source_kind;
                self.observations.pop_front();
                self.per_source[source_slot(kind)] =
                    self.per_source[source_slot(kind)].saturating_sub(1);
                self.metrics.evicted_ttl = self.metrics.evicted_ttl.saturating_add(1);
            } else {
                break;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &PerceptualObservation> {
        self.observations.iter()
    }

    pub fn count_for(&self, kind: PerceptualSourceKind) -> u64 {
        self.per_source[source_slot(kind)]
    }

    pub fn active_sources(&self) -> usize {
        self.per_source.iter().filter(|count| **count > 0).count()
    }

    pub fn active_producers(&self) -> usize {
        self.active_producers.len()
    }

    /// Observations of one modality and measurement mode. The only supported
    /// way to build a comparable set: mixing modalities would read precision
    /// that was never measured.
    pub fn comparable_set(
        &self,
        modality: &str,
        mode: MeasurementMode,
    ) -> Vec<&PerceptualObservation> {
        self.observations
            .iter()
            .filter(|o| o.modality() == modality && o.measurement().measurement_mode == mode)
            .collect()
    }

    /// Mean quality on the 0..=1000 scale, over stored observations only.
    pub fn mean_quality_q(&self) -> u16 {
        if self.observations.is_empty() {
            return 0;
        }
        let sum: u64 = self
            .observations
            .iter()
            .map(|o| u64::from(o.header().quality.overall_q()))
            .sum();
        (sum / self.observations.len() as u64).min(u64::from(QUALITY_SCALE)) as u16
    }
}

/// Versioned persistence. Restored observations keep their provenance: nothing
/// gains confidence by surviving a restart.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PerceptualStorePersisted {
    pub schema_version: u16,
    pub observations: Vec<PerceptualObservation>,
    pub metrics: PerceptualStoreMetrics,
}

impl PerceptualObservationStore {
    pub fn persisted(&self) -> PerceptualStorePersisted {
        PerceptualStorePersisted {
            schema_version: PERCEPTUAL_STORE_SCHEMA_VERSION,
            observations: self.observations.iter().cloned().collect(),
            metrics: self.metrics,
        }
    }

    /// Restore. A state from an unknown schema is discarded rather than
    /// reinterpreted — silently reading unknown bytes as a newer type is how a
    /// measurement acquires precision it never had.
    pub fn restore(state: PerceptualStorePersisted) -> Self {
        let mut store = Self::new();
        if state.schema_version != PERCEPTUAL_STORE_SCHEMA_VERSION {
            return store;
        }
        store.metrics = state.metrics;
        for observation in state.observations.into_iter().take(MAX_OBSERVATIONS) {
            let header = observation.header();
            let producer = header.scope.producer_session_id.bytes();
            if !store.active_producers.contains(&producer)
                && store.active_producers.len() < MAX_ACTIVE_PRODUCERS
            {
                store.active_producers.push(producer);
            }
            store.per_source[source_slot(header.source_kind)] += 1;
            store.observations.push_back(observation);
        }
        store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::perceptual::types::*;

    fn observation(
        at_ms: u64,
        kind: PerceptualSourceKind,
        correlation: CorrelationState,
        producer: u8,
    ) -> PerceptualObservation {
        PerceptualObservation::InstrumentedInteraction(InstrumentedInteractionEpisode {
            header: ObservationHeader {
                observed_at_ms: at_ms,
                source_kind: kind,
                producer_kind: ProducerKind::SyntheticTest,
                scope: InteractionScope {
                    producer_session_id: PerceptualId::new([producer.max(1); 16]).expect("id"),
                    ..InteractionScope::default()
                },
                quality: PerceptualQuality {
                    source_trust_q: 800,
                    measurement_quality_q: 900,
                    temporal_confidence_q: 700,
                    correlation_confidence_q: 900,
                    attribution_confidence_q: 900,
                },
                correlation,
                transport: PerceptualTransportTrace::default(),
                legacy_contract: false,
            },
            measurement: PerceptualMeasurement {
                total_duration_ms: Some(120),
                components: Vec::new(),
                measurement_mode: MeasurementMode::Instrumented,
            },
        })
    }

    fn window(at_ms: u64) -> PerceptualObservation {
        PerceptualObservation::PerceptualWindow(PerceptualWindowObservation {
            header: ObservationHeader {
                observed_at_ms: at_ms,
                source_kind: PerceptualSourceKind::GenericForegroundApplication,
                producer_kind: ProducerKind::MacOsObserver,
                scope: InteractionScope {
                    producer_session_id: PerceptualId::new([5; 16]).expect("id"),
                    ..InteractionScope::default()
                },
                quality: PerceptualQuality {
                    source_trust_q: 500,
                    measurement_quality_q: 400,
                    temporal_confidence_q: 300,
                    correlation_confidence_q: 500,
                    attribution_confidence_q: 400,
                },
                correlation: CorrelationState::CompletedOnly,
                transport: PerceptualTransportTrace::default(),
                legacy_contract: false,
            },
            window_ms: 8_000,
            signal_count: 12,
            sluggishness_q: 300,
            measurement: PerceptualMeasurement {
                measurement_mode: MeasurementMode::AggregateWindow,
                ..PerceptualMeasurement::default()
            },
        })
    }

    #[test]
    fn one_store_holds_every_source_without_a_per_app_collection() {
        let mut store = PerceptualObservationStore::new();
        store.record(observation(
            1,
            PerceptualSourceKind::BrowserChromium,
            CorrelationState::Unique,
            1,
        ));
        store.record(window(2));
        assert_eq!(store.len(), 2);
        assert_eq!(store.count_for(PerceptualSourceKind::BrowserChromium), 1);
        assert_eq!(
            store.count_for(PerceptualSourceKind::GenericForegroundApplication),
            1
        );
        assert_eq!(store.active_sources(), 2);
    }

    #[test]
    fn refused_observations_are_counted_and_the_totals_reconcile() {
        let mut store = PerceptualObservationStore::new();
        store.record(observation(
            1,
            PerceptualSourceKind::Synthetic,
            CorrelationState::Unique,
            1,
        ));
        store.record(observation(
            2,
            PerceptualSourceKind::Synthetic,
            CorrelationState::Ambiguous,
            1,
        ));
        store.record(observation(
            3,
            PerceptualSourceKind::Synthetic,
            CorrelationState::InvalidTiming,
            1,
        ));
        assert_eq!(store.len(), 1);
        assert_eq!(store.metrics.refused_correlation, 2);
        assert!(store.metrics.reconciles());
    }

    #[test]
    fn the_store_is_bounded_and_evicts_the_oldest() {
        let mut store = PerceptualObservationStore::new();
        for index in 0..(MAX_OBSERVATIONS + 10) {
            store.record(observation(
                index as u64,
                PerceptualSourceKind::Synthetic,
                CorrelationState::Unique,
                1,
            ));
        }
        assert_eq!(store.len(), MAX_OBSERVATIONS);
        assert_eq!(store.metrics.evicted_capacity, 10);
        assert!(store.metrics.reconciles());
    }

    #[test]
    fn a_spoofing_producer_cannot_grow_the_active_set_without_limit() {
        let mut store = PerceptualObservationStore::new();
        for producer in 1..=(MAX_ACTIVE_PRODUCERS as u8 + 5) {
            store.record(observation(
                u64::from(producer),
                PerceptualSourceKind::Synthetic,
                CorrelationState::Unique,
                producer,
            ));
        }
        assert_eq!(store.active_producers(), MAX_ACTIVE_PRODUCERS);
        assert_eq!(store.metrics.refused_capacity, 5);
        assert!(store.metrics.reconciles());
    }

    #[test]
    fn stale_observations_expire() {
        let mut store = PerceptualObservationStore::new();
        store.record(observation(
            0,
            PerceptualSourceKind::Synthetic,
            CorrelationState::Unique,
            1,
        ));
        store.record(observation(
            OBSERVATION_TTL_MS + 1,
            PerceptualSourceKind::Synthetic,
            CorrelationState::Unique,
            1,
        ));
        store.expire(OBSERVATION_TTL_MS + 2);
        assert_eq!(store.len(), 1);
        assert_eq!(store.metrics.evicted_ttl, 1);
    }

    #[test]
    fn a_comparable_set_never_mixes_modalities() {
        let mut store = PerceptualObservationStore::new();
        store.record(observation(
            1,
            PerceptualSourceKind::BrowserChromium,
            CorrelationState::Unique,
            1,
        ));
        store.record(window(2));
        let instrumented = store.comparable_set("instrumented", MeasurementMode::Instrumented);
        let windows = store.comparable_set("window", MeasurementMode::AggregateWindow);
        assert_eq!(instrumented.len(), 1);
        assert_eq!(windows.len(), 1);
        assert!(!instrumented[0].comparable_with(windows[0]));
    }

    #[test]
    fn persistence_round_trips_and_keeps_provenance() {
        let mut store = PerceptualObservationStore::new();
        store.record(observation(
            1,
            PerceptualSourceKind::BrowserChromium,
            CorrelationState::Unique,
            1,
        ));
        store.record(window(2));
        let restored = PerceptualObservationStore::restore(store.persisted());
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.metrics.stored_total, store.metrics.stored_total);
        assert_eq!(restored.count_for(PerceptualSourceKind::BrowserChromium), 1);
    }

    #[test]
    fn an_unknown_persisted_schema_is_discarded_rather_than_reinterpreted() {
        let mut store = PerceptualObservationStore::new();
        store.record(observation(
            1,
            PerceptualSourceKind::Synthetic,
            CorrelationState::Unique,
            1,
        ));
        let mut state = store.persisted();
        state.schema_version = PERCEPTUAL_STORE_SCHEMA_VERSION + 1;
        let restored = PerceptualObservationStore::restore(state);
        assert!(restored.is_empty());
        assert_eq!(restored.metrics.stored_total, 0);
    }

    #[test]
    fn quality_reflects_the_weakest_dimension_across_the_store() {
        let mut store = PerceptualObservationStore::new();
        store.record(observation(
            1,
            PerceptualSourceKind::Synthetic,
            CorrelationState::Unique,
            1,
        ));
        // overall_q is the minimum: 700 for the instrumented fixture.
        assert_eq!(store.mean_quality_q(), 700);
    }
}
