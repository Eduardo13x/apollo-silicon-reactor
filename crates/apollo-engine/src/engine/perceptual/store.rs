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
/// Observations written to disk. Far below the in-memory capacity on purpose:
/// persistence exists to survive a restart with recent evidence, not to archive
/// a high-frequency stream. At roughly 500 bytes each this stays well inside a
/// 2 MiB budget with room to spare.
pub const MAX_PERSISTED_OBSERVATIONS: usize = 128;

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

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &PerceptualObservation> {
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
    /// Aggregate mean quality. Derived from the same walk as the per-modality
    /// figures so a publish pays for one pass, not two.
    pub fn mean_quality_q(&self) -> u16 {
        self.quality_by_modality().overall_q.unwrap_or(0)
    }

    /// Mean quality split by modality, each with the resident count it was
    /// computed over.
    ///
    /// The aggregate `mean_quality_q` moves when the *mix* of evidence
    /// changes, not only when measurement gets worse: a burst of low-precision
    /// windows pulls it down while every instrumented episode stays exactly as
    /// good as it was. Read alone it invites the wrong conclusion. Split, each
    /// figure is stable and the mix shift becomes visible instead of hidden.
    ///
    /// A modality with no resident observations reports `None`, never `0`. Zero
    /// is a measurement; absence is not.
    ///
    /// The counts are *resident*, matching the observations actually averaged.
    /// Pairing these means with the lifetime totals would divide by the wrong n.
    pub fn quality_by_modality(&self) -> ModalityQuality {
        let mut acc = [(0u64, 0u32); 3];
        for o in &self.observations {
            let slot = match o {
                PerceptualObservation::InstrumentedInteraction(_) => 0,
                PerceptualObservation::InferredInteraction(_) => 1,
                PerceptualObservation::PerceptualWindow(_) => 2,
            };
            acc[slot].0 += u64::from(o.header().quality.overall_q());
            acc[slot].1 += 1;
        }
        let mean = |(sum, n): (u64, u32)| -> Option<u16> {
            (n > 0).then(|| (sum / u64::from(n)).min(u64::from(QUALITY_SCALE)) as u16)
        };
        let total = acc
            .iter()
            .fold((0u64, 0u32), |(s, n), (bs, bn)| (s + bs, n + bn));
        ModalityQuality {
            overall_q: mean(total),
            instrumented_q: mean(acc[0]),
            instrumented_n: acc[0].1,
            inferred_q: mean(acc[1]),
            inferred_n: acc[1].1,
            window_q: mean(acc[2]),
            window_n: acc[2].1,
        }
    }
}

/// Per-modality quality with the resident count behind each figure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModalityQuality {
    /// Mean across every modality — the figure that follows the evidence mix.
    pub overall_q: Option<u16>,
    pub instrumented_q: Option<u16>,
    pub instrumented_n: u32,
    pub inferred_q: Option<u16>,
    pub inferred_n: u32,
    pub window_q: Option<u16>,
    pub window_n: u32,
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
    /// Snapshot for disk. Only the newest observations are written, and only
    /// ones still inside their TTL as of `now_ms`: writing an already-expired
    /// observation would restore evidence that was never valid again.
    pub fn persisted_at(&self, now_ms: u64) -> PerceptualStorePersisted {
        let fresh: Vec<PerceptualObservation> = self
            .observations
            .iter()
            .filter(|o| now_ms.saturating_sub(o.header().observed_at_ms) <= OBSERVATION_TTL_MS)
            .cloned()
            .collect();
        let start = fresh.len().saturating_sub(MAX_PERSISTED_OBSERVATIONS);
        PerceptualStorePersisted {
            schema_version: PERCEPTUAL_STORE_SCHEMA_VERSION,
            observations: fresh[start..].to_vec(),
            metrics: self.metrics,
        }
    }

    pub fn persisted(&self) -> PerceptualStorePersisted {
        let newest = self
            .observations
            .back()
            .map_or(0, |o| o.header().observed_at_ms);
        self.persisted_at(newest)
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
        for observation in state
            .observations
            .into_iter()
            .take(MAX_PERSISTED_OBSERVATIONS)
        {
            // A partially corrupt entry is skipped, not repaired: an identity
            // we cannot trust must never re-enter the aggregate.
            if observation.header().scope.producer_session_id.bytes() == [0; 16] {
                continue;
            }
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
    fn a_shift_in_the_evidence_mix_is_not_a_drop_in_quality() {
        // The aggregate falls purely because low-precision windows arrive,
        // while every instrumented episode stays exactly as good as it was.
        // Reading the aggregate alone would call that a regression.
        let mut store = PerceptualObservationStore::new();
        store.record(observation(
            1,
            PerceptualSourceKind::BrowserChromium,
            CorrelationState::Unique,
            1,
        ));
        let only_instrumented = store.mean_quality_q();

        for i in 0..9 {
            store.record(window(10 + i));
        }
        let after_windows = store.mean_quality_q();
        assert!(
            after_windows < only_instrumented,
            "aggregate should fall with the mix: {after_windows} vs {only_instrumented}"
        );

        // Split, neither modality moved. Only their proportion did.
        let q = store.quality_by_modality();
        assert_eq!(q.instrumented_q, Some(700));
        assert_eq!(q.instrumented_n, 1);
        assert_eq!(q.window_q, Some(300));
        assert_eq!(q.window_n, 9);
    }

    #[test]
    fn a_modality_with_no_observations_reports_absence_not_zero() {
        let mut store = PerceptualObservationStore::new();
        store.record(window(1));
        let q = store.quality_by_modality();
        assert_eq!(q.window_q, Some(300));
        // Nothing instrumented and nothing inferred has been seen. Zero would
        // claim we measured them and found them worthless.
        assert_eq!(q.instrumented_q, None);
        assert_eq!(q.instrumented_n, 0);
        assert_eq!(q.inferred_q, None);
        assert_eq!(q.inferred_n, 0);
    }

    #[test]
    fn per_modality_counts_are_resident_not_lifetime() {
        let mut store = PerceptualObservationStore::new();
        for i in 0..(MAX_OBSERVATIONS + 20) {
            store.record(window(i as u64 + 1));
        }
        let q = store.quality_by_modality();
        assert_eq!(q.window_n as usize, store.len());
        assert!(
            (q.window_n as u64) < store.metrics.stored_total,
            "resident {} must trail lifetime {}",
            q.window_n,
            store.metrics.stored_total
        );
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

#[cfg(test)]
mod persistence_tests {
    use super::*;
    use crate::engine::perceptual::types::*;

    fn obs(at_ms: u64, producer: u8) -> PerceptualObservation {
        PerceptualObservation::PerceptualWindow(PerceptualWindowObservation {
            header: ObservationHeader {
                observed_at_ms: at_ms,
                source_kind: PerceptualSourceKind::GenericForegroundApplication,
                producer_kind: ProducerKind::MacOsObserver,
                scope: InteractionScope {
                    producer_session_id: PerceptualId::new([producer.max(1); 16]).expect("id"),
                    ..InteractionScope::default()
                },
                quality: PerceptualQuality {
                    source_trust_q: 900,
                    measurement_quality_q: 350,
                    temporal_confidence_q: 300,
                    correlation_confidence_q: 400,
                    attribution_confidence_q: 420,
                },
                correlation: CorrelationState::CompletedOnly,
                transport: PerceptualTransportTrace::default(),
                legacy_contract: false,
            },
            window_ms: 8_000,
            signal_count: 6,
            sluggishness_q: 210,
            measurement: PerceptualMeasurement {
                measurement_mode: MeasurementMode::AggregateWindow,
                ..PerceptualMeasurement::default()
            },
        })
    }

    #[test]
    fn only_a_bounded_recent_slice_reaches_disk() {
        let mut store = PerceptualObservationStore::new();
        for index in 0..(MAX_PERSISTED_OBSERVATIONS + 60) {
            store.record(obs(index as u64, 1));
        }
        let state = store.persisted();
        assert_eq!(state.observations.len(), MAX_PERSISTED_OBSERVATIONS);
        // The newest survive, not the oldest.
        let newest = state
            .observations
            .last()
            .expect("some")
            .header()
            .observed_at_ms;
        assert_eq!(newest, (MAX_PERSISTED_OBSERVATIONS + 59) as u64);
    }

    #[test]
    fn an_already_expired_observation_is_never_written() {
        let mut store = PerceptualObservationStore::new();
        store.record(obs(0, 1));
        store.record(obs(OBSERVATION_TTL_MS + 5_000, 1));
        let state = store.persisted_at(OBSERVATION_TTL_MS + 5_000);
        assert_eq!(
            state.observations.len(),
            1,
            "stale evidence must not be restored as if it were current"
        );
    }

    #[test]
    fn an_empty_store_round_trips_to_an_empty_store() {
        let store = PerceptualObservationStore::new();
        let restored = PerceptualObservationStore::restore(store.persisted());
        assert!(restored.is_empty());
        assert_eq!(restored.active_sources(), 0);
    }

    #[test]
    fn a_partially_corrupt_entry_is_skipped_rather_than_repaired() {
        let mut store = PerceptualObservationStore::new();
        store.record(obs(1, 1));
        store.record(obs(2, 2));
        let mut state = store.persisted();
        // Zero the identity of one entry, as a truncated write would.
        if let PerceptualObservation::PerceptualWindow(ref mut w) = state.observations[0] {
            w.header.scope.producer_session_id = PerceptualId([0; 16]);
        }
        let restored = PerceptualObservationStore::restore(state);
        assert_eq!(restored.len(), 1, "the trustworthy entry survives alone");
    }

    #[test]
    fn a_restart_does_not_duplicate_what_it_restores() {
        let mut store = PerceptualObservationStore::new();
        for index in 0..10u64 {
            store.record(obs(index, 1));
        }
        let restored = PerceptualObservationStore::restore(store.persisted());
        assert_eq!(restored.len(), 10);
        // Restoring the same state again yields the same count, not double.
        let twice = PerceptualObservationStore::restore(restored.persisted());
        assert_eq!(twice.len(), 10);
    }

    #[test]
    fn source_kind_quality_and_correlation_survive_the_round_trip() {
        let mut store = PerceptualObservationStore::new();
        store.record(obs(1, 1));
        let restored = PerceptualObservationStore::restore(store.persisted());
        let header = restored.iter().next().expect("one").header();
        assert_eq!(
            header.source_kind,
            PerceptualSourceKind::GenericForegroundApplication
        );
        assert_eq!(header.quality.measurement_quality_q, 350);
        assert_eq!(header.correlation, CorrelationState::CompletedOnly);
        assert_eq!(
            restored.count_for(PerceptualSourceKind::GenericForegroundApplication),
            1
        );
    }

    #[test]
    fn the_persisted_payload_stays_far_inside_the_disk_budget() {
        let mut store = PerceptualObservationStore::new();
        for index in 0..MAX_OBSERVATIONS {
            store.record(obs(index as u64, 1));
        }
        let json = serde_json::to_vec(&store.persisted()).expect("encodes");
        assert!(
            json.len() < 2 * 1024 * 1024,
            "persisted store is {} bytes, over the 2 MiB budget",
            json.len()
        );
        // Recorded so a future change that inflates the entry is visible.
        assert!(
            json.len() < 256 * 1024,
            "unexpectedly large: {}",
            json.len()
        );
    }
}
