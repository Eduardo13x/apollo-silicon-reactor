//! Synthetic producer for end-to-end tests.
//!
//! Exists to prove the core accepts a source with a shape the browser adapter
//! never has: no INP, no `interactionId`, no input/processing/presentation, an
//! optional total, no start event. If the core ever grows a hidden browser
//! assumption, these tests are where it surfaces.

use crate::engine::perceptual::capabilities::PerceptualCapabilities;
use crate::engine::perceptual::types::*;
use crate::engine::perceptual::validation::{
    validate_envelope, PerceptualValidationError, PERCEPTUAL_SCHEMA_VERSION,
};

use super::{AdapterHealth, PerceptualAdapter};

#[derive(Debug, Default)]
pub struct SyntheticPerceptualAdapter {
    sequence: u64,
    health: AdapterHealth,
}

impl SyntheticPerceptualAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    fn header(&self, at_ms: u64, correlation: CorrelationState) -> ObservationHeader {
        ObservationHeader {
            observed_at_ms: at_ms,
            source_kind: PerceptualSourceKind::Synthetic,
            producer_kind: ProducerKind::SyntheticTest,
            scope: InteractionScope {
                producer_session_id: PerceptualId::new([0x5A; 16]).expect("id"),
                surface_session_id: None,
                activity_session_id: None,
                context_hash: Some(ContextBucket(0x1234)),
            },
            quality: PerceptualQuality {
                source_trust_q: 600,
                measurement_quality_q: 500,
                temporal_confidence_q: 400,
                correlation_confidence_q: 600,
                attribution_confidence_q: 500,
            },
            correlation,
            transport: PerceptualTransportTrace::default(),
            legacy_contract: false,
        }
    }

    fn envelope(
        &mut self,
        observation: PerceptualObservation,
        capabilities: PerceptualCapabilities,
    ) -> PerceptualEventEnvelope {
        self.sequence = self.sequence.saturating_add(1);
        PerceptualEventEnvelope {
            schema_version: PERCEPTUAL_SCHEMA_VERSION,
            producer_id: PerceptualId::new([0x5A; 16]).expect("id"),
            producer_version: BoundedVersion::new("0.1.0").expect("version"),
            producer_kind: ProducerKind::SyntheticTest,
            source_kind: PerceptualSourceKind::Synthetic,
            capabilities,
            sequence: self.sequence,
            observation,
        }
    }

    /// An interaction with a total and no breakdown: the shape a terminal or
    /// editor integration would plausibly produce.
    pub fn total_only_interaction(&mut self, at_ms: u64, total_ms: u32) -> PerceptualEventEnvelope {
        let observation = PerceptualObservation::InferredInteraction(InferredInteractionEpisode {
            header: self.header(at_ms, CorrelationState::CompletedOnly),
            measurement: PerceptualMeasurement {
                total_duration_ms: Some(total_ms),
                components: Vec::new(),
                measurement_mode: MeasurementMode::Inferred,
            },
            inference_basis: InferenceBasis::ProcessActivity,
        });
        let capabilities = PerceptualCapabilities {
            has_completed_event: true,
            has_total_duration: true,
            has_response_signal: true,
            ..PerceptualCapabilities::default()
        };
        self.envelope(observation, capabilities)
    }

    /// A window with neither total nor components.
    pub fn bare_window(&mut self, at_ms: u64, window_ms: u32) -> PerceptualEventEnvelope {
        let observation = PerceptualObservation::PerceptualWindow(PerceptualWindowObservation {
            header: self.header(at_ms, CorrelationState::CompletedOnly),
            window_ms,
            signal_count: 5,
            sluggishness_q: 250,
            measurement: PerceptualMeasurement {
                total_duration_ms: None,
                components: Vec::new(),
                measurement_mode: MeasurementMode::AggregateWindow,
            },
        });
        self.envelope(observation, PerceptualCapabilities::external_observer())
    }
}

impl PerceptualAdapter for SyntheticPerceptualAdapter {
    fn source_kind(&self) -> PerceptualSourceKind {
        PerceptualSourceKind::Synthetic
    }

    fn capabilities(&self) -> PerceptualCapabilities {
        PerceptualCapabilities::external_observer()
    }

    fn validate(
        &self,
        envelope: &PerceptualEventEnvelope,
    ) -> Result<(), PerceptualValidationError> {
        validate_envelope(envelope)
    }

    fn normalize(
        &mut self,
        envelope: PerceptualEventEnvelope,
        _now: MonotonicMillis,
    ) -> Vec<PerceptualObservation> {
        match self.validate(&envelope) {
            Ok(()) => {
                self.health.accepted_total = self.health.accepted_total.saturating_add(1);
                self.health.last_observation_at_ms = envelope.observation.header().observed_at_ms;
                vec![envelope.observation]
            }
            Err(_) => {
                self.health.rejected_total = self.health.rejected_total.saturating_add(1);
                Vec::new()
            }
        }
    }

    fn health(&self) -> AdapterHealth {
        self.health
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::perceptual::store::PerceptualObservationStore;

    #[test]
    fn the_core_accepts_a_source_with_a_total_and_no_breakdown() {
        let mut adapter = SyntheticPerceptualAdapter::new();
        let envelope = adapter.total_only_interaction(1_000, 180);
        assert_eq!(adapter.validate(&envelope), Ok(()));
        let observations = adapter.normalize(envelope, MonotonicMillis(1_000));
        assert_eq!(observations.len(), 1);
        let measurement = observations[0].measurement();
        assert_eq!(measurement.total_duration_ms, Some(180));
        assert!(measurement.components.is_empty());
        assert_eq!(measurement.reconciles_within(8), None);
    }

    #[test]
    fn the_core_accepts_a_window_with_neither_total_nor_components() {
        let mut adapter = SyntheticPerceptualAdapter::new();
        let envelope = adapter.bare_window(2_000, 8_000);
        assert_eq!(adapter.validate(&envelope), Ok(()));
        let observations = adapter.normalize(envelope, MonotonicMillis(2_000));
        assert_eq!(observations.len(), 1);
        assert!(!observations[0].is_individual_interaction());
    }

    #[test]
    fn a_non_browser_source_travels_end_to_end_into_the_store() {
        let mut adapter = SyntheticPerceptualAdapter::new();
        let mut store = PerceptualObservationStore::new();
        for index in 0..5u64 {
            let envelope = adapter.total_only_interaction(index * 100, 150 + index as u32);
            for observation in adapter.normalize(envelope, MonotonicMillis(index * 100)) {
                store.record(observation);
            }
        }
        let envelope = adapter.bare_window(1_000, 8_000);
        for observation in adapter.normalize(envelope, MonotonicMillis(1_000)) {
            store.record(observation);
        }
        assert_eq!(store.len(), 6);
        assert_eq!(store.count_for(PerceptualSourceKind::Synthetic), 6);
        assert_eq!(store.metrics.inferred_total, 5);
        assert_eq!(store.metrics.window_total, 1);
        assert!(store.metrics.reconciles());
    }

    #[test]
    fn a_synthetic_source_never_produces_an_inp_shaped_measurement() {
        let mut adapter = SyntheticPerceptualAdapter::new();
        let envelope = adapter.total_only_interaction(1, 200);
        let json = serde_json::to_string(&envelope).expect("encodes");
        for browser_concept in ["inp", "interactionId", "tab", "navigation", "browser"] {
            assert!(
                !json
                    .to_lowercase()
                    .contains(&browser_concept.to_lowercase()),
                "{browser_concept} leaked into a non-browser payload: {json}"
            );
        }
    }

    #[test]
    fn a_replayed_sequence_is_visible_to_the_caller() {
        let mut adapter = SyntheticPerceptualAdapter::new();
        let first = adapter.total_only_interaction(1, 100).sequence;
        let second = adapter.total_only_interaction(2, 100).sequence;
        assert!(second > first, "sequence must advance for replay detection");
    }
}
