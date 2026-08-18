//! Boundary checks for untrusted producers.
//!
//! Every producer is untrusted, including ones Apollo ships. The checks here are
//! the only thing between a hostile or buggy adapter and the store.

use super::types::{
    LatencyComponentKind, PerceptualEventEnvelope, PerceptualObservation, MAX_COMPONENTS,
    QUALITY_SCALE,
};

/// Newest schema the core interprets.
pub const PERCEPTUAL_SCHEMA_VERSION: u16 = 1;
/// Oldest still accepted. Rejecting an older producer would blind the daemon
/// during a rollout, which is worse than reading a narrower payload.
pub const MIN_PERCEPTUAL_SCHEMA_VERSION: u16 = 1;
/// A single interaction longer than this is a stalled clock, not a measurement.
pub const MAX_DURATION_MS: u32 = 120_000;
pub const MAX_WINDOW_MS: u32 = 600_000;
pub const MAX_WINDOW_SIGNALS: u32 = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptualValidationError {
    UnsupportedSchema,
    ZeroProducerIdentity,
    ZeroSequence,
    TooManyComponents,
    DuplicateComponentKind,
    DurationOutOfRange,
    ComponentOutOfRange,
    QualityOutOfRange,
    WindowOutOfRange,
    /// The producer claims an ability it cannot derive from what it declared.
    IncoherentCapabilities,
    /// The payload claims precision the declared capabilities do not support.
    CapabilityContradiction,
}

impl PerceptualValidationError {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedSchema => "unsupported-schema",
            Self::ZeroProducerIdentity => "zero-producer-identity",
            Self::ZeroSequence => "zero-sequence",
            Self::TooManyComponents => "too-many-components",
            Self::DuplicateComponentKind => "duplicate-component-kind",
            Self::DurationOutOfRange => "duration-out-of-range",
            Self::ComponentOutOfRange => "component-out-of-range",
            Self::QualityOutOfRange => "quality-out-of-range",
            Self::WindowOutOfRange => "window-out-of-range",
            Self::IncoherentCapabilities => "incoherent-capabilities",
            Self::CapabilityContradiction => "capability-contradiction",
        }
    }
}

/// Checks that hold for every producer, whatever it observes.
pub fn validate_envelope(
    envelope: &PerceptualEventEnvelope,
) -> Result<(), PerceptualValidationError> {
    if envelope.schema_version > PERCEPTUAL_SCHEMA_VERSION
        || envelope.schema_version < MIN_PERCEPTUAL_SCHEMA_VERSION
    {
        return Err(PerceptualValidationError::UnsupportedSchema);
    }
    if envelope.producer_id.bytes() == [0; 16] {
        return Err(PerceptualValidationError::ZeroProducerIdentity);
    }
    if envelope.sequence == 0 {
        return Err(PerceptualValidationError::ZeroSequence);
    }
    if !envelope.capabilities.is_coherent() {
        return Err(PerceptualValidationError::IncoherentCapabilities);
    }

    let header = envelope.observation.header();
    if header.scope.producer_session_id.bytes() == [0; 16] {
        return Err(PerceptualValidationError::ZeroProducerIdentity);
    }
    for value in [
        header.quality.source_trust_q,
        header.quality.measurement_quality_q,
        header.quality.temporal_confidence_q,
        header.quality.correlation_confidence_q,
        header.quality.attribution_confidence_q,
        header.transport.transport_quality_q,
    ] {
        if value > QUALITY_SCALE {
            return Err(PerceptualValidationError::QualityOutOfRange);
        }
    }

    let measurement = envelope.observation.measurement();
    if measurement.components.len() > MAX_COMPONENTS {
        return Err(PerceptualValidationError::TooManyComponents);
    }
    let mut seen: Vec<LatencyComponentKind> = Vec::with_capacity(measurement.components.len());
    for component in &measurement.components {
        if component.duration_ms > MAX_DURATION_MS {
            return Err(PerceptualValidationError::ComponentOutOfRange);
        }
        if seen.contains(&component.kind) {
            return Err(PerceptualValidationError::DuplicateComponentKind);
        }
        seen.push(component.kind);
    }
    if measurement
        .total_duration_ms
        .is_some_and(|value| value > MAX_DURATION_MS)
    {
        return Err(PerceptualValidationError::DurationOutOfRange);
    }

    // A producer must not report what it said it cannot see. This is the check
    // that stops an outside observer from emitting internal stages, which would
    // silently upgrade an inference into a fabricated measurement.
    if !measurement.components.is_empty() && !envelope.capabilities.has_latency_breakdown {
        return Err(PerceptualValidationError::CapabilityContradiction);
    }
    if measurement.total_duration_ms.is_some() && !envelope.capabilities.has_total_duration {
        return Err(PerceptualValidationError::CapabilityContradiction);
    }

    if let PerceptualObservation::PerceptualWindow(window) = &envelope.observation {
        if window.window_ms == 0
            || window.window_ms > MAX_WINDOW_MS
            || window.signal_count > MAX_WINDOW_SIGNALS
            || window.sluggishness_q > QUALITY_SCALE
        {
            return Err(PerceptualValidationError::WindowOutOfRange);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::perceptual::capabilities::PerceptualCapabilities;
    use crate::engine::perceptual::types::*;

    fn envelope(observation: PerceptualObservation) -> PerceptualEventEnvelope {
        PerceptualEventEnvelope {
            schema_version: PERCEPTUAL_SCHEMA_VERSION,
            producer_id: PerceptualId::new([3; 16]).expect("id"),
            producer_version: BoundedVersion::new("1.0.0").expect("version"),
            producer_kind: ProducerKind::SyntheticTest,
            source_kind: PerceptualSourceKind::Synthetic,
            capabilities: PerceptualCapabilities::instrumented(),
            sequence: 1,
            observation,
        }
    }

    fn header() -> ObservationHeader {
        ObservationHeader {
            observed_at_ms: 1,
            source_kind: PerceptualSourceKind::Synthetic,
            producer_kind: ProducerKind::SyntheticTest,
            scope: InteractionScope {
                producer_session_id: PerceptualId::new([9; 16]).expect("id"),
                ..InteractionScope::default()
            },
            quality: PerceptualQuality::default(),
            correlation: CorrelationState::Unique,
            transport: PerceptualTransportTrace::default(),
            legacy_contract: false,
        }
    }

    fn instrumented(measurement: PerceptualMeasurement) -> PerceptualObservation {
        PerceptualObservation::InstrumentedInteraction(InstrumentedInteractionEpisode {
            header: header(),
            measurement,
        })
    }

    #[test]
    fn a_valid_envelope_passes() {
        let ok = envelope(instrumented(PerceptualMeasurement {
            total_duration_ms: Some(120),
            components: vec![LatencyComponent {
                kind: LatencyComponentKind::Processing,
                duration_ms: 120,
            }],
            measurement_mode: MeasurementMode::Instrumented,
        }));
        assert_eq!(validate_envelope(&ok), Ok(()));
    }

    #[test]
    fn a_producer_cannot_report_stages_it_declared_it_cannot_see() {
        // The invariant that keeps an inference from becoming a fabrication.
        let mut hostile = envelope(instrumented(PerceptualMeasurement {
            total_duration_ms: None,
            components: vec![LatencyComponent {
                kind: LatencyComponentKind::Presentation,
                duration_ms: 400,
            }],
            measurement_mode: MeasurementMode::Inferred,
        }));
        hostile.capabilities = PerceptualCapabilities::external_observer();
        assert_eq!(
            validate_envelope(&hostile),
            Err(PerceptualValidationError::CapabilityContradiction)
        );
    }

    #[test]
    fn a_total_from_a_producer_that_cannot_measure_one_is_refused() {
        let mut hostile = envelope(instrumented(PerceptualMeasurement {
            total_duration_ms: Some(90),
            components: Vec::new(),
            measurement_mode: MeasurementMode::Inferred,
        }));
        hostile.capabilities = PerceptualCapabilities::external_observer();
        assert_eq!(
            validate_envelope(&hostile),
            Err(PerceptualValidationError::CapabilityContradiction)
        );
    }

    #[test]
    fn incoherent_capabilities_are_refused_before_the_payload_is_read() {
        let mut incoherent = envelope(instrumented(PerceptualMeasurement::default()));
        incoherent.capabilities = PerceptualCapabilities {
            has_latency_breakdown: true,
            has_total_duration: false,
            ..PerceptualCapabilities::default()
        };
        assert_eq!(
            validate_envelope(&incoherent),
            Err(PerceptualValidationError::IncoherentCapabilities)
        );
    }

    #[test]
    fn a_hostile_component_list_is_bounded_and_deduplicated() {
        let many = envelope(instrumented(PerceptualMeasurement {
            total_duration_ms: Some(10),
            components: (0..MAX_COMPONENTS + 1)
                .map(|_| LatencyComponent {
                    kind: LatencyComponentKind::Unknown,
                    duration_ms: 1,
                })
                .collect(),
            measurement_mode: MeasurementMode::Instrumented,
        }));
        assert_eq!(
            validate_envelope(&many),
            Err(PerceptualValidationError::TooManyComponents)
        );

        let duplicated = envelope(instrumented(PerceptualMeasurement {
            total_duration_ms: Some(10),
            components: vec![
                LatencyComponent {
                    kind: LatencyComponentKind::Processing,
                    duration_ms: 5,
                },
                LatencyComponent {
                    kind: LatencyComponentKind::Processing,
                    duration_ms: 5,
                },
            ],
            measurement_mode: MeasurementMode::Instrumented,
        }));
        assert_eq!(
            validate_envelope(&duplicated),
            Err(PerceptualValidationError::DuplicateComponentKind)
        );
    }

    #[test]
    fn zero_identity_and_zero_sequence_are_refused() {
        let mut zero_seq = envelope(instrumented(PerceptualMeasurement::default()));
        zero_seq.sequence = 0;
        assert_eq!(
            validate_envelope(&zero_seq),
            Err(PerceptualValidationError::ZeroSequence)
        );

        let mut zero_id = envelope(instrumented(PerceptualMeasurement::default()));
        zero_id.producer_id = PerceptualId([0; 16]);
        assert_eq!(
            validate_envelope(&zero_id),
            Err(PerceptualValidationError::ZeroProducerIdentity)
        );
    }

    #[test]
    fn a_future_schema_is_refused_and_the_floor_is_accepted() {
        let mut future = envelope(instrumented(PerceptualMeasurement::default()));
        future.schema_version = PERCEPTUAL_SCHEMA_VERSION + 1;
        assert_eq!(
            validate_envelope(&future),
            Err(PerceptualValidationError::UnsupportedSchema)
        );
        let mut floor = envelope(instrumented(PerceptualMeasurement::default()));
        floor.schema_version = MIN_PERCEPTUAL_SCHEMA_VERSION;
        assert_eq!(validate_envelope(&floor), Ok(()));
    }

    #[test]
    fn a_quality_above_the_scale_is_refused() {
        let mut bad = envelope(instrumented(PerceptualMeasurement::default()));
        if let PerceptualObservation::InstrumentedInteraction(ref mut e) = bad.observation {
            e.header.quality.temporal_confidence_q = QUALITY_SCALE + 1;
        }
        assert_eq!(
            validate_envelope(&bad),
            Err(PerceptualValidationError::QualityOutOfRange)
        );
    }

    #[test]
    fn a_window_is_range_checked_on_its_own_terms() {
        let mut window = envelope(PerceptualObservation::PerceptualWindow(
            PerceptualWindowObservation {
                header: header(),
                window_ms: MAX_WINDOW_MS + 1,
                signal_count: 1,
                sluggishness_q: 10,
                measurement: PerceptualMeasurement {
                    measurement_mode: MeasurementMode::AggregateWindow,
                    ..PerceptualMeasurement::default()
                },
            },
        ));
        window.capabilities = PerceptualCapabilities::external_observer();
        assert_eq!(
            validate_envelope(&window),
            Err(PerceptualValidationError::WindowOutOfRange)
        );
    }
}
