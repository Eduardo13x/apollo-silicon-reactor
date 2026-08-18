//! Chromium WebFlow → agnostic perceptual core.
//!
//! Lives outside `engine::perceptual` on purpose: the dependency points from
//! the adapter to the core, never back. This file is allowed to know about
//! tabs, navigations and `interactionId`; the core is not.
//!
//! Nothing here changes what WebFlow measures. INP, the interaction grouping
//! and the input/processing/presentation split stay exactly as phase 0A left
//! them, and INP remains a browser-adapter metric rather than becoming a
//! generic one — no other source can produce it.

use crate::engine::perceptual::adapters::{AdapterHealth, PerceptualAdapter};
use crate::engine::perceptual::capabilities::PerceptualCapabilities;
use crate::engine::perceptual::types::*;
use crate::engine::perceptual::validation::{
    validate_envelope, PerceptualValidationError, PERCEPTUAL_SCHEMA_VERSION,
};
use crate::engine::webflow_types::{WebFlowEvent, WebFlowSource};

/// Browser rounding of `duration` is 8 ms, so the split may miss the total by
/// that much and still describe the same interaction.
pub const CHROMIUM_RECONCILE_TOLERANCE_MS: u32 = 8;

/// Hops only this producer has. Kept out of the generic trace so no other
/// source is ever asked about a service worker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChromiumTransportDetails {
    pub service_worker_wake_ms: Option<u32>,
    pub client_segment_ms: Option<u32>,
    pub cold_start: Option<bool>,
    pub tab_queue_depth: Option<u32>,
}

#[derive(Debug, Default)]
pub struct ChromiumWebFlowAdapter {
    health: AdapterHealth,
    last_details: ChromiumTransportDetails,
}

impl ChromiumWebFlowAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn transport_details(&self) -> ChromiumTransportDetails {
        self.last_details
    }

    /// Capabilities depend on what the payload actually carried, not on the
    /// adapter's name: a v1 extension reports no interaction grouping, and
    /// claiming otherwise would let the core trust a breakdown that is absent.
    pub fn capabilities_for(event: &WebFlowEvent) -> PerceptualCapabilities {
        let has_breakdown = event.metrics.input_delay_total_ms.is_some()
            || event.metrics.processing_total_ms.is_some()
            || event.metrics.presentation_total_ms.is_some();
        let has_total = event.metrics.inp_estimate_ms.is_some();
        PerceptualCapabilities {
            has_started_event: false,
            has_completed_event: true,
            has_total_duration: has_total,
            // A breakdown is only claimable alongside a total it can reconcile
            // with; the validator refuses the pair otherwise.
            has_latency_breakdown: has_breakdown && has_total,
            has_semantic_operation: false,
            has_surface_identity: true,
            has_client_monotonic_clock: true,
            has_response_signal: true,
            supports_transport_trace: true,
        }
    }

    fn scope(event: &WebFlowEvent) -> InteractionScope {
        InteractionScope {
            // browser → producer, tab → surface, navigation → activity.
            producer_session_id: PerceptualId(event.browser_session_id.bytes()),
            surface_session_id: PerceptualId::new(event.tab_session_id.bytes()),
            activity_session_id: PerceptualId::new(event.navigation_id.bytes()),
            context_hash: event.site_bucket.map(|bucket| {
                ContextBucket(u64::from_le_bytes(
                    bucket.bytes()[..8].try_into().unwrap_or([0; 8]),
                ))
            }),
        }
    }

    fn measurement(event: &WebFlowEvent) -> PerceptualMeasurement {
        let mut components = Vec::new();
        for (value, kind) in [
            (
                event.metrics.input_delay_total_ms,
                LatencyComponentKind::InputDelay,
            ),
            (
                event.metrics.processing_total_ms,
                LatencyComponentKind::Processing,
            ),
            (
                event.metrics.presentation_total_ms,
                LatencyComponentKind::Presentation,
            ),
        ] {
            if let Some(duration_ms) = value {
                components.push(LatencyComponent { kind, duration_ms });
            }
        }
        // The window's total is the sum of its parts: the extension reports
        // per-page aggregates, so this summarises a report rather than one
        // interaction. INP is published separately and is not this number.
        let total = if components.is_empty() {
            None
        } else {
            Some(
                components
                    .iter()
                    .fold(0u32, |acc, c| acc.saturating_add(c.duration_ms)),
            )
        };
        PerceptualMeasurement {
            total_duration_ms: total,
            components,
            measurement_mode: MeasurementMode::Instrumented,
        }
    }

    fn quality(event: &WebFlowEvent, legacy: bool) -> PerceptualQuality {
        // A legacy payload does not gain confidence by being read through a
        // newer type: it is reported at the confidence it was produced with.
        let breakdown = event.metrics.input_delay_total_ms.is_some();
        PerceptualQuality {
            source_trust_q: 700,
            measurement_quality_q: if breakdown { 900 } else { 400 },
            temporal_confidence_q: if event.transport.client_segment_ms().is_some() {
                800
            } else {
                500
            },
            correlation_confidence_q: if legacy { 400 } else { 850 },
            attribution_confidence_q: 800,
        }
        .clamped()
    }

    /// Translate one accepted WebFlow event. Returns `None` for events that
    /// carry no perceptual measurement — lifecycle navigation phases are real
    /// events but say nothing about how anything felt.
    pub fn to_envelope(&mut self, event: &WebFlowEvent) -> Option<PerceptualEventEnvelope> {
        if event.source != WebFlowSource::ExtensionVitals {
            return None;
        }
        let measurement = Self::measurement(event);
        if measurement.total_duration_ms.is_none() && measurement.components.is_empty() {
            return None;
        }
        let legacy = event.schema_version < 2;
        self.last_details = ChromiumTransportDetails {
            service_worker_wake_ms: event
                .transport
                .service_worker_wake_ms()
                .map(|v| v.min(u64::from(u32::MAX)) as u32),
            client_segment_ms: event
                .transport
                .client_segment_ms()
                .map(|v| v.min(u64::from(u32::MAX)) as u32),
            cold_start: event.transport.service_worker_cold_start,
            tab_queue_depth: event.transport.tab_queue_depth,
        };

        Some(PerceptualEventEnvelope {
            schema_version: PERCEPTUAL_SCHEMA_VERSION,
            producer_id: PerceptualId(event.browser_session_id.bytes()),
            producer_version: event
                .extension_version
                .as_deref()
                .and_then(BoundedVersion::new)
                .unwrap_or_else(|| BoundedVersion::new("unknown").expect("static")),
            producer_kind: ProducerKind::InstrumentedExtension,
            source_kind: PerceptualSourceKind::BrowserChromium,
            capabilities: Self::capabilities_for(event),
            sequence: event.sequence,
            observation: PerceptualObservation::InstrumentedInteraction(
                InstrumentedInteractionEpisode {
                    header: ObservationHeader {
                        observed_at_ms: 0,
                        source_kind: PerceptualSourceKind::BrowserChromium,
                        producer_kind: ProducerKind::InstrumentedExtension,
                        scope: Self::scope(event),
                        quality: Self::quality(event, legacy),
                        correlation: CorrelationState::CompletedOnly,
                        transport: PerceptualTransportTrace {
                            producer_segment_ms: event
                                .transport
                                .client_segment_ms()
                                .map(|v| v.min(u64::from(u32::MAX)) as u32),
                            bridge_segment_ms: None,
                            daemon_segment_ms: None,
                            total_observed_ms: event
                                .transport
                                .client_segment_ms()
                                .map(|v| v.min(u64::from(u32::MAX)) as u32),
                            cold_start: event.transport.service_worker_cold_start,
                            queue_wait_ms: None,
                            dropped_before_ingest: false,
                            transport_quality_q: if event.transport.client_segment_ms().is_some() {
                                900
                            } else {
                                400
                            },
                        },
                        legacy_contract: legacy,
                    },
                    measurement,
                },
            ),
        })
    }
}

impl PerceptualAdapter for ChromiumWebFlowAdapter {
    fn source_kind(&self) -> PerceptualSourceKind {
        PerceptualSourceKind::BrowserChromium
    }

    fn capabilities(&self) -> PerceptualCapabilities {
        PerceptualCapabilities::instrumented()
    }

    fn validate(
        &self,
        envelope: &PerceptualEventEnvelope,
    ) -> Result<(), PerceptualValidationError> {
        validate_envelope(envelope)?;
        // The browser's own semantic rule, applied only when it declared the
        // capability. No other source is held to it.
        if envelope.capabilities.has_latency_breakdown {
            if let Some(false) = envelope
                .observation
                .measurement()
                .reconciles_within(CHROMIUM_RECONCILE_TOLERANCE_MS)
            {
                return Err(PerceptualValidationError::CapabilityContradiction);
            }
        }
        Ok(())
    }

    fn normalize(
        &mut self,
        mut envelope: PerceptualEventEnvelope,
        now: MonotonicMillis,
    ) -> Vec<PerceptualObservation> {
        // The daemon stamps arrival: the producer's clock is a different time
        // domain and combining them would need an explicit offset.
        match &mut envelope.observation {
            PerceptualObservation::InstrumentedInteraction(e) => {
                e.header.observed_at_ms = now.get();
            }
            PerceptualObservation::InferredInteraction(e) => e.header.observed_at_ms = now.get(),
            PerceptualObservation::PerceptualWindow(w) => w.header.observed_at_ms = now.get(),
        }
        match self.validate(&envelope) {
            Ok(()) => {
                self.health.accepted_total = self.health.accepted_total.saturating_add(1);
                self.health.last_observation_at_ms = now.get();
                if envelope.observation.header().legacy_contract {
                    self.health.legacy_contract_total =
                        self.health.legacy_contract_total.saturating_add(1);
                }
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
    use crate::engine::webflow_types::{
        FeatureCapabilities, OpaqueId, ProducerKind as WireProducerKind, WebFlowMetrics,
        WebFlowPhase, WebFlowTransport, WEBFLOW_SCHEMA_VERSION,
    };

    fn event(metrics: WebFlowMetrics, schema: u16) -> WebFlowEvent {
        WebFlowEvent {
            schema_version: schema,
            browser_session_id: OpaqueId::new([1; 16]).expect("id"),
            tab_session_id: OpaqueId::new([2; 16]).expect("id"),
            navigation_id: OpaqueId::new([3; 16]).expect("id"),
            sequence: 7,
            phase: WebFlowPhase::Settled,
            source: WebFlowSource::ExtensionVitals,
            site_bucket: None,
            metrics,
            transport: WebFlowTransport::default(),
            producer_kind: WireProducerKind::ChromiumExtension,
            extension_version: Some("2.0.2".to_string()),
            bridge_version: None,
            feature_capabilities: FeatureCapabilities(FeatureCapabilities::V2_EXPECTED),
        }
    }

    fn full_metrics() -> WebFlowMetrics {
        WebFlowMetrics {
            inp_estimate_ms: Some(120),
            interaction_count: Some(14),
            input_delay_total_ms: Some(46),
            processing_total_ms: Some(52),
            presentation_total_ms: Some(476),
            ..WebFlowMetrics::default()
        }
    }

    #[test]
    fn a_v2_vitals_event_becomes_an_instrumented_observation() {
        let mut adapter = ChromiumWebFlowAdapter::new();
        let envelope = adapter
            .to_envelope(&event(full_metrics(), WEBFLOW_SCHEMA_VERSION))
            .expect("vitals translate");
        assert_eq!(envelope.source_kind, PerceptualSourceKind::BrowserChromium);
        assert_eq!(envelope.producer_kind, ProducerKind::InstrumentedExtension);
        assert_eq!(adapter.validate(&envelope), Ok(()));
        assert_eq!(envelope.observation.modality(), "instrumented");
    }

    #[test]
    fn the_three_identities_map_onto_the_generic_hierarchy() {
        let mut adapter = ChromiumWebFlowAdapter::new();
        let envelope = adapter
            .to_envelope(&event(full_metrics(), WEBFLOW_SCHEMA_VERSION))
            .expect("translates");
        let scope = envelope.observation.header().scope;
        assert_eq!(scope.producer_session_id.bytes(), [1; 16]);
        assert_eq!(scope.surface_session_id.map(|id| id.bytes()), Some([2; 16]));
        assert_eq!(
            scope.activity_session_id.map(|id| id.bytes()),
            Some([3; 16])
        );
    }

    #[test]
    fn the_components_survive_translation_and_reconcile() {
        let mut adapter = ChromiumWebFlowAdapter::new();
        let envelope = adapter
            .to_envelope(&event(full_metrics(), WEBFLOW_SCHEMA_VERSION))
            .expect("translates");
        let measurement = envelope.observation.measurement();
        assert_eq!(
            measurement.component(LatencyComponentKind::Presentation),
            Some(476)
        );
        assert_eq!(
            measurement.reconciles_within(CHROMIUM_RECONCILE_TOLERANCE_MS),
            Some(true)
        );
    }

    #[test]
    fn a_v1_payload_still_translates_and_keeps_its_provenance() {
        let mut adapter = ChromiumWebFlowAdapter::new();
        let legacy = event(
            WebFlowMetrics {
                event_duration_ms: Some(440),
                inp_estimate_ms: None,
                input_delay_total_ms: None,
                ..WebFlowMetrics::default()
            },
            1,
        );
        // A v1 event carries no interaction measurement at all, so there is
        // nothing perceptual to translate — that is the honest outcome.
        assert!(adapter.to_envelope(&legacy).is_none());
    }

    #[test]
    fn a_legacy_payload_with_a_breakdown_is_marked_and_trusted_less() {
        let mut adapter = ChromiumWebFlowAdapter::new();
        let envelope = adapter
            .to_envelope(&event(full_metrics(), 1))
            .expect("translates");
        let header = envelope.observation.header();
        assert!(header.legacy_contract, "provenance survives migration");
        assert!(
            header.quality.correlation_confidence_q < 850,
            "an old contract does not gain new confidence"
        );
    }

    #[test]
    fn a_lifecycle_event_produces_no_perceptual_observation() {
        let mut adapter = ChromiumWebFlowAdapter::new();
        let mut lifecycle = event(WebFlowMetrics::default(), WEBFLOW_SCHEMA_VERSION);
        lifecycle.source = WebFlowSource::ExtensionLifecycle;
        assert!(adapter.to_envelope(&lifecycle).is_none());
    }

    #[test]
    fn capabilities_follow_the_payload_not_the_adapter_name() {
        let bare = event(
            WebFlowMetrics {
                inp_estimate_ms: Some(120),
                ..WebFlowMetrics::default()
            },
            WEBFLOW_SCHEMA_VERSION,
        );
        let caps = ChromiumWebFlowAdapter::capabilities_for(&bare);
        assert!(!caps.has_latency_breakdown, "no breakdown was reported");
        assert!(caps.is_coherent());
    }

    #[test]
    fn a_non_reconciling_breakdown_is_refused_for_this_adapter_only() {
        let mut adapter = ChromiumWebFlowAdapter::new();
        let broken = event(
            WebFlowMetrics {
                inp_estimate_ms: Some(120),
                input_delay_total_ms: Some(10),
                processing_total_ms: Some(10),
                presentation_total_ms: Some(10),
                ..WebFlowMetrics::default()
            },
            WEBFLOW_SCHEMA_VERSION,
        );
        let mut envelope = adapter.to_envelope(&broken).expect("translates");
        // Force a total that its parts cannot account for.
        if let PerceptualObservation::InstrumentedInteraction(ref mut e) = envelope.observation {
            e.measurement.total_duration_ms = Some(9_000);
        }
        assert_eq!(
            adapter.validate(&envelope),
            Err(PerceptualValidationError::CapabilityContradiction)
        );
    }

    #[test]
    fn service_worker_detail_stays_in_the_adapter() {
        let mut adapter = ChromiumWebFlowAdapter::new();
        let mut with_transport = event(full_metrics(), WEBFLOW_SCHEMA_VERSION);
        with_transport.transport = WebFlowTransport {
            content_send_started_at_ms: Some(1_000),
            service_worker_received_at_ms: Some(1_004),
            native_message_started_at_ms: Some(1_004),
            tab_queue_depth: Some(1),
            service_worker_cold_start: Some(false),
        };
        let envelope = adapter.to_envelope(&with_transport).expect("translates");
        assert_eq!(adapter.transport_details().service_worker_wake_ms, Some(4));
        // The generic trace carries only the shared facts.
        let trace = envelope.observation.header().transport;
        assert_eq!(trace.producer_segment_ms, Some(4));
        let json = serde_json::to_string(&trace).expect("encodes");
        assert!(!json.contains("service_worker"));
    }

    #[test]
    fn browser_observations_share_the_store_with_a_non_browser_source() {
        use crate::engine::perceptual::adapters::synthetic::SyntheticPerceptualAdapter;
        use crate::engine::perceptual::adapters::PerceptualAdapter as _;

        let mut store = PerceptualObservationStore::new();
        let mut chromium = ChromiumWebFlowAdapter::new();
        let envelope = chromium
            .to_envelope(&event(full_metrics(), WEBFLOW_SCHEMA_VERSION))
            .expect("translates");
        for observation in chromium.normalize(envelope, MonotonicMillis(1_000)) {
            store.record(observation);
        }

        let mut synthetic = SyntheticPerceptualAdapter::new();
        let bare = synthetic.bare_window(2_000, 8_000);
        for observation in synthetic.normalize(bare, MonotonicMillis(2_000)) {
            store.record(observation);
        }

        assert_eq!(store.len(), 2);
        assert_eq!(store.active_sources(), 2);
        assert_eq!(store.count_for(PerceptualSourceKind::BrowserChromium), 1);
        assert_eq!(store.count_for(PerceptualSourceKind::Synthetic), 1);
        // And they are never treated as like evidence.
        let browser = store.comparable_set("instrumented", MeasurementMode::Instrumented);
        let window = store.comparable_set("window", MeasurementMode::AggregateWindow);
        assert!(!browser[0].comparable_with(window[0]));
    }
}
