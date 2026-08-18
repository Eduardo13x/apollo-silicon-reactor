//! Real non-browser adapter, built only from signals Apollo already collects.
//!
//! It sees the foreground application, a scheduler/WindowServer responsiveness
//! proxy and a fluidity score. It does **not** see input delay, processing or
//! presentation, so it never emits them — the validator refuses the payload if
//! it tries. It requires no new permission and no kernel access.
//!
//! What it produces is a `PerceptualWindow`: during this span the foreground was
//! X and the machine felt this responsive. That is genuinely weaker evidence
//! than a browser episode, and the modality says so rather than dressing it up.

use crate::engine::perceptual::capabilities::PerceptualCapabilities;
use crate::engine::perceptual::types::*;
use crate::engine::perceptual::validation::{validate_envelope, PerceptualValidationError};

use super::{AdapterHealth, PerceptualAdapter};

/// Span summarised by one window observation.
pub const WINDOW_MS: u32 = 8_000;

/// One sample of what the daemon already knows about responsiveness.
#[derive(Debug, Clone, Copy, Default)]
pub struct ForegroundResponseSample {
    /// Hash of the foreground application. Never its name.
    pub foreground_hash: u64,
    /// 0.0 responsive … 1.0 sluggish, from the scheduler/WindowServer proxy.
    pub perceptual_latency_score: f64,
    /// 0.0 … 1.0, higher is smoother.
    pub fluidity_score: f64,
    pub human_active: bool,
}

#[derive(Debug, Default)]
pub struct MacOsGenericObservationAdapter {
    producer_id: PerceptualId,
    samples: Vec<ForegroundResponseSample>,
    window_started_at_ms: u64,
    sequence: u64,
    health: AdapterHealth,
}

impl MacOsGenericObservationAdapter {
    pub fn new(producer_id: PerceptualId) -> Self {
        Self {
            producer_id,
            ..Self::default()
        }
    }

    /// Feed one cycle's worth of signal. Cheap by construction: this runs on the
    /// daemon's hot path and must not allocate per sample beyond the window.
    pub fn observe(&mut self, sample: ForegroundResponseSample, now_ms: u64) {
        if self.samples.is_empty() {
            self.window_started_at_ms = now_ms;
        }
        // Bounded: a window is a fixed span, not an unbounded accumulation.
        if self.samples.len() < 512 {
            self.samples.push(sample);
        }
    }

    /// Close the window if the span elapsed and there was human activity.
    ///
    /// A window with no human activity is not a perceptual observation — nobody
    /// was waiting for anything — and emitting one would dilute every aggregate
    /// with idle time.
    pub fn close_window(&mut self, now_ms: u64) -> Option<PerceptualEventEnvelope> {
        if self.samples.is_empty()
            || now_ms.saturating_sub(self.window_started_at_ms) < u64::from(WINDOW_MS)
        {
            return None;
        }
        let samples = std::mem::take(&mut self.samples);
        let signal_count = samples.iter().filter(|s| s.human_active).count() as u32;
        if signal_count == 0 {
            return None;
        }
        let sluggishness = samples
            .iter()
            .map(|s| s.perceptual_latency_score.clamp(0.0, 1.0))
            .sum::<f64>()
            / samples.len() as f64;
        let fluidity = samples
            .iter()
            .map(|s| s.fluidity_score.clamp(0.0, 1.0))
            .sum::<f64>()
            / samples.len() as f64;
        let foreground = samples.last().map_or(0, |s| s.foreground_hash);
        self.sequence = self.sequence.saturating_add(1);

        // Confidence is deliberately modest. This is a proxy over a window, and
        // the numbers say so rather than borrowing the browser's precision.
        let quality = PerceptualQuality {
            source_trust_q: 900,
            measurement_quality_q: 350,
            temporal_confidence_q: 300,
            correlation_confidence_q: 400,
            attribution_confidence_q: (fluidity * 500.0) as u16,
        }
        .clamped();

        Some(PerceptualEventEnvelope {
            schema_version: crate::engine::perceptual::validation::PERCEPTUAL_SCHEMA_VERSION,
            producer_id: self.producer_id,
            producer_version: BoundedVersion::new("1.0.0").expect("static version"),
            producer_kind: ProducerKind::MacOsObserver,
            source_kind: PerceptualSourceKind::GenericForegroundApplication,
            capabilities: PerceptualCapabilities::external_observer(),
            sequence: self.sequence,
            observation: PerceptualObservation::PerceptualWindow(PerceptualWindowObservation {
                header: ObservationHeader {
                    observed_at_ms: now_ms,
                    source_kind: PerceptualSourceKind::GenericForegroundApplication,
                    producer_kind: ProducerKind::MacOsObserver,
                    scope: InteractionScope {
                        producer_session_id: self.producer_id,
                        surface_session_id: None,
                        activity_session_id: None,
                        context_hash: Some(ContextBucket(foreground)),
                    },
                    quality,
                    // A window is complete on its own terms: there is no start
                    // event to match, and claiming Unique would imply one.
                    correlation: CorrelationState::CompletedOnly,
                    transport: PerceptualTransportTrace {
                        // In-process: no transport, and no pretence of one.
                        transport_quality_q: QUALITY_SCALE,
                        ..PerceptualTransportTrace::default()
                    },
                    legacy_contract: false,
                },
                window_ms: WINDOW_MS,
                signal_count,
                sluggishness_q: (sluggishness * f64::from(QUALITY_SCALE)) as u16,
                measurement: PerceptualMeasurement {
                    // No total, no components: this source cannot see either,
                    // and absence is the honest report.
                    total_duration_ms: None,
                    components: Vec::new(),
                    measurement_mode: MeasurementMode::AggregateWindow,
                },
            }),
        })
    }
}

impl PerceptualAdapter for MacOsGenericObservationAdapter {
    fn source_kind(&self) -> PerceptualSourceKind {
        PerceptualSourceKind::GenericForegroundApplication
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

    fn adapter() -> MacOsGenericObservationAdapter {
        MacOsGenericObservationAdapter::new(PerceptualId::new([0xA1; 16]).expect("id"))
    }

    fn sample(sluggish: f64, active: bool) -> ForegroundResponseSample {
        ForegroundResponseSample {
            foreground_hash: 0xBEEF,
            perceptual_latency_score: sluggish,
            fluidity_score: 0.84,
            human_active: active,
        }
    }

    #[test]
    fn a_window_needs_its_full_span_before_it_closes() {
        let mut adapter = adapter();
        adapter.observe(sample(0.2, true), 0);
        assert!(adapter.close_window(1_000).is_none());
        adapter.observe(sample(0.2, true), 1_000);
        assert!(adapter.close_window(u64::from(WINDOW_MS) + 1).is_some());
    }

    #[test]
    fn an_idle_window_is_not_a_perceptual_observation() {
        // Nobody was waiting, so there is nothing perceptual to report.
        let mut adapter = adapter();
        adapter.observe(sample(0.9, false), 0);
        adapter.observe(sample(0.9, false), 100);
        assert!(adapter.close_window(u64::from(WINDOW_MS) + 1).is_none());
    }

    #[test]
    fn the_adapter_never_emits_components_it_cannot_see() {
        let mut adapter = adapter();
        adapter.observe(sample(0.4, true), 0);
        let envelope = adapter
            .close_window(u64::from(WINDOW_MS) + 1)
            .expect("window closes");
        let measurement = envelope.observation.measurement();
        assert!(measurement.components.is_empty(), "no fabricated stages");
        assert_eq!(measurement.total_duration_ms, None, "absent, not zero");
        assert_eq!(
            measurement.measurement_mode,
            MeasurementMode::AggregateWindow
        );
    }

    #[test]
    fn what_it_emits_passes_the_core_validator() {
        let mut adapter = adapter();
        adapter.observe(sample(0.4, true), 0);
        let envelope = adapter
            .close_window(u64::from(WINDOW_MS) + 1)
            .expect("window closes");
        assert_eq!(adapter.validate(&envelope), Ok(()));
    }

    #[test]
    fn it_declares_lower_confidence_than_an_instrumented_source() {
        let mut adapter = adapter();
        adapter.observe(sample(0.4, true), 0);
        let envelope = adapter
            .close_window(u64::from(WINDOW_MS) + 1)
            .expect("window closes");
        let quality = envelope.observation.header().quality;
        assert!(
            quality.measurement_quality_q < 500,
            "a proxy must not claim instrumented confidence"
        );
        assert!(quality.overall_q() <= quality.source_trust_q);
    }

    #[test]
    fn the_foreground_travels_as_a_hash_never_as_a_name() {
        let mut adapter = adapter();
        adapter.observe(sample(0.4, true), 0);
        let envelope = adapter
            .close_window(u64::from(WINDOW_MS) + 1)
            .expect("window closes");
        let json = serde_json::to_string(&envelope.observation).expect("encodes");
        assert!(json.contains("context_hash"));
        for forbidden in ["Brave", "Chrome", "url", "title", "\"name\""] {
            assert!(!json.contains(forbidden), "{forbidden} must not appear");
        }
    }

    #[test]
    fn normalize_counts_acceptance_and_rejection_separately() {
        let mut adapter = adapter();
        adapter.observe(sample(0.4, true), 0);
        let envelope = adapter
            .close_window(u64::from(WINDOW_MS) + 1)
            .expect("window closes");
        assert_eq!(adapter.normalize(envelope, MonotonicMillis(1)).len(), 1);
        assert!(adapter.health().has_data());
        assert!(!adapter.health().is_rejecting_everything());
    }

    #[test]
    fn a_sequence_advances_so_replays_are_detectable_downstream() {
        let mut adapter = adapter();
        adapter.observe(sample(0.4, true), 0);
        let first = adapter
            .close_window(u64::from(WINDOW_MS) + 1)
            .expect("first");
        adapter.observe(sample(0.4, true), 20_000);
        let second = adapter.close_window(40_000).expect("second");
        assert!(second.sequence > first.sequence);
    }
}
