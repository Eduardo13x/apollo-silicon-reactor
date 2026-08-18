//! Privacy-safe, bounded contracts for browser navigation observations.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// v2 adds the per-interaction fields (`inp_estimate_ms`, `interaction_count`,
/// component totals) and renames the entry-duration tail. v1 payloads remain
/// readable: every new field is `Option`/`default`, and a v1 producer simply
/// reports no interaction data.
pub const WEBFLOW_SCHEMA_VERSION: u16 = 2;
pub const MIN_SUPPORTED_WEBFLOW_SCHEMA_VERSION: u16 = 1;
/// An untrusted page must not be able to claim an unbounded interaction count.
pub const MAX_WEBFLOW_INTERACTIONS: u32 = 4_096;
pub const MAX_WEBFLOW_MESSAGE_BYTES: usize = 16 * 1024;
pub const MAX_WEBFLOW_INGRESS_EVENTS: usize = 256;
pub const MAX_WEBFLOW_EVENTS_PER_CYCLE: usize = 128;
pub const MAX_WEBFLOW_TIMING_MS: u32 = 120_000;
pub const MAX_WEBFLOW_RESOURCE_COUNT: u32 = 100_000;
pub const MAX_WEBFLOW_TRANSFER_BYTES: u64 = 16 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpaqueId([u8; 16]);

impl OpaqueId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, WebFlowValidationError> {
        if bytes == [0; 16] {
            return Err(WebFlowValidationError::ZeroIdentity);
        }
        Ok(Self(bytes))
    }

    pub const fn bytes(self) -> [u8; 16] {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpaqueBucket([u8; 16]);

impl OpaqueBucket {
    pub fn new(bytes: [u8; 16]) -> Result<Self, WebFlowValidationError> {
        if bytes == [0; 16] {
            return Err(WebFlowValidationError::ZeroIdentity);
        }
        Ok(Self(bytes))
    }

    pub const fn bytes(self) -> [u8; 16] {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebFlowPhase {
    Started,
    Committed,
    DomReady,
    Loaded,
    Settled,
    Failed,
    Abandoned,
}

impl WebFlowPhase {
    pub const fn terminal(self) -> bool {
        matches!(self, Self::Settled | Self::Failed | Self::Abandoned)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebFlowSource {
    ExtensionLifecycle,
    ExtensionVitals,
    DaemonInference,
}

impl WebFlowSource {
    pub const fn confidence_q(self) -> u16 {
        match self {
            Self::ExtensionVitals => 10_000,
            Self::ExtensionLifecycle => 9_000,
            Self::DaemonInference => 5_000,
        }
    }

    pub const fn precision_rank(self) -> u8 {
        match self {
            Self::DaemonInference => 0,
            Self::ExtensionLifecycle => 1,
            Self::ExtensionVitals => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebFlowErrorClass {
    Network,
    NameResolution,
    Connection,
    Tls,
    Timeout,
    Cancelled,
    Browser,
    Unknown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebFlowMetrics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttfb_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dom_ready_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lcp_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Max single `PerformanceEventTiming.duration`, not INP. See
    /// `RuntimeMetrics::browser_event_duration_tail_ms`.
    pub event_duration_ms: Option<u32>,
    /// Web Vitals INP: high percentile over interactions grouped by
    /// `interactionId`. Distinct series from `event_duration_ms`; never join them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inp_estimate_ms: Option<u32>,
    /// Interactions (not entries) behind `inp_estimate_ms`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interaction_count: Option<u32>,
    /// Interactions the collector refused to track once bounded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interactions_dropped: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_delay_total_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub processing_total_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation_total_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cls_milli: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long_task_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long_task_total_ms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_class: Option<WebFlowErrorClass>,
}

impl WebFlowMetrics {
    fn validate(self) -> Result<(), WebFlowValidationError> {
        for value in [
            self.ttfb_ms,
            self.dom_ready_ms,
            self.load_ms,
            self.lcp_ms,
            self.event_duration_ms,
            self.inp_estimate_ms,
            self.input_delay_total_ms,
            self.processing_total_ms,
            self.presentation_total_ms,
            self.long_task_total_ms,
        ]
        .into_iter()
        .flatten()
        {
            if value > MAX_WEBFLOW_TIMING_MS {
                return Err(WebFlowValidationError::MetricOutOfRange);
            }
        }
        if self
            .interaction_count
            .is_some_and(|value| value > MAX_WEBFLOW_INTERACTIONS)
            || self
                .interactions_dropped
                .is_some_and(|value| value > MAX_WEBFLOW_INTERACTIONS)
        {
            return Err(WebFlowValidationError::MetricOutOfRange);
        }
        if self.cls_milli.is_some_and(|value| value > 100_000)
            || self
                .long_task_count
                .is_some_and(|value| value > MAX_WEBFLOW_RESOURCE_COUNT)
            || self
                .resource_count
                .is_some_and(|value| value > MAX_WEBFLOW_RESOURCE_COUNT)
            || self
                .transfer_bytes
                .is_some_and(|value| value > MAX_WEBFLOW_TRANSFER_BYTES)
        {
            return Err(WebFlowValidationError::MetricOutOfRange);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebFlowEvent {
    pub schema_version: u16,
    pub browser_session_id: OpaqueId,
    pub tab_session_id: OpaqueId,
    pub navigation_id: OpaqueId,
    pub sequence: u64,
    pub phase: WebFlowPhase,
    pub source: WebFlowSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site_bucket: Option<OpaqueBucket>,
    #[serde(default)]
    pub metrics: WebFlowMetrics,
}

impl WebFlowEvent {
    pub fn validate(&self) -> Result<(), WebFlowValidationError> {
        if self.schema_version > WEBFLOW_SCHEMA_VERSION
            || self.schema_version < MIN_SUPPORTED_WEBFLOW_SCHEMA_VERSION
        {
            return Err(WebFlowValidationError::UnsupportedSchema);
        }
        if self.sequence == 0 {
            return Err(WebFlowValidationError::ZeroSequence);
        }
        if self.browser_session_id.bytes() == [0; 16]
            || self.tab_session_id.bytes() == [0; 16]
            || self.navigation_id.bytes() == [0; 16]
            || self
                .site_bucket
                .is_some_and(|bucket| bucket.bytes() == [0; 16])
        {
            return Err(WebFlowValidationError::ZeroIdentity);
        }
        self.metrics.validate()
    }

    pub fn bounded_json(&self) -> Result<Vec<u8>, WebFlowValidationError> {
        self.validate()?;
        let bytes =
            serde_json::to_vec(self).map_err(|_| WebFlowValidationError::SerializationFailed)?;
        if bytes.len() > MAX_WEBFLOW_MESSAGE_BYTES {
            return Err(WebFlowValidationError::PayloadTooLarge);
        }
        Ok(bytes)
    }

    pub fn from_bounded_json(bytes: &[u8]) -> Result<Self, WebFlowValidationError> {
        if bytes.len() > MAX_WEBFLOW_MESSAGE_BYTES {
            return Err(WebFlowValidationError::PayloadTooLarge);
        }
        let event: Self =
            serde_json::from_slice(bytes).map_err(|_| WebFlowValidationError::MalformedPayload)?;
        event.validate()?;
        Ok(event)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceivedWebFlowEvent {
    pub event: WebFlowEvent,
    pub received_at_ms: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WebFlowIngressCounters {
    pub accepted: u64,
    pub invalid: u64,
    pub dropped: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebFlowIngressOutcome {
    Accepted,
    Invalid,
    Dropped,
}

impl WebFlowIngressOutcome {
    pub const fn is_accepted(self) -> bool {
        matches!(self, Self::Accepted)
    }
}

#[derive(Debug, Default)]
pub struct WebFlowIngress {
    queue: VecDeque<ReceivedWebFlowEvent>,
    counters: WebFlowIngressCounters,
}

impl WebFlowIngress {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::with_capacity(MAX_WEBFLOW_INGRESS_EVENTS),
            counters: WebFlowIngressCounters::default(),
        }
    }

    pub fn accept_at(&mut self, event: WebFlowEvent, received_at_ms: u64) -> WebFlowIngressOutcome {
        if received_at_ms == 0 || event.validate().is_err() {
            self.counters.invalid = self.counters.invalid.saturating_add(1);
            return WebFlowIngressOutcome::Invalid;
        }
        if self.queue.len() == MAX_WEBFLOW_INGRESS_EVENTS {
            self.counters.dropped = self.counters.dropped.saturating_add(1);
            return WebFlowIngressOutcome::Dropped;
        }
        self.queue.push_back(ReceivedWebFlowEvent {
            event,
            received_at_ms,
        });
        self.counters.accepted = self.counters.accepted.saturating_add(1);
        WebFlowIngressOutcome::Accepted
    }

    pub fn drain(&mut self, maximum: usize) -> Vec<ReceivedWebFlowEvent> {
        let count = maximum
            .min(MAX_WEBFLOW_EVENTS_PER_CYCLE)
            .min(self.queue.len());
        self.queue.drain(..count).collect()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub const fn counters(&self) -> WebFlowIngressCounters {
        self.counters
    }
}

static PROCESS_WEBFLOW_INGRESS: OnceLock<Mutex<WebFlowIngress>> = OnceLock::new();

fn process_webflow_ingress() -> &'static Mutex<WebFlowIngress> {
    PROCESS_WEBFLOW_INGRESS.get_or_init(|| Mutex::new(WebFlowIngress::new()))
}

pub fn webflow_monotonic_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START
        .get_or_init(Instant::now)
        .elapsed()
        .as_millis()
        .clamp(1, u64::MAX as u128) as u64
}

pub fn accept_process_webflow(event: WebFlowEvent) -> WebFlowIngressOutcome {
    accept_process_webflow_at(event, webflow_monotonic_ms())
}

pub fn accept_process_webflow_at(
    event: WebFlowEvent,
    received_at_ms: u64,
) -> WebFlowIngressOutcome {
    process_webflow_ingress()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .accept_at(event, received_at_ms)
}

pub fn drain_process_webflow(maximum: usize) -> Vec<ReceivedWebFlowEvent> {
    process_webflow_ingress()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .drain(maximum)
}

pub fn process_webflow_ingress_counters() -> WebFlowIngressCounters {
    process_webflow_ingress()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .counters()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebFlowValidationError {
    UnsupportedSchema,
    ZeroIdentity,
    ZeroSequence,
    MetricOutOfRange,
    PayloadTooLarge,
    SerializationFailed,
    MalformedPayload,
}

impl fmt::Display for WebFlowValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnsupportedSchema => "unsupported WebFlow schema",
            Self::ZeroIdentity => "WebFlow identity must be nonzero",
            Self::ZeroSequence => "WebFlow sequence must be nonzero",
            Self::MetricOutOfRange => "WebFlow metric is out of range",
            Self::PayloadTooLarge => "WebFlow payload exceeds 16384 bytes",
            Self::SerializationFailed => "WebFlow payload serialization failed",
            Self::MalformedPayload => "malformed WebFlow payload",
        })
    }
}

impl std::error::Error for WebFlowValidationError {}

#[cfg(test)]
mod schema_v2_tests {
    use super::*;

    fn opaque(byte: u8) -> OpaqueId {
        OpaqueId::new([byte.max(1); 16]).expect("nonzero opaque id")
    }

    fn event(metrics: WebFlowMetrics) -> WebFlowEvent {
        WebFlowEvent {
            schema_version: WEBFLOW_SCHEMA_VERSION,
            browser_session_id: opaque(1),
            tab_session_id: opaque(2),
            navigation_id: opaque(3),
            sequence: 1,
            phase: WebFlowPhase::Settled,
            source: WebFlowSource::ExtensionVitals,
            site_bucket: None,
            metrics,
        }
    }

    #[test]
    fn a_v1_producer_still_validates_and_simply_reports_no_interactions() {
        // The deployed extension is v1. Refusing its payloads would blind the
        // daemon during the rollout window.
        let mut legacy = event(WebFlowMetrics {
            event_duration_ms: Some(440),
            ..WebFlowMetrics::default()
        });
        legacy.schema_version = MIN_SUPPORTED_WEBFLOW_SCHEMA_VERSION;
        assert!(legacy.validate().is_ok());
        assert_eq!(legacy.metrics.inp_estimate_ms, None);
        assert_eq!(legacy.metrics.interaction_count, None);
    }

    #[test]
    fn a_future_schema_is_still_refused() {
        let mut future = event(WebFlowMetrics::default());
        future.schema_version = WEBFLOW_SCHEMA_VERSION + 1;
        assert_eq!(
            future.validate(),
            Err(WebFlowValidationError::UnsupportedSchema)
        );
    }

    #[test]
    fn an_untrusted_interaction_count_cannot_be_unbounded() {
        let hostile = event(WebFlowMetrics {
            interaction_count: Some(MAX_WEBFLOW_INTERACTIONS + 1),
            ..WebFlowMetrics::default()
        });
        assert_eq!(
            hostile.validate(),
            Err(WebFlowValidationError::MetricOutOfRange)
        );
        let dropped = event(WebFlowMetrics {
            interactions_dropped: Some(MAX_WEBFLOW_INTERACTIONS + 1),
            ..WebFlowMetrics::default()
        });
        assert_eq!(
            dropped.validate(),
            Err(WebFlowValidationError::MetricOutOfRange)
        );
    }

    #[test]
    fn every_interaction_timing_is_range_checked_like_the_legacy_ones() {
        for build in [
            |value| WebFlowMetrics {
                inp_estimate_ms: Some(value),
                ..WebFlowMetrics::default()
            },
            |value| WebFlowMetrics {
                input_delay_total_ms: Some(value),
                ..WebFlowMetrics::default()
            },
            |value| WebFlowMetrics {
                processing_total_ms: Some(value),
                ..WebFlowMetrics::default()
            },
            |value| WebFlowMetrics {
                presentation_total_ms: Some(value),
                ..WebFlowMetrics::default()
            },
        ] {
            let hostile = event(build(MAX_WEBFLOW_TIMING_MS + 1));
            assert_eq!(
                hostile.validate(),
                Err(WebFlowValidationError::MetricOutOfRange)
            );
        }
    }

    #[test]
    fn the_two_latency_series_travel_as_independent_fields() {
        // Schema discontinuity made structural: a payload can carry the entry
        // tail, the interaction estimate, both, or neither.
        let both = event(WebFlowMetrics {
            event_duration_ms: Some(440),
            inp_estimate_ms: Some(184),
            interaction_count: Some(312),
            ..WebFlowMetrics::default()
        });
        assert!(both.validate().is_ok());
        let round_trip = WebFlowEvent::from_bounded_json(&both.bounded_json().expect("encodes"))
            .expect("decodes");
        assert_eq!(round_trip.metrics.event_duration_ms, Some(440));
        assert_eq!(round_trip.metrics.inp_estimate_ms, Some(184));
        assert_eq!(round_trip.metrics.interaction_count, Some(312));
    }
}
