//! Privacy-safe, bounded contracts for browser navigation observations.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

pub const WEBFLOW_SCHEMA_VERSION: u16 = 1;
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
    pub inp_ms: Option<u32>,
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
            self.inp_ms,
            self.long_task_total_ms,
        ]
        .into_iter()
        .flatten()
        {
            if value > MAX_WEBFLOW_TIMING_MS {
                return Err(WebFlowValidationError::MetricOutOfRange);
            }
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
        if self.schema_version != WEBFLOW_SCHEMA_VERSION {
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
