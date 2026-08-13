//! Bounded, numeric-only context-agent IPC contract and user-session sampler.
//!
//! This module deliberately contains no rich user context. Window/process
//! identifiers are reduced to counts before a sample is built, and no sample
//! is written to disk or retained beyond the latest in-memory value.

use std::collections::HashSet;
use std::fmt;
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::de::{Error as DeError, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::engine::cg_display;
use crate::engine::cg_window;
use crate::engine::coreaudio_active;
use crate::engine::daemon_helpers::socket_path;
use crate::engine::protocol::DaemonRequest;
use crate::engine::user_context;

pub const CONTEXT_SCHEMA_VERSION: u16 = 1;
pub const MAX_CONTEXT_PAYLOAD_BYTES: usize = 4 * 1024;
pub const QUALITY_MIN: f64 = 0.0;
pub const QUALITY_MAX: f64 = 1.0;

#[cfg(target_os = "macos")]
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGPreflightScreenCaptureAccess() -> bool;
}

#[cfg(target_os = "macos")]
#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn AXIsProcessTrusted() -> bool;
}

/// Numeric encoding for a coarse boolean that may be unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TriState {
    #[default]
    Unknown = 0,
    No = 1,
    Yes = 2,
}

impl TriState {
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Unknown),
            1 => Some(Self::No),
            2 => Some(Self::Yes),
            _ => None,
        }
    }

    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

impl Serialize for TriState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(self.as_u8())
    }
}

impl<'de> Deserialize<'de> for TriState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TriStateVisitor;

        impl<'de> Visitor<'de> for TriStateVisitor {
            type Value = TriState;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("numeric tri-state value 0, 1, or 2")
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: DeError,
            {
                u8::try_from(value)
                    .ok()
                    .and_then(TriState::from_u8)
                    .ok_or_else(|| E::custom("tri-state value must be 0, 1, or 2"))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: DeError,
            {
                if value < 0 {
                    return Err(E::custom("tri-state value must not be negative"));
                }
                self.visit_u64(value as u64)
            }
        }

        deserializer.deserialize_u8(TriStateVisitor)
    }
}

/// Numeric permission state. `No` is intentionally not used here because
/// permission denial and an unavailable TCC query are distinct states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum PermissionState {
    #[default]
    Unknown = 0,
    Denied = 1,
    Granted = 2,
}

impl PermissionState {
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Unknown),
            1 => Some(Self::Denied),
            2 => Some(Self::Granted),
            _ => None,
        }
    }

    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

impl Serialize for PermissionState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(self.as_u8())
    }
}

impl<'de> Deserialize<'de> for PermissionState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PermissionVisitor;

        impl<'de> Visitor<'de> for PermissionVisitor {
            type Value = PermissionState;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("numeric permission state 0, 1, or 2")
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: DeError,
            {
                u8::try_from(value)
                    .ok()
                    .and_then(PermissionState::from_u8)
                    .ok_or_else(|| E::custom("permission state must be 0, 1, or 2"))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: DeError,
            {
                if value < 0 {
                    return Err(E::custom("permission state must not be negative"));
                }
                self.visit_u64(value as u64)
            }
        }

        deserializer.deserialize_u8(PermissionVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ContextPermissions {
    pub screen_capture: PermissionState,
    pub microphone: PermissionState,
    pub accessibility: PermissionState,
    pub input_monitoring: PermissionState,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextSummary {
    pub schema_version: u16,
    pub daemon_epoch: u64,
    pub sequence: u64,
    pub monotonic_ns: u64,
    pub audio_output: TriState,
    pub audio_input: TriState,
    pub visual_change_q: f64,
    pub interaction_q: f64,
    pub permissions: ContextPermissions,
}

impl Default for ContextSummary {
    fn default() -> Self {
        Self {
            schema_version: CONTEXT_SCHEMA_VERSION,
            daemon_epoch: 1,
            sequence: 1,
            monotonic_ns: 1,
            audio_output: TriState::Unknown,
            audio_input: TriState::Unknown,
            visual_change_q: 0.0,
            interaction_q: 0.0,
            permissions: ContextPermissions::default(),
        }
    }
}

impl ContextSummary {
    pub fn validate(&self) -> Result<(), ContextValidationError> {
        if self.schema_version != CONTEXT_SCHEMA_VERSION {
            return Err(ContextValidationError::UnsupportedSchema);
        }
        if self.daemon_epoch == 0 {
            return Err(ContextValidationError::ZeroEpoch);
        }
        if self.sequence == 0 {
            return Err(ContextValidationError::ZeroSequence);
        }
        if self.monotonic_ns == 0 {
            return Err(ContextValidationError::ZeroMonotonicTime);
        }
        validate_quality(self.visual_change_q, "visual_change_q")?;
        validate_quality(self.interaction_q, "interaction_q")?;
        Ok(())
    }

    pub fn bounded_request_bytes(&self) -> Result<Vec<u8>, ContextValidationError> {
        self.validate()?;
        let request = DaemonRequest::SubmitContext { summary: *self };
        let bytes = serde_json::to_vec(&request)
            .map_err(|_| ContextValidationError::SerializationFailed)?;
        if bytes.len() > MAX_CONTEXT_PAYLOAD_BYTES {
            return Err(ContextValidationError::PayloadTooLarge);
        }
        Ok(bytes)
    }
}

/// Parse and validate the exact bounded wire payload used by the context
/// agent. Rich or unknown fields fail closed before they can reach state.
pub fn validate_context_payload(bytes: &[u8]) -> Result<ContextSummary, ContextValidationError> {
    if bytes.len() > MAX_CONTEXT_PAYLOAD_BYTES {
        return Err(ContextValidationError::PayloadTooLarge);
    }
    let wire: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|_| ContextValidationError::MalformedPayload)?;
    let Some(object) = wire.as_object() else {
        return Err(ContextValidationError::MalformedPayload);
    };
    if object.keys().any(|key| key != "type" && key != "payload")
        || object.get("type").and_then(serde_json::Value::as_str) != Some("SubmitContext")
    {
        return Err(ContextValidationError::MalformedPayload);
    }
    let Some(payload) = object.get("payload").and_then(serde_json::Value::as_object) else {
        return Err(ContextValidationError::MalformedPayload);
    };
    if payload.keys().any(|key| key != "summary") || !payload.contains_key("summary") {
        return Err(ContextValidationError::MalformedPayload);
    }
    let request: DaemonRequest = serde_json::from_value(wire)
        .map_err(|_| ContextValidationError::MalformedPayload)?;
    match request {
        DaemonRequest::SubmitContext { summary } => {
            summary.validate()?;
            Ok(summary)
        }
        _ => Err(ContextValidationError::MalformedPayload),
    }
}

fn validate_quality(value: f64, field: &'static str) -> Result<(), ContextValidationError> {
    if !value.is_finite() || !(QUALITY_MIN..=QUALITY_MAX).contains(&value) {
        return Err(ContextValidationError::InvalidQuality(field));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextValidationError {
    PayloadTooLarge,
    UnsupportedSchema,
    ZeroEpoch,
    ZeroSequence,
    ZeroMonotonicTime,
    InvalidQuality(&'static str),
    SerializationFailed,
    MalformedPayload,
    EpochRegression,
    SequenceReplay,
    MonotonicRegression,
}

impl fmt::Display for ContextValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PayloadTooLarge => formatter.write_str("context payload exceeds 4096 bytes"),
            Self::UnsupportedSchema => formatter.write_str("unsupported context schema"),
            Self::ZeroEpoch => formatter.write_str("context epoch must be nonzero"),
            Self::ZeroSequence => formatter.write_str("context sequence must be nonzero"),
            Self::ZeroMonotonicTime => {
                formatter.write_str("context monotonic time must be nonzero")
            }
            Self::InvalidQuality(field) => write!(formatter, "{field} must be finite in [0, 1]"),
            Self::SerializationFailed => formatter.write_str("context payload serialization failed"),
            Self::MalformedPayload => formatter.write_str("malformed context payload"),
            Self::EpochRegression => formatter.write_str("context epoch regressed"),
            Self::SequenceReplay => formatter.write_str("context sequence was replayed"),
            Self::MonotonicRegression => formatter.write_str("context monotonic time regressed"),
        }
    }
}

impl std::error::Error for ContextValidationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AntiReplayStore {
    last_epoch: u64,
    last_sequence: u64,
    last_monotonic_ns: u64,
}

impl AntiReplayStore {
    pub fn accept(&mut self, summary: ContextSummary) -> Result<(), ContextValidationError> {
        summary.validate()?;
        if self.last_epoch != 0 {
            if summary.daemon_epoch < self.last_epoch {
                return Err(ContextValidationError::EpochRegression);
            }
            if summary.daemon_epoch == self.last_epoch {
                if summary.sequence <= self.last_sequence {
                    return Err(ContextValidationError::SequenceReplay);
                }
                if summary.monotonic_ns < self.last_monotonic_ns {
                    return Err(ContextValidationError::MonotonicRegression);
                }
            }
        }
        self.last_epoch = summary.daemon_epoch;
        self.last_sequence = summary.sequence;
        self.last_monotonic_ns = summary.monotonic_ns;
        Ok(())
    }

    pub const fn last_epoch(&self) -> u64 {
        self.last_epoch
    }

    pub const fn last_sequence(&self) -> u64 {
        self.last_sequence
    }
}

#[derive(Debug, Default)]
pub struct ContextAgentState {
    replay: AntiReplayStore,
    latest: Option<ContextSummary>,
}

impl ContextAgentState {
    pub fn accept(&mut self, summary: ContextSummary) -> Result<(), ContextValidationError> {
        self.replay.accept(summary)?;
        self.latest = Some(summary);
        Ok(())
    }

    pub fn latest(&self) -> Option<ContextSummary> {
        self.latest
    }

    pub const fn replay(&self) -> AntiReplayStore {
        self.replay
    }
}

static PROCESS_CONTEXT_STATE: OnceLock<Mutex<ContextAgentState>> = OnceLock::new();

pub fn process_context_state() -> &'static Mutex<ContextAgentState> {
    PROCESS_CONTEXT_STATE.get_or_init(|| Mutex::new(ContextAgentState::default()))
}

pub fn accept_process_context(summary: ContextSummary) -> Result<(), ContextValidationError> {
    process_context_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .accept(summary)
}

/// User-session-local sampler. It reduces public API results to coarse numeric
/// aggregates before constructing a wire summary.
pub struct ContextCollector {
    daemon_epoch: u64,
    sequence: u64,
    last_visible_count: Option<usize>,
    last_display_count: Option<u8>,
}

impl ContextCollector {
    pub fn new() -> Self {
        let daemon_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos().clamp(1, u64::MAX as u128) as u64)
            .unwrap_or(1);
        Self {
            daemon_epoch,
            sequence: 0,
            last_visible_count: None,
            last_display_count: None,
        }
    }

    pub fn collect(&mut self) -> Option<ContextSummary> {
        self.sequence = self.sequence.checked_add(1)?;

        let audio = coreaudio_active::audio_activity_snapshot();
        let visible_count = count_visible_windows();
        let display = cg_display::snapshot();
        let visual_change_q = self.visual_change_q(visible_count, display.display_count);
        let interaction_q = interaction_quality();

        Some(ContextSummary {
            schema_version: CONTEXT_SCHEMA_VERSION,
            daemon_epoch: self.daemon_epoch,
            sequence: self.sequence,
            monotonic_ns: monotonic_now_ns(),
            audio_output: tri_state(audio.session_supported, audio.output_active),
            audio_input: tri_state(audio.session_supported, audio.input_active),
            visual_change_q,
            interaction_q,
            permissions: ContextPermissions {
                screen_capture: screen_capture_permission(),
                microphone: if audio.input_probe_available {
                    PermissionState::Granted
                } else {
                    PermissionState::Unknown
                },
                accessibility: accessibility_permission(),
                input_monitoring: PermissionState::Unknown,
            },
        })
    }

    fn visual_change_q(&mut self, visible_count: Option<usize>, display_count: u8) -> f64 {
        let Some(visible_count) = visible_count else {
            self.last_visible_count = None;
            self.last_display_count = Some(display_count);
            return 0.0;
        };
        let count_delta = self
            .last_visible_count
            .map(|previous| previous.abs_diff(visible_count) as f64)
            .unwrap_or(0.0);
        let display_delta = self
            .last_display_count
            .map(|previous| previous.abs_diff(display_count) as f64)
            .unwrap_or(0.0);
        self.last_visible_count = Some(visible_count);
        self.last_display_count = Some(display_count);
        (count_delta / 8.0 + display_delta / 2.0).clamp(0.0, 1.0)
    }
}

impl Default for ContextCollector {
    fn default() -> Self {
        Self::new()
    }
}

fn count_visible_windows() -> Option<usize> {
    let windows: HashSet<u32> = cg_window::visible_pids();
    if windows.is_empty() {
        None
    } else {
        Some(windows.len())
    }
}

fn interaction_quality() -> f64 {
    let events_per_minute = user_context::hid_events_per_minute();
    if events_per_minute.is_finite() && events_per_minute >= 0.0 {
        (events_per_minute / 120.0).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn screen_capture_permission() -> PermissionState {
    #[cfg(target_os = "macos")]
    {
        return if unsafe { CGPreflightScreenCaptureAccess() } {
            PermissionState::Granted
        } else {
            PermissionState::Denied
        };
    }
    #[cfg(not(target_os = "macos"))]
    {
        PermissionState::Unknown
    }
}

fn accessibility_permission() -> PermissionState {
    #[cfg(target_os = "macos")]
    {
        return if unsafe { AXIsProcessTrusted() } {
            PermissionState::Granted
        } else {
            PermissionState::Denied
        };
    }
    #[cfg(not(target_os = "macos"))]
    {
        PermissionState::Unknown
    }
}

fn tri_state(available: bool, active: bool) -> TriState {
    if !available {
        TriState::Unknown
    } else if active {
        TriState::Yes
    } else {
        TriState::No
    }
}

fn monotonic_now_ns() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    let elapsed = START.get_or_init(Instant::now).elapsed().as_nanos();
    elapsed.clamp(1, u64::MAX as u128) as u64
}

pub fn send_once(collector: &mut ContextCollector) -> io::Result<()> {
    let summary = collector
        .collect()
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "context sequence exhausted"))?;
    let bytes = summary
        .bounded_request_bytes()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut stream = UnixStream::connect(socket_path())?;
    stream.set_write_timeout(Some(Duration::from_secs(1)))?;
    stream.set_read_timeout(Some(Duration::from_secs(1)))?;
    stream.write_all(&bytes)?;
    stream.write_all(b"\n")?;
    let _ = stream.flush();
    let mut response = [0u8; 512];
    let _ = stream.read(&mut response);
    Ok(())
}
