//! Application-, sensor- and chip-agnostic perceptual domain.
//!
//! Nothing here knows what a browser surface, a page load, a per-frame timing
//! entry or a background worker is. Those belong to adapters. The core's job is
//! to hold
//! observations of differing precision side by side **without flattening them
//! into a single false shape**: an instrumented browser episode and an inferred
//! window are both evidence, and they are not the same evidence.

use serde::{Deserialize, Serialize};

use super::capabilities::PerceptualCapabilities;

/// Opaque 16-byte identity. Never a name, a URL, a title or a path.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct PerceptualId(pub [u8; 16]);

impl PerceptualId {
    pub fn new(bytes: [u8; 16]) -> Option<Self> {
        (bytes != [0; 16]).then_some(Self(bytes))
    }

    pub const fn bytes(self) -> [u8; 16] {
        self.0
    }
}

/// Bounded, low-cardinality bucket for context. A hash, never the thing hashed.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ContextBucket(pub u64);

pub const MAX_PRODUCER_VERSION_CHARS: usize = 16;
pub const MAX_COMPONENTS: usize = 8;
/// Quantised confidence: 0..=1000, so a quality never becomes a float that
/// silently drifts through serialisation.
pub const QUALITY_SCALE: u16 = 1_000;

/// Milliseconds on the daemon's monotonic clock. A newtype so a producer clock
/// can never be passed where a daemon clock is expected: the two are different
/// time domains and combining them needs an explicit offset with an error bound.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MonotonicMillis(pub u64);

impl MonotonicMillis {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Bounded producer version string.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BoundedVersion(String);

impl BoundedVersion {
    pub fn new(value: &str) -> Option<Self> {
        let trimmed = value.trim();
        (!trimmed.is_empty() && trimmed.chars().count() <= MAX_PRODUCER_VERSION_CHARS)
            .then(|| Self(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// What kind of thing produced an observation. A closed set: a producer is never
/// identified by an application name.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProducerKind {
    #[default]
    Unknown,
    InstrumentedExtension,
    EditorPlugin,
    ShellIntegration,
    MacOsObserver,
    SyntheticTest,
}

impl ProducerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::InstrumentedExtension => "instrumented-extension",
            Self::EditorPlugin => "editor-plugin",
            Self::ShellIntegration => "shell-integration",
            Self::MacOsObserver => "macos-observer",
            Self::SyntheticTest => "synthetic-test",
        }
    }
}

/// The category of surface being observed. Bounded on purpose: "Brave",
/// "Cursor" and "iTerm" are values a hostile or careless producer could send,
/// and dynamic cardinality here would leak the application list.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PerceptualSourceKind {
    #[default]
    Unknown,
    BrowserChromium,
    Editor,
    Terminal,
    NativeApplication,
    ElectronApplication,
    WindowSystem,
    GenericForegroundApplication,
    Synthetic,
}

impl PerceptualSourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::BrowserChromium => "browser-chromium",
            Self::Editor => "editor",
            Self::Terminal => "terminal",
            Self::NativeApplication => "native-app",
            Self::ElectronApplication => "electron-app",
            Self::WindowSystem => "window-system",
            Self::GenericForegroundApplication => "foreground-app",
            Self::Synthetic => "synthetic",
        }
    }
}

/// Generic identity hierarchy. A browser adapter maps its own three levels onto
/// producer/surface/activity; a terminal maps session/pane/command; an outside
/// observer may fill only the first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractionScope {
    pub producer_session_id: PerceptualId,
    pub surface_session_id: Option<PerceptualId>,
    pub activity_session_id: Option<PerceptualId>,
    pub context_hash: Option<ContextBucket>,
}

/// How a measurement was obtained. Carried on every observation so an inferred
/// number can never be read as an instrumented one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MeasurementMode {
    /// The producer measured inside the application.
    Instrumented,
    /// Two independent signals were matched to bound the duration.
    Correlated,
    /// Derived from outside signals; the internal stages are unknown.
    Inferred,
    /// Statistics over a window; no individual interaction is claimed.
    #[default]
    AggregateWindow,
}

impl MeasurementMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Instrumented => "instrumented",
            Self::Correlated => "correlated",
            Self::Inferred => "inferred",
            Self::AggregateWindow => "window",
        }
    }

    /// Whether two observations may be compared as like evidence. An
    /// instrumented episode and an inferred window answer different questions.
    pub fn comparable_with(self, other: Self) -> bool {
        self == other
    }
}

/// Typed latency stage. Free strings are refused: they would become unbounded
/// metric cardinality and an unauditable vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LatencyComponentKind {
    InputDelay,
    Processing,
    Presentation,
    CommandDispatch,
    CommandExecution,
    PromptReady,
    RenderResponse,
    SchedulerWait,
    Unknown,
}

impl LatencyComponentKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InputDelay => "input-delay",
            Self::Processing => "processing",
            Self::Presentation => "presentation",
            Self::CommandDispatch => "command-dispatch",
            Self::CommandExecution => "command-execution",
            Self::PromptReady => "prompt-ready",
            Self::RenderResponse => "render-response",
            Self::SchedulerWait => "scheduler-wait",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatencyComponent {
    pub kind: LatencyComponentKind,
    pub duration_ms: u32,
}

/// A perceptual measurement of any precision.
///
/// `total_duration_ms` and `components` are both optional-by-absence: a source
/// that cannot see them leaves them empty rather than reporting zero.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptualMeasurement {
    pub total_duration_ms: Option<u32>,
    pub components: Vec<LatencyComponent>,
    pub measurement_mode: MeasurementMode,
}

impl PerceptualMeasurement {
    pub fn component(&self, kind: LatencyComponentKind) -> Option<u32> {
        self.components
            .iter()
            .find(|c| c.kind == kind)
            .map(|c| c.duration_ms)
    }

    pub fn components_sum(&self) -> u32 {
        self.components
            .iter()
            .fold(0u32, |acc, c| acc.saturating_add(c.duration_ms))
    }

    /// Only meaningful for a producer that declared a breakdown. Sources that
    /// never claimed one are not asked to satisfy it.
    pub fn reconciles_within(&self, tolerance_ms: u32) -> Option<bool> {
        let total = self.total_duration_ms?;
        if self.components.is_empty() {
            return None;
        }
        Some(self.components_sum().abs_diff(total) <= tolerance_ms)
    }
}

/// Multi-dimensional confidence. One boolean cannot express that a producer is
/// trusted, its clock is not, and its attribution is uncertain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptualQuality {
    pub source_trust_q: u16,
    pub measurement_quality_q: u16,
    pub temporal_confidence_q: u16,
    pub correlation_confidence_q: u16,
    pub attribution_confidence_q: u16,
}

impl PerceptualQuality {
    /// Weakest link: a chain of confidences is worth its lowest element.
    pub fn overall_q(self) -> u16 {
        [
            self.source_trust_q,
            self.measurement_quality_q,
            self.temporal_confidence_q,
            self.correlation_confidence_q,
            self.attribution_confidence_q,
        ]
        .into_iter()
        .min()
        .unwrap_or(0)
    }

    pub fn clamped(self) -> Self {
        Self {
            source_trust_q: self.source_trust_q.min(QUALITY_SCALE),
            measurement_quality_q: self.measurement_quality_q.min(QUALITY_SCALE),
            temporal_confidence_q: self.temporal_confidence_q.min(QUALITY_SCALE),
            correlation_confidence_q: self.correlation_confidence_q.min(QUALITY_SCALE),
            attribution_confidence_q: self.attribution_confidence_q.min(QUALITY_SCALE),
        }
    }
}

/// How well a producer's start and end signals were matched.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CorrelationState {
    Unique,
    Ambiguous,
    StartedOnly,
    CompletedOnly,
    Unmatched,
    Duplicate,
    Expired,
    InvalidSequence,
    #[default]
    InvalidSchema,
    InvalidTiming,
    InvalidMeasurement,
}

impl CorrelationState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unique => "unique",
            Self::Ambiguous => "ambiguous",
            Self::StartedOnly => "started-only",
            Self::CompletedOnly => "completed-only",
            Self::Unmatched => "unmatched",
            Self::Duplicate => "duplicate",
            Self::Expired => "expired",
            Self::InvalidSequence => "invalid-sequence",
            Self::InvalidSchema => "invalid-schema",
            Self::InvalidTiming => "invalid-timing",
            Self::InvalidMeasurement => "invalid-measurement",
        }
    }

    /// Whether this observation may enter an aggregate. Ambiguous, unmatched
    /// and every invalid state are evidence about the *measurement*, not about
    /// the machine, and must never grant credit.
    pub fn admits_to_aggregate(self) -> bool {
        matches!(self, Self::Unique | Self::CompletedOnly)
    }
}

/// Common transport facts. A producer's own hop breakdown stays in its adapter:
/// the core only needs freshness, loss and confidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptualTransportTrace {
    pub producer_segment_ms: Option<u32>,
    pub bridge_segment_ms: Option<u32>,
    pub daemon_segment_ms: Option<u32>,
    pub total_observed_ms: Option<u32>,
    pub cold_start: Option<bool>,
    pub queue_wait_ms: Option<u32>,
    pub dropped_before_ingest: bool,
    pub transport_quality_q: u16,
}

/// Shared header for every observation, whatever its precision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationHeader {
    pub observed_at_ms: u64,
    pub source_kind: PerceptualSourceKind,
    pub producer_kind: ProducerKind,
    pub scope: InteractionScope,
    pub quality: PerceptualQuality,
    pub correlation: CorrelationState,
    pub transport: PerceptualTransportTrace,
    /// True when the observation arrived through a superseded wire contract.
    /// Provenance survives migration: an old observation never gains new
    /// confidence by being re-read through a newer type.
    pub legacy_contract: bool,
}

/// A producer that measured inside the application it observes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrumentedInteractionEpisode {
    pub header: ObservationHeader,
    pub measurement: PerceptualMeasurement,
}

/// Derived from signals outside the application. Knows that something happened
/// and roughly how the machine responded; never claims the internal stages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferredInteractionEpisode {
    pub header: ObservationHeader,
    pub measurement: PerceptualMeasurement,
    /// Why the inference was drawn, as a closed label.
    pub inference_basis: InferenceBasis,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InferenceBasis {
    #[default]
    Unknown,
    ForegroundResponseProxy,
    WindowSystemActivity,
    ProcessActivity,
}

impl InferenceBasis {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::ForegroundResponseProxy => "foreground-response-proxy",
            Self::WindowSystemActivity => "window-system-activity",
            Self::ProcessActivity => "process-activity",
        }
    }
}

/// Statistics over a span where no individual interaction could be closed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualWindowObservation {
    pub header: ObservationHeader,
    pub window_ms: u32,
    /// Perceptual signals seen in the window. Not interactions: a signal is
    /// whatever the source could count, and saying so is the point.
    pub signal_count: u32,
    /// Aggregate responsiveness on 0..=1000, higher is worse.
    pub sluggishness_q: u16,
    pub measurement: PerceptualMeasurement,
}

/// The root abstraction. Three modalities, never collapsed into one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum PerceptualObservation {
    InstrumentedInteraction(InstrumentedInteractionEpisode),
    InferredInteraction(InferredInteractionEpisode),
    PerceptualWindow(PerceptualWindowObservation),
}

impl PerceptualObservation {
    pub fn header(&self) -> &ObservationHeader {
        match self {
            Self::InstrumentedInteraction(e) => &e.header,
            Self::InferredInteraction(e) => &e.header,
            Self::PerceptualWindow(w) => &w.header,
        }
    }

    pub fn measurement(&self) -> &PerceptualMeasurement {
        match self {
            Self::InstrumentedInteraction(e) => &e.measurement,
            Self::InferredInteraction(e) => &e.measurement,
            Self::PerceptualWindow(w) => &w.measurement,
        }
    }

    /// Whether this observation describes one interaction. A window does not,
    /// and must never be presented as if it did.
    pub fn is_individual_interaction(&self) -> bool {
        matches!(
            self,
            Self::InstrumentedInteraction(_) | Self::InferredInteraction(_)
        )
    }

    pub fn modality(&self) -> &'static str {
        match self {
            Self::InstrumentedInteraction(_) => "instrumented",
            Self::InferredInteraction(_) => "inferred",
            Self::PerceptualWindow(_) => "window",
        }
    }

    /// Two observations are comparable evidence only when their modality and
    /// measurement mode agree. Comparing an instrumented browser episode with
    /// an inferred window would read precision that was never measured.
    pub fn comparable_with(&self, other: &Self) -> bool {
        self.modality() == other.modality()
            && self
                .measurement()
                .measurement_mode
                .comparable_with(other.measurement().measurement_mode)
    }

    pub fn admits_to_aggregate(&self) -> bool {
        self.header().correlation.admits_to_aggregate()
    }
}

/// The wire envelope a producer sends. Generic: it carries who is speaking and
/// what they can see, then the payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualEventEnvelope {
    pub schema_version: u16,
    pub producer_id: PerceptualId,
    pub producer_version: BoundedVersion,
    pub producer_kind: ProducerKind,
    pub source_kind: PerceptualSourceKind,
    pub capabilities: PerceptualCapabilities,
    pub sequence: u64,
    pub observation: PerceptualObservation,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(mode: MeasurementMode) -> ObservationHeader {
        let _ = mode;
        ObservationHeader {
            observed_at_ms: 1_000,
            source_kind: PerceptualSourceKind::Synthetic,
            producer_kind: ProducerKind::SyntheticTest,
            scope: InteractionScope {
                producer_session_id: PerceptualId::new([7; 16]).expect("id"),
                ..InteractionScope::default()
            },
            quality: PerceptualQuality::default(),
            correlation: CorrelationState::Unique,
            transport: PerceptualTransportTrace::default(),
            legacy_contract: false,
        }
    }

    #[test]
    fn a_zero_identity_is_refused() {
        assert!(PerceptualId::new([0; 16]).is_none());
        assert!(PerceptualId::new([1; 16]).is_some());
    }

    #[test]
    fn a_producer_version_is_bounded_and_non_empty() {
        assert!(BoundedVersion::new("").is_none());
        assert!(BoundedVersion::new("   ").is_none());
        assert!(BoundedVersion::new(&"x".repeat(MAX_PRODUCER_VERSION_CHARS + 1)).is_none());
        assert_eq!(BoundedVersion::new("2.0.2").unwrap().as_str(), "2.0.2");
    }

    #[test]
    fn a_source_without_components_is_not_asked_to_reconcile() {
        let bare = PerceptualMeasurement {
            total_duration_ms: Some(180),
            components: Vec::new(),
            measurement_mode: MeasurementMode::Inferred,
        };
        assert_eq!(
            bare.reconciles_within(8),
            None,
            "absence of components is not a reconciliation failure"
        );
    }

    #[test]
    fn a_source_with_components_reconciles_within_its_tolerance() {
        let full = PerceptualMeasurement {
            total_duration_ms: Some(100),
            components: vec![
                LatencyComponent {
                    kind: LatencyComponentKind::InputDelay,
                    duration_ms: 10,
                },
                LatencyComponent {
                    kind: LatencyComponentKind::Processing,
                    duration_ms: 20,
                },
                LatencyComponent {
                    kind: LatencyComponentKind::Presentation,
                    duration_ms: 66,
                },
            ],
            measurement_mode: MeasurementMode::Instrumented,
        };
        assert_eq!(full.reconciles_within(8), Some(true));
        assert_eq!(full.reconciles_within(2), Some(false));
    }

    #[test]
    fn quality_is_the_weakest_link_not_an_average() {
        let quality = PerceptualQuality {
            source_trust_q: 1_000,
            measurement_quality_q: 1_000,
            temporal_confidence_q: 120,
            correlation_confidence_q: 1_000,
            attribution_confidence_q: 900,
        };
        assert_eq!(quality.overall_q(), 120);
    }

    #[test]
    fn only_unique_and_completed_only_admit_to_an_aggregate() {
        for state in [CorrelationState::Unique, CorrelationState::CompletedOnly] {
            assert!(state.admits_to_aggregate(), "{}", state.as_str());
        }
        for state in [
            CorrelationState::Ambiguous,
            CorrelationState::Unmatched,
            CorrelationState::Duplicate,
            CorrelationState::Expired,
            CorrelationState::InvalidSequence,
            CorrelationState::InvalidSchema,
            CorrelationState::InvalidTiming,
            CorrelationState::InvalidMeasurement,
            CorrelationState::StartedOnly,
        ] {
            assert!(
                !state.admits_to_aggregate(),
                "{} must not grant credit",
                state.as_str()
            );
        }
    }

    #[test]
    fn a_window_is_never_an_individual_interaction() {
        let window = PerceptualObservation::PerceptualWindow(PerceptualWindowObservation {
            header: header(MeasurementMode::AggregateWindow),
            window_ms: 8_000,
            signal_count: 12,
            sluggishness_q: 300,
            measurement: PerceptualMeasurement {
                measurement_mode: MeasurementMode::AggregateWindow,
                ..PerceptualMeasurement::default()
            },
        });
        assert!(!window.is_individual_interaction());
        assert_eq!(window.modality(), "window");
    }

    #[test]
    fn an_instrumented_episode_is_not_comparable_with_an_inferred_window() {
        let instrumented =
            PerceptualObservation::InstrumentedInteraction(InstrumentedInteractionEpisode {
                header: header(MeasurementMode::Instrumented),
                measurement: PerceptualMeasurement {
                    total_duration_ms: Some(120),
                    components: Vec::new(),
                    measurement_mode: MeasurementMode::Instrumented,
                },
            });
        let window = PerceptualObservation::PerceptualWindow(PerceptualWindowObservation {
            header: header(MeasurementMode::AggregateWindow),
            window_ms: 8_000,
            signal_count: 4,
            sluggishness_q: 200,
            measurement: PerceptualMeasurement {
                measurement_mode: MeasurementMode::AggregateWindow,
                ..PerceptualMeasurement::default()
            },
        });
        assert!(!instrumented.comparable_with(&window));
        assert!(instrumented.comparable_with(&instrumented.clone()));
    }

    #[test]
    fn an_inferred_episode_never_reports_itself_as_instrumented() {
        let inferred = PerceptualObservation::InferredInteraction(InferredInteractionEpisode {
            header: header(MeasurementMode::Inferred),
            measurement: PerceptualMeasurement {
                total_duration_ms: Some(200),
                components: Vec::new(),
                measurement_mode: MeasurementMode::Inferred,
            },
            inference_basis: InferenceBasis::ForegroundResponseProxy,
        });
        assert_eq!(inferred.modality(), "inferred");
        assert_eq!(
            inferred.measurement().measurement_mode,
            MeasurementMode::Inferred
        );
        assert!(inferred.is_individual_interaction());
    }

    #[test]
    fn every_closed_label_is_short_and_stable() {
        let labels: Vec<&str> = [
            ProducerKind::MacOsObserver.as_str(),
            PerceptualSourceKind::BrowserChromium.as_str(),
            MeasurementMode::Inferred.as_str(),
            LatencyComponentKind::Presentation.as_str(),
            CorrelationState::Ambiguous.as_str(),
            InferenceBasis::ForegroundResponseProxy.as_str(),
        ]
        .into_iter()
        .collect();
        assert!(labels.iter().all(|l| !l.is_empty() && l.len() <= 26));
    }
}
