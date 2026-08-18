//! Observational base for phase 0B of the Perceptual Interaction Layer.
//!
//! Records what was true around each interaction and segments sustained windows
//! of stable context. It **observes only**: nothing here emits an action, grants
//! credit, or promotes anything to Gold, and an association it reports is never
//! a causal claim.
//!
//! The distinction that drives the design: OS-level actuators can only
//! plausibly influence input delay and presentation. Processing is the site's
//! own JavaScript, and a regime dominated by it has no lever regardless of what
//! is contending for CPU.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Interactions retained for analysis. Bounded because an untrusted page
/// controls the arrival rate.
pub const MAX_EPISODES: usize = 512;
/// Regimes retained. A regime summarises many episodes, so far fewer are needed.
pub const MAX_REGIMES: usize = 64;
/// An episode older than this cannot describe current conditions.
pub const EPISODE_TTL_MS: u64 = 15 * 60 * 1000;
/// Below this an association is noise, whatever it looks like.
pub const MIN_REGIME_INTERACTIONS: u32 = 20;
/// A gap longer than this ends the current regime: the context in between is
/// unobserved, and stitching across it would invent continuity.
pub const REGIME_MAX_GAP_MS: u64 = 60 * 1000;
/// Contenders below this share of a core are not competition.
pub const CONTENDER_MIN_CPU_PERCENT: f64 = 15.0;

/// Why a contender cannot be acted upon. Closed set, low cardinality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IneligibilityReason {
    /// Chromium family root — the boost veto, three documented regressions.
    BrowserFamily,
    /// Classified interactive: the user is working in it.
    Interactive,
    /// Hard-protected system process.
    Protected,
    /// Below the CPU floor: not actually competing.
    BelowCpuFloor,
    /// Single-threaded: task-level scheduling already covers it.
    SingleThreaded,
}

impl IneligibilityReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BrowserFamily => "browser-family",
            Self::Interactive => "interactive",
            Self::Protected => "protected",
            Self::BelowCpuFloor => "below-cpu-floor",
            Self::SingleThreaded => "single-threaded",
        }
    }
}

/// One process competing for CPU around an interaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contender {
    /// Stable hash of the process name. The name itself is never persisted:
    /// a process list is user data.
    pub name_hash: u64,
    pub family: String,
    pub pid: u32,
    pub cpu_percent: f64,
    pub thread_count: u32,
    /// `None` when the process could be acted upon; `Some` names the veto.
    pub ineligible: Option<IneligibilityReason>,
}

impl Contender {
    pub fn is_actionable(&self) -> bool {
        self.ineligible.is_none()
    }
}

/// System conditions around one interaction. Captured, never inferred.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContentionSnapshot {
    pub foreground_name_hash: u64,
    pub contenders: Vec<Contender>,
    pub total_cpu_percent: f64,
    pub runnable_threads: u32,
    pub context_switches_per_sec: u64,
    pub memory_pressure: f64,
    pub thermal_level: String,
    pub throttled: bool,
    pub energy_watts: f64,
}

impl ContentionSnapshot {
    /// Contender with the most CPU, whether or not it can be acted upon.
    pub fn dominant(&self) -> Option<&Contender> {
        self.contenders
            .iter()
            .max_by(|a, b| a.cpu_percent.total_cmp(&b.cpu_percent))
    }

    pub fn actionable_count(&self) -> usize {
        self.contenders.iter().filter(|c| c.is_actionable()).count()
    }

    /// Coarse band, so strata stay low-cardinality.
    pub fn contention_level(&self) -> ContentionLevel {
        let peak = self
            .contenders
            .iter()
            .map(|c| c.cpu_percent)
            .fold(0.0_f64, f64::max);
        if self.contenders.is_empty() || peak < CONTENDER_MIN_CPU_PERCENT {
            ContentionLevel::None
        } else if peak < 50.0 {
            ContentionLevel::Moderate
        } else {
            ContentionLevel::Heavy
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ContentionLevel {
    #[default]
    None,
    Moderate,
    Heavy,
}

impl ContentionLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Moderate => "moderate",
            Self::Heavy => "heavy",
        }
    }
}

/// How confidently an episode's measurement can be trusted. Anything short of
/// `Unique` is excluded from every aggregate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CorrelationState {
    Unique,
    Ambiguous,
    UnmatchedCompleted,
    Duplicate,
    Expired,
    InvalidComponents,
    InvalidTiming,
    #[default]
    InvalidSchema,
}

impl CorrelationState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unique => "unique",
            Self::Ambiguous => "ambiguous",
            Self::UnmatchedCompleted => "unmatched",
            Self::Duplicate => "duplicate",
            Self::Expired => "expired",
            Self::InvalidComponents => "invalid-components",
            Self::InvalidTiming => "invalid-timing",
            Self::InvalidSchema => "invalid-schema",
        }
    }

    /// Only a uniquely correlated episode may enter an aggregate. Every other
    /// state is evidence about the measurement, not about the machine.
    pub fn is_usable(self) -> bool {
        matches!(self, Self::Unique)
    }
}

/// One measured interaction with the conditions around it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualInteractionEpisode {
    pub observed_at_ms: u64,
    pub context_hash: u64,
    pub total_duration_ms: u32,
    pub input_delay_ms: u32,
    pub processing_ms: u32,
    pub presentation_ms: u32,
    pub correlation: CorrelationState,
    /// Transport segment in the producer's own clock, when it reported one.
    pub transport_ms: Option<u32>,
    pub contention: ContentionSnapshot,
}

impl PerceptualInteractionEpisode {
    /// The three parts must account for the whole, within the browser's own 8 ms
    /// rounding of `duration`. A mismatch means the components describe some
    /// other interaction.
    pub fn components_reconcile(&self) -> bool {
        let sum = self
            .input_delay_ms
            .saturating_add(self.processing_ms)
            .saturating_add(self.presentation_ms);
        sum.abs_diff(self.total_duration_ms) <= 8
    }

    pub fn is_usable(&self) -> bool {
        self.correlation.is_usable() && self.components_reconcile()
    }
}

/// What an observed window supports saying — and nothing more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RegimeClassification {
    /// Not enough usable episodes to say anything.
    InsufficientSamples,
    /// Episodes exist but none survived validation.
    InvalidData,
    /// No process was competing.
    NoContention,
    /// Competition exists, and at least one contender could legally be acted on.
    ContendedActuable,
    /// Competition exists but every contender is vetoed.
    ContendedNonActuable,
    /// Latency tracks contention across this window.
    AssociationObserved,
    /// Contention varied and latency did not follow.
    AssociationNotFound,
}

impl RegimeClassification {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InsufficientSamples => "insufficient-samples",
            Self::InvalidData => "invalid-data",
            Self::NoContention => "no-contention",
            Self::ContendedActuable => "contended-actuable",
            Self::ContendedNonActuable => "contended-nonactuable",
            Self::AssociationObserved => "association-observed",
            Self::AssociationNotFound => "association-not-found",
        }
    }

    /// No observational classification ever grants causal authority. Stated as
    /// code so a later change has to argue with a test.
    pub const fn grants_causal_credit(self) -> bool {
        false
    }
}

/// A sustained window of stable context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentionRegime {
    pub regime_id: u64,
    pub started_at_ms: u64,
    pub ended_at_ms: u64,
    pub foreground_name_hash: u64,
    pub dominant_contender_hash: Option<u64>,
    pub dominant_family: Option<String>,
    pub level: ContentionLevel,
    pub interactions: u32,
    pub median_total_ms: u32,
    pub median_input_delay_ms: u32,
    pub median_processing_ms: u32,
    pub median_presentation_ms: u32,
    pub actionable_contenders: u32,
    pub classification: RegimeClassification,
}

fn median(values: &mut [u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

/// Bounded store of episodes and the regimes derived from them.
#[derive(Debug, Default)]
pub struct PerceptualObservatory {
    episodes: VecDeque<PerceptualInteractionEpisode>,
    regimes: VecDeque<ContentionRegime>,
    next_regime_id: u64,
    pub dropped_capacity: u64,
    pub dropped_ttl: u64,
    pub rejected_unusable: u64,
    correlation_counts: [u64; 8],
}

impl PerceptualObservatory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one episode. Unusable episodes are counted by state and never
    /// stored: keeping them would let a broken measurement reach an aggregate.
    pub fn record(&mut self, episode: PerceptualInteractionEpisode) {
        let slot = match episode.correlation {
            CorrelationState::Unique => 0,
            CorrelationState::Ambiguous => 1,
            CorrelationState::UnmatchedCompleted => 2,
            CorrelationState::Duplicate => 3,
            CorrelationState::Expired => 4,
            CorrelationState::InvalidComponents => 5,
            CorrelationState::InvalidTiming => 6,
            CorrelationState::InvalidSchema => 7,
        };
        self.correlation_counts[slot] = self.correlation_counts[slot].saturating_add(1);
        if !episode.is_usable() {
            self.rejected_unusable = self.rejected_unusable.saturating_add(1);
            return;
        }
        if self.episodes.len() >= MAX_EPISODES {
            self.episodes.pop_front();
            self.dropped_capacity = self.dropped_capacity.saturating_add(1);
        }
        self.episodes.push_back(episode);
    }

    /// Drop episodes that can no longer describe current conditions.
    pub fn expire(&mut self, now_ms: u64) {
        while let Some(front) = self.episodes.front() {
            if now_ms.saturating_sub(front.observed_at_ms) > EPISODE_TTL_MS {
                self.episodes.pop_front();
                self.dropped_ttl = self.dropped_ttl.saturating_add(1);
            } else {
                break;
            }
        }
    }

    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    pub fn regimes(&self) -> impl Iterator<Item = &ContentionRegime> {
        self.regimes.iter()
    }

    pub fn correlation_counts(&self) -> [u64; 8] {
        self.correlation_counts
    }

    /// Segment stored episodes into regimes. A regime breaks when the
    /// foreground changes, the contention band changes, or observation lapses
    /// for longer than `REGIME_MAX_GAP_MS`.
    pub fn segment(&mut self) -> usize {
        self.regimes.clear();
        let mut window: Vec<&PerceptualInteractionEpisode> = Vec::new();
        let mut built = 0usize;
        let mut pending: Vec<ContentionRegime> = Vec::new();
        for episode in &self.episodes {
            let breaks = window.last().is_some_and(|last| {
                last.contention.foreground_name_hash != episode.contention.foreground_name_hash
                    || last.contention.contention_level() != episode.contention.contention_level()
                    || episode.observed_at_ms.saturating_sub(last.observed_at_ms)
                        > REGIME_MAX_GAP_MS
            });
            if breaks {
                if let Some(regime) = build_regime(&window, self.next_regime_id) {
                    self.next_regime_id = self.next_regime_id.wrapping_add(1);
                    pending.push(regime);
                    built += 1;
                }
                window.clear();
            }
            window.push(episode);
        }
        if let Some(regime) = build_regime(&window, self.next_regime_id) {
            self.next_regime_id = self.next_regime_id.wrapping_add(1);
            pending.push(regime);
            built += 1;
        }
        for regime in pending {
            if self.regimes.len() >= MAX_REGIMES {
                self.regimes.pop_front();
            }
            self.regimes.push_back(regime);
        }
        built
    }
}

fn build_regime(
    window: &[&PerceptualInteractionEpisode],
    regime_id: u64,
) -> Option<ContentionRegime> {
    let first = window.first()?;
    let last = window.last()?;
    let mut totals: Vec<u32> = window.iter().map(|e| e.total_duration_ms).collect();
    let mut inputs: Vec<u32> = window.iter().map(|e| e.input_delay_ms).collect();
    let mut processing: Vec<u32> = window.iter().map(|e| e.processing_ms).collect();
    let mut presentation: Vec<u32> = window.iter().map(|e| e.presentation_ms).collect();
    let level = first.contention.contention_level();
    let dominant = first.contention.dominant();
    let actionable = first.contention.actionable_count() as u32;
    let interactions = window.len() as u32;

    let classification = if interactions < MIN_REGIME_INTERACTIONS {
        RegimeClassification::InsufficientSamples
    } else if level == ContentionLevel::None {
        RegimeClassification::NoContention
    } else if actionable == 0 {
        RegimeClassification::ContendedNonActuable
    } else {
        RegimeClassification::ContendedActuable
    };

    Some(ContentionRegime {
        regime_id,
        started_at_ms: first.observed_at_ms,
        ended_at_ms: last.observed_at_ms,
        foreground_name_hash: first.contention.foreground_name_hash,
        dominant_contender_hash: dominant.map(|c| c.name_hash),
        dominant_family: dominant.map(|c| c.family.clone()),
        level,
        interactions,
        median_total_ms: median(&mut totals),
        median_input_delay_ms: median(&mut inputs),
        median_processing_ms: median(&mut processing),
        median_presentation_ms: median(&mut presentation),
        actionable_contenders: actionable,
        classification,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contender(pid: u32, cpu: f64, ineligible: Option<IneligibilityReason>) -> Contender {
        Contender {
            name_hash: u64::from(pid) * 7,
            family: "compiler".to_string(),
            pid,
            cpu_percent: cpu,
            thread_count: 8,
            ineligible,
        }
    }

    fn episode(at_ms: u64, total: u32, contenders: Vec<Contender>) -> PerceptualInteractionEpisode {
        // Components chosen to reconcile exactly with the total.
        let input = total / 10;
        let processing = total / 10;
        let presentation = total - input - processing;
        PerceptualInteractionEpisode {
            observed_at_ms: at_ms,
            context_hash: 42,
            total_duration_ms: total,
            input_delay_ms: input,
            processing_ms: processing,
            presentation_ms: presentation,
            correlation: CorrelationState::Unique,
            transport_ms: Some(1),
            contention: ContentionSnapshot {
                foreground_name_hash: 99,
                contenders,
                ..ContentionSnapshot::default()
            },
        }
    }

    #[test]
    fn components_must_account_for_the_whole_within_browser_rounding() {
        let mut good = episode(0, 100, vec![]);
        assert!(good.components_reconcile());
        // `duration` is rounded to 8 ms by the browser, so small gaps are real.
        good.presentation_ms += 8;
        assert!(good.components_reconcile());
        good.presentation_ms += 8;
        assert!(
            !good.components_reconcile(),
            "a larger gap is another interaction"
        );
    }

    #[test]
    fn only_uniquely_correlated_episodes_enter_the_store() {
        let mut observatory = PerceptualObservatory::new();
        for state in [
            CorrelationState::Ambiguous,
            CorrelationState::Duplicate,
            CorrelationState::InvalidTiming,
            CorrelationState::UnmatchedCompleted,
        ] {
            let mut e = episode(0, 100, vec![]);
            e.correlation = state;
            observatory.record(e);
        }
        assert_eq!(observatory.episode_count(), 0);
        assert_eq!(observatory.rejected_unusable, 4);
        observatory.record(episode(0, 100, vec![]));
        assert_eq!(observatory.episode_count(), 1);
    }

    #[test]
    fn an_episode_whose_components_disagree_is_refused_even_when_unique() {
        let mut broken = episode(0, 100, vec![]);
        broken.presentation_ms = 900;
        assert!(broken.correlation.is_usable());
        let mut observatory = PerceptualObservatory::new();
        observatory.record(broken);
        assert_eq!(observatory.episode_count(), 0);
        assert_eq!(observatory.rejected_unusable, 1);
    }

    #[test]
    fn the_store_is_bounded_and_counts_what_it_drops() {
        let mut observatory = PerceptualObservatory::new();
        for index in 0..(MAX_EPISODES + 20) {
            observatory.record(episode(index as u64, 100, vec![]));
        }
        assert_eq!(observatory.episode_count(), MAX_EPISODES);
        assert_eq!(observatory.dropped_capacity, 20);
    }

    #[test]
    fn stale_episodes_expire_rather_than_describing_the_present() {
        let mut observatory = PerceptualObservatory::new();
        observatory.record(episode(0, 100, vec![]));
        observatory.record(episode(EPISODE_TTL_MS + 1, 100, vec![]));
        observatory.expire(EPISODE_TTL_MS + 2);
        assert_eq!(observatory.episode_count(), 1);
        assert_eq!(observatory.dropped_ttl, 1);
    }

    #[test]
    fn a_window_without_competition_is_no_contention() {
        let mut observatory = PerceptualObservatory::new();
        for index in 0..MIN_REGIME_INTERACTIONS {
            observatory.record(episode(u64::from(index) * 100, 60, vec![]));
        }
        observatory.segment();
        let regime = observatory.regimes().next().expect("one regime");
        assert_eq!(regime.classification, RegimeClassification::NoContention);
        assert_eq!(regime.level, ContentionLevel::None);
    }

    #[test]
    fn competition_that_cannot_be_touched_is_named_nonactuable() {
        // Every contender vetoed: the honest answer is that no lever exists,
        // which is a valid terminal result rather than a failure.
        let vetoed = vec![contender(
            900,
            80.0,
            Some(IneligibilityReason::BrowserFamily),
        )];
        let mut observatory = PerceptualObservatory::new();
        for index in 0..MIN_REGIME_INTERACTIONS {
            observatory.record(episode(u64::from(index) * 100, 300, vetoed.clone()));
        }
        observatory.segment();
        let regime = observatory.regimes().next().expect("one regime");
        assert_eq!(
            regime.classification,
            RegimeClassification::ContendedNonActuable
        );
        assert_eq!(regime.actionable_contenders, 0);
    }

    #[test]
    fn competition_with_a_legal_target_is_named_actuable() {
        let actionable = vec![contender(901, 70.0, None)];
        let mut observatory = PerceptualObservatory::new();
        for index in 0..MIN_REGIME_INTERACTIONS {
            observatory.record(episode(u64::from(index) * 100, 300, actionable.clone()));
        }
        observatory.segment();
        let regime = observatory.regimes().next().expect("one regime");
        assert_eq!(
            regime.classification,
            RegimeClassification::ContendedActuable
        );
        assert_eq!(regime.actionable_contenders, 1);
        assert_eq!(regime.dominant_family.as_deref(), Some("compiler"));
    }

    #[test]
    fn too_few_interactions_say_so_instead_of_guessing() {
        let mut observatory = PerceptualObservatory::new();
        for index in 0..(MIN_REGIME_INTERACTIONS - 1) {
            observatory.record(episode(
                u64::from(index) * 100,
                300,
                vec![contender(902, 70.0, None)],
            ));
        }
        observatory.segment();
        let regime = observatory.regimes().next().expect("one regime");
        assert_eq!(
            regime.classification,
            RegimeClassification::InsufficientSamples
        );
    }

    #[test]
    fn an_observation_gap_breaks_the_regime_rather_than_inventing_continuity() {
        let mut observatory = PerceptualObservatory::new();
        for index in 0..5u64 {
            observatory.record(episode(index * 100, 60, vec![]));
        }
        // Nothing observed for longer than the maximum gap.
        for index in 0..5u64 {
            observatory.record(episode(
                REGIME_MAX_GAP_MS + 10_000 + index * 100,
                60,
                vec![],
            ));
        }
        let built = observatory.segment();
        assert_eq!(built, 2, "the unobserved gap must not be bridged");
    }

    #[test]
    fn a_change_of_contention_band_starts_a_new_regime() {
        let mut observatory = PerceptualObservatory::new();
        for index in 0..5u64 {
            observatory.record(episode(index * 100, 60, vec![]));
        }
        for index in 5..10u64 {
            observatory.record(episode(index * 100, 300, vec![contender(903, 80.0, None)]));
        }
        assert_eq!(observatory.segment(), 2);
    }

    #[test]
    fn no_observational_classification_ever_grants_causal_credit() {
        for classification in [
            RegimeClassification::InsufficientSamples,
            RegimeClassification::InvalidData,
            RegimeClassification::NoContention,
            RegimeClassification::ContendedActuable,
            RegimeClassification::ContendedNonActuable,
            RegimeClassification::AssociationObserved,
            RegimeClassification::AssociationNotFound,
        ] {
            assert!(
                !classification.grants_causal_credit(),
                "{} must never authorise an action",
                classification.as_str()
            );
        }
    }

    #[test]
    fn every_ineligibility_reason_is_a_closed_low_cardinality_label() {
        let labels: Vec<_> = [
            IneligibilityReason::BrowserFamily,
            IneligibilityReason::Interactive,
            IneligibilityReason::Protected,
            IneligibilityReason::BelowCpuFloor,
            IneligibilityReason::SingleThreaded,
        ]
        .into_iter()
        .map(IneligibilityReason::as_str)
        .collect();
        assert_eq!(labels.len(), 5);
        assert!(labels.iter().all(|l| l.len() <= 20 && !l.is_empty()));
    }

    #[test]
    fn correlation_states_are_counted_even_when_the_episode_is_refused() {
        let mut observatory = PerceptualObservatory::new();
        let mut ambiguous = episode(0, 100, vec![]);
        ambiguous.correlation = CorrelationState::Ambiguous;
        observatory.record(ambiguous);
        observatory.record(episode(1, 100, vec![]));
        let counts = observatory.correlation_counts();
        assert_eq!(counts[0], 1, "unique");
        assert_eq!(counts[1], 1, "ambiguous, refused but counted");
    }
}

/// Anonymised evidence export.
///
/// Carries hashes and closed categories only. No URL, page text, process name
/// or any other user data crosses this boundary — a process list is as
/// identifying as a browsing history, so names never leave as names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualExport {
    pub schema_version: u16,
    pub episodes: Vec<ExportedEpisode>,
    pub regimes: Vec<ContentionRegime>,
    pub correlation_counts: ExportedCorrelationCounts,
    pub transport_quality: ExportedTransportQuality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportedEpisode {
    pub observed_at_ms: u64,
    pub context_hash: u64,
    pub total_duration_ms: u32,
    pub input_delay_ms: u32,
    pub processing_ms: u32,
    pub presentation_ms: u32,
    pub contention_level: ContentionLevel,
    pub dominant_contender_hash: Option<u64>,
    pub dominant_family: Option<String>,
    pub actionable_contenders: u32,
    pub transport_ms: Option<u32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedCorrelationCounts {
    pub unique: u64,
    pub ambiguous: u64,
    pub unmatched: u64,
    pub duplicate: u64,
    pub expired: u64,
    pub invalid_components: u64,
    pub invalid_timing: u64,
    pub invalid_schema: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedTransportQuality {
    pub episodes_with_transport: u64,
    pub episodes_without_transport: u64,
    pub dropped_capacity: u64,
    pub dropped_ttl: u64,
    pub rejected_unusable: u64,
}

pub const PERCEPTUAL_EXPORT_SCHEMA_VERSION: u16 = 1;

impl PerceptualObservatory {
    pub fn export(&self) -> PerceptualExport {
        let counts = self.correlation_counts;
        let with_transport = self
            .episodes
            .iter()
            .filter(|e| e.transport_ms.is_some())
            .count() as u64;
        PerceptualExport {
            schema_version: PERCEPTUAL_EXPORT_SCHEMA_VERSION,
            episodes: self
                .episodes
                .iter()
                .map(|e| {
                    let dominant = e.contention.dominant();
                    ExportedEpisode {
                        observed_at_ms: e.observed_at_ms,
                        context_hash: e.context_hash,
                        total_duration_ms: e.total_duration_ms,
                        input_delay_ms: e.input_delay_ms,
                        processing_ms: e.processing_ms,
                        presentation_ms: e.presentation_ms,
                        contention_level: e.contention.contention_level(),
                        dominant_contender_hash: dominant.map(|c| c.name_hash),
                        dominant_family: dominant.map(|c| c.family.clone()),
                        actionable_contenders: e.contention.actionable_count() as u32,
                        transport_ms: e.transport_ms,
                    }
                })
                .collect(),
            regimes: self.regimes.iter().cloned().collect(),
            correlation_counts: ExportedCorrelationCounts {
                unique: counts[0],
                ambiguous: counts[1],
                unmatched: counts[2],
                duplicate: counts[3],
                expired: counts[4],
                invalid_components: counts[5],
                invalid_timing: counts[6],
                invalid_schema: counts[7],
            },
            transport_quality: ExportedTransportQuality {
                episodes_with_transport: with_transport,
                episodes_without_transport: self.episodes.len() as u64 - with_transport,
                dropped_capacity: self.dropped_capacity,
                dropped_ttl: self.dropped_ttl,
                rejected_unusable: self.rejected_unusable,
            },
        }
    }
}

#[cfg(test)]
mod export_tests {
    use super::*;

    #[test]
    fn the_export_carries_no_process_name_or_page_identity() {
        let mut observatory = PerceptualObservatory::new();
        let mut episode = PerceptualInteractionEpisode {
            observed_at_ms: 1,
            context_hash: 0xDEAD_BEEF,
            total_duration_ms: 100,
            input_delay_ms: 10,
            processing_ms: 10,
            presentation_ms: 80,
            correlation: CorrelationState::Unique,
            transport_ms: Some(1),
            contention: ContentionSnapshot::default(),
        };
        episode.contention.contenders.push(Contender {
            name_hash: 0xC0FFEE,
            family: "compiler".to_string(),
            pid: 4242,
            cpu_percent: 80.0,
            thread_count: 8,
            ineligible: None,
        });
        observatory.record(episode);

        let json = serde_json::to_string(&observatory.export()).expect("export encodes");
        let lower = json.to_lowercase();
        for forbidden in [
            "url", "origin", "title", "\"name\"", "path", "host", "cookie",
        ] {
            assert!(
                !lower.contains(forbidden),
                "export must not carry {forbidden}: {json}"
            );
        }
        // The family is a closed category, and the identity is a hash.
        assert!(lower.contains("compiler"));
        assert!(lower.contains("dominant_contender_hash"));
    }

    #[test]
    fn the_export_reconciles_with_what_was_observed() {
        let mut observatory = PerceptualObservatory::new();
        let base = PerceptualInteractionEpisode {
            observed_at_ms: 1,
            context_hash: 1,
            total_duration_ms: 100,
            input_delay_ms: 10,
            processing_ms: 10,
            presentation_ms: 80,
            correlation: CorrelationState::Unique,
            transport_ms: None,
            contention: ContentionSnapshot::default(),
        };
        observatory.record(base.clone());
        let mut ambiguous = base.clone();
        ambiguous.correlation = CorrelationState::Ambiguous;
        observatory.record(ambiguous);

        let export = observatory.export();
        assert_eq!(
            export.episodes.len(),
            1,
            "only usable episodes are exported"
        );
        assert_eq!(export.correlation_counts.unique, 1);
        assert_eq!(export.correlation_counts.ambiguous, 1);
        assert_eq!(export.transport_quality.rejected_unusable, 1);
        assert_eq!(export.transport_quality.episodes_without_transport, 1);
        assert_eq!(export.schema_version, PERCEPTUAL_EXPORT_SCHEMA_VERSION);
    }
}

/// Build a contention snapshot from a process list.
///
/// Eligibility mirrors the actuator's own filters exactly — the Chromium family
/// veto, the interactive classification, hard protection and the CPU floor — so
/// "actionable" here means the same thing it means to `decide_actions`. It is a
/// question about which lever exists, never a decision to pull one.
pub fn snapshot_from_processes<F>(
    foreground_name_hash: u64,
    processes: &[(u32, String, f64, u32)],
    classify: F,
    total_cpu_percent: f64,
    memory_pressure: f64,
    thermal_level: &str,
) -> ContentionSnapshot
where
    F: Fn(&str) -> Option<IneligibilityReason>,
{
    let mut contenders: Vec<Contender> = processes
        .iter()
        .filter(|(_, _, cpu, _)| *cpu >= CONTENDER_MIN_CPU_PERCENT)
        .map(|(pid, name, cpu, threads)| Contender {
            name_hash: stable_name_hash(name),
            family: contender_family(name),
            pid: *pid,
            cpu_percent: *cpu,
            thread_count: *threads,
            ineligible: classify(name).or({
                if *threads < 2 {
                    Some(IneligibilityReason::SingleThreaded)
                } else {
                    None
                }
            }),
        })
        .collect();
    // Bounded: an unbounded contender list would grow persisted state with the
    // machine's process count.
    contenders.sort_by(|a, b| b.cpu_percent.total_cmp(&a.cpu_percent));
    contenders.truncate(8);
    ContentionSnapshot {
        foreground_name_hash,
        contenders,
        total_cpu_percent,
        runnable_threads: 0,
        context_switches_per_sec: 0,
        memory_pressure,
        thermal_level: thermal_level.to_string(),
        throttled: false,
        energy_watts: 0.0,
    }
}

/// FNV-1a over the process name. Names never leave as names: a process list is
/// as identifying as a browsing history.
pub fn stable_name_hash(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// Closed, low-cardinality category. Anything unrecognised is "other" rather
/// than the process name, which would defeat the hashing above.
pub fn contender_family(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    for (needle, family) in [
        ("rustc", "compiler"),
        ("cargo", "compiler"),
        ("clang", "compiler"),
        ("swift", "compiler"),
        ("node", "runtime"),
        ("python", "runtime"),
        ("java", "runtime"),
        ("docker", "container"),
        ("com.docker", "container"),
        ("mdworker", "indexer"),
        ("mds", "indexer"),
        ("spotlight", "indexer"),
        ("backupd", "backup"),
        ("photoanalysisd", "media"),
        ("windowserver", "compositor"),
        ("helper", "browser-helper"),
    ] {
        if lower.contains(needle) {
            return family.to_string();
        }
    }
    "other".to_string()
}

impl PerceptualInteractionEpisode {
    /// Summarise one vitals report as a single sample.
    ///
    /// The extension reports per-page aggregates, not individual interactions,
    /// so the components are sums over `interactions` and `total_duration_ms`
    /// is their sum by construction. That makes the reconciliation check
    /// trivially true here — it still guards the per-interaction path, which is
    /// where a mismatch would mean the components describe another interaction.
    pub fn from_window(
        observed_at_ms: u64,
        context_hash: u64,
        input_delay_ms: u32,
        processing_ms: u32,
        presentation_ms: u32,
        transport_ms: Option<u32>,
        contention: ContentionSnapshot,
    ) -> Self {
        Self {
            observed_at_ms,
            context_hash,
            total_duration_ms: input_delay_ms
                .saturating_add(processing_ms)
                .saturating_add(presentation_ms),
            input_delay_ms,
            processing_ms,
            presentation_ms,
            correlation: CorrelationState::Unique,
            transport_ms,
            contention,
        }
    }
}

#[cfg(test)]
mod wiring_tests {
    use super::*;

    fn classify_none(_: &str) -> Option<IneligibilityReason> {
        None
    }

    #[test]
    fn processes_below_the_cpu_floor_are_not_competition() {
        let procs = vec![
            (1, "rustc".to_string(), 80.0, 8),
            (2, "idle-thing".to_string(), 2.0, 4),
        ];
        let snap = snapshot_from_processes(1, &procs, classify_none, 50.0, 0.3, "nominal");
        assert_eq!(snap.contenders.len(), 1);
        assert_eq!(snap.contenders[0].family, "compiler");
    }

    #[test]
    fn a_single_threaded_contender_is_ineligible_by_the_same_rule_the_actuator_uses() {
        let procs = vec![(1, "rustc".to_string(), 80.0, 1)];
        let snap = snapshot_from_processes(1, &procs, classify_none, 50.0, 0.3, "nominal");
        assert_eq!(
            snap.contenders[0].ineligible,
            Some(IneligibilityReason::SingleThreaded)
        );
        assert_eq!(snap.actionable_count(), 0);
    }

    #[test]
    fn the_caller_veto_wins_over_the_thread_rule() {
        let procs = vec![(1, "Brave Browser Helper".to_string(), 80.0, 12)];
        let snap = snapshot_from_processes(
            1,
            &procs,
            |_| Some(IneligibilityReason::BrowserFamily),
            50.0,
            0.3,
            "nominal",
        );
        assert_eq!(
            snap.contenders[0].ineligible,
            Some(IneligibilityReason::BrowserFamily)
        );
    }

    #[test]
    fn the_contender_list_is_bounded_against_a_busy_machine() {
        let procs: Vec<_> = (0..40)
            .map(|i| (i, format!("proc{i}"), 90.0 - f64::from(i), 8))
            .collect();
        let snap = snapshot_from_processes(1, &procs, classify_none, 90.0, 0.3, "nominal");
        assert!(snap.contenders.len() <= 8);
        assert!(
            snap.contenders[0].cpu_percent >= snap.contenders[1].cpu_percent,
            "the busiest survive truncation"
        );
    }

    #[test]
    fn a_process_name_never_leaves_as_a_name() {
        let procs = vec![(1, "SuperSecretApp".to_string(), 80.0, 8)];
        let snap = snapshot_from_processes(1, &procs, classify_none, 50.0, 0.3, "nominal");
        let json = serde_json::to_string(&snap).expect("encodes");
        assert!(!json.contains("SuperSecretApp"));
        assert!(json.contains("other"), "only the closed family survives");
        assert_ne!(snap.contenders[0].name_hash, 0);
    }

    #[test]
    fn a_window_sample_reconciles_by_construction() {
        let episode = PerceptualInteractionEpisode::from_window(
            1_000,
            42,
            173,
            45,
            1_997,
            Some(4),
            ContentionSnapshot::default(),
        );
        assert_eq!(episode.total_duration_ms, 173 + 45 + 1_997);
        assert!(episode.components_reconcile());
        assert!(episode.is_usable());
    }
}

// ── Source-agnostic evidence ────────────────────────────────────────────────

use crate::engine::perceptual::types::{
    LatencyComponentKind, MeasurementMode, PerceptualObservation, PerceptualSourceKind,
};

/// One observation joined to the machine state around it.
///
/// The join needs nothing browser-specific: no interaction identifier, no
/// surface identity, no latency breakdown. When a source *does* provide a
/// breakdown it enriches the analysis; when it does not, the evidence is
/// weaker and says so through its measurement mode rather than by faking parts.
#[derive(Debug, Clone, PartialEq)]
pub struct PerceptualRegimeEvidence {
    pub observation: PerceptualObservation,
    pub contention: ContentionSnapshot,
}

impl PerceptualRegimeEvidence {
    pub fn source_kind(&self) -> PerceptualSourceKind {
        self.observation.header().source_kind
    }

    pub fn measurement_mode(&self) -> MeasurementMode {
        self.observation.measurement().measurement_mode
    }

    /// Total latency when the source could measure one. `None` is a legitimate
    /// answer, not a zero.
    pub fn total_duration_ms(&self) -> Option<u32> {
        self.observation.measurement().total_duration_ms
    }

    /// Evidence may only be pooled with evidence of the same modality and mode.
    /// An instrumented browser episode and an inferred window answer different
    /// questions, and averaging them would read precision nobody measured.
    pub fn poolable_with(&self, other: &Self) -> bool {
        self.observation.comparable_with(&other.observation)
    }

    pub fn admits_to_aggregate(&self) -> bool {
        self.observation.admits_to_aggregate()
    }
}

/// Stratum key for observational association. Measurement mode is part of the
/// key by construction, so two modes can never land in the same cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceStratum {
    pub source_kind: PerceptualSourceKind,
    pub measurement_mode: MeasurementMode,
    pub contention_level: ContentionLevel,
    pub actionable: bool,
    /// Hashed foreground application. An association that holds only under one
    /// app is a different finding from one that holds everywhere.
    pub foreground_hash: u64,
}

impl EvidenceStratum {
    pub fn of(evidence: &PerceptualRegimeEvidence) -> Self {
        Self {
            source_kind: evidence.source_kind(),
            measurement_mode: evidence.measurement_mode(),
            contention_level: evidence.contention.contention_level(),
            actionable: evidence.contention.actionable_count() > 0,
            foreground_hash: evidence.contention.foreground_name_hash,
        }
    }
}

/// Summary of one stratum. Reports its own insufficiency rather than a number
/// that looks like a finding.
#[derive(Debug, Clone, PartialEq)]
pub struct StratumSummary {
    pub stratum: EvidenceStratum,
    pub samples: u32,
    pub median_total_ms: Option<u32>,
    pub spread_ms: Option<u32>,
    pub classification: RegimeClassification,
}

/// Summarise evidence by stratum. Observational only: a stratum never becomes a
/// causal claim, whatever its numbers look like.
pub fn summarise_strata(evidence: &[PerceptualRegimeEvidence]) -> Vec<StratumSummary> {
    let mut keys: Vec<EvidenceStratum> = Vec::new();
    for item in evidence.iter().filter(|e| e.admits_to_aggregate()) {
        let key = EvidenceStratum::of(item);
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys.into_iter()
        .map(|key| {
            let mut totals: Vec<u32> = evidence
                .iter()
                .filter(|e| e.admits_to_aggregate() && EvidenceStratum::of(e) == key)
                .filter_map(PerceptualRegimeEvidence::total_duration_ms)
                .collect();
            let samples = evidence
                .iter()
                .filter(|e| e.admits_to_aggregate() && EvidenceStratum::of(e) == key)
                .count() as u32;
            let (median_total, spread) = if totals.is_empty() {
                (None, None)
            } else {
                totals.sort_unstable();
                (
                    Some(totals[totals.len() / 2]),
                    Some(totals[totals.len() - 1] - totals[0]),
                )
            };
            let classification = if samples < MIN_REGIME_INTERACTIONS {
                RegimeClassification::InsufficientSamples
            } else if key.contention_level == ContentionLevel::None {
                RegimeClassification::NoContention
            } else if !key.actionable {
                RegimeClassification::ContendedNonActuable
            } else {
                RegimeClassification::ContendedActuable
            };
            StratumSummary {
                stratum: key,
                samples,
                median_total_ms: median_total,
                spread_ms: spread,
                classification,
            }
        })
        .collect()
}

#[cfg(test)]
mod agnostic_evidence_tests {
    use super::*;
    use crate::engine::perceptual::adapters::synthetic::SyntheticPerceptualAdapter;
    use crate::engine::perceptual::adapters::PerceptualAdapter;
    use crate::engine::perceptual::types::MonotonicMillis;

    fn contention(actionable: bool) -> ContentionSnapshot {
        ContentionSnapshot {
            foreground_name_hash: 42,
            contenders: vec![Contender {
                name_hash: 7,
                family: "compiler".to_string(),
                pid: 900,
                cpu_percent: 80.0,
                thread_count: 8,
                ineligible: (!actionable).then_some(IneligibilityReason::BrowserFamily),
            }],
            ..ContentionSnapshot::default()
        }
    }

    fn synthetic_evidence(count: usize, actionable: bool) -> Vec<PerceptualRegimeEvidence> {
        let mut adapter = SyntheticPerceptualAdapter::new();
        (0..count)
            .filter_map(|index| {
                let envelope =
                    adapter.total_only_interaction(index as u64 * 100, 150 + index as u32);
                adapter
                    .normalize(envelope, MonotonicMillis(index as u64 * 100))
                    .pop()
                    .map(|observation| PerceptualRegimeEvidence {
                        observation,
                        contention: contention(actionable),
                    })
            })
            .collect()
    }

    #[test]
    fn a_non_browser_observation_joins_the_machine_state_without_browser_fields() {
        let evidence = synthetic_evidence(1, true).pop().expect("evidence");
        // No interaction id, no surface identity, no breakdown — and it joins.
        assert_eq!(evidence.observation.header().scope.surface_session_id, None);
        assert!(evidence.observation.measurement().components.is_empty());
        assert_eq!(evidence.source_kind(), PerceptualSourceKind::Synthetic);
        assert_eq!(evidence.contention.dominant().map(|c| c.pid), Some(900));
    }

    #[test]
    fn a_source_without_a_total_still_forms_evidence() {
        let mut adapter = SyntheticPerceptualAdapter::new();
        let envelope = adapter.bare_window(1_000, 8_000);
        let observation = adapter
            .normalize(envelope, MonotonicMillis(1_000))
            .pop()
            .expect("window");
        let evidence = PerceptualRegimeEvidence {
            observation,
            contention: contention(true),
        };
        assert_eq!(evidence.total_duration_ms(), None, "absent, not zero");
        assert!(evidence.admits_to_aggregate());
    }

    #[test]
    fn measurement_mode_is_part_of_the_stratum_so_modes_never_pool() {
        let mut mixed = synthetic_evidence(2, true);
        let mut adapter = SyntheticPerceptualAdapter::new();
        let envelope = adapter.bare_window(5_000, 8_000);
        let window = adapter
            .normalize(envelope, MonotonicMillis(5_000))
            .pop()
            .expect("window");
        mixed.push(PerceptualRegimeEvidence {
            observation: window,
            contention: contention(true),
        });

        let strata = summarise_strata(&mixed);
        assert_eq!(strata.len(), 2, "inferred and window are separate strata");
        assert!(!mixed[0].poolable_with(mixed.last().expect("window")));
    }

    #[test]
    fn a_thin_stratum_reports_insufficiency_instead_of_a_number_that_looks_like_a_finding() {
        let strata = summarise_strata(&synthetic_evidence(3, true));
        assert_eq!(strata.len(), 1);
        assert_eq!(
            strata[0].classification,
            RegimeClassification::InsufficientSamples
        );
    }

    #[test]
    fn a_populated_stratum_with_a_legal_target_is_actuable_but_never_causal() {
        let strata = summarise_strata(&synthetic_evidence(
            MIN_REGIME_INTERACTIONS as usize + 2,
            true,
        ));
        assert_eq!(strata.len(), 1);
        assert_eq!(
            strata[0].classification,
            RegimeClassification::ContendedActuable
        );
        assert!(!strata[0].classification.grants_causal_credit());
        assert!(strata[0].median_total_ms.is_some());
    }

    #[test]
    fn vetoed_contention_is_nonactuable_for_any_source() {
        let strata = summarise_strata(&synthetic_evidence(
            MIN_REGIME_INTERACTIONS as usize + 2,
            false,
        ));
        assert_eq!(
            strata[0].classification,
            RegimeClassification::ContendedNonActuable
        );
    }

    #[test]
    fn refused_observations_never_reach_a_stratum() {
        let mut evidence = synthetic_evidence(MIN_REGIME_INTERACTIONS as usize + 2, true);
        for item in evidence.iter_mut().take(5) {
            if let PerceptualObservation::InferredInteraction(ref mut e) = item.observation {
                e.header.correlation =
                    crate::engine::perceptual::types::CorrelationState::Ambiguous;
            }
        }
        let strata = summarise_strata(&evidence);
        assert_eq!(
            strata[0].samples,
            MIN_REGIME_INTERACTIONS + 2 - 5,
            "ambiguous evidence is excluded from the aggregate"
        );
    }
}

// ── Stratified observational association ────────────────────────────────────

/// What a stratified comparison supports saying. Never more than that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AssociationVerdict {
    /// Latency was consistently different when this family was contending.
    Observed,
    /// Contention varied and latency did not follow it.
    NotFound,
    /// Not enough of either side to compare.
    InsufficientSamples,
}

impl AssociationVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Observed => "association-observed",
            Self::NotFound => "association-not-found",
            Self::InsufficientSamples => "insufficient-samples",
        }
    }

    /// Association is not causation. Written as code so a later change that
    /// tries to treat it as causal has to delete a test first.
    pub const fn is_causal(self) -> bool {
        false
    }
}

/// Minimum on each side of a with/without comparison. Below this the
/// difference between two medians is noise wearing a number's clothes.
pub const MIN_ASSOCIATION_SAMPLES_PER_SIDE: u32 = 8;
/// A median shift smaller than this is inside the browser's own 8 ms rounding
/// and the sampling jitter around it.
pub const MIN_ASSOCIATION_DELTA_MS: i32 = 10;

/// One with/without comparison for a single contender family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationalAssociation {
    pub contender_family: String,
    pub actionable: bool,
    pub samples_with: u32,
    pub samples_without: u32,
    pub median_with_ms: Option<u32>,
    pub median_without_ms: Option<u32>,
    pub delta_ms: i32,
    /// Which stage moved most, when the source could break latency down.
    pub worst_component: Option<LatencyComponentKind>,
    pub component_delta_ms: i32,
    pub confidence_q: u16,
    pub verdict: AssociationVerdict,
}

impl ObservationalAssociation {
    /// Whether the pattern disappears when the family is absent — the question
    /// that separates "this always happens" from "this happens with them".
    pub fn improves_without_contender(&self) -> bool {
        self.verdict == AssociationVerdict::Observed && self.delta_ms > 0
    }
}

fn median_of(values: &mut Vec<u32>) -> Option<u32> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[values.len() / 2])
}

/// Compare latency with and without each contender family, within one stratum.
///
/// Only evidence of the same modality and measurement mode is compared, because
/// the stratum key carries both. Nothing here establishes causation: a family
/// that co-occurs with slow interactions may be a symptom of the same load, or
/// may be irrelevant and merely busy at the same time.
pub fn associate_by_contender(
    evidence: &[PerceptualRegimeEvidence],
) -> Vec<ObservationalAssociation> {
    let usable: Vec<&PerceptualRegimeEvidence> = evidence
        .iter()
        .filter(|e| e.admits_to_aggregate() && e.total_duration_ms().is_some())
        .collect();
    if usable.is_empty() {
        return Vec::new();
    }
    let mut families: Vec<String> = Vec::new();
    for item in &usable {
        for contender in &item.contention.contenders {
            if !families.contains(&contender.family) {
                families.push(contender.family.clone());
            }
        }
    }

    families
        .into_iter()
        .map(|family| {
            let present = |e: &PerceptualRegimeEvidence| {
                e.contention.contenders.iter().any(|c| c.family == family)
            };
            let mut with: Vec<u32> = usable
                .iter()
                .filter(|e| present(e))
                .filter_map(|e| e.total_duration_ms())
                .collect();
            let mut without: Vec<u32> = usable
                .iter()
                .filter(|e| !present(e))
                .filter_map(|e| e.total_duration_ms())
                .collect();
            let samples_with = with.len() as u32;
            let samples_without = without.len() as u32;
            let actionable = usable
                .iter()
                .filter(|e| present(e))
                .flat_map(|e| e.contention.contenders.iter())
                .any(|c| c.family == family && c.is_actionable());

            let median_with = median_of(&mut with);
            let median_without = median_of(&mut without);
            let delta = match (median_with, median_without) {
                (Some(a), Some(b)) => i64::from(a) - i64::from(b),
                _ => 0,
            } as i32;

            // Which stage moved most, when both sides could break latency down.
            let component_shift = |kind: LatencyComponentKind| -> Option<i32> {
                let mut a: Vec<u32> = usable
                    .iter()
                    .filter(|e| present(e))
                    .filter_map(|e| e.observation.measurement().component(kind))
                    .collect();
                let mut b: Vec<u32> = usable
                    .iter()
                    .filter(|e| !present(e))
                    .filter_map(|e| e.observation.measurement().component(kind))
                    .collect();
                match (median_of(&mut a), median_of(&mut b)) {
                    (Some(x), Some(y)) => Some((i64::from(x) - i64::from(y)) as i32),
                    _ => None,
                }
            };
            let mut worst_component = None;
            let mut component_delta: i32 = 0;
            for kind in [
                LatencyComponentKind::InputDelay,
                LatencyComponentKind::Processing,
                LatencyComponentKind::Presentation,
            ] {
                if let Some(shift) = component_shift(kind) {
                    if shift.abs() > component_delta.abs() {
                        component_delta = shift;
                        worst_component = Some(kind);
                    }
                }
            }

            let verdict = if samples_with < MIN_ASSOCIATION_SAMPLES_PER_SIDE
                || samples_without < MIN_ASSOCIATION_SAMPLES_PER_SIDE
            {
                AssociationVerdict::InsufficientSamples
            } else if delta.abs() >= MIN_ASSOCIATION_DELTA_MS {
                AssociationVerdict::Observed
            } else {
                AssociationVerdict::NotFound
            };

            // Confidence grows with the thinner side, never with the total: a
            // thousand samples on one side and three on the other is still a
            // three-sample comparison.
            let thinner = samples_with.min(samples_without);
            let confidence_q = if verdict == AssociationVerdict::Observed {
                (u32::from(QUALITY_SCALE_U16).min(thinner * 20)) as u16
            } else {
                0
            };

            ObservationalAssociation {
                contender_family: family,
                actionable,
                samples_with,
                samples_without,
                median_with_ms: median_with,
                median_without_ms: median_without,
                delta_ms: delta,
                worst_component,
                component_delta_ms: component_delta,
                confidence_q,
                verdict,
            }
        })
        .collect()
}

const QUALITY_SCALE_U16: u16 = 1_000;

/// Families that co-occur with the slowest observations, ranked. Answers "what
/// keeps showing up when things feel bad" without asserting they cause it.
pub fn families_co_occurring_with_slowness(
    evidence: &[PerceptualRegimeEvidence],
    slow_threshold_ms: u32,
) -> Vec<(String, u32, bool)> {
    let mut counts: Vec<(String, u32, bool)> = Vec::new();
    for item in evidence
        .iter()
        .filter(|e| e.admits_to_aggregate())
        .filter(|e| {
            e.total_duration_ms()
                .is_some_and(|d| d >= slow_threshold_ms)
        })
    {
        for contender in &item.contention.contenders {
            match counts.iter_mut().find(|(f, _, _)| *f == contender.family) {
                Some(entry) => {
                    entry.1 = entry.1.saturating_add(1);
                    entry.2 |= contender.is_actionable();
                }
                None => counts.push((contender.family.clone(), 1, contender.is_actionable())),
            }
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    counts
}

#[cfg(test)]
mod association_tests {
    use super::*;
    use crate::engine::perceptual::types::{
        InteractionScope, MeasurementMode, ObservationHeader, PerceptualId, PerceptualMeasurement,
        PerceptualQuality, PerceptualTransportTrace, ProducerKind,
    };

    fn evidence(total_ms: u32, family: Option<&str>, actionable: bool) -> PerceptualRegimeEvidence {
        let contenders = family
            .map(|f| {
                vec![Contender {
                    name_hash: 1,
                    family: f.to_string(),
                    pid: 900,
                    cpu_percent: 70.0,
                    thread_count: 8,
                    ineligible: (!actionable).then_some(IneligibilityReason::BrowserFamily),
                }]
            })
            .unwrap_or_default();
        PerceptualRegimeEvidence {
            observation:
                crate::engine::perceptual::types::PerceptualObservation::InferredInteraction(
                    crate::engine::perceptual::types::InferredInteractionEpisode {
                        header: ObservationHeader {
                            observed_at_ms: 1,
                            source_kind: PerceptualSourceKind::Synthetic,
                            producer_kind: ProducerKind::SyntheticTest,
                            scope: InteractionScope {
                                producer_session_id: PerceptualId::new([1; 16]).expect("id"),
                                ..InteractionScope::default()
                            },
                            quality: PerceptualQuality::default(),
                            correlation:
                                crate::engine::perceptual::types::CorrelationState::CompletedOnly,
                            transport: PerceptualTransportTrace::default(),
                            legacy_contract: false,
                        },
                        measurement: PerceptualMeasurement {
                            total_duration_ms: Some(total_ms),
                            components: Vec::new(),
                            measurement_mode: MeasurementMode::Inferred,
                        },
                        inference_basis:
                            crate::engine::perceptual::types::InferenceBasis::ProcessActivity,
                    },
                ),
            contention: ContentionSnapshot {
                foreground_name_hash: 42,
                contenders,
                ..ContentionSnapshot::default()
            },
        }
    }

    fn dataset(
        with_ms: u32,
        without_ms: u32,
        per_side: u32,
        actionable: bool,
    ) -> Vec<PerceptualRegimeEvidence> {
        let mut out = Vec::new();
        for _ in 0..per_side {
            out.push(evidence(with_ms, Some("compiler"), actionable));
            out.push(evidence(without_ms, None, actionable));
        }
        out
    }

    #[test]
    fn a_repeated_shift_with_a_contender_is_observed_but_never_causal() {
        let associations = associate_by_contender(&dataset(300, 120, 12, true));
        let compiler = associations
            .iter()
            .find(|a| a.contender_family == "compiler")
            .expect("family present");
        assert_eq!(compiler.verdict, AssociationVerdict::Observed);
        assert_eq!(compiler.delta_ms, 180);
        assert!(compiler.improves_without_contender());
        assert!(
            !compiler.verdict.is_causal(),
            "association is not causation"
        );
    }

    #[test]
    fn varying_contention_without_a_latency_shift_is_not_found() {
        let associations = associate_by_contender(&dataset(122, 120, 12, true));
        let compiler = associations
            .iter()
            .find(|a| a.contender_family == "compiler")
            .expect("family present");
        assert_eq!(compiler.verdict, AssociationVerdict::NotFound);
        assert_eq!(compiler.confidence_q, 0);
    }

    #[test]
    fn a_one_sided_comparison_reports_insufficiency_instead_of_a_delta() {
        // Nine hundred samples on one side and three on the other is still a
        // three-sample comparison.
        let mut lopsided: Vec<_> = (0..40)
            .map(|_| evidence(300, Some("compiler"), true))
            .collect();
        lopsided.extend((0..3).map(|_| evidence(120, None, true)));
        let associations = associate_by_contender(&lopsided);
        let compiler = &associations[0];
        assert_eq!(compiler.verdict, AssociationVerdict::InsufficientSamples);
        assert_eq!(compiler.confidence_q, 0);
    }

    #[test]
    fn confidence_follows_the_thinner_side_not_the_total() {
        let balanced = associate_by_contender(&dataset(300, 120, 10, true));
        let mut lopsided: Vec<_> = (0..100)
            .map(|_| evidence(300, Some("compiler"), true))
            .collect();
        lopsided.extend((0..10).map(|_| evidence(120, None, true)));
        let skewed = associate_by_contender(&lopsided);
        assert_eq!(balanced[0].confidence_q, skewed[0].confidence_q);
    }

    #[test]
    fn legal_actionability_travels_with_the_association() {
        let vetoed = associate_by_contender(&dataset(300, 120, 12, false));
        assert!(!vetoed[0].actionable, "a vetoed contender offers no lever");
        let legal = associate_by_contender(&dataset(300, 120, 12, true));
        assert!(legal[0].actionable);
    }

    #[test]
    fn families_that_keep_appearing_when_things_feel_slow_are_ranked() {
        let mut mixed = dataset(300, 120, 10, true);
        for _ in 0..4 {
            mixed.push(evidence(400, Some("indexer"), true));
        }
        let ranked = families_co_occurring_with_slowness(&mixed, 250);
        assert_eq!(ranked[0].0, "compiler", "the most frequent leads");
        assert!(ranked.iter().any(|(f, _, _)| f == "indexer"));
    }

    #[test]
    fn an_empty_or_unusable_dataset_yields_no_association_rather_than_a_zero() {
        assert!(associate_by_contender(&[]).is_empty());
        let mut ambiguous = dataset(300, 120, 12, true);
        for item in &mut ambiguous {
            if let crate::engine::perceptual::types::PerceptualObservation::InferredInteraction(
                ref mut e,
            ) = item.observation
            {
                e.header.correlation =
                    crate::engine::perceptual::types::CorrelationState::Ambiguous;
            }
        }
        assert!(associate_by_contender(&ambiguous).is_empty());
    }
}
