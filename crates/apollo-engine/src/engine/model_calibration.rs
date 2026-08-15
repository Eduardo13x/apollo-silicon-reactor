use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::marker::PhantomData;

use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub use crate::engine::decision_ledger::BinaryPredictionTarget;
use crate::engine::decision_ledger::{
    AdviserContribution, CandidateAlternative, DecisionId, HierarchyCoordinates, PredictionRecord,
};
use crate::engine::installation_identity::InstallationId;
use crate::engine::telemetry_medallion::{ActuatorFamily, EvidenceTier, HardwareRegime};

pub const MAX_CALIBRATION_KEYS: usize = 512;
pub const MAX_EXACT_CALIBRATION_KEYS: usize = 384;
pub const MAX_FAMILY_CALIBRATION_KEYS: usize = 128;
pub const MAX_ACCEPTED_DECISION_IDS: usize = 128;
pub const MAX_CALIBRATION_STATE_BYTES: usize = 1_048_576;

const MAX_PRODUCER_CHARS: usize = 48;
const MAX_ACTION_CLASS_CHARS: usize = 96;
const MAX_CALIBRATION_ACTION_CHARS: usize = 320;
const MAX_WORKLOAD_CHARS: usize = 64;
const EMA_ALPHA: f64 = 0.10;
const FLOAT_BOUNDARY_EPSILON: f64 = 1e-12;

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "kebab-case")]
pub enum ProducerId {
    Actuator,
    WorldModel,
    GpuModel,
    Markov,
    CausalGraph,
    Mpc,
    Nars,
    PolicyScorer,
    PredictiveAgent,
    OutcomeTracker,
    LocalConsolidator,
    SurvivalMode,
    Maintenance,
    #[default]
    Other,
}

impl ProducerId {
    pub(crate) fn canonical(source: &str) -> Self {
        let source = source.trim().to_ascii_lowercase();
        match source.as_str() {
            "actuator" | "root-actuator" | "test-actuator" => Self::Actuator,
            "world-model" | "world_model" => Self::WorldModel,
            "gpu-model" | "gpu_model" | "gpu" => Self::GpuModel,
            "markov" | "focus-markov" | "markov-prewarm" => Self::Markov,
            "causal" | "causal-graph" | "causal_graph" => Self::CausalGraph,
            "mpc" | "mpc-horizon" => Self::Mpc,
            "nars" => Self::Nars,
            "policy" | "policy-scorer" | "policy_scorer" => Self::PolicyScorer,
            "predictive-agent" | "predictive_agent" => Self::PredictiveAgent,
            "outcome-tracker" | "outcome_tracker" => Self::OutcomeTracker,
            "local-consolidator" | "local_consolidator" => Self::LocalConsolidator,
            "survival-mode" | "survival_mode" => Self::SurvivalMode,
            "maintenance" | "maintenance-purge" => Self::Maintenance,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum CalibrationActionScope {
    Exact(
        #[serde(
            deserialize_with = "deserialize_calibration_action",
            serialize_with = "serialize_calibration_action"
        )]
        String,
    ),
    Family(ActuatorFamily),
}

impl Default for CalibrationActionScope {
    fn default() -> Self {
        Self::Family(ActuatorFamily::Coordinated)
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "kebab-case")]
pub enum ProcessClass {
    Foreground,
    Background,
    Browser,
    Compiler,
    Media,
    System,
    #[default]
    Other,
}

impl ProcessClass {
    pub fn from_target(family: ActuatorFamily, target: &str, foreground_active: bool) -> Self {
        match family {
            ActuatorFamily::Boost
            | ActuatorFamily::Throttle
            | ActuatorFamily::Freeze
            | ActuatorFamily::Unfreeze
            | ActuatorFamily::Memorystatus
            | ActuatorFamily::ThreadQos
            | ActuatorFamily::PredictivePreThrottle
            | ActuatorFamily::PredictivePurge
            | ActuatorFamily::ChromiumEcore
            | ActuatorFamily::ChromiumPurge
            | ActuatorFamily::ChromiumJetsam => {
                let target = target.to_ascii_lowercase();
                if target.contains("chrome")
                    || target.contains("chromium")
                    || target.contains("brave")
                {
                    Self::Browser
                } else if target.contains("cargo")
                    || target.contains("rustc")
                    || target.contains("clang")
                    || target.contains("xcode")
                {
                    Self::Compiler
                } else if target.contains("audio")
                    || target.contains("video")
                    || target.contains("media")
                {
                    Self::Media
                } else if foreground_active {
                    Self::Foreground
                } else {
                    Self::Background
                }
            }
            ActuatorFamily::Coordinated => Self::Other,
            _ => Self::System,
        }
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "kebab-case")]
pub enum CalibrationHorizon {
    #[default]
    Sec5,
    Sec30,
    Min2,
    Min10,
}

impl CalibrationHorizon {
    pub fn from_cycles(cycles: u64) -> Self {
        match cycles {
            0..=10 => Self::Sec5,
            11..=60 => Self::Sec30,
            61..=240 => Self::Min2,
            _ => Self::Min10,
        }
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "kebab-case")]
pub enum PressureBand {
    Low,
    #[default]
    Moderate,
    High,
    Critical,
}

impl PressureBand {
    pub fn from_fraction(value: f64) -> Option<Self> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return None;
        }
        Some(if value < 0.35 {
            Self::Low
        } else if value < 0.55 {
            Self::Moderate
        } else if value < 0.75 {
            Self::High
        } else {
            Self::Critical
        })
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "kebab-case")]
pub enum ThermalBand {
    Cool,
    #[default]
    Nominal,
    Warm,
    Hot,
}

impl ThermalBand {
    pub fn from_fraction(value: f64) -> Option<Self> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return None;
        }
        Some(if value < 0.25 {
            Self::Cool
        } else if value < 0.50 {
            Self::Nominal
        } else if value < 0.75 {
            Self::Warm
        } else {
            Self::Hot
        })
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "kebab-case")]
pub enum ForegroundContext {
    Active,
    Idle,
    Launching,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(default)]
pub struct CalibrationKey {
    pub producer: ProducerId,
    pub action: CalibrationActionScope,
    #[serde(
        default,
        deserialize_with = "deserialize_calibration_workload",
        serialize_with = "serialize_calibration_workload"
    )]
    pub workload: String,
    pub process_class: ProcessClass,
    pub horizon: CalibrationHorizon,
    pub pressure: PressureBand,
    pub thermal: ThermalBand,
    pub foreground: ForegroundContext,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SeparabilityState {
    #[default]
    Individual,
    Confounded,
    CoordinatedComposite,
    SeparableMember {
        decision_id: DecisionId,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct CalibrationProvenance {
    pub local_authority_eligible: bool,
    #[serde(default, deserialize_with = "deserialize_proposer")]
    pub proposer: String,
    #[serde(default, deserialize_with = "deserialize_alternatives")]
    pub alternatives: Vec<CandidateAlternative>,
    #[serde(default, deserialize_with = "deserialize_predictions")]
    pub predictions: Vec<PredictionRecord>,
    #[serde(default, deserialize_with = "deserialize_advisers")]
    pub adviser_contributions: Vec<AdviserContribution>,
    pub hierarchy: HierarchyCoordinates,
    pub cohort_size: u16,
    pub separability: SeparabilityState,
}

impl CalibrationProvenance {
    pub fn bounded(mut self) -> Self {
        self.proposer = bounded_text(self.proposer.trim(), MAX_PRODUCER_CHARS);
        self.alternatives = self
            .alternatives
            .into_iter()
            .take(8)
            .map(CandidateAlternative::bounded)
            .filter(|alternative| !alternative.action_key.is_empty())
            .collect();
        self.predictions = self
            .predictions
            .into_iter()
            .take(8)
            .map(PredictionRecord::bounded)
            .filter(|prediction| !prediction.source.is_empty())
            .collect();
        self.adviser_contributions = self
            .adviser_contributions
            .into_iter()
            .take(8)
            .map(AdviserContribution::bounded)
            .filter(|adviser| !adviser.adviser.is_empty())
            .collect();
        self
    }
}

#[derive(Debug)]
pub struct CalibrationObservation<'a> {
    pub decision_id: Option<DecisionId>,
    pub tier: EvidenceTier,
    pub installation_id: InstallationId,
    pub hardware_regime: HardwareRegime,
    pub family: ActuatorFamily,
    pub action_key: String,
    pub workload: String,
    pub process_class: ProcessClass,
    pub pressure: PressureBand,
    pub thermal: ThermalBand,
    pub foreground: ForegroundContext,
    pub context_valid: bool,
    pub quality: f64,
    pub actual_utility: f64,
    pub effective: bool,
    pub provenance: &'a CalibrationProvenance,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum TrustState {
    #[default]
    Immature,
    Candidate,
    Validated,
    Trusted,
    Degraded,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct CalibrationRecord {
    pub key: CalibrationKey,
    pub installation_id: InstallationId,
    pub hardware_regime: HardwareRegime,
    pub lifetime_forecast_count: u64,
    pub authority_gold_count: u64,
    pub signed_error_ema: f64,
    pub normalized_mae_ema: f64,
    pub coverage_ema: f64,
    pub brier_ema: Option<f64>,
    pub brier_count: u64,
    pub quality_ema: f64,
    pub welford_count: u64,
    pub welford_mean: f64,
    pub welford_m2: f64,
    pub trust: TrustState,
    pub authority_epoch: u64,
    pub last_update_sequence: u64,
    pub family_fallback: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(default)]
pub struct ModelKey {
    pub producer: ProducerId,
    pub action: CalibrationActionScope,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(default)]
pub struct ContextFingerprint {
    pub workload: String,
    pub process_class: ProcessClass,
    pub horizon: CalibrationHorizon,
    pub pressure: PressureBand,
    pub thermal: ThermalBand,
    pub foreground: ForegroundContext,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct CalibrationWindow {
    pub quality: f64,
    pub normalized_mae: f64,
    pub coverage: f64,
}

impl CalibrationWindow {
    fn is_stable(&self) -> bool {
        [self.quality, self.normalized_mae, self.coverage]
            .into_iter()
            .all(f64::is_finite)
            && self.quality + FLOAT_BOUNDARY_EPSILON >= 0.85
            && self.normalized_mae <= 0.10 + FLOAT_BOUNDARY_EPSILON
            && self.coverage + FLOAT_BOUNDARY_EPSILON >= 0.90
            && self.coverage <= 1.0
    }

    fn is_bad(&self) -> bool {
        ![self.quality, self.normalized_mae, self.coverage]
            .into_iter()
            .all(f64::is_finite)
            || self.quality + FLOAT_BOUNDARY_EPSILON < 0.85
            || self.normalized_mae > 0.15 + FLOAT_BOUNDARY_EPSILON
            || self.coverage + FLOAT_BOUNDARY_EPSILON < 0.80
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct ModelTrustRecord {
    pub key: ModelKey,
    pub installation_id: InstallationId,
    pub hardware_regime: HardwareRegime,
    pub lifetime_forecast_count: u64,
    pub authority_gold_count: u64,
    pub signed_error_ema: f64,
    pub normalized_mae: f64,
    pub coverage_ema: f64,
    pub quality_ema: f64,
    pub welford_count: u64,
    pub welford_mean: f64,
    pub welford_m2: f64,
    #[serde(default, deserialize_with = "deserialize_contexts")]
    pub contexts: Vec<ContextFingerprint>,
    #[serde(default, deserialize_with = "deserialize_windows")]
    pub completed_windows: Vec<CalibrationWindow>,
    pub consecutive_bad_windows: u8,
    pub current_window_count: u8,
    pub current_window_quality_sum: f64,
    pub current_window_mae_sum: f64,
    pub current_window_coverage_sum: f64,
    pub trust: TrustState,
    pub authority_epoch: u64,
    pub recovering_from_degraded: bool,
    pub last_update_sequence: u64,
}

impl ModelTrustRecord {
    pub fn confidence_upper_95(&self) -> Option<f64> {
        if self.welford_count < 50
            || self.welford_count != self.authority_gold_count
            || !self.welford_mean.is_finite()
            || !self.welford_m2.is_finite()
            || self.welford_m2 < 0.0
        {
            return None;
        }
        let variance = self.welford_m2 / (self.welford_count.saturating_sub(1) as f64);
        let standard_error = (variance / self.welford_count as f64).sqrt();
        let upper = self.welford_mean + 1.96 * standard_error;
        upper.is_finite().then_some(upper)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct ModelCalibrationPersisted {
    pub installation_id: InstallationId,
    pub hardware_regime: HardwareRegime,
    #[serde(default, deserialize_with = "deserialize_calibration_records")]
    pub records: Vec<CalibrationRecord>,
    #[serde(default, deserialize_with = "deserialize_model_records")]
    pub models: Vec<ModelTrustRecord>,
    #[serde(default, deserialize_with = "deserialize_decision_ids")]
    pub accepted_decision_ids: Vec<DecisionId>,
    pub accepted_decision_id_high_water: u64,
    pub update_sequence: u64,
    pub metrics: ModelCalibrationMetrics,
}

fn deserialize_proposer<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_text(deserializer, MAX_PRODUCER_CHARS)
}

fn deserialize_calibration_action<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_checked_text(
        deserializer,
        MAX_CALIBRATION_ACTION_CHARS,
        "calibration action",
    )
}

fn deserialize_calibration_workload<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_checked_text(deserializer, MAX_WORKLOAD_CHARS, "calibration workload")
}

fn deserialize_checked_text<'de, D>(
    deserializer: D,
    max_chars: usize,
    description: &'static str,
) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    struct CheckedTextVisitor {
        max_chars: usize,
        description: &'static str,
    }

    impl Visitor<'_> for CheckedTextVisitor {
        type Value = String;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "{} of at most {} characters",
                self.description, self.max_chars
            )
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if value.chars().nth(self.max_chars).is_some() {
                return Err(E::invalid_length(self.max_chars + 1, &self));
            }
            Ok(value.to_string())
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if value.chars().nth(self.max_chars).is_some() {
                return Err(E::invalid_length(self.max_chars + 1, &self));
            }
            Ok(value)
        }
    }

    deserializer.deserialize_str(CheckedTextVisitor {
        max_chars,
        description,
    })
}

fn serialize_calibration_action<S>(value: &String, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serialize_checked_text(
        value,
        serializer,
        MAX_CALIBRATION_ACTION_CHARS,
        "calibration action",
    )
}

fn serialize_calibration_workload<S>(value: &String, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serialize_checked_text(
        value,
        serializer,
        MAX_WORKLOAD_CHARS,
        "calibration workload",
    )
}

fn serialize_checked_text<S>(
    value: &str,
    serializer: S,
    max_chars: usize,
    description: &'static str,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.chars().nth(max_chars).is_some() {
        return Err(serde::ser::Error::custom(format_args!(
            "{description} exceeds {max_chars} characters"
        )));
    }
    serializer.serialize_str(value)
}

fn deserialize_bounded_text<'de, D>(deserializer: D, max_chars: usize) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    struct BoundedTextVisitor {
        max_chars: usize,
    }

    impl Visitor<'_> for BoundedTextVisitor {
        type Value = String;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a bounded calibration string")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(bounded_text(value, self.max_chars))
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(bounded_text(&value, self.max_chars))
        }
    }

    deserializer.deserialize_string(BoundedTextVisitor { max_chars })
}

struct BoundedVecVisitor<T, const MAX: usize>(PhantomData<T>);

impl<'de, T, const MAX: usize> Visitor<'de> for BoundedVecVisitor<T, MAX>
where
    T: Deserialize<'de>,
{
    type Value = Vec<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "a sequence containing at most {MAX} retained items"
        )
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(MAX.min(sequence.size_hint().unwrap_or(MAX)));
        while values.len() < MAX {
            let Some(value) = sequence.next_element()? else {
                return Ok(values);
            };
            values.push(value);
        }
        while sequence.next_element::<IgnoredAny>()?.is_some() {}
        Ok(values)
    }
}

fn deserialize_bounded_vec<'de, D, T, const MAX: usize>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    deserializer.deserialize_seq(BoundedVecVisitor::<T, MAX>(PhantomData))
}

fn deserialize_predictions<'de, D>(deserializer: D) -> Result<Vec<PredictionRecord>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, PredictionRecord, 8>(deserializer)
}

fn deserialize_alternatives<'de, D>(deserializer: D) -> Result<Vec<CandidateAlternative>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, CandidateAlternative, 8>(deserializer)
}

fn deserialize_advisers<'de, D>(deserializer: D) -> Result<Vec<AdviserContribution>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, AdviserContribution, 8>(deserializer)
}

fn deserialize_contexts<'de, D>(deserializer: D) -> Result<Vec<ContextFingerprint>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, ContextFingerprint, 16>(deserializer)
}

fn deserialize_windows<'de, D>(deserializer: D) -> Result<Vec<CalibrationWindow>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, CalibrationWindow, 3>(deserializer)
}

struct RankedCalibrationRecords {
    capacity: usize,
    by_key: BTreeMap<CalibrationKey, CalibrationRecord>,
    by_rank: BTreeSet<(u64, CalibrationKey)>,
}

impl RankedCalibrationRecords {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            by_key: BTreeMap::new(),
            by_rank: BTreeSet::new(),
        }
    }

    fn push(&mut self, record: CalibrationRecord) -> bool {
        let key = record.key.clone();
        let rank = (record.last_update_sequence, key.clone());
        if let Some(existing) = self.by_key.get(&key) {
            if rank.0 <= existing.last_update_sequence {
                return false;
            }
            self.by_rank
                .remove(&(existing.last_update_sequence, key.clone()));
        } else if self.by_key.len() >= self.capacity {
            let Some(victim_rank) = self.by_rank.first().cloned() else {
                return false;
            };
            if rank <= victim_rank {
                return false;
            }
            self.by_rank.remove(&victim_rank);
            self.by_key.remove(&victim_rank.1);
        }
        self.by_rank.insert(rank);
        self.by_key.insert(key, record);
        true
    }

    fn into_values(self) -> impl Iterator<Item = CalibrationRecord> {
        self.by_key.into_values()
    }
}

struct BoundedCalibrationRecords {
    exact: RankedCalibrationRecords,
    family: RankedCalibrationRecords,
}

impl BoundedCalibrationRecords {
    fn new() -> Self {
        Self {
            exact: RankedCalibrationRecords::new(MAX_EXACT_CALIBRATION_KEYS),
            family: RankedCalibrationRecords::new(MAX_FAMILY_CALIBRATION_KEYS),
        }
    }

    fn push(&mut self, record: CalibrationRecord) -> bool {
        if record.family_fallback {
            self.family.push(record)
        } else {
            self.exact.push(record)
        }
    }

    fn into_vec(self) -> Vec<CalibrationRecord> {
        self.exact
            .into_values()
            .chain(self.family.into_values())
            .collect()
    }
}

struct BoundedModelRecords {
    by_key: BTreeMap<ModelKey, ModelTrustRecord>,
    by_rank: BTreeSet<(u64, ModelKey)>,
}

impl BoundedModelRecords {
    fn new() -> Self {
        Self {
            by_key: BTreeMap::new(),
            by_rank: BTreeSet::new(),
        }
    }

    fn push(&mut self, model: ModelTrustRecord) -> bool {
        let key = model.key.clone();
        let rank = (model.last_update_sequence, key.clone());
        if let Some(existing) = self.by_key.get(&key) {
            if rank.0 <= existing.last_update_sequence {
                return false;
            }
            self.by_rank
                .remove(&(existing.last_update_sequence, key.clone()));
        } else if self.by_key.len() >= MAX_CALIBRATION_KEYS {
            let Some(victim_rank) = self.by_rank.first().cloned() else {
                return false;
            };
            if rank <= victim_rank {
                return false;
            }
            self.by_rank.remove(&victim_rank);
            self.by_key.remove(&victim_rank.1);
        }
        self.by_rank.insert(rank);
        self.by_key.insert(key, model);
        true
    }

    fn into_vec(self) -> Vec<ModelTrustRecord> {
        self.by_key.into_values().collect()
    }
}

fn deserialize_calibration_records<'de, D>(
    deserializer: D,
) -> Result<Vec<CalibrationRecord>, D::Error>
where
    D: Deserializer<'de>,
{
    struct RecordsVisitor;

    impl<'de> Visitor<'de> for RecordsVisitor {
        type Value = Vec<CalibrationRecord>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a bounded calibration record sequence")
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut retained = BoundedCalibrationRecords::new();
            while let Some(record) = sequence.next_element()? {
                retained.push(record);
            }
            Ok(retained.into_vec())
        }
    }

    deserializer.deserialize_seq(RecordsVisitor)
}

fn deserialize_model_records<'de, D>(deserializer: D) -> Result<Vec<ModelTrustRecord>, D::Error>
where
    D: Deserializer<'de>,
{
    struct ModelsVisitor;

    impl<'de> Visitor<'de> for ModelsVisitor {
        type Value = Vec<ModelTrustRecord>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a bounded model trust sequence")
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut retained = BoundedModelRecords::new();
            while let Some(model) = sequence.next_element()? {
                retained.push(model);
            }
            Ok(retained.into_vec())
        }
    }

    deserializer.deserialize_seq(ModelsVisitor)
}

fn deserialize_decision_ids<'de, D>(deserializer: D) -> Result<Vec<DecisionId>, D::Error>
where
    D: Deserializer<'de>,
{
    struct DecisionIdsVisitor;

    impl<'de> Visitor<'de> for DecisionIdsVisitor {
        type Value = Vec<DecisionId>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a bounded decision-id sequence")
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut retained = VecDeque::with_capacity(MAX_ACCEPTED_DECISION_IDS);
            let mut seen = BTreeSet::new();
            while let Some(id) = sequence.next_element::<DecisionId>()? {
                if id.0 == 0 || !seen.insert(id) {
                    continue;
                }
                retained.push_back(id);
                if retained.len() > MAX_ACCEPTED_DECISION_IDS {
                    if let Some(oldest) = retained.pop_front() {
                        seen.remove(&oldest);
                    }
                }
            }
            Ok(retained.into_iter().collect())
        }
    }

    deserializer.deserialize_seq(DecisionIdsVisitor)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(default)]
pub struct ModelCalibrationMetrics {
    pub record_count: usize,
    pub exact_record_count: usize,
    pub family_record_count: usize,
    pub accepted_forecasts_total: u64,
    pub ignored_total: u64,
    pub duplicate_total: u64,
    pub unknown_producer_total: u64,
    pub confounded_cohort_ignored_total: u64,
    pub adviser_without_forecast_total: u64,
    pub non_authoritative_total: u64,
    pub non_gold_total: u64,
    pub missing_decision_id_total: u64,
    pub foreign_origin_total: u64,
    pub invalid_total: u64,
    pub missing_forecast_total: u64,
    pub exact_fallback_total: u64,
    pub family_eviction_total: u64,
    pub restore_discarded_total: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelCalibrationSummary {
    pub metrics: ModelCalibrationMetrics,
    pub immature_models: usize,
    pub candidate_models: usize,
    pub validated_models: usize,
    pub trusted_models: usize,
    pub degraded_models: usize,
    pub worst_record: Option<CalibrationKey>,
    pub worst_normalized_mae: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IgnoreReason {
    NonAuthoritative,
    NonGold,
    MissingDecisionId,
    ForeignOrigin,
    InvalidObservation,
    Duplicate,
    Confounded,
    MissingForecast,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ForecastCalibrationDelta {
    pub key: CalibrationKey,
    pub predicted_utility: f64,
    pub actual_utility: f64,
    pub signed_error: f64,
    pub normalized_absolute_error: f64,
    pub uncertainty_covered: bool,
    pub brier: Option<f64>,
    pub trust_before: TrustState,
    pub trust_after: TrustState,
}

impl Default for ForecastCalibrationDelta {
    fn default() -> Self {
        Self {
            key: CalibrationKey::default(),
            predicted_utility: 0.0,
            actual_utility: 0.0,
            signed_error: 0.0,
            normalized_absolute_error: 0.0,
            uncertainty_covered: false,
            brier: None,
            trust_before: TrustState::Immature,
            trust_after: TrustState::Immature,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationUpdate {
    Accepted {
        forecasts: u8,
        deltas: Vec<ForecastCalibrationDelta>,
    },
    Ignored(IgnoreReason),
}

pub(crate) fn project_forecast_delta(
    key: CalibrationKey,
    prediction: &PredictionRecord,
    actual_utility: f64,
    effective: bool,
    trust_before: TrustState,
    trust_after: TrustState,
) -> ForecastCalibrationDelta {
    let signed_error = (actual_utility - prediction.expected_utility).clamp(-2.0, 2.0);
    let normalized_absolute_error = (signed_error.abs() / 2.0).clamp(0.0, 1.0);
    let uncertainty_covered = actual_utility
        >= prediction.expected_utility - prediction.uncertainty
        && actual_utility <= prediction.expected_utility + prediction.uncertainty;
    let brier = if prediction.binary_target == Some(BinaryPredictionTarget::Effective) {
        prediction
            .positive_probability
            .filter(|probability| probability.is_finite() && (0.0..=1.0).contains(probability))
            .map(|probability| (probability - f64::from(effective)).powi(2))
    } else {
        None
    };
    ForecastCalibrationDelta {
        key,
        predicted_utility: prediction.expected_utility,
        actual_utility,
        signed_error,
        normalized_absolute_error,
        uncertainty_covered,
        brier,
        trust_before,
        trust_after,
    }
}

pub(crate) fn valid_forecast_deltas(
    observation: &CalibrationObservation<'_>,
    deltas: &[ForecastCalibrationDelta],
) -> bool {
    if deltas.is_empty() || deltas.len() > 8 {
        return false;
    }
    let exact_action = canonical_action_class(observation.family, &observation.action_key);
    let workload = canonical_workload(&observation.workload);
    let mut seen = BTreeSet::new();
    let mut accepted = 0_usize;

    for prediction in observation.provenance.predictions.iter().take(8) {
        let producer = ProducerId::canonical(&prediction.source);
        if producer == ProducerId::Other
            || !seen.insert(producer)
            || !prediction.expected_utility.is_finite()
            || !(-1.0..=1.0).contains(&prediction.expected_utility)
            || !prediction.uncertainty.is_finite()
            || prediction.uncertainty <= 0.0
        {
            continue;
        }
        let Some(delta) = deltas.get(accepted) else {
            return false;
        };
        let action = match &delta.key.action {
            CalibrationActionScope::Exact(action)
                if exact_action.as_deref() == Some(action.as_str()) =>
            {
                CalibrationActionScope::Exact(action.clone())
            }
            CalibrationActionScope::Family(family) if *family == observation.family => {
                CalibrationActionScope::Family(*family)
            }
            _ => return false,
        };
        let key = CalibrationKey {
            producer,
            action,
            workload: workload.clone(),
            process_class: observation.process_class,
            horizon: CalibrationHorizon::from_cycles(prediction.horizon_cycles),
            pressure: observation.pressure,
            thermal: observation.thermal,
            foreground: observation.foreground,
        };
        if project_forecast_delta(
            key,
            prediction,
            observation.actual_utility,
            observation.effective,
            delta.trust_before,
            delta.trust_after,
        ) != *delta
        {
            return false;
        }
        accepted += 1;
    }

    accepted == deltas.len()
}

#[derive(Debug)]
pub struct ModelCalibrationStore {
    installation_id: InstallationId,
    hardware_regime: HardwareRegime,
    records: BTreeMap<CalibrationKey, CalibrationRecord>,
    models: BTreeMap<ModelKey, ModelTrustRecord>,
    accepted_decision_ids: BTreeSet<DecisionId>,
    accepted_decision_order: VecDeque<DecisionId>,
    accepted_decision_id_high_water: u64,
    update_sequence: u64,
    metrics: ModelCalibrationMetrics,
    exact_count: usize,
    family_count: usize,
}

impl ModelCalibrationStore {
    pub fn new(installation_id: InstallationId) -> Self {
        Self {
            installation_id,
            hardware_regime: HardwareRegime::default(),
            records: BTreeMap::new(),
            models: BTreeMap::new(),
            accepted_decision_ids: BTreeSet::new(),
            accepted_decision_order: VecDeque::new(),
            accepted_decision_id_high_water: 0,
            update_sequence: 0,
            metrics: ModelCalibrationMetrics::default(),
            exact_count: 0,
            family_count: 0,
        }
    }

    pub fn accepted_decision_ids(&self) -> impl Iterator<Item = DecisionId> + '_ {
        self.accepted_decision_order.iter().copied()
    }

    pub fn accepted_decision_count(&self) -> usize {
        self.accepted_decision_order.len()
    }

    pub fn observe_local_gold(
        &mut self,
        observation: &CalibrationObservation<'_>,
    ) -> CalibrationUpdate {
        let Some(decision_id) = observation.decision_id else {
            return self.ignore(IgnoreReason::MissingDecisionId);
        };
        if !observation.provenance.local_authority_eligible {
            return self.ignore(IgnoreReason::NonAuthoritative);
        }
        if observation.tier != EvidenceTier::Gold {
            return self.ignore(IgnoreReason::NonGold);
        }
        if !self.installation_id.is_known()
            || observation.installation_id != self.installation_id
            || !observation.hardware_regime.is_known()
        {
            return self.ignore(IgnoreReason::ForeignOrigin);
        }
        if !observation.quality.is_finite()
            || !observation.context_valid
            || !(0.0..=1.0).contains(&observation.quality)
            || !observation.actual_utility.is_finite()
            || !(-1.0..=1.0).contains(&observation.actual_utility)
        {
            return self.ignore(IgnoreReason::InvalidObservation);
        }
        if self.accepted_decision_ids.contains(&decision_id) {
            self.metrics.duplicate_total = self.metrics.duplicate_total.saturating_add(1);
            return self.ignore(IgnoreReason::Duplicate);
        }
        let invalid_cohort = match observation.provenance.separability {
            SeparabilityState::Individual => observation.provenance.cohort_size > 1,
            SeparabilityState::Confounded => true,
            SeparabilityState::CoordinatedComposite => {
                observation.family != ActuatorFamily::Coordinated
                    || !observation.action_key.starts_with("coordinated:")
            }
            SeparabilityState::SeparableMember {
                decision_id: member,
            } => member != decision_id,
        };
        if invalid_cohort {
            self.metrics.confounded_cohort_ignored_total = self
                .metrics
                .confounded_cohort_ignored_total
                .saturating_add(1);
            return self.ignore(IgnoreReason::Confounded);
        }

        if self.hardware_regime.is_known() && self.hardware_regime != observation.hardware_regime {
            self.reset_authority_for_hardware(observation.hardware_regime);
        } else if !self.hardware_regime.is_known() {
            self.hardware_regime = observation.hardware_regime;
        }

        let exact_action = canonical_action_class(observation.family, &observation.action_key);
        let workload = canonical_workload(&observation.workload);
        let forecast_sources: BTreeSet<ProducerId> = observation
            .provenance
            .predictions
            .iter()
            .take(8)
            .map(|prediction| ProducerId::canonical(&prediction.source))
            .filter(|producer| *producer != ProducerId::Other)
            .collect();
        for adviser in observation.provenance.adviser_contributions.iter().take(8) {
            let producer = ProducerId::canonical(&adviser.adviser);
            if producer != ProducerId::Other && !forecast_sources.contains(&producer) {
                self.metrics.adviser_without_forecast_total = self
                    .metrics
                    .adviser_without_forecast_total
                    .saturating_add(1);
            }
        }
        let mut seen = BTreeSet::new();
        let mut accepted = 0_u8;
        let mut deltas = Vec::with_capacity(8);
        for prediction in observation.provenance.predictions.iter().take(8) {
            let producer = ProducerId::canonical(&prediction.source);
            if producer == ProducerId::Other {
                self.metrics.unknown_producer_total =
                    self.metrics.unknown_producer_total.saturating_add(1);
                continue;
            }
            if !seen.insert(producer)
                || !prediction.expected_utility.is_finite()
                || !(-1.0..=1.0).contains(&prediction.expected_utility)
                || !prediction.uncertainty.is_finite()
                || prediction.uncertainty <= 0.0
            {
                continue;
            }

            let mut key = CalibrationKey {
                producer,
                action: exact_action
                    .clone()
                    .map(CalibrationActionScope::Exact)
                    .unwrap_or(CalibrationActionScope::Family(observation.family)),
                workload: workload.clone(),
                process_class: observation.process_class,
                horizon: CalibrationHorizon::from_cycles(prediction.horizon_cycles),
                pressure: observation.pressure,
                thermal: observation.thermal,
                foreground: observation.foreground,
            };
            if matches!(key.action, CalibrationActionScope::Exact(_))
                && !self.records.contains_key(&key)
                && self.exact_count >= MAX_EXACT_CALIBRATION_KEYS
            {
                key.action = CalibrationActionScope::Family(observation.family);
                self.metrics.exact_fallback_total =
                    self.metrics.exact_fallback_total.saturating_add(1);
            }
            if matches!(key.action, CalibrationActionScope::Family(_))
                && !self.records.contains_key(&key)
                && self.family_count >= MAX_FAMILY_CALIBRATION_KEYS
            {
                self.evict_family_record();
            }

            self.update_sequence = self.update_sequence.saturating_add(1);
            let sequence = self.update_sequence;
            let family_fallback = matches!(key.action, CalibrationActionScope::Family(_));
            let new_record = !self.records.contains_key(&key);
            let record = self
                .records
                .entry(key.clone())
                .or_insert_with(|| CalibrationRecord {
                    key: key.clone(),
                    installation_id: self.installation_id,
                    hardware_regime: observation.hardware_regime,
                    family_fallback,
                    ..CalibrationRecord::default()
                });
            update_record(record, observation, prediction, sequence);
            if new_record {
                if family_fallback {
                    self.family_count = self.family_count.saturating_add(1);
                } else {
                    self.exact_count = self.exact_count.saturating_add(1);
                }
            }
            let model_key = ModelKey {
                producer,
                action: key.action.clone(),
            };
            let trust_before = self
                .models
                .get(&model_key)
                .map_or(TrustState::Immature, |model| model.trust);
            let context = context_fingerprint(&key);
            let model = self
                .models
                .entry(model_key.clone())
                .or_insert_with(|| ModelTrustRecord {
                    key: model_key.clone(),
                    installation_id: self.installation_id,
                    hardware_regime: observation.hardware_regime,
                    ..ModelTrustRecord::default()
                });
            let degraded = update_model(model, observation, prediction, &context, sequence);
            if degraded {
                self.reset_records_for_model(&model_key, observation.hardware_regime);
            }
            if let Some(record) = self.records.get_mut(&key) {
                record.trust = self
                    .models
                    .get(&model_key)
                    .map_or(TrustState::Immature, |model| model.trust);
            }
            let trust_after = self
                .models
                .get(&model_key)
                .map_or(TrustState::Immature, |model| model.trust);
            deltas.push(project_forecast_delta(
                key,
                prediction,
                observation.actual_utility,
                observation.effective,
                trust_before,
                trust_after,
            ));
            accepted = accepted.saturating_add(1);
        }

        if accepted == 0 {
            return self.ignore(IgnoreReason::MissingForecast);
        }
        self.accepted_decision_ids.insert(decision_id);
        self.accepted_decision_order.push_back(decision_id);
        while self.accepted_decision_order.len() > MAX_ACCEPTED_DECISION_IDS {
            if let Some(oldest) = self.accepted_decision_order.pop_front() {
                self.accepted_decision_ids.remove(&oldest);
            }
        }
        self.accepted_decision_id_high_water =
            self.accepted_decision_id_high_water.max(decision_id.0);
        self.metrics.accepted_forecasts_total = self
            .metrics
            .accepted_forecasts_total
            .saturating_add(u64::from(accepted));
        self.refresh_record_metrics();
        CalibrationUpdate::Accepted {
            forecasts: accepted,
            deltas,
        }
    }

    pub fn records(&self) -> impl Iterator<Item = &CalibrationRecord> {
        self.records.values()
    }

    pub fn record(&self, key: &CalibrationKey) -> Option<&CalibrationRecord> {
        self.records.get(key)
    }

    pub fn record_or_family(&self, key: &CalibrationKey) -> Option<&CalibrationRecord> {
        self.records.get(key).or_else(|| {
            let CalibrationActionScope::Exact(action) = &key.action else {
                return None;
            };
            let family = action
                .split_once(':')
                .and_then(|(prefix, _)| actuator_family_from_str(prefix))?;
            let mut fallback = key.clone();
            fallback.action = CalibrationActionScope::Family(family);
            self.records.get(&fallback)
        })
    }

    pub fn trust_for(&self, key: &CalibrationKey) -> TrustState {
        self.model_for(key)
            .map_or(TrustState::Immature, |model| model.trust)
    }

    pub fn model_for(&self, key: &CalibrationKey) -> Option<&ModelTrustRecord> {
        self.record_or_family(key).and_then(|record| {
            self.models.get(&ModelKey {
                producer: record.key.producer,
                action: record.key.action.clone(),
            })
        })
    }

    pub fn metrics(&self) -> ModelCalibrationMetrics {
        self.metrics
    }

    pub fn summary(&self) -> ModelCalibrationSummary {
        let mut summary = ModelCalibrationSummary {
            metrics: self.metrics,
            ..ModelCalibrationSummary::default()
        };
        for model in self.models.values() {
            match model.trust {
                TrustState::Immature => summary.immature_models += 1,
                TrustState::Candidate => summary.candidate_models += 1,
                TrustState::Validated => summary.validated_models += 1,
                TrustState::Trusted => summary.trusted_models += 1,
                TrustState::Degraded => summary.degraded_models += 1,
            }
        }
        for record in self.records.values() {
            let replace = summary.worst_record.as_ref().is_none_or(|worst| {
                record.normalized_mae_ema > summary.worst_normalized_mae
                    || (record.normalized_mae_ema == summary.worst_normalized_mae
                        && &record.key < worst)
            });
            if replace {
                summary.worst_normalized_mae = record.normalized_mae_ema;
                summary.worst_record = Some(record.key.clone());
            }
        }
        summary
    }

    pub fn validate_hardware(&mut self, current: HardwareRegime) {
        if self.records.is_empty() {
            self.hardware_regime = current;
            return;
        }
        if !current.is_known()
            || (self.hardware_regime.is_known() && self.hardware_regime != current)
        {
            self.reset_authority_for_hardware(current);
        } else if !self.hardware_regime.is_known() {
            self.hardware_regime = current;
        }
    }

    pub fn snapshot(&self) -> ModelCalibrationPersisted {
        let state = ModelCalibrationPersisted {
            installation_id: self.installation_id,
            hardware_regime: self.hardware_regime,
            records: self.records.values().cloned().collect(),
            models: self.models.values().cloned().collect(),
            accepted_decision_ids: self.accepted_decision_order.iter().copied().collect(),
            accepted_decision_id_high_water: self.accepted_decision_id_high_water,
            update_sequence: self.update_sequence,
            metrics: self.metrics,
        };
        bound_snapshot_state(state).0
    }

    pub fn restore(&mut self, state: ModelCalibrationPersisted, current: HardwareRegime) {
        let ModelCalibrationPersisted {
            installation_id: persisted_installation,
            hardware_regime: persisted_hardware,
            records,
            models,
            accepted_decision_ids,
            accepted_decision_id_high_water,
            update_sequence,
            metrics,
        } = state;
        self.records.clear();
        self.models.clear();
        self.accepted_decision_ids.clear();
        self.accepted_decision_order.clear();
        self.exact_count = 0;
        self.family_count = 0;
        self.metrics = metrics;
        self.metrics.record_count = 0;
        self.metrics.exact_record_count = 0;
        self.metrics.family_record_count = 0;
        if !self.installation_id.is_known()
            || persisted_installation != self.installation_id
            || !persisted_installation.is_known()
            || !current.is_known()
        {
            self.hardware_regime = current;
            self.metrics.foreign_origin_total = self.metrics.foreign_origin_total.saturating_add(1);
            return;
        }

        self.hardware_regime = current;
        self.update_sequence = update_sequence;
        self.accepted_decision_id_high_water = accepted_decision_id_high_water;
        for id in accepted_decision_ids.into_iter().filter(|id| id.0 > 0) {
            if self.accepted_decision_ids.insert(id) {
                self.accepted_decision_order.push_back(id);
            }
            while self.accepted_decision_order.len() > MAX_ACCEPTED_DECISION_IDS {
                if let Some(oldest) = self.accepted_decision_order.pop_front() {
                    self.accepted_decision_ids.remove(&oldest);
                }
            }
        }
        if !persisted_hardware.is_known() || persisted_hardware != current {
            self.accepted_decision_ids.clear();
            self.accepted_decision_order.clear();
        }

        let mut retained_records = BoundedCalibrationRecords::new();
        for mut record in records {
            if !valid_restored_record(&record)
                || record.installation_id != self.installation_id
                || record.hardware_regime != persisted_hardware
            {
                self.metrics.restore_discarded_total =
                    self.metrics.restore_discarded_total.saturating_add(1);
                continue;
            }
            record.trust = TrustState::Immature;
            if !retained_records.push(record) {
                self.metrics.restore_discarded_total =
                    self.metrics.restore_discarded_total.saturating_add(1);
            }
        }
        for record in retained_records.into_vec() {
            if record.family_fallback {
                self.family_count = self.family_count.saturating_add(1);
            } else {
                self.exact_count = self.exact_count.saturating_add(1);
            }
            self.records.insert(record.key.clone(), record);
        }

        let model_evidence = restored_model_evidence(&self.records);
        let mut retained_models = BoundedModelRecords::new();
        for mut model in models {
            if !model_evidence.contains_key(&model.key) || !sanitize_model(&mut model) {
                self.metrics.restore_discarded_total =
                    self.metrics.restore_discarded_total.saturating_add(1);
                continue;
            }
            if !retained_models.push(model) {
                self.metrics.restore_discarded_total =
                    self.metrics.restore_discarded_total.saturating_add(1);
            }
        }
        for mut model in retained_models.into_vec() {
            let evidence = model_evidence
                .get(&model.key)
                .expect("retained model evidence");
            let cold_reset = persisted_hardware != current
                || !persisted_hardware.is_known()
                || model.hardware_regime != current
                || model.installation_id != self.installation_id
                || !restored_model_matches(&model, evidence);
            if cold_reset {
                cold_reset_restored_model(&mut model, evidence, &mut self.records, current);
                self.metrics.restore_discarded_total =
                    self.metrics.restore_discarded_total.saturating_add(1);
            } else {
                recompute_trust(&mut model);
            }
            self.models.insert(model.key.clone(), model);
        }
        for (model_key, evidence) in &model_evidence {
            if self.models.contains_key(model_key) {
                continue;
            }
            let mut model = ModelTrustRecord {
                key: model_key.clone(),
                installation_id: self.installation_id,
                hardware_regime: current,
                ..ModelTrustRecord::default()
            };
            cold_reset_restored_model(&mut model, evidence, &mut self.records, current);
            self.models.insert(model_key.clone(), model);
        }
        for record in self.records.values_mut() {
            record.trust = self
                .models
                .get(&ModelKey {
                    producer: record.key.producer,
                    action: record.key.action.clone(),
                })
                .map_or(TrustState::Immature, |model| model.trust);
        }
        self.refresh_record_metrics();
    }

    fn ignore(&mut self, reason: IgnoreReason) -> CalibrationUpdate {
        self.metrics.ignored_total = self.metrics.ignored_total.saturating_add(1);
        match reason {
            IgnoreReason::NonAuthoritative => {
                self.metrics.non_authoritative_total =
                    self.metrics.non_authoritative_total.saturating_add(1)
            }
            IgnoreReason::NonGold => {
                self.metrics.non_gold_total = self.metrics.non_gold_total.saturating_add(1)
            }
            IgnoreReason::MissingDecisionId => {
                self.metrics.missing_decision_id_total =
                    self.metrics.missing_decision_id_total.saturating_add(1)
            }
            IgnoreReason::ForeignOrigin => {
                self.metrics.foreign_origin_total =
                    self.metrics.foreign_origin_total.saturating_add(1)
            }
            IgnoreReason::InvalidObservation => {
                self.metrics.invalid_total = self.metrics.invalid_total.saturating_add(1)
            }
            IgnoreReason::MissingForecast => {
                self.metrics.missing_forecast_total =
                    self.metrics.missing_forecast_total.saturating_add(1)
            }
            IgnoreReason::Duplicate | IgnoreReason::Confounded => {}
        }
        CalibrationUpdate::Ignored(reason)
    }

    fn exact_record_count(&self) -> usize {
        self.exact_count
    }

    fn family_record_count(&self) -> usize {
        self.family_count
    }

    fn evict_family_record(&mut self) {
        let victim = self
            .records
            .iter()
            .filter(|(key, _)| matches!(key.action, CalibrationActionScope::Family(_)))
            .min_by(|left, right| {
                left.1
                    .authority_gold_count
                    .cmp(&right.1.authority_gold_count)
                    .then_with(|| {
                        left.1
                            .last_update_sequence
                            .cmp(&right.1.last_update_sequence)
                    })
                    .then_with(|| left.0.cmp(right.0))
            })
            .map(|(key, _)| key.clone());
        if let Some(victim) = victim {
            self.records.remove(&victim);
            self.family_count = self.family_count.saturating_sub(1);
            self.metrics.family_eviction_total =
                self.metrics.family_eviction_total.saturating_add(1);
            self.remove_orphan_models();
        }
    }

    fn reset_authority_for_hardware(&mut self, hardware_regime: HardwareRegime) {
        self.hardware_regime = hardware_regime;
        self.accepted_decision_ids.clear();
        self.accepted_decision_order.clear();
        for record in self.records.values_mut() {
            reset_record_authority(record, hardware_regime);
        }
        for model in self.models.values_mut() {
            reset_model_authority(model, hardware_regime);
        }
    }

    fn refresh_record_metrics(&mut self) {
        self.metrics.record_count = self.records.len();
        self.metrics.exact_record_count = self.exact_record_count();
        self.metrics.family_record_count = self.family_record_count();
    }

    fn reset_records_for_model(&mut self, model_key: &ModelKey, hardware: HardwareRegime) {
        for record in self.records.values_mut().filter(|record| {
            record.key.producer == model_key.producer && record.key.action == model_key.action
        }) {
            reset_record_authority(record, hardware);
        }
    }

    fn remove_orphan_models(&mut self) {
        let retained: BTreeSet<ModelKey> = self
            .records
            .keys()
            .map(|key| ModelKey {
                producer: key.producer,
                action: key.action.clone(),
            })
            .collect();
        self.models.retain(|key, _| retained.contains(key));
    }
}

#[derive(Debug, Default)]
struct RestoredModelEvidence {
    record_keys: Vec<CalibrationKey>,
    contexts: BTreeSet<ContextFingerprint>,
    lifetime_forecast_count: u64,
    authority_gold_count: u64,
    welford_count: u64,
    welford_mean: f64,
    welford_m2: f64,
    authority_epoch: u64,
    last_update_sequence: u64,
}

fn restored_model_evidence(
    records: &BTreeMap<CalibrationKey, CalibrationRecord>,
) -> BTreeMap<ModelKey, RestoredModelEvidence> {
    let mut evidence = BTreeMap::new();
    for record in records.values() {
        let model_key = ModelKey {
            producer: record.key.producer,
            action: record.key.action.clone(),
        };
        let aggregate = evidence
            .entry(model_key)
            .or_insert_with(RestoredModelEvidence::default);
        aggregate.record_keys.push(record.key.clone());
        aggregate.contexts.insert(context_fingerprint(&record.key));
        aggregate.lifetime_forecast_count = aggregate
            .lifetime_forecast_count
            .saturating_add(record.lifetime_forecast_count);
        aggregate.authority_gold_count = aggregate
            .authority_gold_count
            .saturating_add(record.authority_gold_count);
        combine_welford(
            aggregate,
            record.welford_count,
            record.welford_mean,
            record.welford_m2,
        );
        aggregate.authority_epoch = aggregate.authority_epoch.max(record.authority_epoch);
        aggregate.last_update_sequence = aggregate
            .last_update_sequence
            .max(record.last_update_sequence);
    }
    evidence
}

fn combine_welford(aggregate: &mut RestoredModelEvidence, count: u64, mean: f64, m2: f64) {
    if count == 0 {
        return;
    }
    if aggregate.welford_count == 0 {
        aggregate.welford_count = count;
        aggregate.welford_mean = mean;
        aggregate.welford_m2 = m2;
        return;
    }
    let prior_count = aggregate.welford_count;
    let combined_count = prior_count.saturating_add(count);
    let delta = mean - aggregate.welford_mean;
    aggregate.welford_mean += delta * count as f64 / combined_count as f64;
    aggregate.welford_m2 +=
        m2 + delta * delta * prior_count as f64 * count as f64 / combined_count as f64;
    aggregate.welford_count = combined_count;
}

fn restored_model_matches(model: &ModelTrustRecord, evidence: &RestoredModelEvidence) -> bool {
    let expected_contexts = evidence.contexts.len().min(16);
    let contexts_match = if model.authority_gold_count == 0 {
        model.contexts.is_empty()
    } else {
        model.contexts.len() == expected_contexts
            && model
                .contexts
                .iter()
                .all(|context| evidence.contexts.contains(context))
    };
    let expected_windows = (model.authority_gold_count / 10).min(3) as usize;
    let window_cardinality_matches = model.current_window_count as u64
        == model.authority_gold_count % 10
        && model.completed_windows.len() == expected_windows;
    let retained_window_mae_matches = if model.authority_gold_count <= 30 {
        let completed_sum: f64 = model
            .completed_windows
            .iter()
            .map(|window| window.normalized_mae * 10.0)
            .sum();
        approximately_equal(
            completed_sum + model.current_window_mae_sum,
            model.welford_mean * model.welford_count as f64,
        )
    } else {
        true
    };
    model.lifetime_forecast_count == evidence.lifetime_forecast_count
        && model.authority_gold_count == evidence.authority_gold_count
        && model.welford_count == evidence.welford_count
        && approximately_equal(model.welford_mean, evidence.welford_mean)
        && approximately_equal(model.welford_m2, evidence.welford_m2)
        && model.last_update_sequence == evidence.last_update_sequence
        && contexts_match
        && window_cardinality_matches
        && retained_window_mae_matches
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-9 * (1.0 + left.abs().max(right.abs()))
}

fn cold_reset_restored_model(
    model: &mut ModelTrustRecord,
    evidence: &RestoredModelEvidence,
    records: &mut BTreeMap<CalibrationKey, CalibrationRecord>,
    hardware: HardwareRegime,
) {
    model.lifetime_forecast_count = evidence.lifetime_forecast_count;
    model.authority_epoch = evidence.authority_epoch;
    model.last_update_sequence = evidence.last_update_sequence;
    reset_model_authority(model, hardware);
    for key in &evidence.record_keys {
        if let Some(record) = records.get_mut(key) {
            reset_record_authority(record, hardware);
        }
    }
}

fn bound_snapshot_state(
    mut state: ModelCalibrationPersisted,
) -> (ModelCalibrationPersisted, usize) {
    let mut serializations = 1;
    let mut encoded_len = serde_json::to_vec(&state).map_or(usize::MAX, |encoded| encoded.len());
    if encoded_len <= MAX_CALIBRATION_STATE_BYTES {
        return (state, serializations);
    }

    for _ in 0..2 {
        if state.records.is_empty() {
            break;
        }
        let proportional = state
            .records
            .len()
            .saturating_mul(MAX_CALIBRATION_STATE_BYTES)
            .checked_div(encoded_len.max(1))
            .unwrap_or(0);
        let keep = proportional
            .saturating_mul(9)
            .checked_div(10)
            .unwrap_or(0)
            .max(1)
            .min(state.records.len().saturating_sub(1));
        retain_snapshot_records(&mut state, keep);
        serializations += 1;
        encoded_len = serde_json::to_vec(&state).map_or(usize::MAX, |encoded| encoded.len());
        if encoded_len <= MAX_CALIBRATION_STATE_BYTES {
            return (state, serializations);
        }
    }

    // Fixed final fallback: a bounded metadata-only state is always far below
    // the one-MiB calibration budget and requires no fourth serialization.
    state.records.clear();
    state.models.clear();
    (state, serializations)
}

fn retain_snapshot_records(state: &mut ModelCalibrationPersisted, keep: usize) {
    let mut ranked = BTreeMap::new();
    for record in state.records.drain(..) {
        let priority = u8::from(record.family_fallback);
        ranked.insert(
            (
                priority,
                Reverse(record.last_update_sequence),
                record.key.clone(),
            ),
            record,
        );
    }
    state.records = ranked.into_values().take(keep).collect();
    let retained_models: BTreeSet<ModelKey> = state
        .records
        .iter()
        .map(|record| ModelKey {
            producer: record.key.producer,
            action: record.key.action.clone(),
        })
        .collect();
    state
        .models
        .retain(|model| retained_models.contains(&model.key));
}

fn update_record(
    record: &mut CalibrationRecord,
    observation: &CalibrationObservation<'_>,
    prediction: &PredictionRecord,
    sequence: u64,
) {
    let signed_error = (observation.actual_utility - prediction.expected_utility).clamp(-2.0, 2.0);
    let normalized_error = (signed_error.abs() / 2.0).clamp(0.0, 1.0);
    let covered = f64::from(
        observation.actual_utility >= prediction.expected_utility - prediction.uncertainty
            && observation.actual_utility <= prediction.expected_utility + prediction.uncertainty,
    );
    let alpha = if record.authority_gold_count == 0 {
        1.0
    } else {
        EMA_ALPHA
    };
    record.lifetime_forecast_count = record.lifetime_forecast_count.saturating_add(1);
    record.authority_gold_count = record.authority_gold_count.saturating_add(1);
    record.signed_error_ema = ema(record.signed_error_ema, signed_error, alpha);
    record.normalized_mae_ema = ema(record.normalized_mae_ema, normalized_error, alpha);
    record.coverage_ema = ema(record.coverage_ema, covered, alpha);
    record.quality_ema = ema(record.quality_ema, observation.quality, alpha);
    record.welford_count = record.welford_count.saturating_add(1);
    let delta = normalized_error - record.welford_mean;
    record.welford_mean += delta / record.welford_count as f64;
    let delta_after = normalized_error - record.welford_mean;
    record.welford_m2 += delta * delta_after;
    if prediction.binary_target == Some(BinaryPredictionTarget::Effective) {
        if let Some(probability) = prediction
            .positive_probability
            .filter(|probability| probability.is_finite() && (0.0..=1.0).contains(probability))
        {
            let target = f64::from(observation.effective);
            let brier = (probability - target).powi(2);
            record.brier_ema = Some(ema(
                record.brier_ema.unwrap_or(0.0),
                brier,
                if record.brier_count == 0 {
                    1.0
                } else {
                    EMA_ALPHA
                },
            ));
            record.brier_count = record.brier_count.saturating_add(1);
        }
    }
    record.last_update_sequence = sequence;
}

fn update_model(
    model: &mut ModelTrustRecord,
    observation: &CalibrationObservation<'_>,
    prediction: &PredictionRecord,
    context: &ContextFingerprint,
    sequence: u64,
) -> bool {
    let signed_error = (observation.actual_utility - prediction.expected_utility).clamp(-2.0, 2.0);
    let normalized_error = (signed_error.abs() / 2.0).clamp(0.0, 1.0);
    let covered = f64::from(
        observation.actual_utility >= prediction.expected_utility - prediction.uncertainty
            && observation.actual_utility <= prediction.expected_utility + prediction.uncertainty,
    );
    let alpha = if model.authority_gold_count == 0 {
        1.0
    } else {
        EMA_ALPHA
    };
    model.lifetime_forecast_count = model.lifetime_forecast_count.saturating_add(1);
    model.authority_gold_count = model.authority_gold_count.saturating_add(1);
    model.signed_error_ema = ema(model.signed_error_ema, signed_error, alpha);
    model.normalized_mae = ema(model.normalized_mae, normalized_error, alpha);
    model.coverage_ema = ema(model.coverage_ema, covered, alpha);
    model.quality_ema = ema(model.quality_ema, observation.quality, alpha);
    model.welford_count = model.welford_count.saturating_add(1);
    let delta = normalized_error - model.welford_mean;
    model.welford_mean += delta / model.welford_count as f64;
    let delta_after = normalized_error - model.welford_mean;
    model.welford_m2 += delta * delta_after;
    if let Err(index) = model.contexts.binary_search(context) {
        if model.contexts.len() < 16 {
            model.contexts.insert(index, context.clone());
        }
    }
    model.current_window_count = model.current_window_count.saturating_add(1);
    model.current_window_quality_sum += observation.quality;
    model.current_window_mae_sum += normalized_error;
    model.current_window_coverage_sum += covered;
    model.last_update_sequence = sequence;
    model.hardware_regime = observation.hardware_regime;
    model.installation_id = observation.installation_id;

    if model.current_window_count == 10 {
        let window = CalibrationWindow {
            quality: model.current_window_quality_sum / 10.0,
            normalized_mae: model.current_window_mae_sum / 10.0,
            coverage: model.current_window_coverage_sum / 10.0,
        };
        if model.completed_windows.len() >= 3 {
            model.completed_windows.remove(0);
        }
        model.completed_windows.push(window.clone());
        model.current_window_count = 0;
        model.current_window_quality_sum = 0.0;
        model.current_window_mae_sum = 0.0;
        model.current_window_coverage_sum = 0.0;
        if window.is_stable() {
            model.consecutive_bad_windows = 0;
        } else if window.is_bad() {
            model.consecutive_bad_windows = model.consecutive_bad_windows.saturating_add(1);
        }
        if model.consecutive_bad_windows >= 2 {
            reset_model_authority(model, observation.hardware_regime);
            return true;
        }
    }
    recompute_trust(model);
    false
}

fn recompute_trust(model: &mut ModelTrustRecord) {
    let candidate = model.authority_gold_count >= 10
        && model.quality_ema.is_finite()
        && model.quality_ema + FLOAT_BOUNDARY_EPSILON >= 0.85;
    if !candidate {
        model.trust = if model.recovering_from_degraded {
            TrustState::Degraded
        } else {
            TrustState::Immature
        };
        return;
    }
    model.trust = TrustState::Candidate;
    model.recovering_from_degraded = false;
    let validated = model.authority_gold_count >= 20
        && model.contexts.len() >= 3
        && model.normalized_mae.is_finite()
        && model.normalized_mae <= 0.15 + FLOAT_BOUNDARY_EPSILON;
    if !validated {
        return;
    }
    model.trust = TrustState::Validated;
    let trusted = model.authority_gold_count >= 50
        && model.contexts.len() >= 5
        && model.normalized_mae <= 0.10 + FLOAT_BOUNDARY_EPSILON
        && model
            .confidence_upper_95()
            .is_some_and(|upper| upper <= 0.10 + FLOAT_BOUNDARY_EPSILON)
        && model.completed_windows.len() == 3
        && model
            .completed_windows
            .iter()
            .all(CalibrationWindow::is_stable);
    if trusted {
        model.trust = TrustState::Trusted;
    }
}

fn reset_model_authority(model: &mut ModelTrustRecord, hardware: HardwareRegime) {
    model.authority_epoch = model.authority_epoch.saturating_add(1);
    model.authority_gold_count = 0;
    model.signed_error_ema = 0.0;
    model.normalized_mae = 0.0;
    model.coverage_ema = 0.0;
    model.quality_ema = 0.0;
    model.welford_count = 0;
    model.welford_mean = 0.0;
    model.welford_m2 = 0.0;
    model.contexts.clear();
    model.completed_windows.clear();
    model.consecutive_bad_windows = 0;
    model.current_window_count = 0;
    model.current_window_quality_sum = 0.0;
    model.current_window_mae_sum = 0.0;
    model.current_window_coverage_sum = 0.0;
    model.trust = TrustState::Degraded;
    model.recovering_from_degraded = true;
    model.hardware_regime = hardware;
}

fn reset_record_authority(record: &mut CalibrationRecord, hardware: HardwareRegime) {
    record.authority_epoch = record.authority_epoch.saturating_add(1);
    record.authority_gold_count = 0;
    record.signed_error_ema = 0.0;
    record.normalized_mae_ema = 0.0;
    record.coverage_ema = 0.0;
    record.brier_ema = None;
    record.brier_count = 0;
    record.quality_ema = 0.0;
    record.welford_count = 0;
    record.welford_mean = 0.0;
    record.welford_m2 = 0.0;
    record.trust = TrustState::Degraded;
    record.hardware_regime = hardware;
}

fn context_fingerprint(key: &CalibrationKey) -> ContextFingerprint {
    ContextFingerprint {
        workload: key.workload.clone(),
        process_class: key.process_class,
        horizon: key.horizon,
        pressure: key.pressure,
        thermal: key.thermal,
        foreground: key.foreground,
    }
}

fn valid_restored_record(record: &CalibrationRecord) -> bool {
    let valid_action = valid_action_scope(&record.key.action)
        && matches!(&record.key.action, CalibrationActionScope::Family(_))
            == record.family_fallback;
    record.key.producer != ProducerId::Other
        && valid_action
        && record.key.workload.chars().count() <= MAX_WORKLOAD_CHARS
        && canonical_workload(&record.key.workload) == record.key.workload
        && record.installation_id.is_known()
        && record.hardware_regime.is_known()
        && record.authority_gold_count <= record.lifetime_forecast_count
        && record.welford_count == record.authority_gold_count
        && record.brier_count <= record.authority_gold_count
        && (record.brier_count == 0) == record.brier_ema.is_none()
        && [
            record.signed_error_ema,
            record.normalized_mae_ema,
            record.coverage_ema,
            record.quality_ema,
            record.welford_mean,
            record.welford_m2,
        ]
        .into_iter()
        .all(f64::is_finite)
        && record.brier_ema.is_none_or(f64::is_finite)
        && (-2.0..=2.0).contains(&record.signed_error_ema)
        && (0.0..=1.0).contains(&record.normalized_mae_ema)
        && (0.0..=1.0).contains(&record.coverage_ema)
        && (0.0..=1.0).contains(&record.quality_ema)
        && (0.0..=1.0).contains(&record.welford_mean)
        && record.welford_m2 >= 0.0
        && record.welford_m2 <= record.welford_count as f64
        && record
            .brier_ema
            .is_none_or(|brier| (0.0..=1.0).contains(&brier))
}

fn sanitize_model(model: &mut ModelTrustRecord) -> bool {
    if model.key.producer == ProducerId::Other
        || !valid_action_scope(&model.key.action)
        || model.authority_gold_count > model.lifetime_forecast_count
        || model.welford_count != model.authority_gold_count
        || model.current_window_count >= 10
        || model.completed_windows.len() > 3
        || model.contexts.len() > 16
        || [
            model.signed_error_ema,
            model.normalized_mae,
            model.coverage_ema,
            model.quality_ema,
            model.welford_mean,
            model.welford_m2,
            model.current_window_quality_sum,
            model.current_window_mae_sum,
            model.current_window_coverage_sum,
        ]
        .into_iter()
        .any(|value| !value.is_finite())
        || model.welford_m2 < 0.0
        || model.welford_m2 > model.welford_count as f64
        || !(-2.0..=2.0).contains(&model.signed_error_ema)
        || !(0.0..=1.0).contains(&model.normalized_mae)
        || !(0.0..=1.0).contains(&model.coverage_ema)
        || !(0.0..=1.0).contains(&model.quality_ema)
        || !(0.0..=1.0).contains(&model.welford_mean)
        || !(0.0..=f64::from(model.current_window_count))
            .contains(&model.current_window_quality_sum)
        || !(0.0..=f64::from(model.current_window_count)).contains(&model.current_window_mae_sum)
        || !(0.0..=f64::from(model.current_window_count))
            .contains(&model.current_window_coverage_sum)
        || model.contexts.iter().any(|context| {
            context.workload.chars().count() > MAX_WORKLOAD_CHARS
                || canonical_workload(&context.workload) != context.workload
        })
        || model.completed_windows.iter().any(|window| {
            ![window.quality, window.normalized_mae, window.coverage]
                .into_iter()
                .all(f64::is_finite)
                || !(0.0..=1.0).contains(&window.quality)
                || !(0.0..=1.0).contains(&window.normalized_mae)
                || !(0.0..=1.0).contains(&window.coverage)
        })
    {
        return false;
    }
    model.contexts.sort();
    model.contexts.dedup();
    model.trust = TrustState::Immature;
    true
}

fn valid_action_scope(action: &CalibrationActionScope) -> bool {
    match action {
        CalibrationActionScope::Family(_) => true,
        CalibrationActionScope::Exact(action) => {
            let Some((prefix, _)) = action.split_once(':') else {
                return false;
            };
            let Some(family) = actuator_family_from_str(prefix) else {
                return false;
            };
            canonical_action_class(family, action).as_deref() == Some(action.as_str())
        }
    }
}

fn actuator_family_from_str(value: &str) -> Option<ActuatorFamily> {
    ActuatorFamily::ALL
        .into_iter()
        .find(|family| family.as_str() == value)
}

fn ema(previous: f64, observation: f64, alpha: f64) -> f64 {
    alpha * observation + (1.0 - alpha) * previous
}

pub(crate) fn canonical_action_class(family: ActuatorFamily, action_key: &str) -> Option<String> {
    if action_key.chars().count() > 320 {
        return None;
    }
    let (prefix, suffix) = action_key.split_once(':')?;
    if prefix != family.as_str() || suffix.is_empty() {
        return None;
    }
    if family == ActuatorFamily::Coordinated {
        let mut previous = None;
        for member in suffix.split('+') {
            let member_family = actuator_family_from_str(member)?;
            if member_family == ActuatorFamily::Coordinated
                || previous.is_some_and(|prior: &str| prior >= member)
            {
                return None;
            }
            previous = Some(member);
        }
    }
    let class = match family {
        ActuatorFamily::Sysctl => suffix.split('=').next().unwrap_or_default(),
        ActuatorFamily::Spotlight => suffix,
        ActuatorFamily::ThreadQos => suffix.rsplit(':').next().unwrap_or_default(),
        ActuatorFamily::Memorystatus => suffix.rsplit(':').next().unwrap_or_default(),
        ActuatorFamily::Quarantine => suffix.rsplit(':').next().unwrap_or_default(),
        ActuatorFamily::Coordinated => suffix,
        ActuatorFamily::PredictiveThreshold
        | ActuatorFamily::PredictiveProfile
        | ActuatorFamily::PredictivePreThrottle
        | ActuatorFamily::PredictivePurge
        | ActuatorFamily::InteractionQos
        | ActuatorFamily::IoShaping
        | ActuatorFamily::ChromiumEcore
        | ActuatorFamily::ChromiumPurge
        | ActuatorFamily::ChromiumJetsam
        | ActuatorFamily::MarkovPrewarm => suffix.split('@').next().unwrap_or_default(),
        _ => "action",
    };
    if class.is_empty()
        || class.chars().count() > MAX_ACTION_CLASS_CHARS
        || !class
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_-.+".contains(&byte))
    {
        return None;
    }
    Some(format!("{}:{class}", family.as_str()))
}

fn canonical_workload(workload: &str) -> String {
    let workload = workload.trim().to_ascii_lowercase();
    if workload.is_empty()
        || workload.chars().count() > MAX_WORKLOAD_CHARS
        || !workload
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte))
    {
        "other".to_string()
    } else {
        workload
    }
}

fn bounded_text(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::decision_ledger::{DecisionId, PredictionRecord};
    use crate::engine::installation_identity::InstallationId;
    use crate::engine::telemetry_medallion::{ActuatorFamily, EvidenceTier, HardwareRegime};

    const LOCAL_ID: InstallationId = InstallationId(7);
    const HARDWARE: HardwareRegime = HardwareRegime {
        p_core_count: 4,
        e_core_count: 4,
        ram_gib: 8,
    };

    fn prediction(source: &str, expected: f64, uncertainty: f64) -> PredictionRecord {
        PredictionRecord {
            source: source.to_string(),
            expected_utility: expected,
            uncertainty,
            horizon_cycles: 10,
            positive_probability: None,
            binary_target: None,
        }
    }

    fn observe_sample(
        store: &mut ModelCalibrationStore,
        id: u64,
        expected: f64,
        actual: f64,
        uncertainty: f64,
        quality: f64,
        context: usize,
    ) -> CalibrationUpdate {
        let provenance = CalibrationProvenance {
            local_authority_eligible: true,
            predictions: vec![prediction("world-model", expected, uncertainty)],
            ..CalibrationProvenance::default()
        };
        store.observe_local_gold(&CalibrationObservation {
            decision_id: Some(DecisionId(id)),
            tier: EvidenceTier::Gold,
            installation_id: LOCAL_ID,
            hardware_regime: HARDWARE,
            family: ActuatorFamily::Boost,
            action_key: "boost:Editor:pid:42".to_string(),
            workload: format!("context-{}", context % 5),
            process_class: match context % 5 {
                0 => ProcessClass::Foreground,
                1 => ProcessClass::Background,
                2 => ProcessClass::Browser,
                3 => ProcessClass::Compiler,
                _ => ProcessClass::Media,
            },
            pressure: PressureBand::Moderate,
            thermal: ThermalBand::Nominal,
            foreground: ForegroundContext::Active,
            context_valid: true,
            quality,
            actual_utility: actual,
            effective: actual > 0.0,
            provenance: &provenance,
        })
    }

    fn first_key(store: &ModelCalibrationStore) -> CalibrationKey {
        store
            .records()
            .next()
            .expect("calibration record")
            .key
            .clone()
    }

    fn cohort_observation<'a>(
        id: u64,
        family: ActuatorFamily,
        action_key: &str,
        provenance: &'a CalibrationProvenance,
    ) -> CalibrationObservation<'a> {
        CalibrationObservation {
            decision_id: Some(DecisionId(id)),
            tier: EvidenceTier::Gold,
            installation_id: LOCAL_ID,
            hardware_regime: HARDWARE,
            family,
            action_key: action_key.to_string(),
            workload: "idle".to_string(),
            process_class: ProcessClass::System,
            pressure: PressureBand::High,
            thermal: ThermalBand::Nominal,
            foreground: ForegroundContext::Idle,
            context_valid: true,
            quality: 0.95,
            actual_utility: 0.1,
            effective: true,
            provenance,
        }
    }

    #[test]
    fn local_gold_updates_incremental_forecast_statistics_once() {
        let provenance = CalibrationProvenance {
            local_authority_eligible: true,
            predictions: vec![PredictionRecord {
                source: "world-model".to_string(),
                expected_utility: 0.20,
                uncertainty: 0.20,
                horizon_cycles: 10,
                positive_probability: Some(0.75),
                binary_target: Some(BinaryPredictionTarget::Effective),
            }],
            ..CalibrationProvenance::default()
        };
        let observation = CalibrationObservation {
            decision_id: Some(DecisionId(11)),
            tier: EvidenceTier::Gold,
            installation_id: LOCAL_ID,
            hardware_regime: HARDWARE,
            family: ActuatorFamily::Boost,
            action_key: "boost:Editor:pid:42".to_string(),
            workload: "interactive".to_string(),
            process_class: ProcessClass::Foreground,
            pressure: PressureBand::Moderate,
            thermal: ThermalBand::Nominal,
            foreground: ForegroundContext::Active,
            context_valid: true,
            quality: 0.95,
            actual_utility: 0.10,
            effective: true,
            provenance: &provenance,
        };
        let mut store = ModelCalibrationStore::new(LOCAL_ID);

        assert!(matches!(
            store.observe_local_gold(&observation),
            CalibrationUpdate::Accepted { forecasts: 1, .. }
        ));

        let record = store.records().next().expect("one calibration record");
        assert_eq!(store.metrics().record_count, 1);
        assert_eq!(record.authority_gold_count, 1);
        assert!((record.signed_error_ema - -0.10).abs() < 1e-12);
        assert!((record.normalized_mae_ema - 0.05).abs() < 1e-12);
        assert_eq!(record.coverage_ema, 1.0);
        assert_eq!(record.welford_count, 1);
        assert!((record.brier_ema.expect("binary sample") - 0.0625).abs() < 1e-12);
        assert_eq!(record.brier_count, 1);
    }

    #[test]
    fn absent_or_invalid_binary_probability_does_not_synthesize_brier() {
        for probability in [None, Some(-0.1), Some(1.1), Some(f64::NAN)] {
            let provenance = CalibrationProvenance {
                local_authority_eligible: true,
                predictions: vec![PredictionRecord {
                    positive_probability: probability,
                    binary_target: Some(BinaryPredictionTarget::Effective),
                    ..prediction("world-model", 0.0, 0.2)
                }],
                ..CalibrationProvenance::default()
            };
            let mut store = ModelCalibrationStore::new(LOCAL_ID);
            let observation =
                cohort_observation(1, ActuatorFamily::Boost, "boost:Editor", &provenance);
            assert!(matches!(
                store.observe_local_gold(&observation),
                CalibrationUpdate::Accepted { forecasts: 1, .. }
            ));
            let record = store.records().next().unwrap();
            assert_eq!(record.brier_ema, None);
            assert_eq!(record.brier_count, 0);
        }
    }

    #[test]
    fn promotion_boundaries_and_exact_error_equality_are_inclusive() {
        let mut store = ModelCalibrationStore::new(LOCAL_ID);
        for id in 1..=50 {
            assert!(matches!(
                observe_sample(&mut store, id, 0.0, 0.0, 0.2, 0.85, id as usize),
                CalibrationUpdate::Accepted { forecasts: 1, .. }
            ));
            let key = first_key(&store);
            let expected = match id {
                1..=9 => TrustState::Immature,
                10..=19 => TrustState::Candidate,
                20..=49 => TrustState::Validated,
                _ => TrustState::Trusted,
            };
            assert_eq!(store.trust_for(&key), expected, "threshold at sample {id}");
        }
        assert_eq!(store.summary().trusted_models, 1);
        assert!(store.summary().worst_record.is_some());

        let mut equality = ModelCalibrationStore::new(LOCAL_ID);
        for id in 1..=20 {
            observe_sample(&mut equality, id, -0.15, 0.15, 0.4, 0.85, id as usize);
        }
        assert_eq!(
            equality.trust_for(&first_key(&equality)),
            TrustState::Validated
        );
        assert!(
            (equality
                .model_for(&first_key(&equality))
                .unwrap()
                .normalized_mae
                - 0.15)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn decisive_interval_requires_its_upper_endpoint_at_or_below_point_one() {
        let mut variable = ModelCalibrationStore::new(LOCAL_ID);
        for id in 1..=50 {
            let normalized_error = if id % 2 == 0 { 0.09 } else { 0.11 };
            observe_sample(
                &mut variable,
                id,
                0.0,
                normalized_error * 2.0,
                0.3,
                0.95,
                id as usize,
            );
        }
        let variable_model = variable.model_for(&first_key(&variable)).unwrap();
        assert!(variable_model.confidence_upper_95().unwrap() > 0.10);
        assert_ne!(variable_model.trust, TrustState::Trusted);

        let mut exact = ModelCalibrationStore::new(LOCAL_ID);
        for id in 1..=50 {
            observe_sample(&mut exact, id, 0.0, 0.20, 0.3, 0.95, id as usize);
        }
        let exact_model = exact.model_for(&first_key(&exact)).unwrap();
        assert!((exact_model.confidence_upper_95().unwrap() - 0.10).abs() < 1e-12);
        assert_eq!(exact_model.trust, TrustState::Trusted);
    }

    #[test]
    fn stable_neutral_and_bad_windows_apply_exact_hysteresis() {
        let mut store = ModelCalibrationStore::new(LOCAL_ID);
        for id in 1..=10 {
            observe_sample(&mut store, id, 0.0, 0.0, 0.2, 0.95, id as usize);
        }
        let key = first_key(&store);
        assert_eq!(store.model_for(&key).unwrap().completed_windows.len(), 1);

        for id in 11..=20 {
            observe_sample(&mut store, id, 0.0, 0.20, 0.01, 0.95, id as usize);
        }
        assert_eq!(store.model_for(&key).unwrap().consecutive_bad_windows, 1);
        assert_ne!(store.trust_for(&key), TrustState::Degraded);

        for id in 21..=30 {
            let uncertainty = if id <= 28 { 0.3 } else { 0.01 };
            observe_sample(&mut store, id, 0.0, 0.20, uncertainty, 0.95, id as usize);
        }
        assert_eq!(store.model_for(&key).unwrap().consecutive_bad_windows, 1);

        for id in 31..=40 {
            observe_sample(&mut store, id, 0.0, 0.0, 0.2, 0.95, id as usize);
        }
        assert_eq!(store.model_for(&key).unwrap().consecutive_bad_windows, 0);

        for id in 41..=60 {
            observe_sample(&mut store, id, 0.0, 0.20, 0.01, 0.95, id as usize);
        }
        let degraded = store.model_for(&key).unwrap();
        assert_eq!(degraded.trust, TrustState::Degraded);
        assert_eq!(degraded.authority_gold_count, 0);

        for id in 61..=69 {
            observe_sample(&mut store, id, 0.0, 0.0, 0.2, 0.95, id as usize);
        }
        assert_eq!(store.trust_for(&key), TrustState::Degraded);
        observe_sample(&mut store, 70, 0.0, 0.0, 0.2, 0.95, 70);
        assert_eq!(store.trust_for(&key), TrustState::Candidate);
    }

    #[test]
    fn three_complete_stable_windows_are_required_not_two_plus_a_partial() {
        let mut partial = ModelCalibrationStore::new(LOCAL_ID);
        for id in 1..=29 {
            observe_sample(&mut partial, id, 0.0, 0.0, 0.2, 0.95, id as usize);
        }
        let partial_key = first_key(&partial);
        assert_eq!(
            partial
                .model_for(&partial_key)
                .unwrap()
                .completed_windows
                .len(),
            2
        );

        let mut store = ModelCalibrationStore::new(LOCAL_ID);
        for id in 1..=49 {
            observe_sample(&mut store, id, 0.0, 0.0, 0.2, 0.95, id as usize);
        }
        let key = first_key(&store);
        assert_eq!(store.trust_for(&key), TrustState::Validated);
        assert_eq!(store.model_for(&key).unwrap().completed_windows.len(), 3);

        observe_sample(&mut store, 50, 0.0, 0.0, 0.2, 0.95, 50);
        assert_eq!(store.trust_for(&key), TrustState::Trusted);
    }

    #[test]
    fn admission_gates_and_support_only_advisers_are_inert() {
        let base_prediction = prediction("world-model", 0.0, 0.2);
        let cases = [
            (
                None,
                EvidenceTier::Gold,
                LOCAL_ID,
                true,
                SeparabilityState::Individual,
            ),
            (
                Some(DecisionId(1)),
                EvidenceTier::Silver,
                LOCAL_ID,
                true,
                SeparabilityState::Individual,
            ),
            (
                Some(DecisionId(2)),
                EvidenceTier::Gold,
                InstallationId(99),
                true,
                SeparabilityState::Individual,
            ),
            (
                Some(DecisionId(3)),
                EvidenceTier::Gold,
                LOCAL_ID,
                false,
                SeparabilityState::Individual,
            ),
            (
                Some(DecisionId(4)),
                EvidenceTier::Gold,
                LOCAL_ID,
                true,
                SeparabilityState::Confounded,
            ),
        ];
        for (decision_id, tier, installation_id, eligible, separability) in cases {
            let provenance = CalibrationProvenance {
                local_authority_eligible: eligible,
                predictions: vec![base_prediction.clone()],
                separability,
                ..CalibrationProvenance::default()
            };
            let mut store = ModelCalibrationStore::new(LOCAL_ID);
            let update = store.observe_local_gold(&CalibrationObservation {
                decision_id,
                tier,
                installation_id,
                hardware_regime: HARDWARE,
                family: ActuatorFamily::Boost,
                action_key: "boost:Editor".to_string(),
                workload: "idle".to_string(),
                process_class: ProcessClass::Foreground,
                pressure: PressureBand::Low,
                thermal: ThermalBand::Cool,
                foreground: ForegroundContext::Active,
                context_valid: true,
                quality: 0.95,
                actual_utility: 0.1,
                effective: true,
                provenance: &provenance,
            });
            assert!(matches!(update, CalibrationUpdate::Ignored(_)));
            assert_eq!(store.metrics().record_count, 0);
        }

        let support_only = CalibrationProvenance {
            local_authority_eligible: true,
            adviser_contributions: vec![AdviserContribution {
                adviser: "world-model".to_string(),
                support: 1.0,
                uncertainty: 0.1,
            }],
            ..CalibrationProvenance::default()
        };
        let mut store = ModelCalibrationStore::new(LOCAL_ID);
        let update = store.observe_local_gold(&CalibrationObservation {
            decision_id: Some(DecisionId(9)),
            tier: EvidenceTier::Gold,
            installation_id: LOCAL_ID,
            hardware_regime: HARDWARE,
            family: ActuatorFamily::Boost,
            action_key: "boost:Editor".to_string(),
            workload: "idle".to_string(),
            process_class: ProcessClass::Foreground,
            pressure: PressureBand::Low,
            thermal: ThermalBand::Cool,
            foreground: ForegroundContext::Active,
            context_valid: true,
            quality: 0.95,
            actual_utility: 0.1,
            effective: true,
            provenance: &support_only,
        });
        assert_eq!(
            update,
            CalibrationUpdate::Ignored(IgnoreReason::MissingForecast)
        );
        assert_eq!(store.metrics().record_count, 0);
    }

    #[test]
    fn eight_distinct_predictions_survive_and_duplicate_producer_metadata_is_once() {
        let sources = [
            "world-model",
            "gpu-model",
            "markov",
            "causal-graph",
            "mpc",
            "nars",
            "policy-scorer",
            "predictive-agent",
        ];
        let provenance = CalibrationProvenance {
            local_authority_eligible: true,
            proposer: "world-model".to_string(),
            predictions: sources
                .into_iter()
                .map(|source| prediction(source, 0.0, 0.2))
                .chain(std::iter::once(prediction("world-model", 1.0, 0.2)))
                .collect(),
            adviser_contributions: vec![AdviserContribution {
                adviser: "world-model".to_string(),
                support: 1.0,
                uncertainty: 0.1,
            }],
            ..CalibrationProvenance::default()
        };
        let mut store = ModelCalibrationStore::new(LOCAL_ID);
        let update = store.observe_local_gold(&CalibrationObservation {
            decision_id: Some(DecisionId(1)),
            tier: EvidenceTier::Gold,
            installation_id: LOCAL_ID,
            hardware_regime: HARDWARE,
            family: ActuatorFamily::Boost,
            action_key: "boost:Editor".to_string(),
            workload: "idle".to_string(),
            process_class: ProcessClass::Foreground,
            pressure: PressureBand::Low,
            thermal: ThermalBand::Cool,
            foreground: ForegroundContext::Active,
            context_valid: true,
            quality: 0.95,
            actual_utility: 0.0,
            effective: false,
            provenance: &provenance,
        });

        assert!(matches!(
            update,
            CalibrationUpdate::Accepted { forecasts: 8, .. }
        ));
        assert_eq!(store.metrics().record_count, 8);
    }

    #[test]
    fn cohort_calibration_is_composite_only_unless_member_is_explicitly_separable() {
        let mut store = ModelCalibrationStore::new(LOCAL_ID);
        let mut provenance = CalibrationProvenance {
            local_authority_eligible: true,
            predictions: vec![prediction("world-model", 0.0, 0.2)],
            cohort_size: 2,
            separability: SeparabilityState::Confounded,
            ..CalibrationProvenance::default()
        };
        assert_eq!(
            store.observe_local_gold(&cohort_observation(
                1,
                ActuatorFamily::Boost,
                "boost:Editor",
                &provenance
            )),
            CalibrationUpdate::Ignored(IgnoreReason::Confounded)
        );

        provenance.separability = SeparabilityState::Individual;
        assert_eq!(
            store.observe_local_gold(&cohort_observation(
                2,
                ActuatorFamily::Boost,
                "boost:Editor",
                &provenance
            )),
            CalibrationUpdate::Ignored(IgnoreReason::Confounded)
        );

        provenance.separability = SeparabilityState::CoordinatedComposite;
        assert!(matches!(
            store.observe_local_gold(&cohort_observation(
                3,
                ActuatorFamily::Coordinated,
                "coordinated:boost+thread_qos",
                &provenance,
            )),
            CalibrationUpdate::Accepted { forecasts: 1, .. }
        ));

        provenance.separability = SeparabilityState::SeparableMember {
            decision_id: DecisionId(4),
        };
        assert!(matches!(
            store.observe_local_gold(&cohort_observation(
                4,
                ActuatorFamily::Boost,
                "boost:Editor",
                &provenance
            )),
            CalibrationUpdate::Accepted { forecasts: 1, .. }
        ));
        assert_eq!(store.metrics().record_count, 2);
    }

    #[test]
    fn fixed_partitions_fallback_and_evict_deterministically_without_record_513() {
        fn fill() -> ModelCalibrationStore {
            let mut store = ModelCalibrationStore::new(LOCAL_ID);
            for id in 1..=(MAX_EXACT_CALIBRATION_KEYS + MAX_FAMILY_CALIBRATION_KEYS + 3) {
                let provenance = CalibrationProvenance {
                    local_authority_eligible: true,
                    predictions: vec![prediction("world-model", 0.0, 0.2)],
                    ..CalibrationProvenance::default()
                };
                store.observe_local_gold(&CalibrationObservation {
                    decision_id: Some(DecisionId(id as u64)),
                    tier: EvidenceTier::Gold,
                    installation_id: LOCAL_ID,
                    hardware_regime: HARDWARE,
                    family: ActuatorFamily::Boost,
                    action_key: "boost:Editor".to_string(),
                    workload: format!("work-{id}"),
                    process_class: ProcessClass::Foreground,
                    pressure: PressureBand::Moderate,
                    thermal: ThermalBand::Nominal,
                    foreground: ForegroundContext::Active,
                    context_valid: true,
                    quality: 0.95,
                    actual_utility: 0.0,
                    effective: false,
                    provenance: &provenance,
                });
                assert!(store.metrics().record_count <= MAX_CALIBRATION_KEYS);
            }
            store
        }

        let first = fill();
        let second = fill();
        assert_eq!(
            first.metrics().exact_record_count,
            MAX_EXACT_CALIBRATION_KEYS
        );
        assert_eq!(
            first.metrics().family_record_count,
            MAX_FAMILY_CALIBRATION_KEYS
        );
        assert_eq!(first.metrics().record_count, MAX_CALIBRATION_KEYS);
        assert!(first
            .records()
            .any(|record| { record.family_fallback && record.key.workload.starts_with("work-") }));
        let family_record = first
            .records()
            .find(|record| record.family_fallback)
            .unwrap();
        let mut exact_lookup = family_record.key.clone();
        exact_lookup.action = CalibrationActionScope::Exact("boost:action".to_string());
        assert!(first.record(&exact_lookup).is_none());
        assert_eq!(
            first
                .record_or_family(&exact_lookup)
                .map(|record| &record.key),
            Some(&family_record.key)
        );
        exact_lookup.workload = "different-context".to_string();
        assert!(first.record_or_family(&exact_lookup).is_none());
        assert_eq!(
            first
                .snapshot()
                .records
                .iter()
                .map(|record| &record.key)
                .collect::<Vec<_>>(),
            second
                .snapshot()
                .records
                .iter()
                .map(|record| &record.key)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn restart_dedup_retains_128_ids_but_accepts_unseen_late_completion_once() {
        let mut store = ModelCalibrationStore::new(LOCAL_ID);
        for id in 1..=129 {
            observe_sample(&mut store, id, 0.0, 0.0, 0.2, 0.95, id as usize);
        }
        let snapshot = store.snapshot();
        assert_eq!(
            snapshot.accepted_decision_ids.len(),
            MAX_ACCEPTED_DECISION_IDS
        );
        assert!(!snapshot.accepted_decision_ids.contains(&DecisionId(1)));
        assert_eq!(snapshot.accepted_decision_id_high_water, 129);

        let mut restored = ModelCalibrationStore::new(LOCAL_ID);
        restored.restore(snapshot, HARDWARE);
        assert_eq!(
            observe_sample(&mut restored, 129, 0.0, 0.0, 0.2, 0.95, 0),
            CalibrationUpdate::Ignored(IgnoreReason::Duplicate)
        );
        assert!(matches!(
            observe_sample(&mut restored, 1, 0.0, 0.0, 0.2, 0.95, 0),
            CalibrationUpdate::Accepted { forecasts: 1, .. }
        ));
        assert_eq!(
            observe_sample(&mut restored, 1, 0.0, 0.0, 0.2, 0.95, 0),
            CalibrationUpdate::Ignored(IgnoreReason::Duplicate)
        );
    }

    #[test]
    fn restore_recomputes_trust_quarantines_origin_and_caps_hostile_state() {
        let mut source = ModelCalibrationStore::new(LOCAL_ID);
        for id in 1..=10 {
            observe_sample(&mut source, id, 0.0, 0.0, 0.2, 0.95, id as usize);
        }
        let key = first_key(&source);
        let mut snapshot = source.snapshot();
        for record in &mut snapshot.records {
            record.trust = TrustState::Trusted;
        }

        let mut same = ModelCalibrationStore::new(LOCAL_ID);
        same.restore(snapshot.clone(), HARDWARE);
        assert_eq!(same.trust_for(&key), TrustState::Candidate);

        let changed_hardware = HardwareRegime {
            p_core_count: 4,
            e_core_count: 6,
            ram_gib: 16,
        };
        same.validate_hardware(changed_hardware);
        assert_eq!(same.model_for(&key).unwrap().trust, TrustState::Degraded);
        assert_eq!(same.model_for(&key).unwrap().authority_gold_count, 0);
        assert_eq!(same.accepted_decision_ids().count(), 0);
        let recovery_provenance = CalibrationProvenance {
            local_authority_eligible: true,
            predictions: vec![prediction("world-model", 0.0, 0.2)],
            ..CalibrationProvenance::default()
        };
        let mut recovery = cohort_observation(
            11,
            ActuatorFamily::Boost,
            "boost:Editor",
            &recovery_provenance,
        );
        recovery.hardware_regime = changed_hardware;
        assert!(matches!(
            same.observe_local_gold(&recovery),
            CalibrationUpdate::Accepted { forecasts: 1, .. }
        ));
        assert_eq!(same.model_for(&key).unwrap().trust, TrustState::Degraded);
        assert_eq!(same.model_for(&key).unwrap().authority_gold_count, 1);

        let mut changed = ModelCalibrationStore::new(LOCAL_ID);
        changed.restore(snapshot.clone(), changed_hardware);
        assert_eq!(changed.model_for(&key).unwrap().trust, TrustState::Degraded);
        assert_eq!(changed.model_for(&key).unwrap().authority_gold_count, 0);
        assert_eq!(changed.accepted_decision_ids().count(), 0);

        let mut foreign = ModelCalibrationStore::new(InstallationId(99));
        foreign.restore(snapshot.clone(), HARDWARE);
        assert_eq!(foreign.metrics().record_count, 0);

        let template = snapshot.records[0].clone();
        let mut duplicate_state = snapshot.clone();
        let mut newer_duplicate = template.clone();
        newer_duplicate.last_update_sequence = newer_duplicate.last_update_sequence + 1_000;
        newer_duplicate.signed_error_ema = 0.123;
        duplicate_state.records.push(newer_duplicate.clone());
        duplicate_state.models[0].last_update_sequence = newer_duplicate.last_update_sequence;
        let mut deduplicated = ModelCalibrationStore::new(LOCAL_ID);
        deduplicated.restore(duplicate_state, HARDWARE);
        assert_eq!(
            deduplicated
                .record(&newer_duplicate.key)
                .unwrap()
                .signed_error_ema,
            0.123
        );

        snapshot.records.clear();
        for id in 0..700_u64 {
            let mut record = template.clone();
            record.key.workload = format!("hostile-{id}");
            record.last_update_sequence = id;
            record.trust = TrustState::Trusted;
            snapshot.records.push(record);
        }
        let mut invalid = template;
        invalid.key.workload = "x".repeat(MAX_WORKLOAD_CHARS + 1);
        invalid.normalized_mae_ema = f64::NAN;
        snapshot.records.push(invalid);
        let mut restored_a = ModelCalibrationStore::new(LOCAL_ID);
        let mut restored_b = ModelCalibrationStore::new(LOCAL_ID);
        restored_a.restore(snapshot.clone(), HARDWARE);
        restored_b.restore(snapshot, HARDWARE);
        assert!(restored_a.metrics().record_count <= MAX_CALIBRATION_KEYS);
        assert_eq!(restored_a.snapshot().records, restored_b.snapshot().records);
        let bytes = serde_json::to_vec(&restored_a.snapshot()).unwrap();
        assert!(
            bytes.len() <= MAX_CALIBRATION_STATE_BYTES,
            "{} bytes",
            bytes.len()
        );
        assert!(serde_json::from_str::<ModelCalibrationPersisted>("{not-json").is_err());
        assert!(serde_json::from_str::<ModelCalibrationPersisted>(
            r#"{"records":[{"key":{"producer":"future-producer"}}]}"#,
        )
        .is_err());
    }

    #[test]
    fn restore_cold_resets_a_forged_model_disconnected_from_retained_records() {
        let mut source = ModelCalibrationStore::new(LOCAL_ID);
        observe_sample(&mut source, 1, 0.0, 0.0, 0.2, 0.95, 0);
        let key = first_key(&source);
        let mut state = source.snapshot();
        let model = state.models.first_mut().expect("model fixture");
        model.lifetime_forecast_count = 50;
        model.authority_gold_count = 50;
        model.welford_count = 50;
        model.welford_mean = 0.0;
        model.welford_m2 = 0.0;
        model.contexts = (0..5)
            .map(|index| ContextFingerprint {
                workload: format!("forged-{index}"),
                process_class: ProcessClass::Foreground,
                horizon: CalibrationHorizon::Sec5,
                pressure: PressureBand::Moderate,
                thermal: ThermalBand::Nominal,
                foreground: ForegroundContext::Active,
            })
            .collect();
        model.completed_windows = vec![
            CalibrationWindow {
                quality: 0.99,
                normalized_mae: 0.0,
                coverage: 1.0,
            };
            3
        ];
        model.trust = TrustState::Trusted;

        let mut restored = ModelCalibrationStore::new(LOCAL_ID);
        restored.restore(state, HARDWARE);

        let restored_model = restored.model_for(&key).expect("cold model retained");
        assert_ne!(restored_model.trust, TrustState::Trusted);
        assert_eq!(restored_model.authority_gold_count, 0);
        assert_eq!(restored_model.welford_count, 0);
        assert!(restored_model.contexts.is_empty());
        assert!(restored_model.completed_windows.is_empty());
        assert_eq!(restored.record(&key).unwrap().authority_gold_count, 0);
    }

    #[test]
    fn persisted_deserialization_caps_hostile_collections_before_restore() {
        let mut state = ModelCalibrationPersisted::default();
        for index in 0..700_u64 {
            let action = CalibrationActionScope::Exact(format!("boost:model-{index}"));
            state.records.push(CalibrationRecord {
                key: CalibrationKey {
                    producer: ProducerId::WorldModel,
                    action: action.clone(),
                    workload: format!("context-{index}"),
                    ..CalibrationKey::default()
                },
                last_update_sequence: index,
                ..CalibrationRecord::default()
            });
            state.models.push(ModelTrustRecord {
                key: ModelKey {
                    producer: ProducerId::WorldModel,
                    action,
                },
                contexts: (0..40)
                    .map(|context| ContextFingerprint {
                        workload: format!("context-{context}"),
                        ..ContextFingerprint::default()
                    })
                    .collect(),
                completed_windows: vec![CalibrationWindow::default(); 12],
                last_update_sequence: index,
                ..ModelTrustRecord::default()
            });
        }
        state.accepted_decision_ids = (1..=500).map(DecisionId).collect();

        let encoded = serde_json::to_vec(&state).unwrap();
        let bounded: ModelCalibrationPersisted = serde_json::from_slice(&encoded).unwrap();

        assert!(bounded.records.len() <= MAX_CALIBRATION_KEYS);
        assert!(bounded.models.len() <= MAX_CALIBRATION_KEYS);
        assert!(bounded.accepted_decision_ids.len() <= MAX_ACCEPTED_DECISION_IDS);
        assert!(bounded
            .models
            .iter()
            .all(|model| model.contexts.len() <= 16));
        assert!(bounded
            .models
            .iter()
            .all(|model| model.completed_windows.len() <= 3));
    }

    #[test]
    fn over_budget_snapshot_prunes_in_a_fixed_serialization_budget() {
        let mut state = ModelCalibrationPersisted::default();
        for index in 0..MAX_EXACT_CALIBRATION_KEYS {
            let action = CalibrationActionScope::Exact(format!("boost:model-{index:084}"));
            state.records.push(CalibrationRecord {
                key: CalibrationKey {
                    producer: ProducerId::WorldModel,
                    action: action.clone(),
                    workload: format!("workload-{index:055}"),
                    ..CalibrationKey::default()
                },
                last_update_sequence: index as u64,
                ..CalibrationRecord::default()
            });
            state.models.push(ModelTrustRecord {
                key: ModelKey {
                    producer: ProducerId::WorldModel,
                    action,
                },
                contexts: (0..16)
                    .map(|context| ContextFingerprint {
                        workload: format!("context-{index:039}-{context:02}"),
                        ..ContextFingerprint::default()
                    })
                    .collect(),
                completed_windows: vec![CalibrationWindow::default(); 3],
                last_update_sequence: index as u64,
                ..ModelTrustRecord::default()
            });
        }
        for index in 0..MAX_FAMILY_CALIBRATION_KEYS {
            state.records.push(CalibrationRecord {
                key: CalibrationKey {
                    producer: ProducerId::GpuModel,
                    action: CalibrationActionScope::Family(ActuatorFamily::Boost),
                    workload: format!("family-{index:056}"),
                    ..CalibrationKey::default()
                },
                family_fallback: true,
                last_update_sequence: (MAX_EXACT_CALIBRATION_KEYS + index) as u64,
                ..CalibrationRecord::default()
            });
        }
        assert!(serde_json::to_vec(&state).unwrap().len() > MAX_CALIBRATION_STATE_BYTES);

        let (bounded, serializations) = bound_snapshot_state(state);
        let encoded = serde_json::to_vec(&bounded).unwrap();
        let exact = bounded
            .records
            .iter()
            .filter(|record| !record.family_fallback)
            .count();
        let family = bounded.records.len() - exact;

        assert!(encoded.len() <= MAX_CALIBRATION_STATE_BYTES);
        assert!(serializations <= 3, "serializations={serializations}");
        assert!(family == 0 || exact == MAX_EXACT_CALIBRATION_KEYS);
    }

    #[test]
    fn single_admission_returns_exact_delta_and_trust_transition() {
        let mut store = ModelCalibrationStore::new(LOCAL_ID);
        for id in 1..10 {
            let _ = observe_sample(&mut store, id, 0.05, 0.08, 0.10, 0.95, id as usize);
        }

        let CalibrationUpdate::Accepted { forecasts, deltas } =
            observe_sample(&mut store, 10, 0.05, 0.08, 0.10, 0.95, 0)
        else {
            panic!("tenth local Gold forecast must be accepted");
        };

        assert_eq!(forecasts, 1);
        assert_eq!(deltas.len(), 1);
        let delta = &deltas[0];
        assert_eq!(delta.key.producer, ProducerId::WorldModel);
        assert_eq!(delta.key.horizon, CalibrationHorizon::Sec5);
        assert!((delta.predicted_utility - 0.05).abs() < f64::EPSILON);
        assert!((delta.actual_utility - 0.08).abs() < f64::EPSILON);
        assert!((delta.signed_error - 0.03).abs() < 1e-12);
        assert!((delta.normalized_absolute_error - 0.015).abs() < 1e-12);
        assert!(delta.uncertainty_covered);
        assert_eq!(delta.trust_before, TrustState::Immature);
        assert_eq!(delta.trust_after, TrustState::Candidate);
    }
}
