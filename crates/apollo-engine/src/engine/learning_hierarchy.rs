//! Bounded, installation-local Goal -> Strategy -> Tactic -> Action memory.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::marker::PhantomData;

use serde::de::{Error as DeError, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::engine::decision_ledger::{
    AdviserContribution, CandidateAlternative, DecisionId, DecisionLifecycle, PredictionRecord,
};
use crate::engine::installation_identity::InstallationId;
use crate::engine::model_calibration::{
    canonical_action_class, CalibrationActionScope, CalibrationHorizon, ForecastCalibrationDelta,
    ForegroundContext, PressureBand, ProducerId, SeparabilityState, ThermalBand, TrustState,
};
use crate::engine::telemetry_medallion::{ActuatorEpisodeContext, ActuatorFamily, HardwareRegime};

pub const MAX_PROTOTYPES: usize = 256;
pub const MAX_CONTEXTS_PER_FAMILY: usize = 8;
pub const MAX_REPRESENTATIVE_ACTIONS: usize = 4;
pub const RETRIEVAL_TOP_K: usize = 4;
pub const MAX_PROCESSED_DECISION_IDS: usize = 128;
pub const MAX_HIERARCHY_PROPOSITIONS: usize = 4;

const MAX_ACTION_KEY_CHARS: usize = 320;
const MAX_TARGET_CHARS: usize = 256;
const MAX_SOURCE_CHARS: usize = 48;
const MAX_PROTOTYPE_OBSERVATIONS: u32 = 65_535;
const MAX_EVIDENCE_MASS: f64 = 64.0;
const PROTOTYPE_EMA_ALPHA: f64 = 0.20;
const UTILITY_DEADBAND: f64 = 0.005;
const SEVEN_DAYS_SECS: i64 = 7 * 24 * 60 * 60;
const FOURTEEN_DAYS_SECS: i64 = 14 * 24 * 60 * 60;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum Goal {
    Stability,
    Responsiveness,
    MemoryHeadroom,
    ThermalSafety,
    EnergyEfficiency,
}

impl Goal {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stability => "stability",
            Self::Responsiveness => "responsiveness",
            Self::MemoryHeadroom => "memory-headroom",
            Self::ThermalSafety => "thermal-safety",
            Self::EnergyEfficiency => "energy-efficiency",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum Strategy {
    ProtectForeground,
    PredictNextUse,
    RelievePressure,
    ShiftBackgroundWork,
    RecoverState,
    ReduceEnergy,
}

impl Strategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProtectForeground => "protect-foreground",
            Self::PredictNextUse => "predict-next-use",
            Self::RelievePressure => "relieve-pressure",
            Self::ShiftBackgroundWork => "shift-background-work",
            Self::RecoverState => "recover-state",
            Self::ReduceEnergy => "reduce-energy",
        }
    }
}

pub const fn classify_family(family: ActuatorFamily) -> (Goal, Strategy) {
    match family {
        ActuatorFamily::Boost => (Goal::Responsiveness, Strategy::ProtectForeground),
        ActuatorFamily::Throttle => (Goal::MemoryHeadroom, Strategy::RelievePressure),
        ActuatorFamily::Freeze => (Goal::MemoryHeadroom, Strategy::RelievePressure),
        ActuatorFamily::Unfreeze => (Goal::Stability, Strategy::RecoverState),
        ActuatorFamily::Memorystatus => (Goal::MemoryHeadroom, Strategy::RelievePressure),
        ActuatorFamily::Sysctl => (Goal::Stability, Strategy::RecoverState),
        ActuatorFamily::Spotlight => (Goal::EnergyEfficiency, Strategy::ShiftBackgroundWork),
        ActuatorFamily::Quarantine => (Goal::EnergyEfficiency, Strategy::ShiftBackgroundWork),
        ActuatorFamily::ThreadQos => (Goal::Responsiveness, Strategy::ProtectForeground),
        ActuatorFamily::MarkovPrewarm => (Goal::Responsiveness, Strategy::PredictNextUse),
        ActuatorFamily::InteractionQos => (Goal::Responsiveness, Strategy::ProtectForeground),
        ActuatorFamily::IoShaping => (Goal::Responsiveness, Strategy::ProtectForeground),
        ActuatorFamily::PredictiveThreshold => (Goal::Stability, Strategy::PredictNextUse),
        ActuatorFamily::PredictiveProfile => (Goal::Responsiveness, Strategy::PredictNextUse),
        ActuatorFamily::PredictivePreThrottle => (Goal::MemoryHeadroom, Strategy::RelievePressure),
        ActuatorFamily::PredictivePurge => (Goal::MemoryHeadroom, Strategy::RelievePressure),
        ActuatorFamily::ChromiumEcore => (Goal::EnergyEfficiency, Strategy::ReduceEnergy),
        ActuatorFamily::ChromiumPurge => (Goal::MemoryHeadroom, Strategy::RelievePressure),
        ActuatorFamily::ChromiumJetsam => (Goal::MemoryHeadroom, Strategy::RelievePressure),
        ActuatorFamily::Coordinated => (Goal::Stability, Strategy::RecoverState),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(default)]
pub struct HierarchyPath {
    pub goal: Goal,
    pub strategy: Strategy,
    pub family: ActuatorFamily,
    #[serde(default, deserialize_with = "deserialize_hierarchy_action")]
    pub action: String,
}

impl Default for HierarchyPath {
    fn default() -> Self {
        Self {
            goal: Goal::Stability,
            strategy: Strategy::RecoverState,
            family: ActuatorFamily::Coordinated,
            action: String::new(),
        }
    }
}

impl HierarchyPath {
    pub fn classify(family: ActuatorFamily, action: &str) -> Option<Self> {
        let action = action.trim();
        if action.is_empty()
            || action.chars().count() > MAX_ACTION_KEY_CHARS
            || action.chars().any(char::is_control)
            || action.contains(":pid:")
        {
            return None;
        }
        let action = canonical_action_class(family, action)?;
        let (goal, strategy) = classify_family(family);
        Some(Self {
            goal,
            strategy,
            family,
            action,
        })
    }

    pub fn is_canonical(&self) -> bool {
        Self::classify(self.family, &self.action).is_some_and(|canonical| canonical == *self)
    }

    pub fn propositions(&self, context: HierarchyContext) -> [String; 4] {
        let goal = self.goal.as_str();
        let strategy = self.strategy.as_str();
        let family = self.family.as_str();
        [
            format!("goal:{goal}"),
            format!("strategy:{goal}:{strategy}"),
            format!("tactic:{goal}:{strategy}:{family}"),
            format!(
                "context:{goal}:{strategy}:{family}:{}",
                context.canonical_key()
            ),
        ]
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "kebab-case")]
pub enum WorkloadClass {
    Build,
    LlmInference,
    Browsing,
    Idle,
    #[default]
    Unknown,
}

impl WorkloadClass {
    fn classify(value: &str) -> Self {
        match value.trim() {
            "build" => Self::Build,
            "llm-inference" => Self::LlmInference,
            "browsing" => Self::Browsing,
            "idle" => Self::Idle,
            _ => Self::Unknown,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::LlmInference => "llm-inference",
            Self::Browsing => "browsing",
            Self::Idle => "idle",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "kebab-case")]
pub enum ForegroundBand {
    Foreground,
    #[default]
    Background,
}

impl ForegroundBand {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Foreground => "foreground",
            Self::Background => "background",
        }
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "kebab-case")]
pub enum MediaState {
    #[default]
    Quiet,
    Audio,
    Call,
}

impl MediaState {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Quiet => "quiet",
            Self::Audio => "audio",
            Self::Call => "call",
        }
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(default)]
pub struct HierarchyContext {
    pub workload: WorkloadClass,
    pub pressure: PressureBand,
    pub thermal: ThermalBand,
    pub foreground: ForegroundBand,
    pub media: MediaState,
}

impl HierarchyContext {
    pub fn classify(workload: &str, context: &ActuatorEpisodeContext) -> Option<Self> {
        if !context.valid || !context.is_finite() {
            return None;
        }
        let workload = WorkloadClass::classify(workload);
        if workload == WorkloadClass::Unknown {
            return None;
        }
        Some(Self {
            workload,
            pressure: PressureBand::from_fraction(context.memory_pressure)?,
            thermal: ThermalBand::from_fraction(context.thermal_score)?,
            foreground: if context.foreground_app_hash != 0 || context.app_launching {
                ForegroundBand::Foreground
            } else {
                ForegroundBand::Background
            },
            media: if context.user_call_in_progress {
                MediaState::Call
            } else if context.user_audio_active {
                MediaState::Audio
            } else {
                MediaState::Quiet
            },
        })
    }

    fn canonical_key(self) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            self.workload.as_str(),
            pressure_name(self.pressure),
            thermal_name(self.thermal),
            self.foreground.as_str(),
            self.media.as_str()
        )
    }

    fn is_authoritative(self) -> bool {
        self.workload != WorkloadClass::Unknown
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct ResolvedLearningDetails {
    pub decision_id: DecisionId,
    pub lifecycle: DecisionLifecycle,
    pub hierarchy: HierarchyPath,
    pub context: HierarchyContext,
    #[serde(default, deserialize_with = "deserialize_bounded_eight")]
    pub alternatives: Vec<CandidateAlternative>,
    #[serde(default, deserialize_with = "deserialize_bounded_eight")]
    pub predictions: Vec<PredictionRecord>,
    #[serde(default, deserialize_with = "deserialize_bounded_eight")]
    pub adviser_contributions: Vec<AdviserContribution>,
    pub expected_utility: f64,
    pub actual_utility: f64,
    pub raw_utility_delta: f64,
    pub counterfactual_delta: f64,
    pub quality: f64,
    pub causal_quality: f64,
    pub confounder_count: u8,
    pub separability: SeparabilityState,
    #[serde(default, deserialize_with = "deserialize_bounded_eight")]
    pub calibration_deltas: Vec<ForecastCalibrationDelta>,
    pub installation_id: InstallationId,
    pub hardware_regime: HardwareRegime,
    pub resolved_cycle: u64,
    pub resolved_timestamp_unix: i64,
}

impl ResolvedLearningDetails {
    pub fn is_authoritative(&self) -> bool {
        if self.decision_id.0 == 0
            || self.lifecycle != DecisionLifecycle::Applied
            || !self.hierarchy.is_canonical()
            || !self.context.is_authoritative()
            || !self.installation_id.is_known()
            || !self.hardware_regime.is_known()
            || self.resolved_timestamp_unix <= 0
            || self.confounder_count != 0
            || self.calibration_deltas.is_empty()
            || self.calibration_deltas.len() > 8
            || self.predictions.is_empty()
            || self.predictions.len() > 8
            || self.alternatives.len() > 8
            || self.adviser_contributions.len() > 8
            || ![
                self.expected_utility,
                self.actual_utility,
                self.raw_utility_delta,
                self.counterfactual_delta,
                self.quality,
                self.causal_quality,
            ]
            .into_iter()
            .all(f64::is_finite)
            || !(-1.0..=1.0).contains(&self.expected_utility)
            || !(-1.0..=1.0).contains(&self.actual_utility)
            || !(-1.0..=1.0).contains(&self.raw_utility_delta)
            || !(-1.0..=1.0).contains(&self.counterfactual_delta)
            || !(0.85..=1.0).contains(&self.quality)
            || !(0.0..=1.0).contains(&self.causal_quality)
        {
            return false;
        }
        let valid_separability = match self.hierarchy.family {
            ActuatorFamily::Coordinated => {
                self.separability == SeparabilityState::CoordinatedComposite
            }
            _ => self.separability == SeparabilityState::Individual,
        };
        valid_separability
            && valid_alternatives(&self.alternatives)
            && valid_predictions(&self.predictions)
            && valid_advisers(&self.adviser_contributions)
            && valid_deltas(self)
    }

    fn prototype_key(&self) -> PrototypeKey {
        PrototypeKey {
            goal: self.hierarchy.goal,
            strategy: self.hierarchy.strategy,
            family: self.hierarchy.family,
            context: self.context,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(default)]
pub struct PrototypeKey {
    pub goal: Goal,
    pub strategy: Strategy,
    pub family: ActuatorFamily,
    pub context: HierarchyContext,
}

impl Default for PrototypeKey {
    fn default() -> Self {
        Self {
            goal: Goal::Stability,
            strategy: Strategy::RecoverState,
            family: ActuatorFamily::Coordinated,
            context: HierarchyContext::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct RepresentativeAction {
    #[serde(default, deserialize_with = "deserialize_hierarchy_action")]
    pub action: String,
    pub observations: u32,
    pub utility_ema: f64,
    pub last_resolved_cycle: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(default)]
pub struct TrustTransitionSummary {
    pub forecasts: u8,
    pub promotions: u8,
    pub degradations: u8,
    pub strongest_before: TrustState,
    pub strongest_after: TrustState,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct LearningPrototype {
    pub key: PrototypeKey,
    pub observations: u32,
    pub effective: u32,
    pub regressions: u32,
    pub neutral: u32,
    pub utility_ema: f64,
    pub quality_ema: f64,
    pub causal_quality_ema: f64,
    pub calibration_error_ema: f64,
    pub evidence_mass: f64,
    pub last_resolved_cycle: u64,
    pub last_observed_unix: i64,
    pub last_decay_unix: i64,
    pub last_trust_transition: TrustTransitionSummary,
    #[serde(default, deserialize_with = "deserialize_bounded_four")]
    pub representative_actions: Vec<RepresentativeAction>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrototypeRecommendation {
    pub key: PrototypeKey,
    pub utility: f64,
    pub quality: f64,
    pub causal_quality: f64,
    pub evidence_mass: f64,
    pub representative_actions: Vec<RepresentativeAction>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HierarchyConsolidationOutcome {
    Improved,
    Worsened,
    Neutral,
    Duplicate,
    #[default]
    Rejected,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HierarchyConsolidation {
    pub outcome: HierarchyConsolidationOutcome,
    pub propositions: Option<[String; MAX_HIERARCHY_PROPOSITIONS]>,
}

impl HierarchyConsolidation {
    pub fn accepted(&self) -> bool {
        matches!(
            self.outcome,
            HierarchyConsolidationOutcome::Improved
                | HierarchyConsolidationOutcome::Worsened
                | HierarchyConsolidationOutcome::Neutral
        )
    }

    pub fn duplicate(&self) -> bool {
        self.outcome == HierarchyConsolidationOutcome::Duplicate
    }

    pub fn rejected(&self) -> bool {
        self.outcome == HierarchyConsolidationOutcome::Rejected
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LearningHierarchy {
    installation_id: InstallationId,
    hardware_regime: HardwareRegime,
    #[serde(
        default,
        serialize_with = "serialize_prototypes",
        deserialize_with = "deserialize_prototypes"
    )]
    prototypes: BTreeMap<PrototypeKey, LearningPrototype>,
    #[serde(skip)]
    family_index: BTreeMap<ActuatorFamily, BTreeSet<PrototypeKey>>,
    #[serde(skip)]
    indices_valid: bool,
    #[serde(default, deserialize_with = "deserialize_processed_ids")]
    processed_decision_ids: VecDeque<DecisionId>,
    pub duplicate_total: u64,
    pub rejected_total: u64,
    pub eviction_total: u64,
}

impl PartialEq for LearningHierarchy {
    fn eq(&self, other: &Self) -> bool {
        self.installation_id == other.installation_id
            && self.hardware_regime == other.hardware_regime
            && self.prototypes == other.prototypes
            && self.processed_decision_ids == other.processed_decision_ids
            && self.duplicate_total == other.duplicate_total
            && self.rejected_total == other.rejected_total
            && self.eviction_total == other.eviction_total
    }
}

impl Default for LearningHierarchy {
    fn default() -> Self {
        Self::new(InstallationId::UNKNOWN, HardwareRegime::default())
    }
}

impl LearningHierarchy {
    pub fn new(installation_id: InstallationId, hardware_regime: HardwareRegime) -> Self {
        Self {
            installation_id,
            hardware_regime,
            prototypes: BTreeMap::new(),
            family_index: BTreeMap::new(),
            indices_valid: true,
            processed_decision_ids: VecDeque::new(),
            duplicate_total: 0,
            rejected_total: 0,
            eviction_total: 0,
        }
    }

    pub fn consolidate(&mut self, details: &ResolvedLearningDetails) -> HierarchyConsolidation {
        self.ensure_indices();
        if !details.is_authoritative()
            || details.installation_id != self.installation_id
            || details.hardware_regime != self.hardware_regime
        {
            self.rejected_total = self.rejected_total.saturating_add(1);
            return HierarchyConsolidation::default();
        }
        if self.processed_decision_ids.contains(&details.decision_id) {
            self.duplicate_total = self.duplicate_total.saturating_add(1);
            return HierarchyConsolidation {
                outcome: HierarchyConsolidationOutcome::Duplicate,
                propositions: None,
            };
        }

        self.processed_decision_ids.push_back(details.decision_id);
        while self.processed_decision_ids.len() > MAX_PROCESSED_DECISION_IDS {
            self.processed_decision_ids.pop_front();
        }
        let key = details.prototype_key();
        if !self.prototypes.contains_key(&key) {
            self.ensure_context_capacity(&key, details.resolved_timestamp_unix);
            self.ensure_global_capacity(details.resolved_timestamp_unix);
            self.family_index
                .entry(key.family)
                .or_default()
                .insert(key.clone());
        }
        let prototype = self
            .prototypes
            .entry(key.clone())
            .or_insert_with(|| LearningPrototype {
                key,
                last_decay_unix: details.resolved_timestamp_unix,
                ..LearningPrototype::default()
            });
        decay_prototype(prototype, details.resolved_timestamp_unix);
        observe_prototype(prototype, details);

        let outcome = if details.actual_utility > UTILITY_DEADBAND {
            HierarchyConsolidationOutcome::Improved
        } else if details.actual_utility < -UTILITY_DEADBAND {
            HierarchyConsolidationOutcome::Worsened
        } else {
            HierarchyConsolidationOutcome::Neutral
        };
        let propositions = matches!(
            outcome,
            HierarchyConsolidationOutcome::Improved | HierarchyConsolidationOutcome::Worsened
        )
        .then(|| details.hierarchy.propositions(details.context));
        HierarchyConsolidation {
            outcome,
            propositions,
        }
    }

    pub fn restore_for_origin(
        &mut self,
        installation_id: InstallationId,
        hardware_regime: HardwareRegime,
    ) -> bool {
        if !installation_id.is_known()
            || !hardware_regime.is_known()
            || self.installation_id != installation_id
            || self.hardware_regime != hardware_regime
        {
            *self = Self::new(installation_id, hardware_regime);
            return true;
        }
        self.normalize();
        false
    }

    pub fn checkpoint_snapshot(&self, now_unix: i64) -> Self {
        let mut snapshot = self.clone();
        for prototype in snapshot.prototypes.values_mut() {
            decay_prototype(prototype, now_unix);
        }
        snapshot
            .prototypes
            .retain(|_, prototype| !weak_and_stale(prototype, now_unix));
        snapshot.normalize();
        snapshot
    }

    pub fn retrieve(
        &mut self,
        context: &HierarchyContext,
        family: ActuatorFamily,
        now_unix: i64,
    ) -> Vec<PrototypeRecommendation> {
        self.ensure_indices();
        let matching = self
            .family_index
            .get(&family)
            .map(|keys| keys.iter().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let mut top = Vec::with_capacity(RETRIEVAL_TOP_K);
        for key in matching {
            let Some(prototype) = self.prototypes.get_mut(&key) else {
                continue;
            };
            decay_prototype(prototype, now_unix);
            let recommendation = PrototypeRecommendation {
                key: prototype.key.clone(),
                utility: prototype.utility_ema,
                quality: prototype.quality_ema,
                causal_quality: prototype.causal_quality_ema,
                evidence_mass: prototype.evidence_mass,
                representative_actions: prototype.representative_actions.clone(),
            };
            let index = top
                .iter()
                .position(|existing| recommendation_better(&recommendation, existing, context))
                .unwrap_or(top.len());
            top.insert(index, recommendation);
            if top.len() > RETRIEVAL_TOP_K {
                top.pop();
            }
        }
        top
    }

    pub fn prototype_count(&self) -> usize {
        self.prototypes.len()
    }

    pub fn processed_decision_count(&self) -> usize {
        self.processed_decision_ids.len()
    }

    pub fn prototypes(&self) -> impl Iterator<Item = &LearningPrototype> {
        self.prototypes.values()
    }

    pub fn prototype_for(&self, details: &ResolvedLearningDetails) -> Option<&LearningPrototype> {
        self.prototypes.get(&details.prototype_key())
    }

    pub fn variant_count(&self, family: ActuatorFamily) -> usize {
        if self.indices_valid {
            self.family_index.get(&family).map_or(0, BTreeSet::len)
        } else {
            self.prototypes
                .keys()
                .filter(|key| key.family == family)
                .take(MAX_CONTEXTS_PER_FAMILY + 1)
                .count()
        }
    }

    fn ensure_context_capacity(&mut self, key: &PrototypeKey, now_unix: i64) {
        if self.variant_count(key.family) < MAX_CONTEXTS_PER_FAMILY {
            return;
        }
        if let Some(victim) = self.weakest_key(Some(key.family), now_unix) {
            self.remove_prototype(&victim);
            self.eviction_total = self.eviction_total.saturating_add(1);
        }
    }

    fn ensure_global_capacity(&mut self, now_unix: i64) {
        if self.prototypes.len() < MAX_PROTOTYPES {
            return;
        }
        if let Some(victim) = self.weakest_key(None, now_unix) {
            self.remove_prototype(&victim);
            self.eviction_total = self.eviction_total.saturating_add(1);
        }
    }

    fn weakest_key(&self, family: Option<ActuatorFamily>, now_unix: i64) -> Option<PrototypeKey> {
        let family_keys = family.and_then(|family| self.family_index.get(&family));
        let mut candidates =
            Vec::with_capacity(family_keys.map_or(self.prototypes.len(), BTreeSet::len));
        if let Some(keys) = family_keys {
            for key in keys {
                if let Some(prototype) = self.prototypes.get(key) {
                    candidates.push((key, prototype));
                }
            }
        } else {
            candidates.extend(self.prototypes.iter());
        }
        let has_weak_stale = candidates
            .iter()
            .any(|(_, prototype)| weak_and_stale(prototype, now_unix));
        candidates
            .into_iter()
            .filter(|(_, prototype)| !has_weak_stale || weak_and_stale(prototype, now_unix))
            .min_by(|(_, left), (_, right)| weakness_cmp(left, right, now_unix))
            .map(|(key, _)| key.clone())
    }

    fn normalize(&mut self) {
        let mut normalized = BTreeMap::new();
        let mut family_index: BTreeMap<ActuatorFamily, BTreeSet<PrototypeKey>> = BTreeMap::new();
        for (_, mut prototype) in std::mem::take(&mut self.prototypes) {
            if classify_family(prototype.key.family) != (prototype.key.goal, prototype.key.strategy)
                || !prototype.key.context.is_authoritative()
                || !normalize_prototype(&mut prototype)
            {
                continue;
            }
            let family_keys = family_index.entry(prototype.key.family).or_default();
            if family_keys.len() >= MAX_CONTEXTS_PER_FAMILY {
                continue;
            }
            family_keys.insert(prototype.key.clone());
            normalized.insert(prototype.key.clone(), prototype);
            if normalized.len() >= MAX_PROTOTYPES {
                break;
            }
        }
        self.prototypes = normalized;
        self.family_index = family_index;
        self.indices_valid = true;
        let mut seen = BTreeSet::new();
        self.processed_decision_ids
            .retain(|id| id.0 != 0 && seen.insert(*id));
        while self.processed_decision_ids.len() > MAX_PROCESSED_DECISION_IDS {
            self.processed_decision_ids.pop_front();
        }
    }

    fn ensure_indices(&mut self) {
        if !self.indices_valid {
            self.normalize();
        }
    }

    fn remove_prototype(&mut self, key: &PrototypeKey) {
        self.prototypes.remove(key);
        if let Some(keys) = self.family_index.get_mut(&key.family) {
            keys.remove(key);
            if keys.is_empty() {
                self.family_index.remove(&key.family);
            }
        }
    }
}

fn observe_prototype(prototype: &mut LearningPrototype, details: &ResolvedLearningDetails) {
    let alpha = if prototype.observations == 0 {
        1.0
    } else {
        PROTOTYPE_EMA_ALPHA
    };
    prototype.observations = prototype
        .observations
        .saturating_add(1)
        .min(MAX_PROTOTYPE_OBSERVATIONS);
    if details.actual_utility > UTILITY_DEADBAND {
        prototype.effective = prototype.effective.saturating_add(1);
    } else if details.actual_utility < -UTILITY_DEADBAND {
        prototype.regressions = prototype.regressions.saturating_add(1);
    } else {
        prototype.neutral = prototype.neutral.saturating_add(1);
    }
    let calibration_error = details
        .calibration_deltas
        .iter()
        .map(|delta| delta.normalized_absolute_error)
        .sum::<f64>()
        / details.calibration_deltas.len() as f64;
    prototype.utility_ema = ema(
        prototype.utility_ema,
        details.actual_utility,
        alpha,
        -1.0,
        1.0,
    );
    prototype.quality_ema = ema(prototype.quality_ema, details.quality, alpha, 0.0, 1.0);
    prototype.causal_quality_ema = ema(
        prototype.causal_quality_ema,
        details.causal_quality,
        alpha,
        0.0,
        1.0,
    );
    prototype.calibration_error_ema = ema(
        prototype.calibration_error_ema,
        calibration_error,
        alpha,
        0.0,
        1.0,
    );
    prototype.evidence_mass = (prototype.evidence_mass + details.quality).min(MAX_EVIDENCE_MASS);
    prototype.last_resolved_cycle = details.resolved_cycle;
    prototype.last_observed_unix = details.resolved_timestamp_unix;
    prototype.last_decay_unix = details.resolved_timestamp_unix;
    prototype.last_trust_transition = summarize_trust(&details.calibration_deltas);
    observe_representative(
        &mut prototype.representative_actions,
        &details.hierarchy.action,
        details.actual_utility,
        details.resolved_cycle,
    );
}

fn summarize_trust(deltas: &[ForecastCalibrationDelta]) -> TrustTransitionSummary {
    let mut summary = TrustTransitionSummary {
        forecasts: deltas.len().min(u8::MAX as usize) as u8,
        ..TrustTransitionSummary::default()
    };
    for delta in deltas {
        if trust_rank(delta.trust_after) > trust_rank(delta.trust_before) {
            summary.promotions = summary.promotions.saturating_add(1);
        }
        if delta.trust_after == TrustState::Degraded && delta.trust_before != TrustState::Degraded {
            summary.degradations = summary.degradations.saturating_add(1);
        }
        if trust_rank(delta.trust_before) >= trust_rank(summary.strongest_before) {
            summary.strongest_before = delta.trust_before;
        }
        if trust_rank(delta.trust_after) >= trust_rank(summary.strongest_after) {
            summary.strongest_after = delta.trust_after;
        }
    }
    summary
}

fn observe_representative(
    representatives: &mut Vec<RepresentativeAction>,
    action: &str,
    utility: f64,
    cycle: u64,
) {
    if let Some(existing) = representatives
        .iter_mut()
        .find(|representative| representative.action == action)
    {
        let alpha = if existing.observations == 0 {
            1.0
        } else {
            PROTOTYPE_EMA_ALPHA
        };
        existing.observations = existing.observations.saturating_add(1);
        existing.utility_ema = ema(existing.utility_ema, utility, alpha, -1.0, 1.0);
        existing.last_resolved_cycle = cycle;
        return;
    }
    let candidate = RepresentativeAction {
        action: action.to_string(),
        observations: 1,
        utility_ema: utility,
        last_resolved_cycle: cycle,
    };
    if representatives.len() < MAX_REPRESENTATIVE_ACTIONS {
        representatives.push(candidate);
        return;
    }
    let victim = representatives
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| representative_cmp(left, right))
        .map(|(index, _)| index)
        .unwrap_or(0);
    representatives[victim] = candidate;
}

fn decay_prototype(prototype: &mut LearningPrototype, now_unix: i64) {
    if now_unix <= 0 || prototype.last_decay_unix <= 0 || now_unix <= prototype.last_decay_unix {
        return;
    }
    let elapsed = now_unix.saturating_sub(prototype.last_decay_unix) as f64;
    let factor = 0.5_f64.powf(elapsed / SEVEN_DAYS_SECS as f64);
    prototype.evidence_mass = (prototype.evidence_mass * factor).clamp(0.0, MAX_EVIDENCE_MASS);
    prototype.last_decay_unix = now_unix;
}

fn weak_and_stale(prototype: &LearningPrototype, now_unix: i64) -> bool {
    effective_mass(prototype, now_unix) < 0.5
        && prototype.last_observed_unix > 0
        && now_unix.saturating_sub(prototype.last_observed_unix) >= FOURTEEN_DAYS_SECS
}

fn weakness_cmp(left: &LearningPrototype, right: &LearningPrototype, now_unix: i64) -> Ordering {
    effective_mass(left, now_unix)
        .total_cmp(&effective_mass(right, now_unix))
        .then_with(|| finite_unit(left.quality_ema).total_cmp(&finite_unit(right.quality_ema)))
        .then_with(|| {
            finite_unit(left.causal_quality_ema).total_cmp(&finite_unit(right.causal_quality_ema))
        })
        .then_with(|| left.last_observed_unix.cmp(&right.last_observed_unix))
        .then_with(|| left.key.cmp(&right.key))
}

fn recommendation_better(
    left: &PrototypeRecommendation,
    right: &PrototypeRecommendation,
    context: &HierarchyContext,
) -> bool {
    context_similarity(left.key.context, *context)
        .cmp(&context_similarity(right.key.context, *context))
        .then_with(|| left.evidence_mass.total_cmp(&right.evidence_mass))
        .then_with(|| left.quality.total_cmp(&right.quality))
        .then_with(|| left.causal_quality.total_cmp(&right.causal_quality))
        .then_with(|| right.key.cmp(&left.key))
        == Ordering::Greater
}

fn context_similarity(left: HierarchyContext, right: HierarchyContext) -> u8 {
    u8::from(left.workload == right.workload)
        + u8::from(left.pressure == right.pressure)
        + u8::from(left.thermal == right.thermal)
        + u8::from(left.foreground == right.foreground)
        + u8::from(left.media == right.media)
}

fn effective_mass(prototype: &LearningPrototype, now_unix: i64) -> f64 {
    let mass = finite_mass(prototype.evidence_mass);
    if now_unix <= prototype.last_decay_unix || prototype.last_decay_unix <= 0 {
        mass
    } else {
        let elapsed = now_unix.saturating_sub(prototype.last_decay_unix) as f64;
        (mass * 0.5_f64.powf(elapsed / SEVEN_DAYS_SECS as f64)).clamp(0.0, MAX_EVIDENCE_MASS)
    }
}

fn normalize_prototype(prototype: &mut LearningPrototype) -> bool {
    if prototype.last_observed_unix <= 0
        || ![
            prototype.utility_ema,
            prototype.quality_ema,
            prototype.causal_quality_ema,
            prototype.calibration_error_ema,
            prototype.evidence_mass,
        ]
        .into_iter()
        .all(f64::is_finite)
    {
        return false;
    }
    prototype.observations = prototype.observations.min(MAX_PROTOTYPE_OBSERVATIONS);
    prototype.effective = prototype.effective.min(prototype.observations);
    prototype.regressions = prototype.regressions.min(prototype.observations);
    prototype.neutral = prototype.neutral.min(prototype.observations);
    prototype.utility_ema = prototype.utility_ema.clamp(-1.0, 1.0);
    prototype.quality_ema = prototype.quality_ema.clamp(0.0, 1.0);
    prototype.causal_quality_ema = prototype.causal_quality_ema.clamp(0.0, 1.0);
    prototype.calibration_error_ema = prototype.calibration_error_ema.clamp(0.0, 1.0);
    prototype.evidence_mass = prototype.evidence_mass.clamp(0.0, MAX_EVIDENCE_MASS);
    prototype.last_decay_unix = prototype.last_decay_unix.max(prototype.last_observed_unix);
    let mut seen = BTreeSet::new();
    prototype
        .representative_actions
        .retain_mut(|representative| {
            let canonical = HierarchyPath::classify(prototype.key.family, &representative.action);
            let valid = canonical.is_some()
                && representative.utility_ema.is_finite()
                && seen.insert(canonical.as_ref().unwrap().action.clone());
            if valid {
                representative.action = canonical.unwrap().action;
                representative.utility_ema = representative.utility_ema.clamp(-1.0, 1.0);
                representative.observations = representative.observations.max(1);
            }
            valid
        });
    prototype
        .representative_actions
        .truncate(MAX_REPRESENTATIVE_ACTIONS);
    true
}

fn valid_alternatives(alternatives: &[CandidateAlternative]) -> bool {
    let mut seen = BTreeSet::new();
    alternatives.iter().all(|alternative| {
        !alternative.action_key.is_empty()
            && alternative.action_key.chars().count() <= MAX_ACTION_KEY_CHARS
            && alternative.target.chars().count() <= MAX_TARGET_CHARS
            && alternative.expected_utility.is_finite()
            && (-1.0..=1.0).contains(&alternative.expected_utility)
            && alternative.uncertainty.is_finite()
            && (0.0..=1.0).contains(&alternative.uncertainty)
            && seen.insert(alternative.action_key.as_str())
    })
}

fn valid_predictions(predictions: &[PredictionRecord]) -> bool {
    let mut seen = BTreeSet::new();
    predictions.iter().all(|prediction| {
        !prediction.source.is_empty()
            && prediction.source.chars().count() <= MAX_SOURCE_CHARS
            && prediction.expected_utility.is_finite()
            && (-1.0..=1.0).contains(&prediction.expected_utility)
            && prediction.uncertainty.is_finite()
            && prediction.uncertainty > 0.0
            && prediction.uncertainty <= 1.0
            && prediction
                .positive_probability
                .is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
            && seen.insert(prediction.source.as_str())
    })
}

fn valid_advisers(advisers: &[AdviserContribution]) -> bool {
    let mut seen = BTreeSet::new();
    advisers.iter().all(|adviser| {
        !adviser.adviser.is_empty()
            && adviser.adviser.chars().count() <= MAX_SOURCE_CHARS
            && adviser.support.is_finite()
            && (-1.0..=1.0).contains(&adviser.support)
            && adviser.uncertainty.is_finite()
            && (0.0..=1.0).contains(&adviser.uncertainty)
            && seen.insert(adviser.adviser.as_str())
    })
}

fn valid_deltas(details: &ResolvedLearningDetails) -> bool {
    let mut seen = BTreeSet::new();
    let mut accepted = 0_usize;
    for prediction in details.predictions.iter().take(8) {
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
        let Some(delta) = details.calibration_deltas.get(accepted) else {
            return false;
        };
        let action_matches = match &delta.key.action {
            CalibrationActionScope::Exact(action) => action == &details.hierarchy.action,
            CalibrationActionScope::Family(family) => *family == details.hierarchy.family,
        };
        let expected_error = delta.actual_utility - delta.predicted_utility;
        let coverage = details.actual_utility
            >= prediction.expected_utility - prediction.uncertainty
            && details.actual_utility <= prediction.expected_utility + prediction.uncertainty;
        let foreground_matches = match details.context.foreground {
            ForegroundBand::Foreground => matches!(
                delta.key.foreground,
                ForegroundContext::Active | ForegroundContext::Launching
            ),
            ForegroundBand::Background => matches!(
                delta.key.foreground,
                ForegroundContext::Idle | ForegroundContext::Unknown
            ),
        };
        let brier_matches = match (prediction.binary_target, prediction.positive_probability) {
            (Some(crate::engine::decision_ledger::BinaryPredictionTarget::Effective), Some(p))
                if p.is_finite() && (0.0..=1.0).contains(&p) =>
            {
                delta.brier.is_some_and(|brier| {
                    (brier - p.powi(2)).abs() <= 1.0e-12
                        || (brier - (p - 1.0).powi(2)).abs() <= 1.0e-12
                })
            }
            _ => delta.brier.is_none(),
        };
        if !(delta.key.producer == producer
            && action_matches
            && delta.key.workload == details.context.workload.as_str()
            && delta.key.horizon == CalibrationHorizon::from_cycles(prediction.horizon_cycles)
            && delta.key.pressure == details.context.pressure
            && delta.key.thermal == details.context.thermal
            && foreground_matches
            && (delta.predicted_utility - prediction.expected_utility).abs() <= f64::EPSILON
            && (delta.actual_utility - details.actual_utility).abs() <= f64::EPSILON
            && (delta.signed_error - expected_error).abs() <= 1.0e-12
            && (delta.normalized_absolute_error - expected_error.abs() / 2.0).abs() <= 1.0e-12
            && delta.uncertainty_covered == coverage
            && brier_matches)
        {
            return false;
        }
        accepted += 1;
    }
    accepted == details.calibration_deltas.len()
}

fn representative_cmp(left: &RepresentativeAction, right: &RepresentativeAction) -> Ordering {
    left.observations
        .cmp(&right.observations)
        .then_with(|| left.utility_ema.abs().total_cmp(&right.utility_ema.abs()))
        .then_with(|| left.last_resolved_cycle.cmp(&right.last_resolved_cycle))
        .then_with(|| left.action.cmp(&right.action))
}

fn trust_rank(state: TrustState) -> u8 {
    match state {
        TrustState::Immature => 0,
        TrustState::Degraded => 1,
        TrustState::Candidate => 2,
        TrustState::Validated => 3,
        TrustState::Trusted => 4,
    }
}

fn pressure_name(value: PressureBand) -> &'static str {
    match value {
        PressureBand::Low => "low",
        PressureBand::Moderate => "moderate",
        PressureBand::High => "high",
        PressureBand::Critical => "critical",
    }
}

fn thermal_name(value: ThermalBand) -> &'static str {
    match value {
        ThermalBand::Cool => "cool",
        ThermalBand::Nominal => "nominal",
        ThermalBand::Warm => "warm",
        ThermalBand::Hot => "hot",
    }
}

fn ema(previous: f64, observation: f64, alpha: f64, min: f64, max: f64) -> f64 {
    ((1.0 - alpha) * previous + alpha * observation).clamp(min, max)
}

fn finite_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn finite_mass(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, MAX_EVIDENCE_MASS)
    } else {
        0.0
    }
}

struct BoundedVecVisitor<T, const MAX: usize>(PhantomData<T>);

struct BoundedHierarchyActionVisitor;

impl<'de> Visitor<'de> for BoundedHierarchyActionVisitor {
    type Value = String;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "a hierarchy action of at most {MAX_ACTION_KEY_CHARS} characters"
        )
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: DeError,
    {
        if value.chars().nth(MAX_ACTION_KEY_CHARS).is_some() {
            return Err(E::invalid_length(MAX_ACTION_KEY_CHARS + 1, &self));
        }
        Ok(value.to_string())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: DeError,
    {
        if value.chars().nth(MAX_ACTION_KEY_CHARS).is_some() {
            return Err(E::invalid_length(MAX_ACTION_KEY_CHARS + 1, &self));
        }
        Ok(value)
    }
}

fn deserialize_hierarchy_action<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_string(BoundedHierarchyActionVisitor)
}

impl<'de, T, const MAX: usize> Visitor<'de> for BoundedVecVisitor<T, MAX>
where
    T: Deserialize<'de>,
{
    type Value = Vec<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "at most {MAX} records")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(MAX.min(sequence.size_hint().unwrap_or(MAX)));
        while let Some(value) = sequence.next_element()? {
            if values.len() == MAX {
                return Err(A::Error::invalid_length(MAX + 1, &self));
            }
            values.push(value);
        }
        Ok(values)
    }
}

fn deserialize_bounded_eight<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    deserializer.deserialize_seq(BoundedVecVisitor::<T, 8>(PhantomData))
}

fn deserialize_bounded_four<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    deserializer.deserialize_seq(BoundedVecVisitor::<T, 4>(PhantomData))
}

fn serialize_prototypes<S>(
    prototypes: &BTreeMap<PrototypeKey, LearningPrototype>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut sequence = serializer.serialize_seq(Some(prototypes.len()))?;
    for prototype in prototypes.values() {
        sequence.serialize_element(prototype)?;
    }
    sequence.end()
}

struct PrototypeVisitor;

impl<'de> Visitor<'de> for PrototypeVisitor {
    type Value = BTreeMap<PrototypeKey, LearningPrototype>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "at most {MAX_PROTOTYPES} hierarchy prototypes")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = BTreeMap::new();
        while let Some(prototype) = sequence.next_element::<LearningPrototype>()? {
            if values.len() == MAX_PROTOTYPES {
                return Err(A::Error::invalid_length(MAX_PROTOTYPES + 1, &self));
            }
            if values.contains_key(&prototype.key) {
                return Err(A::Error::custom("duplicate hierarchy prototype key"));
            }
            values.insert(prototype.key.clone(), prototype);
        }
        Ok(values)
    }
}

fn deserialize_prototypes<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<PrototypeKey, LearningPrototype>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(PrototypeVisitor)
}

fn deserialize_processed_ids<'de, D>(deserializer: D) -> Result<VecDeque<DecisionId>, D::Error>
where
    D: Deserializer<'de>,
{
    let values = deserializer
        .deserialize_seq(BoundedVecVisitor::<DecisionId, MAX_PROCESSED_DECISION_IDS>(
            PhantomData,
        ))?;
    Ok(values.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::decision_ledger::PredictionRecord;
    use crate::engine::model_calibration::{CalibrationHorizon, CalibrationKey};

    const ORIGIN: InstallationId = InstallationId(41);
    const HARDWARE: HardwareRegime = HardwareRegime {
        p_core_count: 4,
        e_core_count: 6,
        ram_gib: 16,
    };

    fn context(index: usize) -> HierarchyContext {
        let workloads = [
            WorkloadClass::Build,
            WorkloadClass::LlmInference,
            WorkloadClass::Browsing,
            WorkloadClass::Idle,
        ];
        let pressures = [
            PressureBand::Low,
            PressureBand::Moderate,
            PressureBand::High,
            PressureBand::Critical,
        ];
        let thermals = [
            ThermalBand::Cool,
            ThermalBand::Nominal,
            ThermalBand::Warm,
            ThermalBand::Hot,
        ];
        let foregrounds = [ForegroundBand::Background, ForegroundBand::Foreground];
        let media = [MediaState::Quiet, MediaState::Audio, MediaState::Call];
        HierarchyContext {
            workload: workloads[index % workloads.len()],
            pressure: pressures[(index / 4) % pressures.len()],
            thermal: thermals[(index / 16) % thermals.len()],
            foreground: foregrounds[(index / 64) % foregrounds.len()],
            media: media[(index / 128) % media.len()],
        }
    }

    fn details(
        id: u64,
        family: ActuatorFamily,
        hierarchy_context: HierarchyContext,
        utility: f64,
        action: &str,
        timestamp: i64,
    ) -> ResolvedLearningDetails {
        ResolvedLearningDetails {
            decision_id: DecisionId(id),
            lifecycle: DecisionLifecycle::Applied,
            hierarchy: HierarchyPath::classify(family, action).expect("canonical action"),
            context: hierarchy_context,
            predictions: vec![PredictionRecord {
                source: "world-model".to_string(),
                expected_utility: 0.0,
                uncertainty: 0.2,
                horizon_cycles: 10,
                positive_probability: None,
                binary_target: None,
            }],
            expected_utility: 0.0,
            actual_utility: utility,
            raw_utility_delta: utility,
            quality: 0.95,
            causal_quality: 0.95,
            separability: if family == ActuatorFamily::Coordinated {
                SeparabilityState::CoordinatedComposite
            } else {
                SeparabilityState::Individual
            },
            calibration_deltas: vec![ForecastCalibrationDelta {
                key: CalibrationKey {
                    producer: ProducerId::WorldModel,
                    action: CalibrationActionScope::Family(family),
                    workload: hierarchy_context.workload.as_str().to_string(),
                    horizon: CalibrationHorizon::Sec5,
                    pressure: hierarchy_context.pressure,
                    thermal: hierarchy_context.thermal,
                    foreground: match hierarchy_context.foreground {
                        ForegroundBand::Foreground => ForegroundContext::Active,
                        ForegroundBand::Background => ForegroundContext::Unknown,
                    },
                    ..CalibrationKey::default()
                },
                predicted_utility: 0.0,
                actual_utility: utility,
                signed_error: utility,
                normalized_absolute_error: utility.abs() / 2.0,
                uncertainty_covered: true,
                brier: None,
                trust_before: TrustState::Candidate,
                trust_after: TrustState::Validated,
            }],
            installation_id: ORIGIN,
            hardware_regime: HARDWARE,
            resolved_cycle: id,
            resolved_timestamp_unix: timestamp,
            ..ResolvedLearningDetails::default()
        }
    }

    fn hierarchy() -> LearningHierarchy {
        LearningHierarchy::new(ORIGIN, HARDWARE)
    }

    #[test]
    fn exact_incremental_update_and_directional_semantics() {
        let now = 1_700_000_000;
        let mut memory = hierarchy();
        let positive = details(
            1,
            ActuatorFamily::Boost,
            context(0),
            0.06,
            "boost:editor",
            now,
        );
        let result = memory.consolidate(&positive);
        assert_eq!(result.outcome, HierarchyConsolidationOutcome::Improved);
        assert_eq!(
            result.propositions.as_ref().map(|items| items.len()),
            Some(4)
        );
        let prototype = memory.prototype_for(&positive).unwrap();
        assert_eq!((prototype.observations, prototype.effective), (1, 1));
        assert_eq!(prototype.utility_ema, 0.06);
        assert_eq!(prototype.quality_ema, 0.95);
        assert_eq!(prototype.calibration_error_ema, 0.03);
        assert_eq!(prototype.last_trust_transition.promotions, 1);

        let negative = details(
            2,
            ActuatorFamily::Throttle,
            context(1),
            -0.04,
            "throttle:editor",
            now,
        );
        assert_eq!(
            memory.consolidate(&negative).outcome,
            HierarchyConsolidationOutcome::Worsened
        );
        assert_eq!(memory.prototype_for(&negative).unwrap().regressions, 1);

        let neutral = details(
            3,
            ActuatorFamily::Freeze,
            context(2),
            0.0,
            "freeze:editor",
            now,
        );
        let result = memory.consolidate(&neutral);
        assert_eq!(result.outcome, HierarchyConsolidationOutcome::Neutral);
        assert!(result.propositions.is_none());
        assert_eq!(memory.prototype_for(&neutral).unwrap().neutral, 1);
    }

    #[test]
    fn family_context_cap_and_tie_eviction_are_exact_and_deterministic() {
        let now = 1_700_000_000;
        let mut memory = hierarchy();
        for index in 0..MAX_CONTEXTS_PER_FAMILY {
            let item = details(
                index as u64 + 1,
                ActuatorFamily::Boost,
                context(index),
                0.02,
                "boost:editor",
                now,
            );
            assert!(memory.consolidate(&item).accepted());
        }
        let lexical_victim = memory
            .prototypes
            .keys()
            .min()
            .expect("eight prototypes")
            .clone();
        let ninth = details(
            9,
            ActuatorFamily::Boost,
            context(8),
            0.02,
            "boost:editor",
            now,
        );
        assert!(memory.consolidate(&ninth).accepted());
        assert_eq!(memory.variant_count(ActuatorFamily::Boost), 8);
        assert!(!memory.prototypes.contains_key(&lexical_victim));
        assert_eq!(memory.eviction_total, 1);
    }

    #[test]
    fn representative_cap_processed_id_cap_and_replay_dedup_survive_serde() {
        let now = 1_700_000_000;
        let mut memory = hierarchy();
        for index in 0..=MAX_REPRESENTATIVE_ACTIONS {
            let action = format!("sysctl:rep-{}=1", char::from(b'e' - index as u8));
            let mut item = details(
                index as u64 + 1,
                ActuatorFamily::Sysctl,
                context(0),
                0.02,
                &action,
                now,
            );
            item.resolved_cycle = 1;
            assert!(memory.consolidate(&item).accepted());
        }
        let probe = details(
            200,
            ActuatorFamily::Sysctl,
            context(0),
            0.02,
            "sysctl:probe=1",
            now,
        );
        let representatives = &memory.prototype_for(&probe).unwrap().representative_actions;
        assert_eq!(representatives.len(), 4);
        assert!(!representatives
            .iter()
            .any(|item| item.action == "sysctl:rep-b"));

        for id in 6..=140 {
            let item = details(
                id,
                ActuatorFamily::Boost,
                context(0),
                0.02,
                "boost:editor",
                now,
            );
            memory.consolidate(&item);
        }
        assert_eq!(
            memory.processed_decision_count(),
            MAX_PROCESSED_DECISION_IDS
        );
        let retained = details(
            140,
            ActuatorFamily::Boost,
            context(0),
            0.02,
            "boost:editor",
            now,
        );
        let encoded = serde_json::to_vec(&memory).unwrap();
        let mut restored: LearningHierarchy = serde_json::from_slice(&encoded).unwrap();
        assert!(!restored.restore_for_origin(ORIGIN, HARDWARE));
        assert_eq!(
            restored.consolidate(&retained).outcome,
            HierarchyConsolidationOutcome::Duplicate
        );
    }

    #[test]
    fn lazy_decay_stale_pruning_negative_retention_and_top_four_are_bounded() {
        let now = 1_700_000_000;
        let mut memory = hierarchy();
        let negative = details(
            1,
            ActuatorFamily::Boost,
            context(0),
            -0.06,
            "boost:negative",
            now,
        );
        memory.consolidate(&negative);
        assert_eq!(memory.checkpoint_snapshot(now + 1).prototype_count(), 1);

        for index in 1..MAX_CONTEXTS_PER_FAMILY {
            let item = details(
                index as u64 + 1,
                ActuatorFamily::Boost,
                context(index),
                0.01 * index as f64,
                "boost:editor",
                now,
            );
            memory.consolidate(&item);
        }
        let first = memory.retrieve(&context(0), ActuatorFamily::Boost, now + SEVEN_DAYS_SECS);
        let second = memory.retrieve(&context(0), ActuatorFamily::Boost, now + SEVEN_DAYS_SECS);
        assert_eq!(first.len(), RETRIEVAL_TOP_K);
        assert_eq!(
            first.iter().map(|item| &item.key).collect::<Vec<_>>(),
            second.iter().map(|item| &item.key).collect::<Vec<_>>()
        );
        assert!((first[0].evidence_mass - 0.475).abs() < 1.0e-12);
        assert_eq!(
            memory
                .checkpoint_snapshot(now + FOURTEEN_DAYS_SECS)
                .prototype_count(),
            0
        );
    }

    #[test]
    fn restore_repairs_caps_drops_nonfinite_and_resets_foreign_origin() {
        let now = 1_700_000_000;
        let mut memory = hierarchy();
        let item = details(
            1,
            ActuatorFamily::Boost,
            context(0),
            0.03,
            "boost:editor",
            now,
        );
        memory.consolidate(&item);
        memory
            .prototypes
            .get_mut(&item.prototype_key())
            .unwrap()
            .quality_ema = f64::NAN;
        assert!(!memory.restore_for_origin(ORIGIN, HARDWARE));
        assert_eq!(memory.prototype_count(), 0);

        memory.consolidate(&details(
            2,
            ActuatorFamily::Boost,
            context(0),
            0.03,
            "boost:editor",
            now,
        ));
        assert!(memory.restore_for_origin(
            InstallationId(99),
            HardwareRegime {
                ram_gib: 32,
                ..HARDWARE
            }
        ));
        assert_eq!(memory.prototype_count(), 0);
        assert_eq!(memory.processed_decision_count(), 0);
    }

    #[test]
    fn persisted_prototype_allocation_is_hard_bounded_and_under_budget() {
        let now = 1_700_000_000;
        let mut memory = hierarchy();
        for index in 0..=MAX_PROTOTYPES {
            let hierarchy_context = context(index);
            let key = PrototypeKey {
                goal: Goal::Responsiveness,
                strategy: Strategy::ProtectForeground,
                family: ActuatorFamily::Boost,
                context: hierarchy_context,
            };
            memory.prototypes.insert(
                key.clone(),
                LearningPrototype {
                    key,
                    observations: 1,
                    effective: 1,
                    utility_ema: 0.02,
                    quality_ema: 0.95,
                    causal_quality_ema: 0.95,
                    calibration_error_ema: 0.01,
                    evidence_mass: 0.95,
                    last_resolved_cycle: index as u64 + 1,
                    last_observed_unix: now,
                    last_decay_unix: now,
                    ..LearningPrototype::default()
                },
            );
        }
        assert_eq!(memory.prototypes.len(), MAX_PROTOTYPES + 1);
        let hostile = serde_json::to_vec(&memory).unwrap();
        assert!(serde_json::from_slice::<LearningHierarchy>(&hostile).is_err());

        memory.prototypes.pop_last();
        let saturated = serde_json::to_vec(&memory).unwrap();
        assert!(saturated.len() < 2 * 1024 * 1024);
        let mut restored: LearningHierarchy = serde_json::from_slice(&saturated).unwrap();
        assert!(!restored.restore_for_origin(ORIGIN, HARDWARE));
        assert!(restored.prototype_count() <= MAX_PROTOTYPES);
        assert!(restored.variant_count(ActuatorFamily::Boost) <= MAX_CONTEXTS_PER_FAMILY);
    }

    #[test]
    fn rich_nested_vectors_reject_the_ninth_element_before_authority() {
        let mut item = details(
            1,
            ActuatorFamily::Boost,
            context(0),
            0.02,
            "boost:editor",
            1_700_000_000,
        );
        item.predictions = (0..9)
            .map(|index| PredictionRecord {
                source: format!("source-{index}"),
                expected_utility: 0.0,
                uncertainty: 0.2,
                horizon_cycles: 10,
                positive_probability: None,
                binary_target: None,
            })
            .collect();
        let encoded = serde_json::to_vec(&item).unwrap();
        assert!(serde_json::from_slice::<ResolvedLearningDetails>(&encoded).is_err());

        let mut hostile = serde_json::to_value(details(
            2,
            ActuatorFamily::Boost,
            context(0),
            0.02,
            "boost:editor",
            1_700_000_000,
        ))
        .unwrap();
        hostile["hierarchy"]["action"] = serde_json::Value::String("x".repeat(321));
        assert!(serde_json::from_value::<ResolvedLearningDetails>(hostile).is_err());
    }

    #[test]
    fn task4_review_hostile_oversized_rich_strings_fail_serde() {
        let mut item = details(
            3,
            ActuatorFamily::Boost,
            context(0),
            0.02,
            "boost:editor",
            1_700_000_000,
        );
        item.alternatives.push(CandidateAlternative {
            action_key: "boost:editor".to_string(),
            target: "Editor".to_string(),
            expected_utility: 0.02,
            uncertainty: 0.2,
        });

        let mut hostile = serde_json::to_value(&item).unwrap();
        hostile["alternatives"][0]["action_key"] =
            serde_json::Value::String("x".repeat(MAX_ACTION_KEY_CHARS + 1));
        assert!(serde_json::from_value::<ResolvedLearningDetails>(hostile).is_err());

        let mut hostile = serde_json::to_value(&item).unwrap();
        hostile["alternatives"][0]["target"] =
            serde_json::Value::String("x".repeat(MAX_TARGET_CHARS + 1));
        assert!(serde_json::from_value::<ResolvedLearningDetails>(hostile).is_err());

        let mut hostile = serde_json::to_value(&item).unwrap();
        hostile["calibration_deltas"][0]["key"]["action"] =
            serde_json::json!({ "exact": "x".repeat(MAX_ACTION_KEY_CHARS + 1) });
        assert!(serde_json::from_value::<ResolvedLearningDetails>(hostile).is_err());

        let mut hostile = serde_json::to_value(&item).unwrap();
        hostile["calibration_deltas"][0]["key"]["workload"] =
            serde_json::Value::String("x".repeat(65));
        assert!(serde_json::from_value::<ResolvedLearningDetails>(hostile).is_err());

        item.alternatives[0].target = "x".repeat(MAX_TARGET_CHARS + 1);
        assert!(serde_json::to_vec(&item).is_err());
    }
}
