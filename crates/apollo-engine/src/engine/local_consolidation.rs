//! Local System 2 -> System 1 consolidation from measured actuator outcomes.
//!
//! This module replaces the former prompt-teacher transfer. Its input is the
//! universal Gold stream curated by `TelemetryMedallion`, so every update is
//! tied to an action Apollo actually issued and an outcome measured on this
//! installation. No text prompt, API response, or free-form JSON can enter the
//! control path.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::engine::installation_identity::InstallationId;
use crate::engine::nars_belief::{ArousalState, DriftDetector, Salience};
use crate::engine::telemetry_medallion::{
    ActuatorFamily, EvidenceTier, HardwareRegime, ResolvedActuatorEvidence,
};

const MIN_GOLD_QUALITY: f64 = 0.85;
const UTILITY_DEADBAND: f64 = 0.005;
const FAMILY_EMA_ALPHA: f64 = 0.20;
const MAX_FAMILY_EVIDENCE: u32 = 256;
const MAX_RECENT_IDS: usize = 128;
const MAX_BELIEF_KEY_CHARS: usize = 192;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LocalConsolidationVerdict {
    Improved,
    Worsened,
    Neutral,
    Duplicate,
    #[default]
    Rejected,
}

impl LocalConsolidationVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Improved => "improved",
            Self::Worsened => "worsened",
            Self::Neutral => "neutral",
            Self::Duplicate => "duplicate",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LocalConsolidationReport {
    pub verdict: LocalConsolidationVerdict,
    pub action_key: String,
    pub family: String,
    pub utility: f64,
    pub quality: f64,
    pub salience: Salience,
    pub system1_updates: u32,
    pub family_confidence: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FamilyReflexMemory {
    pub observations: u32,
    pub improvements: u32,
    pub regressions: u32,
    pub neutral: u32,
    pub utility_ema: f64,
    pub quality_ema: f64,
    pub last_resolved_cycle: u64,
}

impl FamilyReflexMemory {
    fn observe(&mut self, evidence: &ResolvedActuatorEvidence, verdict: LocalConsolidationVerdict) {
        let alpha = if self.observations == 0 {
            1.0
        } else {
            FAMILY_EMA_ALPHA
        };
        self.observations = self.observations.saturating_add(1).min(MAX_FAMILY_EVIDENCE);
        match verdict {
            LocalConsolidationVerdict::Improved => {
                self.improvements = self.improvements.saturating_add(1)
            }
            LocalConsolidationVerdict::Worsened => {
                self.regressions = self.regressions.saturating_add(1)
            }
            LocalConsolidationVerdict::Neutral => self.neutral = self.neutral.saturating_add(1),
            LocalConsolidationVerdict::Duplicate | LocalConsolidationVerdict::Rejected => return,
        }
        self.utility_ema = ema(
            self.utility_ema,
            evidence.net_utility_delta,
            alpha,
            -1.0,
            1.0,
        );
        self.quality_ema = ema(self.quality_ema, evidence.quality, alpha, 0.0, 1.0);
        self.last_resolved_cycle = evidence.resolved_cycle;

        if self
            .improvements
            .saturating_add(self.regressions)
            .saturating_add(self.neutral)
            > MAX_FAMILY_EVIDENCE
        {
            self.improvements = halve_rounded(self.improvements);
            self.regressions = halve_rounded(self.regressions);
            self.neutral = halve_rounded(self.neutral);
        }
    }

    pub fn confidence(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        let evidence = f64::from(self.observations);
        (self.quality_ema.clamp(0.0, 1.0) * evidence / (evidence + 4.0)).clamp(0.0, 1.0)
    }

    pub fn advisory_scale(&self) -> f64 {
        if self.observations < 3 {
            1.0
        } else {
            (0.75 + self.confidence() * 0.25).clamp(0.75, 1.0)
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LocalConsolidationView {
    pub confidence: f64,
    pub families_with_evidence: u32,
    pub total_consolidations: u64,
    pub family_scales: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LocalConsolidator {
    installation_id: InstallationId,
    hardware_regime: HardwareRegime,
    families: BTreeMap<String, FamilyReflexMemory>,
    recent_evidence_ids: VecDeque<u64>,
    pub total_consolidations: u64,
    pub total_improvements: u64,
    pub total_regressions: u64,
    pub total_neutral: u64,
    pub total_system1_updates: u64,
}

impl LocalConsolidator {
    pub fn retain_only_installation(&mut self, installation_id: InstallationId) {
        if !installation_id.is_known() || self.installation_id != installation_id {
            *self = Self::default();
        }
    }

    pub fn consolidate(
        &mut self,
        evidence: &ResolvedActuatorEvidence,
        drift_detector: &mut DriftDetector,
        arousal_state: &mut ArousalState,
        system1_struggling: bool,
        dr_zero_self_challenge: f64,
    ) -> LocalConsolidationReport {
        let family = evidence.family.as_str().to_string();
        let action_key: String = evidence
            .action_key
            .chars()
            .take(MAX_BELIEF_KEY_CHARS)
            .collect();
        let mut report = LocalConsolidationReport {
            action_key: action_key.clone(),
            family: family.clone(),
            utility: evidence.net_utility_delta,
            quality: evidence.quality,
            ..LocalConsolidationReport::default()
        };
        if !valid_gold(evidence) {
            return report;
        }
        self.bind_to_evidence_source(evidence);
        if self.recent_evidence_ids.contains(&evidence.id) {
            report.verdict = LocalConsolidationVerdict::Duplicate;
            return report;
        }
        self.recent_evidence_ids.push_back(evidence.id);
        while self.recent_evidence_ids.len() > MAX_RECENT_IDS {
            self.recent_evidence_ids.pop_front();
        }

        let verdict = if evidence.effective && evidence.net_utility_delta > UTILITY_DEADBAND {
            LocalConsolidationVerdict::Improved
        } else if evidence.net_utility_delta < -UTILITY_DEADBAND {
            LocalConsolidationVerdict::Worsened
        } else {
            LocalConsolidationVerdict::Neutral
        };
        report.verdict = verdict;
        report.salience = local_salience(
            evidence,
            system1_struggling,
            dr_zero_self_challenge.clamp(0.0, 1.0),
        );

        self.total_consolidations = self.total_consolidations.saturating_add(1);
        match verdict {
            LocalConsolidationVerdict::Improved => {
                self.total_improvements = self.total_improvements.saturating_add(1)
            }
            LocalConsolidationVerdict::Worsened => {
                self.total_regressions = self.total_regressions.saturating_add(1)
            }
            LocalConsolidationVerdict::Neutral => {
                self.total_neutral = self.total_neutral.saturating_add(1)
            }
            LocalConsolidationVerdict::Duplicate | LocalConsolidationVerdict::Rejected => {}
        }

        let family_memory = self.families.entry(family.clone()).or_default();
        family_memory.observe(evidence, verdict);
        report.family_confidence = family_memory.confidence();

        // Neutral outcomes still calibrate family confidence, but they do not
        // become negative NARS evidence. Only a measured directional utility
        // change is compiled into System 1.
        if matches!(
            verdict,
            LocalConsolidationVerdict::Improved | LocalConsolidationVerdict::Worsened
        ) {
            let success = verdict == LocalConsolidationVerdict::Improved;
            let pressure = evidence.context_before.memory_pressure.clamp(0.0, 1.0);
            let media_state = if evidence.context_before.user_call_in_progress {
                "call"
            } else if evidence.context_before.user_audio_active {
                "audio"
            } else {
                "quiet"
            };
            let mut keys = vec![
                format!("actuator:{action_key}"),
                format!("family:{family}"),
                format!("media:{media_state}:family:{family}"),
            ];
            if !evidence.target.is_empty() {
                let target: String = evidence
                    .target
                    .chars()
                    .take(MAX_BELIEF_KEY_CHARS / 2)
                    .collect();
                keys.push(format!("target:{family}:{target}"));
            }
            keys.sort();
            keys.dedup();
            for key in keys {
                drift_detector.observe_contextual(&key, success, report.salience, pressure);
                report.system1_updates = report.system1_updates.saturating_add(1);
            }
            self.total_system1_updates = self
                .total_system1_updates
                .saturating_add(u64::from(report.system1_updates));
            arousal_state.update(report.salience);
        }
        report
    }

    pub fn view(&self) -> LocalConsolidationView {
        let mut family_scales = BTreeMap::new();
        let mut weighted_confidence = 0.0;
        let mut weight = 0.0;
        for (family, memory) in &self.families {
            family_scales.insert(family.clone(), memory.advisory_scale());
            let family_weight = f64::from(memory.observations.min(16));
            weighted_confidence += memory.confidence() * family_weight;
            weight += family_weight;
        }
        LocalConsolidationView {
            confidence: if weight > 0.0 {
                (weighted_confidence / weight).clamp(0.0, 1.0)
            } else {
                0.0
            },
            families_with_evidence: self.families.len().min(u32::MAX as usize) as u32,
            total_consolidations: self.total_consolidations,
            family_scales,
        }
    }

    pub fn view_for_installation(&self, installation_id: InstallationId) -> LocalConsolidationView {
        if installation_id.is_known() && self.installation_id == installation_id {
            self.view()
        } else {
            LocalConsolidationView::default()
        }
    }

    pub fn family_memory(&self, family: ActuatorFamily) -> Option<&FamilyReflexMemory> {
        self.families.get(family.as_str())
    }

    fn bind_to_evidence_source(&mut self, evidence: &ResolvedActuatorEvidence) {
        if self.installation_id == evidence.installation_id
            && self.hardware_regime == evidence.hardware_regime
        {
            return;
        }
        // Consolidated reflexes are installation-local. A copied checkpoint,
        // hardware change, or installation-id rotation starts clean before the
        // first new Gold result is compiled into System 1.
        *self = Self {
            installation_id: evidence.installation_id,
            hardware_regime: evidence.hardware_regime,
            ..Self::default()
        };
    }
}

fn valid_gold(evidence: &ResolvedActuatorEvidence) -> bool {
    evidence.tier == EvidenceTier::Gold
        && evidence.quality.is_finite()
        && evidence.quality >= MIN_GOLD_QUALITY
        && evidence.net_utility_delta.is_finite()
        && evidence.context_before.valid
        && evidence.installation_id.is_known()
        && evidence.hardware_regime.is_known()
        && !evidence.action_key.is_empty()
        && evidence.action_key.len() <= 320
        && evidence.workload.len() <= 96
}

fn local_salience(
    evidence: &ResolvedActuatorEvidence,
    system1_struggling: bool,
    dr_zero_self_challenge: f64,
) -> Salience {
    let pressure = evidence.context_before.memory_pressure.clamp(0.0, 1.0);
    let magnitude = (evidence.net_utility_delta.abs() / 0.08).clamp(0.0, 1.0);
    let epistemic_gate =
        (1.0 - dr_zero_self_challenge * 0.50) * if system1_struggling { 0.80 } else { 1.0 };
    let arousal =
        ((pressure * 0.55 + magnitude * 0.45) * evidence.quality.clamp(0.0, 1.0) * epistemic_gate)
            .clamp(0.0, 1.0) as f32;
    let valence = (evidence.net_utility_delta / 0.08).clamp(-1.0, 1.0) as f32;
    Salience { arousal, valence }
}

fn ema(previous: f64, observation: f64, alpha: f64, min: f64, max: f64) -> f64 {
    ((1.0 - alpha) * previous + alpha * observation).clamp(min, max)
}

fn halve_rounded(value: u32) -> u32 {
    value.saturating_add(1) / 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::telemetry_medallion::{
        ActuatorEpisodeContext, ActuatorObjective, HardwareRegime, WorldStateDelta,
    };

    fn gold(
        id: u64,
        family: ActuatorFamily,
        utility: f64,
        effective: bool,
    ) -> ResolvedActuatorEvidence {
        ResolvedActuatorEvidence {
            id,
            family,
            objective: ActuatorObjective::BalancedUtility,
            action_key: format!("{}:Editor", family.as_str()),
            target: "Editor".to_string(),
            workload: "build".to_string(),
            issued_cycle: 10,
            resolved_cycle: 20,
            resolved_timestamp_unix: 1_700_000_000,
            hardware_regime: HardwareRegime {
                p_core_count: 4,
                e_core_count: 6,
                ram_gib: 16,
            },
            installation_id: InstallationId(7),
            horizon_cycles: 10,
            tier: EvidenceTier::Gold,
            quality: 0.95,
            raw_utility_delta: utility,
            counterfactual_delta: 0.0,
            net_utility_delta: utility,
            net_state_delta: WorldStateDelta::default(),
            context_before: ActuatorEpisodeContext {
                valid: true,
                memory_pressure: 0.42,
                ..ActuatorEpisodeContext::default()
            },
            effective,
            confounder_count: 0,
            target_present_after: Some(true),
        }
    }

    #[test]
    fn gold_outcome_compiles_into_system1_and_family_memory() {
        let mut consolidator = LocalConsolidator::default();
        let mut nars = DriftDetector::new();
        let mut arousal = ArousalState::default();
        let report = consolidator.consolidate(
            &gold(1, ActuatorFamily::Boost, 0.06, true),
            &mut nars,
            &mut arousal,
            false,
            0.10,
        );

        assert_eq!(report.verdict, LocalConsolidationVerdict::Improved);
        assert_eq!(report.system1_updates, 4);
        assert!(nars.belief("actuator:boost:Editor").is_some());
        assert!(nars.belief("family:boost").is_some());
        assert!(nars.belief("media:quiet:family:boost").is_some());
        assert_eq!(consolidator.total_consolidations, 1);
        assert_eq!(consolidator.view().families_with_evidence, 1);
    }

    #[test]
    fn media_context_is_compiled_into_system1_with_call_priority() {
        let mut consolidator = LocalConsolidator::default();
        let mut nars = DriftDetector::new();
        let mut arousal = ArousalState::default();
        let mut evidence = gold(91, ActuatorFamily::ThreadQos, 0.04, true);
        evidence.context_before.user_audio_active = true;
        evidence.context_before.user_call_in_progress = true;

        let report = consolidator.consolidate(&evidence, &mut nars, &mut arousal, false, 0.15);

        assert_eq!(report.verdict, LocalConsolidationVerdict::Improved);
        assert!(nars.belief("media:call:family:thread_qos").is_some());
        assert!(nars.belief("media:audio:family:thread_qos").is_none());
    }

    #[test]
    fn duplicate_gold_is_not_replayed() {
        let mut consolidator = LocalConsolidator::default();
        let mut nars = DriftDetector::new();
        let mut arousal = ArousalState::default();
        let evidence = gold(2, ActuatorFamily::MarkovPrewarm, 0.03, true);
        let first = consolidator.consolidate(&evidence, &mut nars, &mut arousal, false, 0.0);
        let second = consolidator.consolidate(&evidence, &mut nars, &mut arousal, false, 0.0);

        assert_eq!(first.verdict, LocalConsolidationVerdict::Improved);
        assert_eq!(second.verdict, LocalConsolidationVerdict::Duplicate);
        assert_eq!(consolidator.total_consolidations, 1);
    }

    #[test]
    fn bronze_or_invalid_context_never_reaches_system1() {
        let mut consolidator = LocalConsolidator::default();
        let mut nars = DriftDetector::new();
        let beliefs_before = nars.len();
        let mut arousal = ArousalState::default();
        let mut evidence = gold(3, ActuatorFamily::Throttle, -0.04, false);
        evidence.tier = EvidenceTier::Bronze;
        let report = consolidator.consolidate(&evidence, &mut nars, &mut arousal, false, 0.0);

        assert_eq!(report.verdict, LocalConsolidationVerdict::Rejected);
        assert_eq!(nars.len(), beliefs_before);
        assert_eq!(consolidator.total_consolidations, 0);
    }

    #[test]
    fn family_scale_is_neutral_until_three_gold_outcomes() {
        let mut consolidator = LocalConsolidator::default();
        let mut nars = DriftDetector::new();
        let mut arousal = ArousalState::default();
        for id in 10..12 {
            consolidator.consolidate(
                &gold(id, ActuatorFamily::ChromiumEcore, 0.02, true),
                &mut nars,
                &mut arousal,
                false,
                0.0,
            );
        }
        assert_eq!(
            consolidator
                .family_memory(ActuatorFamily::ChromiumEcore)
                .expect("family memory")
                .advisory_scale(),
            1.0
        );
        consolidator.consolidate(
            &gold(12, ActuatorFamily::ChromiumEcore, 0.02, true),
            &mut nars,
            &mut arousal,
            false,
            0.0,
        );
        let scale = consolidator
            .family_memory(ActuatorFamily::ChromiumEcore)
            .expect("family memory")
            .advisory_scale();
        assert!((0.75..=1.0).contains(&scale));
    }

    #[test]
    fn copied_memory_resets_on_installation_or_hardware_change() {
        let mut consolidator = LocalConsolidator::default();
        let mut nars = DriftDetector::new();
        let mut arousal = ArousalState::default();
        consolidator.consolidate(
            &gold(20, ActuatorFamily::Boost, 0.03, true),
            &mut nars,
            &mut arousal,
            false,
            0.0,
        );
        assert!(consolidator.family_memory(ActuatorFamily::Boost).is_some());

        let mut foreign = gold(21, ActuatorFamily::Throttle, -0.03, false);
        foreign.installation_id = InstallationId(8);
        foreign.hardware_regime.ram_gib = 24;
        consolidator.consolidate(&foreign, &mut nars, &mut arousal, false, 0.0);

        assert_eq!(consolidator.total_consolidations, 1);
        assert!(consolidator.family_memory(ActuatorFamily::Boost).is_none());
        assert!(consolidator
            .family_memory(ActuatorFamily::Throttle)
            .is_some());

        consolidator.retain_only_installation(InstallationId(7));
        assert_eq!(consolidator.total_consolidations, 0);
    }
}
