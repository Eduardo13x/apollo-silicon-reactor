//! Local System 2 -> System 1 consolidation from measured actuator outcomes.
//!
//! This module replaces the former prompt-teacher transfer. Its input is the
//! universal Gold stream curated by `TelemetryMedallion`, so every update is
//! tied to an action Apollo actually issued and an outcome measured on this
//! installation. No text prompt, API response, or free-form JSON can enter the
//! control path.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::engine::installation_identity::InstallationId;
use crate::engine::learning_hierarchy::{
    HierarchyConsolidationOutcome, LearningHierarchy, ResolvedLearningDetails,
};
use crate::engine::nars_belief::{ArousalState, DriftDetector, Salience};
use crate::engine::telemetry_medallion::{
    valid_learning_details, ActuatorFamily, HardwareRegime, ResolvedActuatorEvidence,
};

const FAMILY_EMA_ALPHA: f64 = 0.20;
const MAX_FAMILY_EVIDENCE: u32 = 256;
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
            evidence.utility.apollo_utility,
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
    #[serde(default)]
    learning_hierarchy: LearningHierarchy,
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

    /// Bind restored learning to the live machine. A copied checkpoint or a
    /// hardware change cold-resets all installation-local consolidation.
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
            *self = Self {
                installation_id,
                hardware_regime,
                learning_hierarchy: LearningHierarchy::new(installation_id, hardware_regime),
                ..Self::default()
            };
            return true;
        }
        if self
            .learning_hierarchy
            .restore_for_origin(installation_id, hardware_regime)
        {
            *self = Self {
                installation_id,
                hardware_regime,
                learning_hierarchy: LearningHierarchy::new(installation_id, hardware_regime),
                ..Self::default()
            };
            true
        } else {
            false
        }
    }

    pub fn checkpoint_snapshot(&self, now_unix: i64) -> Self {
        let mut snapshot = self.clone();
        snapshot.learning_hierarchy = self.learning_hierarchy.checkpoint_snapshot(now_unix);
        snapshot
    }

    pub fn learning_hierarchy(&self) -> &LearningHierarchy {
        &self.learning_hierarchy
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
            utility: evidence.utility.apollo_utility,
            quality: evidence.quality,
            ..LocalConsolidationReport::default()
        };
        let Some(details) =
            authoritative_details(evidence, self.installation_id, self.hardware_regime)
        else {
            return report;
        };

        let hierarchy = self.learning_hierarchy.consolidate(details);
        let verdict = match hierarchy.outcome {
            HierarchyConsolidationOutcome::Improved => LocalConsolidationVerdict::Improved,
            HierarchyConsolidationOutcome::Worsened => LocalConsolidationVerdict::Worsened,
            HierarchyConsolidationOutcome::Neutral => LocalConsolidationVerdict::Neutral,
            HierarchyConsolidationOutcome::Duplicate => {
                report.verdict = LocalConsolidationVerdict::Duplicate;
                return report;
            }
            HierarchyConsolidationOutcome::Rejected => return report,
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

        if let Some(propositions) = hierarchy.propositions {
            let success = verdict == LocalConsolidationVerdict::Improved;
            let pressure = evidence.context_before.memory_pressure.clamp(0.0, 1.0);
            for key in propositions {
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
}

fn authoritative_details(
    evidence: &ResolvedActuatorEvidence,
    installation_id: InstallationId,
    hardware_regime: HardwareRegime,
) -> Option<&ResolvedLearningDetails> {
    let details = evidence.learning_details.as_ref()?;
    valid_learning_details(evidence, installation_id, hardware_regime).then_some(details)
}

fn local_salience(
    evidence: &ResolvedActuatorEvidence,
    system1_struggling: bool,
    dr_zero_self_challenge: f64,
) -> Salience {
    let pressure = evidence.context_before.memory_pressure.clamp(0.0, 1.0);
    let magnitude = (evidence.utility.apollo_utility.abs() / 0.08).clamp(0.0, 1.0);
    let epistemic_gate =
        (1.0 - dr_zero_self_challenge * 0.50) * if system1_struggling { 0.80 } else { 1.0 };
    let arousal =
        ((pressure * 0.55 + magnitude * 0.45) * evidence.quality.clamp(0.0, 1.0) * epistemic_gate)
            .clamp(0.0, 1.0) as f32;
    let valence = (evidence.utility.apollo_utility / 0.08).clamp(-1.0, 1.0) as f32;
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
    use crate::engine::decision_ledger::{
        CandidateAlternative, DecisionId, DecisionLifecycle, PredictionRecord,
    };
    use crate::engine::learning_hierarchy::{
        HierarchyContext, HierarchyPath, ResolvedLearningDetails,
    };
    use crate::engine::model_calibration::{
        project_forecast_delta, CalibrationActionScope, CalibrationHorizon, CalibrationKey,
        CalibrationProvenance, ForegroundContext, PressureBand, ProcessClass, ProducerId,
        SeparabilityState, ThermalBand, TrustState,
    };
    use crate::engine::telemetry_medallion::{
        ActuatorEpisodeContext, ActuatorObjective, EvidenceTier, HardwareRegime, WorldStateDelta,
    };

    fn gold(
        id: u64,
        family: ActuatorFamily,
        utility: f64,
        effective: bool,
    ) -> ResolvedActuatorEvidence {
        let mut evidence = ResolvedActuatorEvidence {
            id,
            decision_id: Some(DecisionId(id)),
            family,
            objective: ActuatorObjective::BalancedUtility,
            action_key: if family == ActuatorFamily::Coordinated {
                "coordinated:boost+throttle".to_string()
            } else {
                format!("{}:Editor", family.as_str())
            },
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
            attribution: Default::default(),
            calibration_provenance: CalibrationProvenance {
                local_authority_eligible: true,
                proposer: "world-model".to_string(),
                alternatives: vec![CandidateAlternative {
                    action_key: format!("{}:alternative", family.as_str()),
                    target: "background".to_string(),
                    expected_utility: 0.01,
                    uncertainty: 0.2,
                }],
                predictions: vec![PredictionRecord {
                    source: "world-model".to_string(),
                    expected_utility: 0.02,
                    uncertainty: 0.2,
                    horizon_cycles: 10,
                    positive_probability: None,
                    binary_target: None,
                }],
                cohort_size: 1,
                separability: if family == ActuatorFamily::Coordinated {
                    SeparabilityState::CoordinatedComposite
                } else {
                    SeparabilityState::Individual
                },
                ..CalibrationProvenance::default()
            },
            learning_details: None,
            utility: crate::engine::telemetry_medallion::UtilityDecomposition {
                apollo_utility: utility,
                ..Default::default()
            },
            perceptual_latency_improvement: 0.0,
            net_state_delta: WorldStateDelta::default(),
            context_before: ActuatorEpisodeContext {
                valid: true,
                memory_pressure: 0.42,
                ..ActuatorEpisodeContext::default()
            },
            effective,
            confounder_count: 0,
            target_present_after: Some(true),
        };
        sync_learning_details(&mut evidence);
        evidence
    }

    fn sync_learning_details(evidence: &mut ResolvedActuatorEvidence) {
        let decision_id = evidence.decision_id.expect("fixture decision id");
        let prediction = evidence.calibration_provenance.predictions[0].clone();
        let foreground = if evidence.context_before.app_launching {
            ForegroundContext::Launching
        } else if evidence.context_before.foreground_idle {
            ForegroundContext::Idle
        } else if evidence.context_before.foreground_app_hash != 0 {
            ForegroundContext::Active
        } else {
            ForegroundContext::Unknown
        };
        let process_class = ProcessClass::from_target(
            evidence.family,
            &evidence.target,
            matches!(
                foreground,
                ForegroundContext::Active | ForegroundContext::Launching
            ),
        );
        let delta = project_forecast_delta(
            CalibrationKey {
                producer: ProducerId::WorldModel,
                action: CalibrationActionScope::Family(evidence.family),
                workload: evidence.workload.clone(),
                process_class,
                horizon: CalibrationHorizon::from_cycles(prediction.horizon_cycles),
                pressure: PressureBand::from_fraction(evidence.context_before.memory_pressure)
                    .unwrap(),
                thermal: ThermalBand::from_fraction(evidence.context_before.thermal_score).unwrap(),
                foreground,
            },
            &prediction,
            evidence.utility.apollo_utility,
            evidence.effective,
            TrustState::Immature,
            TrustState::Immature,
        );
        evidence.learning_details = Some(ResolvedLearningDetails {
            decision_id,
            lifecycle: DecisionLifecycle::Applied,
            hierarchy: HierarchyPath::classify(evidence.family, &evidence.action_key).unwrap(),
            context: HierarchyContext::classify(&evidence.workload, &evidence.context_before)
                .unwrap(),
            alternatives: evidence.calibration_provenance.alternatives.clone(),
            predictions: evidence.calibration_provenance.predictions.clone(),
            adviser_contributions: vec![],
            expected_utility: 0.02,
            actual_utility: evidence.utility.apollo_utility,
            raw_utility_delta: evidence.raw_utility_delta,
            counterfactual_delta: evidence.counterfactual_delta,
            quality: evidence.quality,
            causal_quality: evidence.quality,
            confounder_count: evidence.confounder_count,
            separability: evidence.calibration_provenance.separability,
            calibration_deltas: vec![delta],
            installation_id: evidence.installation_id,
            hardware_regime: evidence.hardware_regime,
            resolved_cycle: evidence.resolved_cycle,
            resolved_timestamp_unix: evidence.resolved_timestamp_unix,
        });
    }

    fn configured(nars: &mut DriftDetector) -> LocalConsolidator {
        let mut consolidator = LocalConsolidator::default();
        assert!(consolidator.restore_for_origin(
            InstallationId(7),
            gold(999, ActuatorFamily::Boost, 0.01, true).hardware_regime
        ));
        nars.clear_hierarchy_beliefs();
        consolidator
    }

    #[test]
    fn gold_outcome_compiles_into_system1_and_family_memory() {
        let mut nars = DriftDetector::new();
        let mut consolidator = configured(&mut nars);
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
        assert!(nars.belief("goal:responsiveness").is_some());
        assert!(nars
            .belief("strategy:responsiveness:protect-foreground")
            .is_some());
        assert!(nars
            .belief("tactic:responsiveness:protect-foreground:boost")
            .is_some());
        assert!(nars.belief("actuator:boost:Editor").is_none());
        assert!(nars.belief("target:boost:Editor").is_none());
        assert_eq!(consolidator.total_consolidations, 1);
        assert_eq!(consolidator.view().families_with_evidence, 1);
    }

    #[test]
    fn media_context_is_compiled_into_system1_with_call_priority() {
        let mut nars = DriftDetector::new();
        let mut consolidator = configured(&mut nars);
        let mut arousal = ArousalState::default();
        let mut evidence = gold(91, ActuatorFamily::ThreadQos, 0.04, true);
        evidence.context_before.user_audio_active = true;
        evidence.context_before.user_call_in_progress = true;
        evidence.context_before.foreground_app_hash = 1;
        sync_learning_details(&mut evidence);

        let report = consolidator.consolidate(&evidence, &mut nars, &mut arousal, false, 0.15);

        assert_eq!(report.verdict, LocalConsolidationVerdict::Improved);
        assert!(nars
            .belief("context:responsiveness:protect-foreground:thread_qos:build:moderate:cool:foreground:call")
            .is_some());
    }

    #[test]
    fn duplicate_gold_is_not_replayed() {
        let mut nars = DriftDetector::new();
        let mut consolidator = configured(&mut nars);
        let mut arousal = ArousalState::default();
        let evidence = gold(2, ActuatorFamily::MarkovPrewarm, 0.03, true);
        let first = consolidator.consolidate(&evidence, &mut nars, &mut arousal, false, 0.0);
        let second = consolidator.consolidate(&evidence, &mut nars, &mut arousal, false, 0.0);

        assert_eq!(first.verdict, LocalConsolidationVerdict::Improved);
        assert_eq!(second.verdict, LocalConsolidationVerdict::Duplicate);
        assert_eq!(consolidator.total_consolidations, 1);
    }

    #[test]
    fn checkpoint_roundtrip_retains_aggregate_and_dedup_without_replay_queue() {
        let mut nars = DriftDetector::new();
        let mut consolidator = configured(&mut nars);
        let mut arousal = ArousalState::default();
        let evidence = gold(222, ActuatorFamily::Boost, 0.03, true);
        assert_eq!(
            consolidator
                .consolidate(&evidence, &mut nars, &mut arousal, false, 0.0)
                .verdict,
            LocalConsolidationVerdict::Improved
        );

        let snapshot = consolidator.checkpoint_snapshot(evidence.resolved_timestamp_unix);
        let encoded = serde_json::to_vec(&snapshot).unwrap();
        let mut restored: LocalConsolidator = serde_json::from_slice(&encoded).unwrap();
        assert!(!restored.restore_for_origin(evidence.installation_id, evidence.hardware_regime));
        let report = restored.consolidate(
            &evidence,
            &mut DriftDetector::new(),
            &mut ArousalState::default(),
            false,
            0.0,
        );
        assert_eq!(report.verdict, LocalConsolidationVerdict::Duplicate);
        assert_eq!(restored.total_consolidations, 1);
        assert_eq!(restored.learning_hierarchy().prototype_count(), 1);
    }

    #[test]
    fn bronze_or_invalid_context_never_reaches_system1() {
        let mut nars = DriftDetector::new();
        let mut consolidator = configured(&mut nars);
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
    fn every_authority_predicate_fails_closed_independently() {
        let base = gold(300, ActuatorFamily::Boost, 0.04, true);
        let mut cases = Vec::new();

        let mut silver = base.clone();
        silver.tier = EvidenceTier::Silver;
        cases.push(silver);

        let mut missing_id = base.clone();
        missing_id.decision_id = None;
        cases.push(missing_id);

        let mut missing_detail = base.clone();
        missing_detail.learning_details = None;
        cases.push(missing_detail);

        let mut unattributed = base.clone();
        unattributed.calibration_provenance.local_authority_eligible = false;
        cases.push(unattributed);

        let mut confounded = base.clone();
        confounded.calibration_provenance.cohort_size = 2;
        confounded.calibration_provenance.separability = SeparabilityState::Confounded;
        sync_learning_details(&mut confounded);
        cases.push(confounded);

        let mut bad_origin = base.clone();
        bad_origin.installation_id = InstallationId(8);
        sync_learning_details(&mut bad_origin);
        cases.push(bad_origin);

        let mut bad_hardware = base.clone();
        bad_hardware.hardware_regime.ram_gib = 32;
        sync_learning_details(&mut bad_hardware);
        cases.push(bad_hardware);

        let mut bad_context = base.clone();
        bad_context.context_before.valid = false;
        cases.push(bad_context);

        let mut no_calibration = base.clone();
        no_calibration
            .learning_details
            .as_mut()
            .unwrap()
            .calibration_deltas
            .clear();
        cases.push(no_calibration);

        let mut nonfinite = base.clone();
        nonfinite.learning_details.as_mut().unwrap().causal_quality = f64::NAN;
        cases.push(nonfinite);

        for lifecycle in [
            DecisionLifecycle::Proposed,
            DecisionLifecycle::Rejected,
            DecisionLifecycle::Vetoed,
            DecisionLifecycle::Blocked,
            DecisionLifecycle::Executing,
            DecisionLifecycle::Failed,
            DecisionLifecycle::NoOp,
            DecisionLifecycle::Reverted,
            DecisionLifecycle::Expired,
            DecisionLifecycle::Settled,
        ] {
            let mut inactive = base.clone();
            inactive.learning_details.as_mut().unwrap().lifecycle = lifecycle;
            cases.push(inactive);
        }

        for evidence in cases {
            let mut nars = DriftDetector::new();
            let beliefs_before = nars.len();
            let mut consolidator = configured(&mut nars);
            let report = consolidator.consolidate(
                &evidence,
                &mut nars,
                &mut ArousalState::default(),
                false,
                0.0,
            );
            assert_eq!(report.verdict, LocalConsolidationVerdict::Rejected);
            assert_eq!(consolidator.total_consolidations, 0);
            assert_eq!(consolidator.learning_hierarchy().prototype_count(), 0);
            assert_eq!(nars.len(), beliefs_before);
        }
    }

    #[test]
    fn coordinated_gold_is_one_composite_and_never_member_credit() {
        let mut nars = DriftDetector::new();
        let mut consolidator = configured(&mut nars);
        let mut evidence = gold(350, ActuatorFamily::Coordinated, 0.05, true);
        evidence.calibration_provenance.cohort_size = 3;
        sync_learning_details(&mut evidence);
        let report = consolidator.consolidate(
            &evidence,
            &mut nars,
            &mut ArousalState::default(),
            false,
            0.0,
        );

        assert_eq!(report.verdict, LocalConsolidationVerdict::Improved);
        assert_eq!(report.system1_updates, 4);
        assert_eq!(consolidator.learning_hierarchy().prototype_count(), 1);
        assert_eq!(consolidator.view().families_with_evidence, 1);
        assert!(nars
            .belief("tactic:stability:recover-state:coordinated")
            .is_some());
        assert!(nars
            .belief("tactic:responsiveness:protect-foreground:boost")
            .is_none());
    }

    #[test]
    fn origin_reset_clears_all_local_memory_and_only_hierarchy_nars() {
        let mut nars = DriftDetector::new();
        let mut consolidator = configured(&mut nars);
        let evidence = gold(360, ActuatorFamily::Boost, 0.05, true);
        consolidator.consolidate(
            &evidence,
            &mut nars,
            &mut ArousalState::default(),
            false,
            0.0,
        );
        nars.observe("legacy:descriptive", true);
        assert!(consolidator.restore_for_origin(
            InstallationId(7),
            HardwareRegime {
                ram_gib: 32,
                ..evidence.hardware_regime
            }
        ));
        nars.clear_hierarchy_beliefs();

        assert_eq!(consolidator.total_consolidations, 0);
        assert_eq!(consolidator.learning_hierarchy().prototype_count(), 0);
        assert!(nars.belief("goal:responsiveness").is_none());
        assert!(nars.belief("legacy:descriptive").is_some());
        assert!(nars.belief("apple-owned").is_some());
    }

    #[test]
    fn family_scale_is_neutral_until_three_gold_outcomes() {
        let mut nars = DriftDetector::new();
        let mut consolidator = configured(&mut nars);
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
        let mut nars = DriftDetector::new();
        let mut consolidator = configured(&mut nars);
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
        sync_learning_details(&mut foreign);
        consolidator.consolidate(&foreign, &mut nars, &mut arousal, false, 0.0);

        assert_eq!(consolidator.total_consolidations, 1);
        assert!(consolidator.family_memory(ActuatorFamily::Boost).is_some());
        assert!(consolidator
            .family_memory(ActuatorFamily::Throttle)
            .is_none());

        consolidator.retain_only_installation(InstallationId(7));
        assert_eq!(consolidator.total_consolidations, 1);
    }

    #[test]
    fn neutral_gold_updates_prototype_without_nars_and_conflicting_identity_is_duplicate() {
        let mut nars = DriftDetector::new();
        let mut consolidator = configured(&mut nars);
        let mut arousal = ArousalState::default();
        let neutral = gold(70, ActuatorFamily::Boost, 0.0, false);
        let beliefs_before = nars.len();

        let first = consolidator.consolidate(&neutral, &mut nars, &mut arousal, false, 0.0);
        let mut conflict = neutral.clone();
        conflict.action_key = "boost:Other".to_string();
        sync_learning_details(&mut conflict);
        let duplicate = consolidator.consolidate(&conflict, &mut nars, &mut arousal, false, 0.0);

        assert_eq!(first.verdict, LocalConsolidationVerdict::Neutral);
        assert_eq!(first.system1_updates, 0);
        assert_eq!(nars.len(), beliefs_before);
        assert_eq!(duplicate.verdict, LocalConsolidationVerdict::Duplicate);
        assert_eq!(consolidator.learning_hierarchy().prototype_count(), 1);
        assert_eq!(
            consolidator
                .learning_hierarchy()
                .prototype_for(neutral.learning_details.as_ref().unwrap())
                .unwrap()
                .observations,
            1
        );
    }
}
