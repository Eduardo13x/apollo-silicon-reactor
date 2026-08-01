//! Central intent planning for already-admitted root actions.
//!
//! Producers remain responsible for proposing actions and specialist safety
//! gates remain authoritative. This planner is the final, mutation-free pass
//! that removes contradictory intents and orders equivalent accelerators.

use std::collections::{HashMap, HashSet};

use crate::engine::action_types::RootAction;
use crate::engine::telemetry_medallion::actuator_action_key;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IntentEvidence {
    /// Conservative expected benefit in Apollo's normalized utility space.
    pub expected_benefit: f64,
    /// Epistemic uncertainty in [0, 1]. Higher values lower dispatch rank.
    pub uncertainty: f64,
}

#[derive(Debug, Clone, Default)]
pub struct PlanningContext {
    pub memory_pressure: f64,
    pub fluidity_degraded: bool,
    pub app_launching: bool,
    pub utility_evidence: HashMap<String, IntentEvidence>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanReport {
    pub proposed: u64,
    pub admitted: u64,
    pub conflict_drops: u64,
    pub reordered: u64,
    pub evidence_ranked: u64,
    pub family_priors_used: u64,
    pub temporal_candidates: u64,
    pub temporal_rollouts: u64,
    pub temporal_promotions: u64,
    pub temporal_memory_samples: u64,
    pub temporal_expected_gain: f64,
    pub temporal_uncertainty: f64,
    pub temporal_pressure_delta: f64,
    pub temporal_fluidity_delta: f64,
    pub temporal_energy_delta: f64,
    pub temporal_authoritative: bool,
    pub temporal_best_first: Option<String>,
    pub temporal_best_second: Option<String>,
    pub temporal_abstention_reason: Option<String>,
    pub last_resolution: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ProcessIntent {
    Boost,
    Throttle,
    Freeze,
    Unfreeze,
}

#[derive(Debug)]
struct RankedAction {
    original_index: usize,
    action: RootAction,
    priority: u8,
    score: f64,
}

/// Plan actions after safety admission and before execution.
///
/// Recovery dominates restriction for the same PID, and stronger restriction
/// dominates weaker restriction. This prevents one cycle from issuing pairs
/// such as freeze+unfreeze or boost+throttle. Independent controls such as
/// jetsam priority and thread QoS may still coexist with a process action.
pub fn plan_actions(
    actions: Vec<RootAction>,
    context: &PlanningContext,
) -> (Vec<RootAction>, PlanReport) {
    let mut report = PlanReport {
        proposed: actions.len() as u64,
        ..PlanReport::default()
    };
    if actions.len() < 2 {
        report.admitted = actions.len() as u64;
        report.evidence_ranked = actions
            .iter()
            .filter_map(actuator_action_key)
            .filter(|key| {
                context
                    .utility_evidence
                    .get(key)
                    .is_some_and(|evidence| evidence.expected_benefit > 0.0)
            })
            .count() as u64;
        return (actions, report);
    }
    report.evidence_ranked = actions
        .iter()
        .filter_map(actuator_action_key)
        .filter(|key| {
            context
                .utility_evidence
                .get(key)
                .is_some_and(|evidence| evidence.expected_benefit > 0.0)
        })
        .count() as u64;

    let mut dominant_by_pid: HashMap<u32, ProcessIntent> = HashMap::new();
    for action in &actions {
        if let Some((pid, intent)) = process_intent(action) {
            dominant_by_pid
                .entry(pid)
                .and_modify(|current| *current = (*current).max(intent))
                .or_insert(intent);
        }
    }

    let mut retained = Vec::with_capacity(actions.len());
    for (index, action) in actions.into_iter().enumerate() {
        if let Some((pid, intent)) = process_intent(&action) {
            if dominant_by_pid
                .get(&pid)
                .is_some_and(|winner| *winner != intent)
            {
                report.conflict_drops = report.conflict_drops.saturating_add(1);
                report.last_resolution = Some(format!(
                    "pid={pid} kept={} dropped={}",
                    process_intent_name(*dominant_by_pid.get(&pid).expect("winner exists")),
                    process_intent_name(intent)
                ));
                continue;
            }
        }
        retained.push(RankedAction {
            original_index: index,
            priority: action_priority(&action),
            score: action_score(&action, context),
            action,
        });
    }

    // Preserve specialist ordering across priority classes. Only accelerator
    // candidates are utility-ranked, so learned evidence cannot jump ahead of
    // recovery or pressure protection.
    let accelerator_indexes: HashSet<usize> = retained
        .iter()
        .enumerate()
        .filter_map(|(index, ranked)| (ranked.priority == 4).then_some(index))
        .collect();
    let mut accelerator_actions: Vec<RankedAction> = retained
        .iter()
        .filter(|ranked| ranked.priority == 4)
        .map(|ranked| RankedAction {
            original_index: ranked.original_index,
            action: ranked.action.clone(),
            priority: ranked.priority,
            score: ranked.score,
        })
        .collect();
    accelerator_actions.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.original_index.cmp(&right.original_index))
    });
    let mut accelerators = accelerator_actions.into_iter();
    for (index, slot) in retained.iter_mut().enumerate() {
        if accelerator_indexes.contains(&index) {
            let replacement = accelerators.next().expect("accelerator slots match");
            if replacement.original_index != slot.original_index {
                report.reordered = report.reordered.saturating_add(1);
            }
            *slot = replacement;
        }
    }

    report.admitted = retained.len() as u64;
    (
        retained.into_iter().map(|ranked| ranked.action).collect(),
        report,
    )
}

fn process_intent(action: &RootAction) -> Option<(u32, ProcessIntent)> {
    match action {
        RootAction::BoostProcess { pid, .. } => Some((*pid, ProcessIntent::Boost)),
        RootAction::ThrottleProcess { pid, .. } => Some((*pid, ProcessIntent::Throttle)),
        RootAction::FreezeProcess { pid, .. } => Some((*pid, ProcessIntent::Freeze)),
        RootAction::UnfreezeProcess { pid, .. } => Some((*pid, ProcessIntent::Unfreeze)),
        _ => None,
    }
}

fn process_intent_name(intent: ProcessIntent) -> &'static str {
    match intent {
        ProcessIntent::Boost => "boost",
        ProcessIntent::Throttle => "throttle",
        ProcessIntent::Freeze => "freeze",
        ProcessIntent::Unfreeze => "unfreeze",
    }
}

fn action_priority(action: &RootAction) -> u8 {
    match action {
        RootAction::UnfreezeProcess { .. } => 0,
        RootAction::FreezeProcess { .. } => 1,
        RootAction::ThrottleProcess { .. } | RootAction::SetMemorystatus { .. } => 2,
        RootAction::SetThreadQoS { tier, .. } if tier != "interactive" => 2,
        RootAction::BoostProcess { .. } => 4,
        RootAction::SetThreadQoS { tier, .. } if tier == "interactive" => 4,
        _ => 3,
    }
}

fn action_score(action: &RootAction, context: &PlanningContext) -> f64 {
    let (base_benefit, cost, base_uncertainty) = match action {
        RootAction::UnfreezeProcess { .. } => (1.0, 0.05, 0.0),
        RootAction::FreezeProcess { .. } => (0.55 + context.memory_pressure * 0.35, 0.75, 0.20),
        RootAction::ThrottleProcess { .. } => (0.45 + context.memory_pressure * 0.30, 0.30, 0.15),
        RootAction::BoostProcess { .. } => {
            let urgency = (context.app_launching as u8 + context.fluidity_degraded as u8) as f64;
            (0.45 + urgency * 0.20, 0.20, 0.25)
        }
        RootAction::SetThreadQoS { tier, .. } if tier == "interactive" => {
            let urgency = (context.app_launching as u8 + context.fluidity_degraded as u8) as f64;
            (0.40 + urgency * 0.20, 0.12, 0.20)
        }
        RootAction::SetThreadQoS { .. } => (0.45 + context.memory_pressure * 0.15, 0.10, 0.15),
        RootAction::SetMemorystatus { .. } => (0.45, 0.20, 0.20),
        RootAction::SetSysctl(_) => (0.35, 0.40, 0.30),
        RootAction::ToggleSpotlight { .. } | RootAction::QuarantineDaemon { .. } => {
            (0.30, 0.45, 0.35)
        }
    };
    let evidence = actuator_action_key(action)
        .and_then(|key| context.utility_evidence.get(&key))
        .copied()
        .unwrap_or_default();
    base_benefit + evidence.expected_benefit
        - cost
        - (base_uncertainty + evidence.uncertainty) * 0.25
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::audit_types::DecisionReason;

    fn reason() -> DecisionReason {
        DecisionReason::PressureContext
    }

    fn boost(pid: u32, name: &str) -> RootAction {
        RootAction::BoostProcess {
            pid,
            name: name.to_string(),
            reason: "test".to_string(),
            decision_reason: reason(),
            start_sec: 1,
            start_usec: 0,
        }
    }

    fn throttle(pid: u32) -> RootAction {
        RootAction::ThrottleProcess {
            pid,
            name: "worker".to_string(),
            aggressive: false,
            reason: "test".to_string(),
            decision_reason: reason(),
            start_sec: 1,
            start_usec: 0,
        }
    }

    fn unfreeze(pid: u32) -> RootAction {
        RootAction::UnfreezeProcess {
            pid,
            name: "worker".to_string(),
            reason: "test".to_string(),
            decision_reason: reason(),
            start_sec: 1,
            start_usec: 0,
        }
    }

    #[test]
    fn recovery_removes_opposing_process_intents() {
        let (planned, report) = plan_actions(
            vec![boost(42, "Editor"), throttle(42), unfreeze(42)],
            &PlanningContext::default(),
        );
        assert_eq!(planned.len(), 1);
        assert!(matches!(planned[0], RootAction::UnfreezeProcess { .. }));
        assert_eq!(report.conflict_drops, 2);
    }

    #[test]
    fn evidence_reorders_only_accelerator_slots() {
        let mut context = PlanningContext::default();
        context.utility_evidence.insert(
            "boost:Fast".to_string(),
            IntentEvidence {
                expected_benefit: 0.30,
                uncertainty: 0.05,
            },
        );
        let actions = vec![boost(1, "Slow"), throttle(2), boost(3, "Fast")];
        let (planned, report) = plan_actions(actions, &context);
        assert!(matches!(
            &planned[0],
            RootAction::BoostProcess { name, .. } if name == "Fast"
        ));
        assert!(matches!(planned[1], RootAction::ThrottleProcess { .. }));
        assert!(matches!(
            &planned[2],
            RootAction::BoostProcess { name, .. } if name == "Slow"
        ));
        assert_eq!(report.reordered, 2);
    }

    #[test]
    fn distinct_pids_and_independent_controls_survive() {
        let actions = vec![
            boost(1, "Editor"),
            throttle(2),
            RootAction::SetMemorystatus {
                pid: 1,
                priority: 10,
                reason: "test".to_string(),
                decision_reason: reason(),
            },
        ];
        let (planned, report) = plan_actions(actions, &PlanningContext::default());
        assert_eq!(planned.len(), 3);
        assert_eq!(report.conflict_drops, 0);
    }
}
