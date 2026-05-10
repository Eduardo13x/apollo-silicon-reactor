//! # Daemon Skill Tick
//!
//! Per-cycle skill application extracted from main.rs (Wave 16).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Apply learned skills: throttle processes with solid causal links (matching_skills)
//! - Run trial skill: try one unproven induced skill per cycle; record result next cycle
//!   via WAL (BUG-01 crash recovery [Gray & Reuter 1992 §11])
//!
//! ## Ordering invariant
//! Must run AFTER decide_actions populates `current_actions` (for dedup), and AFTER
//! signal_digest is available (pressure threshold gating). Returns new throttle actions
//! to append; caller merges into the main actions vec.

use std::collections::HashSet;

use std::path::Path;

use apollo_engine::collector::SystemCollector;
use apollo_engine::collector::SystemSnapshot;
use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::llm::{
    delete_file_best_effort, pending_trial_path, write_json_critical,
};
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::optimization_skills::SkillRegistry;
use apollo_engine::engine::outcome_tracker::OutcomeTracker;
use apollo_engine::engine::process_identity::is_apple_platform_process;
use apollo_engine::engine::safety::{is_protected_name, protected_processes};
use apollo_engine::engine::types::RootAction;

/// Per-cycle skill application tick.
///
/// # Parameters
/// - `skill_registry` — mutable ref to lctx.skill_registry
/// - `snapshot` — system snapshot for this cycle
/// - `state` — SharedState (policy lock for learned protection patterns)
/// - `collector` — SystemCollector (process iterator for target matching)
/// - `foreground_pid` — current foreground PID (foreground gate for trial)
/// - `workload_mode` — current workload mode string (skill selection)
/// - `is_root` — whether daemon runs as root (for WAL path)
/// - `current_actions` — actions accumulated so far (for dedup)
/// - `pending_trial_skill` — mutable: resolved on entry, set on new trial
///
/// # Returns
/// New throttle actions from skills and trial (caller appends to main actions vec).
#[allow(clippy::too_many_arguments)]
pub fn run_skill_tick(
    skill_registry: &mut SkillRegistry,
    snapshot: &SystemSnapshot,
    state: &SharedState,
    collector: &SystemCollector,
    foreground_pid: Option<u32>,
    workload_mode: &str,
    is_root: bool,
    current_actions: &[RootAction],
    pending_trial_skill: &mut Option<(String, f64)>,
) -> Vec<RootAction> {
    let mut new_actions: Vec<RootAction> = Vec::new();

    // ── Apply learned skills ─────────────────────────────────────────────────
    // Throttle processes with solid causal links (confidence × avg_delta).
    // matching_skills() gates on pressure ≥ skill.min_pressure AND is_reliable()
    // (≥5 observations, ≥60% success rate). [Sutton & Barto 2018]
    {
        let skill_matches =
            skill_registry.matching_skills(snapshot.pressure.memory_pressure as f32, workload_mode);
        if !skill_matches.is_empty() {
            let already_actioned: HashSet<String> = current_actions
                .iter()
                .filter_map(|a| match a {
                    RootAction::ThrottleProcess { name, .. }
                    | RootAction::FreezeProcess { name, .. } => Some(name.clone()),
                    _ => None,
                })
                .collect();
            let skill_targets: HashSet<String> = skill_matches
                .iter()
                .flat_map(|s| s.throttle_targets.iter().cloned())
                .collect();
            for (pid, process) in collector.system().processes() {
                let name = process.name().to_string();
                // [Saltzer & Kaashoek 2009] never throttle protected daemons via skills
                // even if the registry learned them as "effective" (stale correlation).
                if is_protected_name(&name) {
                    continue;
                }
                let pid_u32 = pid.as_u32();
                // ApplePlatform pre-filter (SuperPlan post-debrief 2026-05-06):
                // SIP-protected Apple binaries reject task_for_pid + memorystatus_control.
                // Apollo's skill_registry was learning "throttle:kernelmanagerd" etc. as
                // "effective" but the kernel ALWAYS rejects → 271/500 journal entries
                // were `success: false` with BlockReason::ApplePlatform. Skip emission
                // for any Apple platform binary (csops CS_PLATFORM_BINARY check, ~1µs).
                if is_apple_platform_process(pid_u32) {
                    continue;
                }
                if skill_targets.contains(&name) && !already_actioned.contains(&name) {
                    let skill_name = skill_matches
                        .iter()
                        .find(|s| s.throttle_targets.contains(&name))
                        .map(|s| s.name.as_str())
                        .unwrap_or("skill");
                    new_actions.push(RootAction::throttle(
                        pid_u32,
                        name,
                        false,
                        format!("skill:{}", skill_name),
                        DecisionReason::MLWorkload,
                    ));
                }
            }
        }
    }

    // ── Trial induced skills ─────────────────────────────────────────────────
    // Each cycle at elevated pressure: try one unproven skill; record next cycle.
    // WAL write-ahead ensures trial survives daemon crash [Gray & Reuter 1992 §11].
    {
        // Resolve pending trial from previous cycle.
        if let Some((ref pending_name, pressure_before)) = *pending_trial_skill {
            let effective = snapshot.pressure.memory_pressure < pressure_before - 0.01;
            skill_registry.record_result_with_pressure(
                pending_name,
                effective,
                pressure_before as f32,
            );
            *pending_trial_skill = None;
            delete_file_best_effort(&pending_trial_path(is_root));
        }

        let trial = skill_registry
            .next_trial_skill(snapshot.pressure.memory_pressure as f32, workload_mode);
        if let Some(skill) = trial {
            let skill_name = skill.name.clone();
            let pressure_before = snapshot.pressure.memory_pressure;
            let policy_prot = state
                .policy
                .lock_recover()
                .learned_policy
                .protected_patterns
                .clone();
            let already_actioned: HashSet<String> = current_actions
                .iter()
                .chain(new_actions.iter())
                .filter_map(|a| match a {
                    RootAction::ThrottleProcess { name, .. } => Some(name.clone()),
                    _ => None,
                })
                .collect();
            let mut trialed = false;
            // "Foreground-blocked" ≠ "ineffective": skill couldn't run this cycle.
            let mut targets_found_but_skipped = false;
            for target in &skill.throttle_targets.clone() {
                // [Saltzer & Kaashoek 2009] is_protected_name is the single truth point.
                if is_protected_name(target)
                    || policy_prot.iter().any(|p| target.contains(p.as_str()))
                {
                    continue;
                }
                for (pid, process) in collector.system().processes() {
                    if process.name() == target {
                        let pid_u32 = pid.as_u32();
                        // ApplePlatform pre-filter (also applies to trial skills).
                        if is_apple_platform_process(pid_u32) {
                            // Mark trial as not-runnable so skill_registry doesn't
                            // penalize the skill for SIP-blocked targets.
                            targets_found_but_skipped = true;
                            break;
                        }
                        if Some(pid_u32) == foreground_pid {
                            targets_found_but_skipped = true;
                        } else {
                            if !already_actioned.contains(target) {
                                new_actions.push(RootAction::throttle(
                                    pid_u32,
                                    target.clone(),
                                    false,
                                    format!("trial:{}", skill_name),
                                    DecisionReason::MLWorkload,
                                ));
                            }
                            trialed = true;
                        }
                        break;
                    }
                }
            }
            if trialed {
                *pending_trial_skill = Some((skill_name.clone(), pressure_before));
                write_json_critical(
                    &pending_trial_path(is_root),
                    &*pending_trial_skill,
                    Some(0o600),
                );
            } else if targets_found_but_skipped {
                // Foreground-blocked — skill is not penalised; wait for next cycle.
            } else {
                // Targets absent: mark ineffective so registry GC's the skill.
                skill_registry.record_result(&skill_name, false);
            }
        }
    }

    new_actions
}

/// Run autonomous rule induction every 100 cycles.
///
/// Mines experience memory + co-occurrence graph for new skills. Filters
/// targets that are protected (static or policy-learned) so induced skills
/// are always executable. Persists when new skills are crystallised.
///
/// # Parameters
/// - `skill_registry` — mutable skill registry (receives induced skills)
/// - `outcome_tracker` — source of experience memory + causal pairs
/// - `state` — SharedState (policy lock for protected_patterns)
/// - `workload_mode` — current workload string (skill scope filter)
/// - `skills_path` — path to persist registry after induction
/// - `cycle_count` — caller passes only when gate fires (% 100 == 0)
pub fn run_rule_induction(
    skill_registry: &mut SkillRegistry,
    outcome_tracker: &OutcomeTracker,
    state: &SharedState,
    workload_mode: &str,
    skills_path: &Path,
) {
    let existing_names = skill_registry.name_set();
    let top_pairs = outcome_tracker.top_causal_pairs(100);
    let protected_set = protected_processes();
    let policy_prot = state
        .policy
        .lock_recover()
        .learned_policy
        .protected_patterns
        .clone();
    let policy_prot_refs: Vec<&str> = policy_prot.iter().map(|s| s.as_str()).collect();
    let mut all_protected: Vec<&str> = protected_set.iter().copied().collect();
    all_protected.extend_from_slice(&policy_prot_refs);
    let new_skills = apollo_engine::engine::rule_inducer::induce(
        &outcome_tracker.experience,
        &top_pairs,
        &existing_names,
        &all_protected,
        workload_mode,
    );
    let induced_count = new_skills.len();
    for skill in new_skills {
        skill_registry.register_induced(skill);
    }
    skill_registry.purge_unexecutable(&all_protected);
    if induced_count > 0 {
        println!(
            "rule_inducer: {} new skills crystallized (total={})",
            induced_count,
            skill_registry.len()
        );
        skill_registry.persist(skills_path);
    }
}
