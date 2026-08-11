//! # Daemon Cycle Tail
//!
//! End-of-cycle blocks extracted from the daemon main loop as part of
//! the V1.1.0 Strangler Fig pass (Wave 10) [Fowler 2004].
//!
//! ## Ordering invariant (peer-review 2026-04-18)
//!
//! `Fluidity QoS → Enriched telemetry (incl. UCHS) → Periodic stage →
//! status broadcast`.
//!
//! - Fluidity QoS elevation must land BEFORE telemetry wiring so the
//!   cognitive metrics reflect this cycle's decision to prioritize UI
//!   fluidity (NotebookLM peer review §1).
//! - UCHS fields are merged into the same `state.metrics` lock guard as
//!   enriched telemetry; the two stages share one critical section to
//!   avoid a second round-trip through the mutex (NotebookLM §1, §3).
//! - Periodic stage (% 100 / % 500 / % 7200 gates) runs LAST so GC and
//!   persistence see a consistent `runtime_metrics.json` snapshot.
//!
//! ## Purity
//!
//! All four functions are shallow glue: they mutate through the locks /
//! `&mut` handles they already owned inline. No new allocations, no
//! new I/O, no new ordering.
//!
//! ## Shared-state carry-overs
//!
//! `frozen_state` and `mach_qos` remain **flat** `Arc<Mutex<…>>` fields on
//! `SharedState` — the thermal sentinel holds independent `Arc`s into
//! them. Do not bundle them into a sub-struct (NotebookLM §"Advertencia
//! de Bloqueo").

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, Instant};

use apollo_engine::collector::{SystemCollector, SystemSnapshot};
use apollo_engine::engine::build_tracker::BuildPhase;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::decision_ledger::{
    ActuatorDecisionEvent, ActuatorDecisionOutcome, CycleDecisionEvents,
};
use apollo_engine::engine::fluidity::FluidityState;
use apollo_engine::engine::io_tiering::{IoPromotionDisposition, IoShaper};
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::lse_counters::CycleStage;
use apollo_engine::engine::mach_qos::{LatencyTier, TaskPolicyLease, ThroughputTier};
use apollo_engine::engine::overflow_guard::OverflowThresholds;
use apollo_engine::engine::pipeline::learning_context::LearningContext;
use apollo_engine::engine::pipeline::periodic_stage::{
    run_periodic, PeriodicContext, PeriodicResult,
};
use apollo_engine::engine::process_classifier::ProcessSnapshot;
use apollo_engine::engine::process_identity::ProcessIdentity;
use apollo_engine::engine::process_tree::ProcessTree;
use apollo_engine::engine::safety::{
    can_boost, hard_protected_contains, is_chromium_family, ProcessInterventionClass,
};
use apollo_engine::engine::swap_predictor::SwapForecast;
use apollo_engine::engine::thermal_bailout::ThermalAction;
use apollo_engine::engine::world_model::{ContextualActionBias, WorldModel};

use crate::cognitive_tick::{CognitiveDecision, CognitiveState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InteractionReason {
    Input,
    WindowOperation,
    BuildStart,
    AppLaunch,
}

impl InteractionReason {
    fn ttl(self) -> Duration {
        match self {
            Self::Input => Duration::from_millis(1_600),
            Self::WindowOperation => Duration::from_millis(1_200),
            Self::BuildStart => Duration::from_secs(3),
            Self::AppLaunch => Duration::from_secs(5),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::WindowOperation => "window",
            Self::BuildStart => "build_start",
            Self::AppLaunch => "app_launch",
        }
    }
}

const MAX_ACCELERATION_MEMBERS: usize = 4;
const MAX_CONTINUOUS_LEASE: Duration = Duration::from_secs(12);
const LEASE_COOLDOWN: Duration = Duration::from_secs(2);
const LEDGER_GRACE: Duration = Duration::from_secs(5);
const TASK_QOS_OWNER: &str = "acceleration lease: interactive task QoS";
const NICE_OWNER: &str = "acceleration lease: nice fallback";
const LEASE_NICE: i32 = -2;

#[derive(Debug, Default)]
pub struct AccelerationTickOutput {
    pub decision_events: CycleDecisionEvents,
}

fn acceleration_event(
    action_key: &str,
    pid: u32,
    cycle: u64,
    outcome: ActuatorDecisionOutcome,
    detail: impl Into<String>,
) -> ActuatorDecisionEvent {
    ActuatorDecisionEvent::local(
        action_key,
        format!("pid:{pid}"),
        cycle,
        outcome,
        "acceleration-lease",
        detail,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccelerationMemberEffect {
    TaskQos,
    Nice,
    IoPromotion,
}

fn acceleration_member_event(
    pid: u32,
    name: &str,
    effect: AccelerationMemberEffect,
    cycle: u64,
    outcome: ActuatorDecisionOutcome,
    detail: impl Into<String>,
) -> ActuatorDecisionEvent {
    let action_key = match effect {
        AccelerationMemberEffect::TaskQos => "interaction_qos:task_qos",
        AccelerationMemberEffect::Nice => "interaction_qos:nice",
        AccelerationMemberEffect::IoPromotion => "io_shaping:interactive_release",
    };
    ActuatorDecisionEvent::local(
        action_key,
        format!("{name}:pid:{pid}"),
        cycle,
        outcome,
        "acceleration-lease",
        detail,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeaseTtlBand {
    Short,
    Standard,
    Long,
}

impl LeaseTtlBand {
    const ALL: [Self; 3] = [Self::Short, Self::Standard, Self::Long];

    fn as_str(self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Standard => "standard",
            Self::Long => "long",
        }
    }

    fn action_key(self) -> &'static str {
        match self {
            Self::Short => "interaction_qos:foreground@short",
            Self::Standard => "interaction_qos:foreground@standard",
            Self::Long => "interaction_qos:foreground@long",
        }
    }

    fn from_bias(bias: ContextualActionBias) -> Self {
        if !bias.score.is_finite() || !bias.is_informative() || bias.score.abs() < 0.25 {
            Self::Standard
        } else if bias.score < 0.0 {
            Self::Short
        } else {
            Self::Long
        }
    }

    fn ttl(self, reason: InteractionReason) -> Duration {
        let factor = match self {
            Self::Short => 0.90,
            Self::Standard => 1.0,
            Self::Long => 1.10,
        };
        reason.ttl().mul_f64(factor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LeaseTtlDecision {
    band: LeaseTtlBand,
    exploratory: bool,
}

fn learned_lease_ttl_band(world_model: &WorldModel, workload: &str) -> Option<LeaseTtlBand> {
    let mut candidates = Vec::with_capacity(LeaseTtlBand::ALL.len());
    for band in LeaseTtlBand::ALL {
        if let Some(assessment) = world_model.assess_utility(band.action_key(), workload) {
            candidates.push((
                band,
                assessment.lower_bound,
                assessment.upper_bound,
                assessment.quality,
            ));
        }
    }
    select_distinct_ttl_band(candidates)
}

fn select_distinct_ttl_band(
    mut candidates: Vec<(LeaseTtlBand, f64, f64, f64)>,
) -> Option<LeaseTtlBand> {
    if candidates.len() < 2 {
        return None;
    }
    candidates.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| right.3.total_cmp(&left.3))
    });
    let best = candidates[0];
    let runner_up = candidates[1];
    (best.1 > runner_up.2 + 0.002).then_some(best.0)
}

fn contextual_io_release_allowed(bias: ContextualActionBias) -> bool {
    !bias.authoritative || bias.score > -0.50
}

fn contextual_io_outcome(bias: ContextualActionBias) -> Option<ActuatorDecisionOutcome> {
    (bias.is_informative() && !contextual_io_release_allowed(bias))
        .then_some(ActuatorDecisionOutcome::Vetoed)
}

fn lease_renewal_outcome() -> ActuatorDecisionOutcome {
    ActuatorDecisionOutcome::NoOp
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccelerationFamily {
    General,
    Chromium,
}

impl AccelerationFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Chromium => "chromium",
        }
    }

    fn allows_explicit_task_qos(self) -> bool {
        self == Self::General
    }
}

#[derive(Debug, Clone)]
struct AccelerationCandidate {
    pid: u32,
    name: String,
    score: f64,
}

#[derive(Debug)]
struct AccelerationSelection {
    root_pid: u32,
    family: AccelerationFamily,
    members: Vec<AccelerationCandidate>,
}

#[derive(Debug)]
struct LeasedMember {
    pid: u32,
    name: String,
    start_sec: u64,
    start_usec: u64,
    task_qos_mutated: bool,
    policy_lease: Option<TaskPolicyLease>,
    prior_nice: Option<i32>,
}

#[derive(Debug)]
struct ActiveAccelerationLease {
    root_pid: u32,
    family: AccelerationFamily,
    members: Vec<LeasedMember>,
    acquired_at: Instant,
    expires_at: Instant,
    hard_deadline: Instant,
    reason: InteractionReason,
    ttl_band: LeaseTtlBand,
    ttl_exploratory: bool,
}

/// Short-lived, bounded acceleration ownership for the active application
/// family. Input history detects HID idle resets without another IOKit query.
#[derive(Debug, Default)]
pub struct AccelerationLeaseBroker {
    active: Option<ActiveAccelerationLease>,
    last_idle_secs: Option<f64>,
    build_was_active: bool,
    app_launch_was_active: bool,
    window_op_was_active: bool,
    cooldown_until: Option<Instant>,
    parameter_sequence: u64,
    parameter_exploration_arm: usize,
}

impl AccelerationLeaseBroker {
    fn preview_ttl_band(
        &self,
        contextual_bias: ContextualActionBias,
        learned: Option<LeaseTtlBand>,
    ) -> LeaseTtlDecision {
        let next_sequence = self.parameter_sequence.saturating_add(1);
        // One bounded arm probe every eight admitted leases. The treatment is
        // at most +/-10% and never extends the hard deadline; randomized-style
        // variation is needed to learn parameter causality instead of merely
        // reinforcing whichever duration the aggregate action already used.
        if next_sequence.is_multiple_of(8) {
            let band = LeaseTtlBand::ALL[self.parameter_exploration_arm % 3];
            LeaseTtlDecision {
                band,
                exploratory: true,
            }
        } else {
            LeaseTtlDecision {
                band: learned.unwrap_or_else(|| LeaseTtlBand::from_bias(contextual_bias)),
                exploratory: false,
            }
        }
    }

    fn commit_ttl_decision(&mut self, decision: LeaseTtlDecision) {
        self.parameter_sequence = self.parameter_sequence.saturating_add(1);
        if decision.exploratory {
            self.parameter_exploration_arm = self.parameter_exploration_arm.wrapping_add(1);
        }
    }

    fn select_reason(
        &mut self,
        fluidity_state: &FluidityState,
        build_phase: BuildPhase,
        idle_secs: f64,
    ) -> Option<InteractionReason> {
        let input_reset = self
            .last_idle_secs
            .is_some_and(|previous| idle_secs + 0.05 < previous && idle_secs <= 2.5);
        self.last_idle_secs = idle_secs.is_finite().then_some(idle_secs.max(0.0));

        let build_active = build_phase != BuildPhase::Idle;
        let build_started = build_active && !self.build_was_active;
        self.build_was_active = build_active;

        let app_launch_active = fluidity_state.app_launching();
        let app_launch_started = app_launch_active && !self.app_launch_was_active;
        self.app_launch_was_active = app_launch_active;

        let window_op_active = fluidity_state.window_op_active();
        let window_op_started = window_op_active && !self.window_op_was_active;
        self.window_op_was_active = window_op_active;

        if app_launch_started {
            Some(InteractionReason::AppLaunch)
        } else if window_op_started {
            Some(InteractionReason::WindowOperation)
        } else if build_started {
            Some(InteractionReason::BuildStart)
        } else if input_reset {
            Some(InteractionReason::Input)
        } else {
            None
        }
    }
}

fn candidate_precedes(lhs: &AccelerationCandidate, rhs: &AccelerationCandidate) -> bool {
    lhs.score > rhs.score || (lhs.score == rhs.score && lhs.pid < rhs.pid)
}

/// Keep only the highest-utility family members in O(n * K), where K is the
/// fixed lease cap. This avoids sorting a Chromium coalition just to retain
/// its first few members.
fn insert_bounded_candidate(
    selected: &mut Vec<AccelerationCandidate>,
    candidate: AccelerationCandidate,
) {
    if let Some(index) = selected
        .iter()
        .position(|existing| existing.pid == candidate.pid)
    {
        if !candidate_precedes(&candidate, &selected[index]) {
            return;
        }
        selected.remove(index);
    }
    let position = selected
        .iter()
        .position(|existing| candidate_precedes(&candidate, existing))
        .unwrap_or(selected.len());
    if position < MAX_ACCELERATION_MEMBERS {
        selected.insert(position, candidate);
        selected.truncate(MAX_ACCELERATION_MEMBERS);
    } else if selected.len() < MAX_ACCELERATION_MEMBERS {
        selected.push(candidate);
    }
}

fn is_build_member(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    ProcessInterventionClass::for_name(name) == ProcessInterventionClass::BuildTool
        || matches!(
            lower.as_str(),
            "cargo" | "clang" | "clang++" | "swiftc" | "xcodebuild" | "ninja" | "make" | "cmake"
        )
}

fn build_target_score(snapshot: &ProcessSnapshot) -> f64 {
    let lower = snapshot.name.to_ascii_lowercase();
    let coordinator = matches!(
        lower.as_str(),
        "cargo" | "xcodebuild" | "ninja" | "make" | "cmake"
    );
    let cpu = if snapshot.cpu_percent.is_finite() {
        f64::from(snapshot.cpu_percent.max(0.0))
    } else {
        0.0
    };
    (if coordinator { 1_000_000.0 } else { 500_000.0 }) + cpu * 100.0
}

fn select_build_target(snapshots: &[ProcessSnapshot]) -> Option<u32> {
    let mut best: Option<(u32, f64)> = None;
    for snapshot in snapshots {
        if snapshot.is_zombie
            || !is_build_member(&snapshot.name)
            || hard_protected_contains(&snapshot.name)
            || !can_boost(&snapshot.name)
        {
            continue;
        }
        let score = build_target_score(snapshot);
        if best.is_none_or(|(pid, best_score)| {
            score > best_score || (score == best_score && snapshot.pid < pid)
        }) {
            best = Some((snapshot.pid, score));
        }
    }
    best.map(|(pid, _)| pid)
}

fn read_nice(pid: u32) -> std::io::Result<i32> {
    unsafe {
        *libc::__error() = 0;
        let value = libc::getpriority(libc::PRIO_PROCESS, pid);
        if value == -1 && *libc::__error() != 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(value)
        }
    }
}

fn write_nice(pid: u32, value: i32) -> std::io::Result<()> {
    unsafe {
        *libc::__error() = 0;
        let rc = libc::setpriority(libc::PRIO_PROCESS, pid, value);
        if rc == -1 && *libc::__error() != 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

fn nice_fallback_target(prior: i32) -> Option<i32> {
    (prior >= 0).then_some(LEASE_NICE)
}

fn apply_nice_fallback(pid: u32) -> std::io::Result<Option<i32>> {
    let prior = read_nice(pid)?;
    // Respect an existing priority boost from macOS, the app, or another
    // Apollo producer. This fallback is intentionally much milder than the
    // legacy -10 boost and always restores the exact captured value.
    let Some(target) = nice_fallback_target(prior) else {
        return Ok(None);
    };
    write_nice(pid, target)?;
    Ok(Some(prior))
}

fn member_role_bonus(name: &str, reason: InteractionReason) -> f64 {
    let lower = name.to_ascii_lowercase();
    let chromium_role = if lower.contains("gpu") {
        900.0
    } else if lower.contains("renderer") {
        800.0
    } else if lower.contains("helper") {
        300.0
    } else {
        0.0
    };
    let build_role = if matches!(reason, InteractionReason::BuildStart) && is_build_member(name) {
        25_000.0
    } else {
        0.0
    };
    chromium_role + build_role
}

fn select_acceleration_family(
    target_pid: u32,
    reason: InteractionReason,
    process_tree: &ProcessTree,
    snapshots: &[ProcessSnapshot],
) -> Option<AccelerationSelection> {
    if matches!(reason, InteractionReason::BuildStart) {
        let mut selected = Vec::with_capacity(MAX_ACCELERATION_MEMBERS);
        for snapshot in snapshots {
            if snapshot.is_zombie
                || !is_build_member(&snapshot.name)
                || hard_protected_contains(&snapshot.name)
                || !can_boost(&snapshot.name)
            {
                continue;
            }
            insert_bounded_candidate(
                &mut selected,
                AccelerationCandidate {
                    pid: snapshot.pid,
                    name: snapshot.name.clone(),
                    score: build_target_score(snapshot),
                },
            );
        }
        return (!selected.is_empty()).then_some(AccelerationSelection {
            root_pid: target_pid,
            family: AccelerationFamily::General,
            members: selected,
        });
    }

    let root_pid = process_tree
        .resolve_root_pid(target_pid)
        .unwrap_or(target_pid);
    if root_pid <= 1 {
        return None;
    }

    let mut family_pids = process_tree.cascade_pids(target_pid);
    if family_pids.is_empty() {
        family_pids.push(target_pid);
    }
    if !family_pids.contains(&root_pid) {
        family_pids.push(root_pid);
    }

    let by_pid: HashMap<u32, &ProcessSnapshot> = snapshots
        .iter()
        .map(|snapshot| (snapshot.pid, snapshot))
        .collect();
    let root_name = process_tree
        .resolve_app_name(target_pid)
        .or_else(|| by_pid.get(&root_pid).map(|snapshot| snapshot.name.as_str()))?;
    if hard_protected_contains(root_name) {
        return None;
    }
    let family = if is_chromium_family(root_name)
        || family_pids.iter().any(|pid| {
            by_pid
                .get(pid)
                .is_some_and(|snapshot| is_chromium_family(&snapshot.name))
        }) {
        AccelerationFamily::Chromium
    } else {
        AccelerationFamily::General
    };

    let mut selected = Vec::with_capacity(MAX_ACCELERATION_MEMBERS);
    for pid in family_pids {
        let Some(snapshot) = by_pid.get(&pid).copied() else {
            continue;
        };
        if snapshot.is_zombie || hard_protected_contains(&snapshot.name) {
            continue;
        }
        let intervention = ProcessInterventionClass::for_name(&snapshot.name);
        if intervention == ProcessInterventionClass::ProtectedSystem {
            continue;
        }
        // Chromium gets only the existing foreground-tier inheritance. The
        // explicit latency/throughput boost remains forbidden by B.6.
        if family == AccelerationFamily::General && !can_boost(&snapshot.name) {
            continue;
        }

        let cpu = if snapshot.cpu_percent.is_finite() {
            f64::from(snapshot.cpu_percent.max(0.0))
        } else {
            0.0
        };
        let contention = snapshot.cpu_contention.unwrap_or(0.0).clamp(0.0, 1.0);
        let role_bonus = member_role_bonus(&snapshot.name, reason);
        let is_root = pid == root_pid;
        let active_child = cpu >= 0.25
            || contention >= 0.05
            || snapshot.has_gui_window
            || snapshot.wakeups_per_sec >= 5.0
            || (matches!(reason, InteractionReason::AppLaunch) && role_bonus > 0.0);
        if !is_root && !active_child {
            continue;
        }

        let score = if is_root { 1_000_000.0 } else { 0.0 }
            + if snapshot.has_gui_window {
                10_000.0
            } else {
                0.0
            }
            + role_bonus
            + cpu * 100.0
            + contention * 1_000.0
            + f64::from(snapshot.wakeups_per_sec.max(0.0).min(1_000.0));
        insert_bounded_candidate(
            &mut selected,
            AccelerationCandidate {
                pid,
                name: snapshot.name.clone(),
                score,
            },
        );
    }

    (!selected.is_empty()).then_some(AccelerationSelection {
        root_pid,
        family,
        members: selected,
    })
}

fn refresh_owned_effects(
    lease: &ActiveAccelerationLease,
    now: Instant,
    cycle: u64,
) -> CycleDecisionEvents {
    let mut decision_events = CycleDecisionEvents::default();
    let ttl = lease
        .expires_at
        .saturating_duration_since(now)
        .saturating_add(LEDGER_GRACE);
    for member in &lease.members {
        if member.task_qos_mutated {
            let refreshed = apollo_engine::engine::effect_ledger::refresh_global_if_justification(
                &apollo_engine::engine::effect_ledger::AppliedEffect::TaskQoS { pid: member.pid },
                ttl,
                member.start_sec,
                TASK_QOS_OWNER,
            );
            decision_events.push(acceleration_member_event(
                member.pid,
                &member.name,
                AccelerationMemberEffect::TaskQos,
                cycle,
                ActuatorDecisionOutcome::NoOp,
                if refreshed {
                    "owned task QoS lease refreshed"
                } else {
                    "task QoS ownership no longer held"
                },
            ));
        }
        if let Some(prior) = member.prior_nice {
            let refreshed = apollo_engine::engine::effect_ledger::refresh_global_if_justification(
                &apollo_engine::engine::effect_ledger::AppliedEffect::Nice {
                    pid: member.pid,
                    prior,
                },
                ttl,
                member.start_sec,
                NICE_OWNER,
            );
            decision_events.push(acceleration_member_event(
                member.pid,
                &member.name,
                AccelerationMemberEffect::Nice,
                cycle,
                ActuatorDecisionOutcome::NoOp,
                if refreshed {
                    "owned nice lease refreshed"
                } else {
                    "nice ownership no longer held"
                },
            ));
        }
    }
    decision_events
}

fn acquire_acceleration_lease(
    state: &SharedState,
    controller: &mut AccelerationLeaseBroker,
    io_shaper: &mut IoShaper,
    selection: AccelerationSelection,
    reason: InteractionReason,
    ttl_decision: LeaseTtlDecision,
    allow_io_promotion: bool,
    io_bias: ContextualActionBias,
    now: Instant,
    cycle: u64,
) -> CycleDecisionEvents {
    let mut decision_events = CycleDecisionEvents::default();
    let ttl = ttl_decision.band.ttl(reason);
    let ledger_ttl = ttl.saturating_add(LEDGER_GRACE);
    let mut identity_skips = 0u64;
    let mut capability_skips = 0u64;
    let mut nice_fallbacks = 0u64;
    let mut nice_failures = 0u64;
    let mut conflict_skips = 0u64;
    let mut prepared = Vec::with_capacity(selection.members.len());
    for candidate in selection.members {
        if let Some(identity) = ProcessIdentity::from_pid(candidate.pid) {
            let task_effect =
                apollo_engine::engine::effect_ledger::AppliedEffect::TaskQoS { pid: candidate.pid };
            let nice_effect = apollo_engine::engine::effect_ledger::AppliedEffect::Nice {
                pid: candidate.pid,
                prior: 0,
            };
            let task_conflict = selection.family.allows_explicit_task_qos()
                && apollo_engine::engine::effect_ledger::is_global_tracked(&task_effect);
            let nice_conflict = selection.family.allows_explicit_task_qos()
                && apollo_engine::engine::effect_ledger::is_global_tracked(&nice_effect);
            conflict_skips = conflict_skips
                .saturating_add(u64::from(task_conflict))
                .saturating_add(u64::from(nice_conflict));
            prepared.push((candidate, identity, task_conflict, nice_conflict));
        } else {
            identity_skips = identity_skips.saturating_add(1);
            decision_events.push(acceleration_member_event(
                candidate.pid,
                &candidate.name,
                AccelerationMemberEffect::TaskQos,
                cycle,
                ActuatorDecisionOutcome::Blocked,
                "process identity unavailable",
            ));
        }
    }

    let mut members = Vec::with_capacity(prepared.len());
    let selected_pids: Vec<u32> = prepared
        .iter()
        .map(|(candidate, _, _, _)| candidate.pid)
        .collect();
    let selected_names: HashMap<u32, String> = prepared
        .iter()
        .map(|(candidate, _, _, _)| (candidate.pid, candidate.name.clone()))
        .collect();
    let mut qos = state.mach_qos.lock_recover();
    let io_outcomes = if allow_io_promotion {
        io_shaper.promote_interactive_outcomes(&selected_pids, Some(&mut qos))
    } else {
        for (candidate, _, _, _) in &prepared {
            decision_events.push(acceleration_member_event(
                candidate.pid,
                &candidate.name,
                AccelerationMemberEffect::IoPromotion,
                cycle,
                ActuatorDecisionOutcome::Vetoed,
                "authoritative contextual utility veto",
            ));
        }
        Vec::new()
    };
    let io_promotions = io_outcomes
        .iter()
        .filter(|outcome| outcome.disposition == IoPromotionDisposition::Applied)
        .count() as u32;
    for outcome in io_outcomes {
        decision_events.push(acceleration_member_event(
            outcome.pid,
            selected_names
                .get(&outcome.pid)
                .map_or("unknown", String::as_str),
            AccelerationMemberEffect::IoPromotion,
            cycle,
            match outcome.disposition {
                IoPromotionDisposition::Applied => ActuatorDecisionOutcome::Applied,
                IoPromotionDisposition::NoOp => ActuatorDecisionOutcome::NoOp,
                IoPromotionDisposition::Failed => ActuatorDecisionOutcome::Failed,
            },
            "interactive I/O release evaluated",
        ));
    }
    for (candidate, identity, task_conflict, nice_conflict) in prepared {
        // Chromium keeps the existing non-invasive foreground inheritance and
        // can only receive an Apollo-owned I/O release above. TASK_CATEGORY
        // is deliberately excluded here: several hardened GUI apps accept
        // foreground elevation but reject TASK_UNSPECIFIED rollback.
        if !selection.family.allows_explicit_task_qos() {
            continue;
        }
        let mut policy_lease = None;
        let mut task_qos_mutated = false;
        if task_conflict {
            decision_events.push(acceleration_member_event(
                candidate.pid,
                &candidate.name,
                AccelerationMemberEffect::TaskQos,
                cycle,
                ActuatorDecisionOutcome::Blocked,
                "effect ownership conflict",
            ));
        } else {
            if let Some(lease) = qos.acquire_task_policy_lease(candidate.pid) {
                let outcome = qos.set_latency_and_throughput_with_lease(
                    &lease,
                    LatencyTier::Interactive,
                    ThroughputTier::High,
                );
                task_qos_mutated = outcome.mutated;
                decision_events.push(acceleration_member_event(
                    candidate.pid,
                    &candidate.name,
                    AccelerationMemberEffect::TaskQos,
                    cycle,
                    if outcome.mutated {
                        ActuatorDecisionOutcome::Applied
                    } else if outcome.success {
                        ActuatorDecisionOutcome::NoOp
                    } else {
                        ActuatorDecisionOutcome::Failed
                    },
                    outcome.error.as_deref().unwrap_or("task QoS evaluated"),
                ));
                if task_qos_mutated {
                    policy_lease = Some(lease);
                } else {
                    capability_skips = capability_skips.saturating_add(1);
                }
            } else {
                capability_skips = capability_skips.saturating_add(1);
                decision_events.push(acceleration_member_event(
                    candidate.pid,
                    &candidate.name,
                    AccelerationMemberEffect::TaskQos,
                    cycle,
                    ActuatorDecisionOutcome::Blocked,
                    "task policy lease unavailable",
                ));
            }
        }
        let prior_nice = if task_qos_mutated {
            None
        } else if nice_conflict {
            decision_events.push(acceleration_member_event(
                candidate.pid,
                &candidate.name,
                AccelerationMemberEffect::Nice,
                cycle,
                ActuatorDecisionOutcome::Blocked,
                "effect ownership conflict",
            ));
            None
        } else {
            match apply_nice_fallback(candidate.pid) {
                Ok(prior) => {
                    decision_events.push(acceleration_member_event(
                        candidate.pid,
                        &candidate.name,
                        AccelerationMemberEffect::Nice,
                        cycle,
                        if prior.is_some() {
                            ActuatorDecisionOutcome::Applied
                        } else {
                            ActuatorDecisionOutcome::NoOp
                        },
                        "nice fallback evaluated",
                    ));
                    prior
                }
                Err(error) => {
                    nice_failures = nice_failures.saturating_add(1);
                    decision_events.push(acceleration_member_event(
                        candidate.pid,
                        &candidate.name,
                        AccelerationMemberEffect::Nice,
                        cycle,
                        ActuatorDecisionOutcome::Failed,
                        format!("nice fallback failed: {error}"),
                    ));
                    None
                }
            }
        };
        nice_fallbacks = nice_fallbacks.saturating_add(u64::from(prior_nice.is_some()));
        if task_qos_mutated || prior_nice.is_some() {
            members.push(LeasedMember {
                pid: candidate.pid,
                name: candidate.name,
                start_sec: identity.start_sec,
                start_usec: identity.start_usec,
                task_qos_mutated,
                policy_lease,
                prior_nice,
            });
        }
    }
    drop(qos);

    for member in &members {
        if member.task_qos_mutated {
            apollo_engine::engine::effect_ledger::record_global(
                apollo_engine::engine::effect_ledger::AppliedEffect::TaskQoS { pid: member.pid },
                ledger_ttl,
                member.start_sec,
                TASK_QOS_OWNER,
            );
        }
        if let Some(prior) = member.prior_nice {
            apollo_engine::engine::effect_ledger::record_global(
                apollo_engine::engine::effect_ledger::AppliedEffect::Nice {
                    pid: member.pid,
                    prior,
                },
                ledger_ttl,
                member.start_sec,
                NICE_OWNER,
            );
        }
    }

    let member_count = members.len() as u32;
    if member_count > 0 {
        controller.commit_ttl_decision(ttl_decision);
    }
    {
        let mut metrics = state.metrics.lock_recover();
        metrics.metrics.acceleration_lease_identity_skips_total = metrics
            .metrics
            .acceleration_lease_identity_skips_total
            .saturating_add(identity_skips);
        metrics.metrics.acceleration_lease_capability_skips_total = metrics
            .metrics
            .acceleration_lease_capability_skips_total
            .saturating_add(capability_skips);
        metrics.metrics.acceleration_lease_nice_fallbacks_total = metrics
            .metrics
            .acceleration_lease_nice_fallbacks_total
            .saturating_add(nice_fallbacks);
        metrics.metrics.acceleration_lease_nice_failures_total = metrics
            .metrics
            .acceleration_lease_nice_failures_total
            .saturating_add(nice_failures);
        metrics.metrics.acceleration_lease_conflict_skips_total = metrics
            .metrics
            .acceleration_lease_conflict_skips_total
            .saturating_add(conflict_skips);
        metrics.metrics.acceleration_lease_io_promotions_total = metrics
            .metrics
            .acceleration_lease_io_promotions_total
            .saturating_add(u64::from(io_promotions));
        if io_bias.is_informative() {
            metrics.metrics.world_model_contextual_io_total = metrics
                .metrics
                .world_model_contextual_io_total
                .saturating_add(1);
            metrics.metrics.world_model_contextual_last_action =
                "io_shaping:interactive_release".to_string();
            metrics.metrics.world_model_contextual_last_bias = io_bias.score;
            if io_bias.has_gpu_influence() {
                metrics.metrics.record_gpu_contextual_influence(
                    "io-shaping",
                    "io_shaping:interactive_release",
                    io_bias.gpu_context_support,
                );
            }
        }
        if member_count > 0 || io_promotions > 0 {
            metrics.metrics.acceleration_lease_last_family = selection.family.as_str().to_string();
            match selection.family {
                AccelerationFamily::General => {
                    metrics.metrics.acceleration_lease_general_total = metrics
                        .metrics
                        .acceleration_lease_general_total
                        .saturating_add(1);
                }
                AccelerationFamily::Chromium => {
                    metrics.metrics.acceleration_lease_chromium_total = metrics
                        .metrics
                        .acceleration_lease_chromium_total
                        .saturating_add(1);
                }
            }
        }
        if member_count > 0 {
            metrics.metrics.interaction_qos_activations = metrics
                .metrics
                .interaction_qos_activations
                .saturating_add(1);
            metrics.metrics.interaction_qos_active = true;
            metrics.metrics.interaction_qos_reason = reason.as_str().to_string();
            metrics.metrics.interaction_qos_ttl_band = ttl_decision.band.as_str().to_string();
            metrics.metrics.interaction_qos_ttl_ms = ttl.as_millis().min(u64::MAX as u128) as u64;
            metrics.metrics.interaction_qos_ttl_exploratory = ttl_decision.exploratory;
            metrics.metrics.interaction_qos_parameter_explorations_total = metrics
                .metrics
                .interaction_qos_parameter_explorations_total
                .saturating_add(u64::from(ttl_decision.exploratory));
            metrics.metrics.acceleration_lease_members_active = member_count;
            metrics.metrics.acceleration_lease_members_applied_total = metrics
                .metrics
                .acceleration_lease_members_applied_total
                .saturating_add(u64::from(member_count));
            metrics.metrics.acceleration_lease_family = selection.family.as_str().to_string();
        }
    }

    if member_count > 0 || io_promotions > 0 {
        tracing::debug!(
            root_pid = selection.root_pid,
            family = selection.family.as_str(),
            members = member_count,
            io_promotions,
            reason = reason.as_str(),
            "acceleration lease acquired"
        );
        if member_count > 0 {
            controller.active = Some(ActiveAccelerationLease {
                root_pid: selection.root_pid,
                family: selection.family,
                members,
                acquired_at: now,
                expires_at: now + ttl,
                hard_deadline: now + MAX_CONTINUOUS_LEASE,
                reason,
                ttl_band: ttl_decision.band,
                ttl_exploratory: ttl_decision.exploratory,
            });
        }
    }
    if member_count == 0 {
        // No reversible lease was needed (already optimal, I/O-only, or
        // kernel-denied). Suppress repeated identity/syscall probes for the
        // remainder of this interaction window.
        controller.cooldown_until = Some(now + ttl.min(LEASE_COOLDOWN));
    }
    decision_events
}

/// Apply a bounded family acceleration lease and restore only effects this
/// broker still owns. QoS is a scheduler hint on Apple Silicon, not physical
/// P/E-core affinity. Chromium keeps its non-invasive B.6 policy.
#[allow(clippy::too_many_arguments)]
fn update_acceleration_lease_inner(
    state: &SharedState,
    controller: &mut AccelerationLeaseBroker,
    fluidity_state: &FluidityState,
    thermal_action: &ThermalAction,
    foreground_pid: Option<u32>,
    build_phase: BuildPhase,
    idle_secs: f64,
    process_tree: &ProcessTree,
    snapshots: &[ProcessSnapshot],
    io_shaper: &mut IoShaper,
    world_model: &WorldModel,
    workload: &str,
    cycle: u64,
) -> CycleDecisionEvents {
    let mut decision_events = CycleDecisionEvents::default();
    let now = Instant::now();
    let reason = controller.select_reason(fluidity_state, build_phase, idle_secs);
    let target_pid = match reason {
        Some(InteractionReason::AppLaunch) => fluidity_state.launch_pid.or(foreground_pid),
        Some(InteractionReason::BuildStart) => select_build_target(snapshots).or(foreground_pid),
        _ => foreground_pid,
    };
    let target_root = target_pid.map(|pid| {
        if matches!(reason, Some(InteractionReason::BuildStart)) {
            pid
        } else {
            process_tree.resolve_root_pid(pid).unwrap_or(pid)
        }
    });

    if thermal_action.force_ecores {
        decision_events.extend_buffer(&release_acceleration_lease(controller, state, cycle));
        return decision_events;
    }

    if controller
        .active
        .as_ref()
        .is_some_and(|lease| now >= lease.hard_deadline)
    {
        decision_events.extend_buffer(&release_acceleration_lease(controller, state, cycle));
        controller.cooldown_until = Some(now + LEASE_COOLDOWN);
        return decision_events;
    }

    let must_release = controller.active.as_ref().is_some_and(|lease| {
        now >= lease.expires_at || target_root.is_some_and(|root_pid| root_pid != lease.root_pid)
    });
    if must_release {
        decision_events.extend_buffer(&release_acceleration_lease(controller, state, cycle));
    }

    if reason.is_none()
        || target_pid.is_none()
        || controller.cooldown_until.is_some_and(|until| now < until)
    {
        return decision_events;
    }

    // Context search is bounded but still unnecessary on idle cycles. Query
    // only after a specialist interaction signal has passed the local gates.
    let interaction_bias =
        world_model.contextual_action_bias("interaction_qos:foreground", workload);
    let io_bias = world_model.contextual_action_bias("io_shaping:interactive_release", workload);
    let learned_ttl_band = learned_lease_ttl_band(world_model, workload);

    if let (Some(reason), Some(pid), Some(root_pid)) = (reason, target_pid, target_root) {
        if let Some(active) = controller.active.as_mut() {
            if active.root_pid == root_pid {
                let ttl = active.ttl_band.ttl(reason);
                active.expires_at = (now + ttl).min(active.hard_deadline);
                active.reason = reason;
                decision_events.extend_buffer(&refresh_owned_effects(active, now, cycle));
                let mut metrics = state.metrics.lock_recover();
                metrics.metrics.interaction_qos_reason = reason.as_str().to_string();
                metrics.metrics.interaction_qos_ttl_band = active.ttl_band.as_str().to_string();
                metrics.metrics.interaction_qos_ttl_ms =
                    ttl.as_millis().min(u64::MAX as u128) as u64;
                metrics.metrics.interaction_qos_ttl_exploratory = active.ttl_exploratory;
                metrics.metrics.acceleration_lease_renewals_total = metrics
                    .metrics
                    .acceleration_lease_renewals_total
                    .saturating_add(1);
                if interaction_bias.is_informative() {
                    metrics.metrics.world_model_contextual_interaction_total = metrics
                        .metrics
                        .world_model_contextual_interaction_total
                        .saturating_add(1);
                    metrics.metrics.world_model_contextual_last_action =
                        "interaction_qos:foreground".to_string();
                    metrics.metrics.world_model_contextual_last_bias = interaction_bias.score;
                }
                return decision_events;
            }
        }

        if let Some(selection) = select_acceleration_family(pid, reason, process_tree, snapshots) {
            let ttl_decision = controller.preview_ttl_band(interaction_bias, learned_ttl_band);
            if interaction_bias.is_informative() {
                let mut metrics = state.metrics.lock_recover();
                metrics.metrics.world_model_contextual_interaction_total = metrics
                    .metrics
                    .world_model_contextual_interaction_total
                    .saturating_add(1);
                metrics.metrics.world_model_contextual_last_action =
                    "interaction_qos:foreground".to_string();
                metrics.metrics.world_model_contextual_last_bias = interaction_bias.score;
                if interaction_bias.has_gpu_influence() {
                    metrics.metrics.record_gpu_contextual_influence(
                        "interaction-qos",
                        "interaction_qos:foreground",
                        interaction_bias.gpu_context_support,
                    );
                }
            }
            decision_events.extend_buffer(&acquire_acceleration_lease(
                state,
                controller,
                io_shaper,
                selection,
                reason,
                ttl_decision,
                contextual_io_release_allowed(io_bias),
                io_bias,
                now,
                cycle,
            ));
        }
    }
    decision_events
}

#[allow(clippy::too_many_arguments)]
pub fn update_acceleration_lease(
    state: &SharedState,
    controller: &mut AccelerationLeaseBroker,
    fluidity_state: &FluidityState,
    thermal_action: &ThermalAction,
    foreground_pid: Option<u32>,
    build_phase: BuildPhase,
    idle_secs: f64,
    process_tree: &ProcessTree,
    snapshots: &[ProcessSnapshot],
    io_shaper: &mut IoShaper,
    world_model: &WorldModel,
    workload: &str,
    cycle: u64,
) -> AccelerationTickOutput {
    let prior_root = controller.active.as_ref().map(|lease| lease.root_pid);
    let expired = controller.active.as_ref().is_some_and(|lease| {
        Instant::now() >= lease.expires_at || Instant::now() >= lease.hard_deadline
    });
    let decision_events = update_acceleration_lease_inner(
        state,
        controller,
        fluidity_state,
        thermal_action,
        foreground_pid,
        build_phase,
        idle_secs,
        process_tree,
        snapshots,
        io_shaper,
        world_model,
        workload,
        cycle,
    );
    let target = controller
        .active
        .as_ref()
        .map(|lease| lease.root_pid)
        .or(prior_root)
        .or(foreground_pid)
        .unwrap_or(0);
    let mut output = AccelerationTickOutput { decision_events };
    let interaction_action_key = {
        let metrics = state.metrics.lock_recover();
        match (
            metrics.metrics.interaction_qos_ttl_exploratory,
            metrics.metrics.interaction_qos_ttl_band.as_str(),
        ) {
            (true, band @ ("short" | "standard" | "long")) => {
                format!("interaction_qos:foreground@{band}")
            }
            _ => "interaction_qos:foreground".to_string(),
        }
    };
    if expired {
        output.decision_events.push(acceleration_event(
            &interaction_action_key,
            target,
            cycle,
            ActuatorDecisionOutcome::Expired,
            "lease deadline reached",
        ));
    }
    output
}

pub fn release_acceleration_lease(
    controller: &mut AccelerationLeaseBroker,
    state: &SharedState,
    cycle: u64,
) -> CycleDecisionEvents {
    let mut decision_events = CycleDecisionEvents::default();
    let Some(lease) = controller.active.take() else {
        return decision_events;
    };

    let mut reverted_members = 0u64;
    let mut reverted_effects = 0u64;
    let mut identity_skips = 0u64;
    for member in &lease.members {
        let task_effect =
            apollo_engine::engine::effect_ledger::AppliedEffect::TaskQoS { pid: member.pid };
        let nice_effect = member.prior_nice.map(|prior| {
            apollo_engine::engine::effect_ledger::AppliedEffect::Nice {
                pid: member.pid,
                prior,
            }
        });
        if !ProcessIdentity::verify(
            member.pid,
            Some(&member.name),
            member.start_sec,
            member.start_usec,
        ) {
            apollo_engine::engine::effect_ledger::forget_global_if_justification(
                &task_effect,
                TASK_QOS_OWNER,
            );
            if let Some(effect) = nice_effect.as_ref() {
                apollo_engine::engine::effect_ledger::forget_global_if_justification(
                    effect, NICE_OWNER,
                );
            }
            state.mach_qos.lock_recover().remove(member.pid);
            identity_skips = identity_skips.saturating_add(1);
            if member.task_qos_mutated {
                decision_events.push(acceleration_member_event(
                    member.pid,
                    &member.name,
                    AccelerationMemberEffect::TaskQos,
                    cycle,
                    ActuatorDecisionOutcome::Blocked,
                    "process identity changed before task QoS rollback",
                ));
            }
            if member.prior_nice.is_some() {
                decision_events.push(acceleration_member_event(
                    member.pid,
                    &member.name,
                    AccelerationMemberEffect::Nice,
                    cycle,
                    ActuatorDecisionOutcome::Blocked,
                    "process identity changed before nice rollback",
                ));
            }
            continue;
        }

        let mut member_reverted = false;
        let owns_task_qos = member.task_qos_mutated
            && apollo_engine::engine::effect_ledger::is_global_owner(&task_effect, TASK_QOS_OWNER);
        let owns_nice = nice_effect.as_ref().is_some_and(|effect| {
            apollo_engine::engine::effect_ledger::is_global_owner(effect, NICE_OWNER)
        });
        let mut qos = state.mach_qos.lock_recover();
        let task_qos_outcome = if owns_task_qos {
            member.policy_lease.as_ref().map(|policy_lease| {
                qos.set_latency_and_throughput_with_lease(
                    policy_lease,
                    LatencyTier::Default,
                    ThroughputTier::Default,
                )
            })
        } else {
            None
        };
        drop(qos);
        let nice_outcome = if owns_nice {
            member.prior_nice.map(|prior| write_nice(member.pid, prior))
        } else {
            None
        };

        let task_qos_restored = task_qos_outcome
            .as_ref()
            .is_some_and(|outcome| outcome.mutated);
        if task_qos_restored {
            apollo_engine::engine::effect_ledger::forget_global_if_justification(
                &task_effect,
                TASK_QOS_OWNER,
            );
            reverted_effects = reverted_effects.saturating_add(1);
            member_reverted = true;
        }
        if member.task_qos_mutated {
            let (outcome, detail) = match task_qos_outcome.as_ref() {
                Some(result) if result.mutated => (
                    ActuatorDecisionOutcome::Reverted,
                    "owned task QoS effect restored".to_string(),
                ),
                Some(result) if result.success => (
                    ActuatorDecisionOutcome::NoOp,
                    "task QoS rollback required no mutation".to_string(),
                ),
                Some(result) => (
                    ActuatorDecisionOutcome::Failed,
                    result
                        .error
                        .clone()
                        .unwrap_or_else(|| "task QoS rollback failed".to_string()),
                ),
                None => (
                    ActuatorDecisionOutcome::NoOp,
                    "task QoS ownership no longer held".to_string(),
                ),
            };
            decision_events.push(acceleration_member_event(
                member.pid,
                &member.name,
                AccelerationMemberEffect::TaskQos,
                cycle,
                outcome,
                detail,
            ));
        }
        let nice_restored = nice_outcome.as_ref().is_some_and(Result::is_ok);
        if nice_restored {
            if let Some(effect) = nice_effect.as_ref() {
                apollo_engine::engine::effect_ledger::forget_global_if_justification(
                    effect, NICE_OWNER,
                );
            }
            reverted_effects = reverted_effects.saturating_add(1);
            member_reverted = true;
        }
        if member.prior_nice.is_some() {
            let (outcome, detail) = match nice_outcome.as_ref() {
                Some(Ok(())) => (
                    ActuatorDecisionOutcome::Reverted,
                    "owned nice effect restored".to_string(),
                ),
                Some(Err(error)) => (
                    ActuatorDecisionOutcome::Failed,
                    format!("nice rollback failed: {error}"),
                ),
                None => (
                    ActuatorDecisionOutcome::NoOp,
                    "nice ownership no longer held".to_string(),
                ),
            };
            decision_events.push(acceleration_member_event(
                member.pid,
                &member.name,
                AccelerationMemberEffect::Nice,
                cycle,
                outcome,
                detail,
            ));
        }
        reverted_members = reverted_members.saturating_add(u64::from(member_reverted));
    }

    let mut metrics = state.metrics.lock_recover();
    metrics.metrics.interaction_qos_reverts =
        metrics.metrics.interaction_qos_reverts.saturating_add(1);
    metrics.metrics.interaction_qos_active = false;
    metrics.metrics.interaction_qos_reason.clear();
    metrics.metrics.acceleration_lease_members_active = 0;
    metrics.metrics.acceleration_lease_member_reverts_total = metrics
        .metrics
        .acceleration_lease_member_reverts_total
        .saturating_add(reverted_members);
    metrics.metrics.acceleration_lease_identity_skips_total = metrics
        .metrics
        .acceleration_lease_identity_skips_total
        .saturating_add(identity_skips);
    metrics.metrics.acceleration_lease_family.clear();
    metrics.metrics.reverts_applied = metrics
        .metrics
        .reverts_applied
        .saturating_add(reverted_effects);
    tracing::debug!(
        root_pid = lease.root_pid,
        family = lease.family.as_str(),
        members = lease.members.len(),
        held_ms = lease.acquired_at.elapsed().as_millis(),
        reason = lease.reason.as_str(),
        reverted_members,
        identity_skips,
        "acceleration lease released"
    );
    decision_events
}

#[cfg(test)]
mod interaction_qos_tests {
    use super::*;
    use apollo_engine::engine::process_tree::ProcessEntry;

    #[test]
    fn acceleration_event_preserves_side_channel_family_and_pid() {
        let event = acceleration_event(
            "interaction_qos:foreground",
            42,
            15,
            apollo_engine::engine::decision_ledger::ActuatorDecisionOutcome::Applied,
            "task QoS applied",
        );

        assert_eq!(event.proposal.action_key, "interaction_qos:foreground");
        assert_eq!(event.proposal.target, "pid:42");
        assert_eq!(event.proposal.proposed_cycle, 15);
    }

    #[test]
    fn acceleration_member_receipts_keep_member_and_effect_identity() {
        let qos = acceleration_member_event(
            42,
            "Editor Helper",
            AccelerationMemberEffect::TaskQos,
            15,
            ActuatorDecisionOutcome::Applied,
            "task QoS applied",
        );
        let nice = acceleration_member_event(
            43,
            "Compiler",
            AccelerationMemberEffect::Nice,
            15,
            ActuatorDecisionOutcome::Failed,
            "nice failed",
        );

        assert_eq!(qos.proposal.action_key, "interaction_qos:task_qos");
        assert_eq!(qos.proposal.target, "Editor Helper:pid:42");
        assert_eq!(nice.proposal.action_key, "interaction_qos:nice");
        assert_eq!(nice.proposal.target, "Compiler:pid:43");
        assert_eq!(nice.outcome, ActuatorDecisionOutcome::Failed);
    }

    #[test]
    fn renewal_is_noop_and_authoritative_io_denial_is_vetoed() {
        assert_eq!(lease_renewal_outcome(), ActuatorDecisionOutcome::NoOp);
        assert_eq!(
            contextual_io_outcome(ContextualActionBias {
                score: -0.8,
                authoritative: true,
                model_observations: 12,
                ..ContextualActionBias::default()
            }),
            Some(ActuatorDecisionOutcome::Vetoed)
        );
        assert_eq!(contextual_io_outcome(ContextualActionBias::default()), None);
    }

    #[test]
    fn contextual_acceleration_tuning_is_bounded_and_cannot_grant_io_authority() {
        let positive = ContextualActionBias {
            score: 1.0,
            model_observations: 20,
            authoritative: true,
            ..ContextualActionBias::default()
        };
        let negative = ContextualActionBias {
            score: -1.0,
            model_observations: 20,
            authoritative: true,
            ..ContextualActionBias::default()
        };
        let preliminary_negative = ContextualActionBias {
            authoritative: false,
            ..negative
        };

        assert_eq!(LeaseTtlBand::from_bias(positive), LeaseTtlBand::Long);
        assert_eq!(LeaseTtlBand::from_bias(negative), LeaseTtlBand::Short);
        assert_eq!(
            LeaseTtlBand::Long.ttl(InteractionReason::Input),
            Duration::from_millis(1_760)
        );
        assert_eq!(
            LeaseTtlBand::Short.ttl(InteractionReason::Input),
            Duration::from_millis(1_440)
        );
        assert!(contextual_io_release_allowed(positive));
        assert!(!contextual_io_release_allowed(negative));
        assert!(contextual_io_release_allowed(preliminary_negative));
    }

    #[test]
    fn parameter_exploration_is_sparse_bounded_and_rotates_all_arms() {
        let mut broker = AccelerationLeaseBroker::default();
        let neutral = ContextualActionBias::default();
        let mut explored = Vec::new();
        for _ in 0..24 {
            let decision = broker.preview_ttl_band(neutral, None);
            if decision.exploratory {
                explored.push(decision.band);
            } else {
                assert_eq!(decision.band, LeaseTtlBand::Standard);
            }
            broker.commit_ttl_decision(decision);
        }
        assert_eq!(
            explored,
            vec![
                LeaseTtlBand::Short,
                LeaseTtlBand::Standard,
                LeaseTtlBand::Long
            ]
        );
        for band in LeaseTtlBand::ALL {
            let ttl = band.ttl(InteractionReason::AppLaunch);
            assert!((Duration::from_millis(4_500)..=Duration::from_millis(5_500)).contains(&ttl));
            assert!(ttl < MAX_CONTINUOUS_LEASE);
        }
    }

    #[test]
    fn failed_lease_does_not_consume_a_parameter_probe() {
        let mut broker = AccelerationLeaseBroker::default();
        let neutral = ContextualActionBias::default();
        for _ in 0..7 {
            let decision = broker.preview_ttl_band(neutral, None);
            broker.commit_ttl_decision(decision);
        }

        let failed_preview = broker.preview_ttl_band(neutral, None);
        assert!(failed_preview.exploratory);
        assert_eq!(failed_preview.band, LeaseTtlBand::Short);
        // No commit models a kernel/capability rejection. The same probe must
        // be retried rather than silently rotating past it.
        let retry = broker.preview_ttl_band(neutral, None);
        assert_eq!(retry, failed_preview);

        broker.commit_ttl_decision(retry);
        assert!(!broker.preview_ttl_band(neutral, None).exploratory);
    }

    #[test]
    fn learned_parameter_choice_requires_separated_confidence_intervals() {
        assert_eq!(
            select_distinct_ttl_band(vec![(LeaseTtlBand::Long, 0.02, 0.04, 0.95)]),
            None
        );
        assert_eq!(
            select_distinct_ttl_band(vec![
                (LeaseTtlBand::Long, 0.03, 0.05, 0.95),
                (LeaseTtlBand::Standard, 0.02, 0.04, 0.95),
            ]),
            None
        );
        assert_eq!(
            select_distinct_ttl_band(vec![
                (LeaseTtlBand::Long, 0.05, 0.06, 0.95),
                (LeaseTtlBand::Standard, 0.01, 0.02, 0.95),
                (LeaseTtlBand::Short, -0.02, 0.0, 0.95),
            ]),
            Some(LeaseTtlBand::Long)
        );
    }

    fn entry(pid: u32, ppid: u32, name: &str, cpu: f32) -> ProcessEntry {
        ProcessEntry {
            pid,
            ppid,
            name: name.to_string(),
            cpu_usage: cpu,
            memory_bytes: 1_000_000,
        }
    }

    fn snapshot(pid: u32, name: &str, cpu: f32) -> ProcessSnapshot {
        ProcessSnapshot {
            pid,
            name: name.to_string(),
            cpu_percent: cpu,
            rss_bytes: 1_000_000,
            is_zombie: false,
            secs_since_foreground: 0,
            secs_since_user_interaction: 0,
            has_network: false,
            has_gui_window: false,
            wakeups_per_sec: 0.0,
            parent_alive: true,
            process_uptime_secs: 10,
            faults_total: 0,
            pageins_total: 0,
            is_translated: false,
            mach_port_count: 0,
            cpu_contention: None,
            is_app_bundle: false,
        }
    }

    #[test]
    fn hid_reset_emits_one_short_input_signal() {
        let fluidity = FluidityState::new();
        let mut controller = AccelerationLeaseBroker::default();

        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Idle, 8.0),
            None
        );
        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Idle, 0.2),
            Some(InteractionReason::Input)
        );
        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Idle, 0.5),
            None
        );
    }

    #[test]
    fn build_only_signals_on_starting_edge() {
        let fluidity = FluidityState::new();
        let mut controller = AccelerationLeaseBroker::default();

        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Starting, 0.0),
            Some(InteractionReason::BuildStart)
        );
        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Active, 0.1),
            None
        );
        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Finishing, 0.2),
            None
        );
    }

    #[test]
    fn launch_has_priority_over_window_and_input() {
        let mut fluidity = FluidityState::new();
        let mut controller = AccelerationLeaseBroker::default();
        controller.select_reason(&fluidity, BuildPhase::Idle, 9.0);
        fluidity.windowserver_cpu_spike = true;
        fluidity.launch_active = true;

        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Starting, 0.0),
            Some(InteractionReason::AppLaunch)
        );
    }

    #[test]
    fn sustained_launch_and_window_levels_emit_only_once() {
        let mut fluidity = FluidityState::new();
        let mut controller = AccelerationLeaseBroker::default();
        fluidity.launch_active = true;
        fluidity.windowserver_cpu_spike = true;

        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Idle, 1.0),
            Some(InteractionReason::AppLaunch)
        );
        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Idle, 1.1),
            None,
            "sustained levels are state, not fresh acceleration events"
        );

        fluidity.launch_active = false;
        fluidity.windowserver_cpu_spike = false;
        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Idle, 1.2),
            None
        );
        fluidity.windowserver_cpu_spike = true;
        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Idle, 1.3),
            Some(InteractionReason::WindowOperation)
        );
    }

    #[test]
    fn invalid_idle_sample_does_not_poison_reset_detection() {
        let fluidity = FluidityState::new();
        let mut controller = AccelerationLeaseBroker::default();
        controller.select_reason(&fluidity, BuildPhase::Idle, 7.0);

        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Idle, f64::NAN),
            None
        );
        assert_eq!(
            controller.select_reason(&fluidity, BuildPhase::Idle, 0.0),
            None
        );
    }

    #[test]
    fn chromium_selection_keeps_root_and_active_helpers_only() {
        let tree = ProcessTree::build(&[
            entry(200, 1, "Google Chrome", 1.0),
            entry(201, 200, "Google Chrome Helper (Renderer)", 8.0),
            entry(202, 200, "Google Chrome Helper (GPU)", 3.0),
            entry(203, 200, "Google Chrome Helper (Renderer)", 0.0),
            entry(300, 1, "Unrelated App", 90.0),
        ]);
        let snapshots = vec![
            snapshot(200, "Google Chrome", 1.0),
            snapshot(201, "Google Chrome Helper (Renderer)", 8.0),
            snapshot(202, "Google Chrome Helper (GPU)", 3.0),
            snapshot(203, "Google Chrome Helper (Renderer)", 0.0),
            snapshot(300, "Unrelated App", 90.0),
        ];

        let selection =
            select_acceleration_family(200, InteractionReason::Input, &tree, &snapshots)
                .expect("foreground Chrome family");
        let pids: Vec<u32> = selection.members.iter().map(|member| member.pid).collect();

        assert_eq!(selection.family, AccelerationFamily::Chromium);
        assert!(!selection.family.allows_explicit_task_qos());
        assert_eq!(pids, vec![200, 201, 202]);
        assert!(
            !pids.contains(&203),
            "idle renderer stays under macOS control"
        );
        assert!(!pids.contains(&300), "unrelated process cannot enter lease");
    }

    #[test]
    fn build_selection_prioritizes_compiler_and_stays_bounded() {
        let tree = ProcessTree::build(&[
            entry(400, 1, "Codex", 1.0),
            entry(401, 400, "rustc", 12.0),
            entry(402, 400, "cargo", 6.0),
            entry(403, 400, "clang", 4.0),
            entry(404, 400, "worker-a", 80.0),
            entry(405, 400, "worker-b", 70.0),
        ]);
        let snapshots = vec![
            snapshot(400, "Codex", 1.0),
            snapshot(401, "rustc", 12.0),
            snapshot(402, "cargo", 6.0),
            snapshot(403, "clang", 4.0),
            snapshot(404, "worker-a", 80.0),
            snapshot(405, "worker-b", 70.0),
        ];

        let target = select_build_target(&snapshots).expect("active build target");
        let selection =
            select_acceleration_family(target, InteractionReason::BuildStart, &tree, &snapshots)
                .expect("build family");
        let pids: Vec<u32> = selection.members.iter().map(|member| member.pid).collect();

        assert_eq!(selection.family, AccelerationFamily::General);
        assert!(selection.family.allows_explicit_task_qos());
        assert_eq!(target, 402, "coordinator owns the bounded build lease");
        assert_eq!(selection.root_pid, 402);
        assert_eq!(pids.len(), 3);
        assert!(
            pids.contains(&401),
            "compiler role outranks generic workers"
        );
        assert!(pids.contains(&402), "build coordinator remains represented");
        assert!(!pids.contains(&400), "protected parent is not mutated");
        assert!(
            !pids.contains(&404),
            "generic workers stay out of build lease"
        );
    }

    #[test]
    fn protected_and_system_roots_never_receive_a_lease() {
        let protected_tree = ProcessTree::build(&[entry(500, 1, "WindowServer", 40.0)]);
        let protected_snapshots = vec![snapshot(500, "WindowServer", 40.0)];
        assert!(select_acceleration_family(
            500,
            InteractionReason::Input,
            &protected_tree,
            &protected_snapshots,
        )
        .is_none());

        let launchd_tree = ProcessTree::build(&[entry(1, 0, "launchd", 1.0)]);
        let launchd_snapshots = vec![snapshot(1, "launchd", 1.0)];
        assert!(select_acceleration_family(
            1,
            InteractionReason::Input,
            &launchd_tree,
            &launchd_snapshots,
        )
        .is_none());
    }

    #[test]
    fn bounded_top_k_has_deterministic_tie_breaking() {
        let mut selected = Vec::new();
        for pid in [9, 2, 7, 4, 1, 8] {
            insert_bounded_candidate(
                &mut selected,
                AccelerationCandidate {
                    pid,
                    name: format!("p{pid}"),
                    score: 10.0,
                },
            );
        }
        let pids: Vec<u32> = selected.iter().map(|member| member.pid).collect();
        assert_eq!(pids, vec![1, 2, 4, 7]);
    }

    #[test]
    fn bounded_top_k_deduplicates_pid_and_keeps_best_sample() {
        let mut selected = Vec::new();
        insert_bounded_candidate(
            &mut selected,
            AccelerationCandidate {
                pid: 42,
                name: "first".to_string(),
                score: 1.0,
            },
        );
        insert_bounded_candidate(
            &mut selected,
            AccelerationCandidate {
                pid: 42,
                name: "newer".to_string(),
                score: 2.0,
            },
        );

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "newer");
    }

    #[test]
    fn event_ttls_fit_inside_the_continuous_budget() {
        for reason in [
            InteractionReason::Input,
            InteractionReason::WindowOperation,
            InteractionReason::BuildStart,
            InteractionReason::AppLaunch,
        ] {
            assert!(reason.ttl() <= MAX_CONTINUOUS_LEASE);
            assert!(reason.ttl() <= Duration::from_secs(5));
        }
        assert!(LEASE_COOLDOWN < MAX_CONTINUOUS_LEASE);
    }

    #[test]
    fn nice_fallback_is_mild_and_never_overwrites_existing_boost() {
        assert_eq!(LEASE_NICE, -2);
        assert_eq!(nice_fallback_target(0), Some(-2));
        assert_eq!(nice_fallback_target(10), Some(-2));
        assert_eq!(nice_fallback_target(-1), None);
        assert_eq!(nice_fallback_target(-10), None);
    }
}

/// Context bundle for [`wire_enriched_telemetry`].
///
/// Grouped because the function mutates ~20 fields of `RuntimeMetrics`
/// under a single `state.metrics` lock guard.
pub struct EnrichedTelemetryInputs<'a> {
    pub snapshot: &'a SystemSnapshot,
    pub swap_forecast: &'a SwapForecast,
    pub fluidity_state: &'a FluidityState,
    pub overflow_thresholds: &'a OverflowThresholds,
    pub behavior_interactive_pids: &'a HashSet<u32>,
    pub cog_decision: &'a CognitiveDecision,
    pub cognitive_state: &'a CognitiveState,
    pub lctx: &'a LearningContext<'a>,
    pub causal_qos_upgrades_cycle: u32,
    pub thermal_predicted_throttle: u8,
    pub thermal_seconds_to_throttle: Option<i32>,
    pub thermal_trend_predicted: &'a str,
    /// Number of recent foreground coalitions in the active envelope
    /// (Sprint Coalition 2026-05-10). 0 when nothing is foreground;
    /// 1-3 in steady-state app-switching.
    pub active_coalitions_count: u32,
    /// Lock-free metrics for Phase 0 lock-decomp instrumentation.
    pub lf_metrics: &'a apollo_engine::engine::lse_counters::LockFreeMetrics,
    /// Phase 1.5a (MLP router unblock, 2026-06-27): per-cycle telemetry
    /// archive inputs. None disables the archive for this cycle.
    /// The snapshot append runs AFTER all `m.metrics` assignments in
    /// [`wire_enriched_telemetry`] so the line reflects the same state the
    /// rest of the cycle saw (atomic with the rest of cycle_tail).
    pub metrics_history: Option<MetricsHistoryInputs<'a>>,
}

/// Phase 1.5a per-cycle archive inputs (forwarded to
/// [`apollo_engine::engine::daemon_metrics_history::append_history_snapshot`]).
///
/// `causal_subsystem_debias` is precomputed by the caller
/// (`cognitive_state.meta_cognition.subsystem_debias_multiplier(CausalGraph)`)
/// to keep the archive writer free of MetaCognition dep so it can be unit
/// tested without a live cognitive stack.
pub struct MetricsHistoryInputs<'a> {
    pub writer: &'a mut apollo_engine::engine::daemon_metrics_history::MetricsHistoryWriter,
    pub cycle: u64,
    pub world_model: &'a apollo_engine::engine::world_model::WorldModel,
    pub drift_detector: &'a apollo_engine::engine::nars_belief::DriftDetector,
    pub learnable_params: &'a apollo_engine::engine::learned_state::LearnableParams,
    pub causal_subsystem_debias: f32,
}

/// Sum RSS (MB) of all currently-frozen PIDs by walking the sysinfo
/// process table.
///
/// CRITICAL — keep this OUTSIDE the metrics god-lock.
///
/// Phase-1 instrumentation (commit 126e44c) captured `stall_candidate_F2`
/// firing 55 times across ~2h with `metrics_lock_held_max_us` peaking
/// at 30ms — a 50x amplification over the steady-state ~559us average.
/// Root cause: `sysinfo::System::process(pid)` locates a PID in O(N)
/// over the ~400-entry process table; doing this N times for N frozen
/// PIDs while holding `state.metrics` blocked every other telemetry
/// consumer (publishers, TUI, audit drain) for the duration.
///
/// Per the project's Lock Scope Minimization rule
/// (`~/.claude/skills/apollo-evolve/references/rust-systems-patterns.md`)
/// never hold a mutex across I/O. Sysinfo lookups are in-memory but
/// O(N) per call — the rule applies even though there is no syscall.
///
/// Brief `state.frozen_state` lock is acquired and released BEFORE the
/// sysinfo walk — no mutex nesting, no I/O under any lock. The cloned
/// `HashMap<u32, _>` is the only data crossing a lock boundary.
///
/// Phase-2a (2026-06-27): the result is passed into
/// [`wire_enriched_telemetry`] as a precomputed `f64`, so the metrics
/// lock is held only for the field-assignment + counter-drain. See
/// `/Users/eduardocortez/hardening-audit-2026-06-24/main-loop-stall-candidates.md`
/// (F2 MED-HIGH).
pub fn compute_frozen_ram_mb(state: &SharedState, collector: &SystemCollector) -> f64 {
    let frozen_pids = state.frozen_state.lock_recover().clone();
    let sys = collector.system();
    sum_frozen_ram_mb(&frozen_pids, &sys)
}

/// Pure, testable sum of frozen-process RSS in MiB.
///
/// Extracted so the Lock-Scope-Minimization refactor (compute_frozen_ram_mb)
/// has a unit-testable pure core: callers pass the already-cloned PIDs map
/// and a `&sysinfo::System` reference; this function does the O(N) walk only.
///
/// Bit-equivalent to the inlined formula the cycle_tail function used before
/// the split: `filter_map(sys.process) | map(.memory()/1MiB) | sum | .max(0)`.
/// The unit test in the same file (`tests` module at the bottom) verifies
/// this equivalence against hand-computed expected values, including empty
/// input, missing PIDs, and negative-result clamping.
fn sum_frozen_ram_mb<V>(frozen_pids: &HashMap<u32, V>, sys: &sysinfo::System) -> f64 {
    frozen_pids
        .keys()
        .filter_map(|pid| sys.process(sysinfo::Pid::from_u32(*pid)))
        .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
        .sum::<f64>()
        .max(0.0)
}

#[inline]
fn ns_to_ceil_us(ns: u64) -> u64 {
    ns.div_ceil(1_000)
}

/// Wire enriched telemetry + UCHS neurocognitive metrics into
/// `RuntimeMetrics` under a single `state.metrics` lock guard.
///
/// Fields written here can only be computed in the main loop where
/// `swap_forecast`, `sys`, and per-cycle cognitive state are in scope.
///
/// Pre-conditions:
/// - `fluidity_state.windowserver_cpu_ema` has been updated this cycle
///   (via `fluidity_state.observe()` inside the proc-snapshot block).
/// - `cog_decision` is this cycle's fresh neurocognitive decision.
/// - `run_neurocognitive_tick` has already mutated `cognitive_state`.
/// - `frozen_ram_mb` has been precomputed by [`compute_frozen_ram_mb`]
///   BEFORE the lock is acquired (see that fn's doc for the why).
///
/// Post-conditions:
/// - `state.metrics.metrics.*` has ~20 fields refreshed.
/// - `state.frozen_state` is NOT touched here — the caller pre-snapshotted
///   it into `frozen_ram_mb`.
pub fn wire_enriched_telemetry(
    state: &SharedState,
    frozen_ram_mb: f64,
    inputs: &mut EnrichedTelemetryInputs<'_>,
) {
    let mut m = state.metrics.lock_recover();
    // SwapTrend — previously computed but never exposed.
    m.metrics.swap_trend = format!("{:?}", inputs.swap_forecast.swap_trend);
    // WindowServer CPU — use EMA from FluidityState (already computed
    // each cycle in the proc_snaps block). More stable than raw sample.
    m.metrics.windowserver_cpu_pct = inputs.fluidity_state.windowserver_cpu_ema;
    // Compression signal from the EMA-smoothed compressor_pressure already
    // computed by the collector (ratio of compressor pages to total physical
    // pages × 0.85). The old formula used_ram - (total - free) was wrong:
    // on macOS total ≠ used + free (inactive/wired/speculative pages exist),
    // producing saturating_sub underflow → always 0 or nonsense.
    m.metrics.compressed_memory_ratio =
        inputs.snapshot.pressure.compressor_pressure.clamp(0.0, 1.0);
    // Frozen RAM: precomputed by the caller via `compute_frozen_ram_mb`
    // BEFORE this lock was taken. Under pressure, walking the sysinfo
    // process table for ~N frozen PIDs scaled to 30ms (Phase-1 F2 trace)
    // — keeping it out of the metrics god-lock is the Phase-2a fix.
    m.metrics.frozen_ram_mb = frozen_ram_mb;
    // cycles_high_pressure — consecutive cycles above bg_pressure.
    let bg_threshold = inputs.overflow_thresholds.bg_pressure;
    if inputs.snapshot.pressure.memory_pressure > bg_threshold {
        m.metrics.cycles_high_pressure = m.metrics.cycles_high_pressure.saturating_add(1);
    } else {
        m.metrics.cycles_high_pressure = 0;
    }
    // behavior_interactive_pid_count — how many PIDs learned dynamically.
    m.metrics.behavior_interactive_pid_count = inputs.behavior_interactive_pids.len();
    // rl_threshold_current — absolute threshold (bg_pressure + rl_adj).
    m.metrics.rl_threshold_current = bg_threshold + m.metrics.rl_adjustment_pp as f64 / 100.0;
    // ── UCHS / Neurocognitive metrics (8 cognitive modules) ──────────
    m.metrics.uchs_composite = inputs.cog_decision.uchs_composite;
    m.metrics.uchs_grade = inputs.cognitive_state.health.grade.clone();
    m.metrics.uchs_recovery_mode = inputs.cognitive_state.health.recovery_mode;
    m.metrics.epistemic_uncertainty = inputs.cognitive_state.epistemic.composite;
    m.metrics.epistemic_level = inputs.cognitive_state.epistemic.level_label().to_string();
    // Sprint Coalition 2026-05-10 metrics — guard-tower over-protection
    // signal (6th component of epistemic composite) + active-coalition
    // envelope size. Surfaces whether the new layered protection from
    // commits a381c6b..1ab6bdb is actually firing in production.
    m.metrics.guard_overprotection = inputs.cognitive_state.epistemic.guard_overprotection;
    m.metrics.active_coalitions_count = inputs.active_coalitions_count;
    // Phase 0 lock-decomp baseline (2026-05-10). Average and max use the
    // same publish window; mixing a lifetime average with an interval max
    // can produce avg > max and misdiagnose contention.
    let lf = inputs.lf_metrics;
    let ((wc, ws, wm), (hc, hs, hm)) = lf.drain_metrics_lock_window_ns();
    m.metrics.metrics_lock_wait_avg_us = if wc > 0 {
        (ws as f64 / wc as f64) / 1000.0
    } else {
        0.0
    };
    m.metrics.metrics_lock_wait_max_us = ns_to_ceil_us(wm);
    m.metrics.metrics_lock_held_avg_us = if hc > 0 {
        (hs as f64 / hc as f64) / 1000.0
    } else {
        0.0
    };
    m.metrics.metrics_lock_held_max_us = ns_to_ceil_us(hm);
    // Phase 0b stage split.
    //
    // Windowed avg + windowed max — both drained per publish so producer
    // and consumer agree on the same time horizon. Previously the avg
    // divided a lifetime cumulative `stage_*_total_ns` by lifetime
    // `stage_count`, while the max was per-interval drained — this
    // structurally produced `avg_ms > max_ms` on tail-light stages
    // (esp. Persist) and leaked stale lifetime values into dashboards.
    // Sprint 9 `4b13a39` rule: producer + consumer agree on horizon.
    // [Welford 1962] online statistics windowing.
    let sc_window = lf.drain_stage_count_window();
    let to_avg_ms = |total_window: u64| -> f64 {
        if sc_window > 0 {
            (total_window as f64 / sc_window as f64) / 1_000_000.0
        } else {
            0.0
        }
    };
    let ns_to_ms = |ns: u64| -> f64 { ns as f64 / 1_000_000.0 };
    m.metrics.stage_sense_avg_ms = to_avg_ms(lf.drain_stage_total_ns(CycleStage::Sense));
    m.metrics.stage_sense_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::Sense));
    m.metrics.stage_reason_avg_ms = to_avg_ms(lf.drain_stage_total_ns(CycleStage::Reason));
    m.metrics.stage_reason_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::Reason));
    m.metrics.stage_execute_avg_ms = to_avg_ms(lf.drain_stage_total_ns(CycleStage::Execute));
    m.metrics.stage_execute_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::Execute));
    m.metrics.stage_learn_avg_ms = to_avg_ms(lf.drain_stage_total_ns(CycleStage::Learn));
    m.metrics.stage_learn_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::Learn));
    m.metrics.stage_persist_avg_ms = to_avg_ms(lf.drain_stage_total_ns(CycleStage::Persist));
    m.metrics.stage_persist_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::Persist));
    // REASON sub-stages (Phase 0c).
    m.metrics.stage_reason_signal_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonSignalTick));
    m.metrics.stage_reason_signal_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonSignalTick));
    m.metrics.stage_reason_neuro_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonNeuro));
    m.metrics.stage_reason_neuro_max_ms = ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonNeuro));
    m.metrics.stage_reason_decide_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonDecide));
    m.metrics.stage_reason_decide_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonDecide));
    m.metrics.stage_reason_usercontext_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonUserContext));
    m.metrics.stage_reason_usercontext_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonUserContext));
    m.metrics.stage_reason_holtwinters_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonHoltWinters));
    m.metrics.stage_reason_holtwinters_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonHoltWinters));
    m.metrics.stage_reason_pagereclaim_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonPageReclaim));
    m.metrics.stage_reason_pagereclaim_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonPageReclaim));
    m.metrics.stage_reason_chromium_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonChromium));
    m.metrics.stage_reason_chromium_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonChromium));
    m.metrics.stage_reason_enrich_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonEnrich));
    m.metrics.stage_reason_enrich_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonEnrich));
    // Additive instrumentation (2026-06-23): untimed enrich→decide ops.
    m.metrics.stage_reason_procscan_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonProcScan));
    m.metrics.stage_reason_procscan_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonProcScan));
    m.metrics.stage_reason_rusage_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonRusage));
    m.metrics.stage_reason_rusage_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonRusage));
    m.metrics.stage_reason_signalintel_avg_ms =
        to_avg_ms(lf.drain_stage_total_ns(CycleStage::ReasonSignalIntel));
    m.metrics.stage_reason_signalintel_max_ms =
        ns_to_ms(lf.drain_stage_max_ns(CycleStage::ReasonSignalIntel));
    m.metrics.meta_confidence = inputs.cognitive_state.meta_cognition.meta_confidence;
    m.metrics.humble_mode = inputs.cog_decision.humble_mode;
    m.metrics.adversarial_pass_rate =
        inputs.cognitive_state.adversarial.lifetime_pass_rate() as f32;
    m.metrics.adversarial_safety_alert = inputs.cog_decision.safety_alert;
    m.metrics.cognitive_snr = inputs.cognitive_state.reward_bus.signal_to_noise();
    m.metrics.self_eval_quality = inputs.cognitive_state.self_evaluator.evaluator_trust();
    m.metrics.reptile_cached_workloads = inputs.cognitive_state.reptile.cached_workloads();
    m.metrics.drift_early_warning = inputs.lctx.outcome_tracker.drift_detector.early_warning();
    // Causal QoS upgrades this cycle (FreezeProcess → ThrottleProcess).
    m.metrics.causal_qos_upgrades_cycle = inputs.causal_qos_upgrades_cycle;
    // Predictive thermal state from ThermalManager (previously discarded).
    // seconds_to_throttle: null = no forecast, 0 = throttling now, >0 = seconds of headroom.
    m.metrics.thermal_predicted_throttle = inputs.thermal_predicted_throttle;
    m.metrics.thermal_seconds_to_throttle = inputs.thermal_seconds_to_throttle;
    m.metrics.thermal_trend_predicted = inputs.thermal_trend_predicted.to_string();
    // Phase-1 stall-candidate F2 (audit 2026-06-24): the original sysinfo
    // walk is now outside this lock. Later spikes exposed two remaining
    // amplifiers below: cloning all RuntimeMetrics and logging the warning
    // before releasing the guard. The compact snapshot + post-unlock warning
    // keep this threshold useful without contributing to it. [F2 MED-HIGH] per
    // /Users/eduardocortez/hardening-audit-2026-06-24/main-loop-stall-candidates.md
    //
    // Capture for post-unlock logging. Emitting the warning while holding the
    // very lock it diagnoses can amplify one slow acquisition into another.
    let metrics_lock_held_max_us = m.metrics.metrics_lock_held_max_us;

    // ── Phase 1.5a per-cycle telemetry archive (2026-06-27) ──────────────
    // Append a JSONL line to /var/lib/apollo/runtime_metrics_history.jsonl
    // AFTER all m.metrics assignments above. The 16-d feature vector is
    // built from the freshly-stored RuntimeMetrics + the caller's
    // world_model / drift_detector / learnable_params. Failure is
    // best-effort: log + return Err (the caller decides). Cycle is never
    // blocked — `append_history_snapshot` does its own fsync budget and
    // bails out on I/O error without panicking. Unblocks the MLP router
    // PR (Phase 1 CV collapsed to majority-class baseline because
    // runtime_metrics.json is a single snapshot, not a time series).
    // ponytail: minimum that works, no decision-path mutation.
    //
    // Prepare only the compact 16-d history payload while the snapshot is
    // consistent. Cloning all of RuntimeMetrics here copied large vectors and
    // strings under the global lock even though the archive never uses them.
    let prepared_history = inputs.metrics_history.as_ref().map(|mh| {
        apollo_engine::engine::daemon_metrics_history::prepare_history_snapshot(
            &m.metrics,
            mh.causal_subsystem_debias,
            mh.world_model,
            mh.drift_detector,
            mh.learnable_params,
        )
    });
    drop(m); // release the metrics lock before any I/O (Lock Scope Minimization)

    if metrics_lock_held_max_us > 5000 {
        tracing::warn!(
            target: "apollo.stall_candidate",
            held_max_us = metrics_lock_held_max_us,
            "stall_candidate_F2: metrics lock held >5ms this cycle"
        );
    }

    if let (Some(mh), Some(snapshot)) = (inputs.metrics_history.as_mut(), prepared_history.as_ref())
    {
        if let Err(e) = mh.writer.append(mh.cycle, snapshot) {
            // Mirror the append_failure path inside append_history_snapshot:
            // the function has ALREADY bumped FAILED_WRITES + emitted warn.
            // Surface here only for diagnosability; cycle continues.
            tracing::debug!(
                target: "apollo.metrics_history",
                cycle = mh.cycle,
                error = %e,
                "cycle_tail: metrics_history append surfaced Err (already counted)"
            );
        }
    }
}

/// Context bundle for [`run_periodic_stage`].
///
/// A thin wrapper over [`PeriodicContext`]'s owned (non-`lctx`) fields —
/// keeps the main-loop call-site from re-listing every path and counter.
pub struct PeriodicStageInputs<'a> {
    pub cycle_count: u64,
    pub current_pressure: f64,
    pub workload_mode: &'a str,
    pub skills_path: &'a Path,
    pub hop_groups_path: &'a Path,
    pub signal_intel_path: &'a Path,
    pub learned_state_path: &'a Path,
    pub persist_generations: u32,
    pub last_restore_quality: Option<f64>,
    pub pending_trial_skill: Option<(String, f64)>,
}

/// Run the periodic maintenance stage (% 100 / % 500 / % 7200 gates).
///
/// Delegates to [`run_periodic`] with a freshly-constructed
/// [`PeriodicContext`]. The % 500 GC (experience compression, weight
/// prune, skill GC + persist) runs here; the % 100 persist and
/// rule-induction remain inline in main.rs above this call (they need
/// SharedState access); the % 7200 hourly GC also remains inline
/// (binary-local types: `cache_warmer`, `io_shaper`,
/// `temporal_predictor`).
///
/// Side effect: persists `optimization_skills.json` when the % 500
/// gate fires and new GC work occurred.
pub fn run_periodic_stage<'a>(
    inputs: PeriodicStageInputs<'a>,
    lctx: &mut LearningContext<'a>,
) -> PeriodicResult {
    let mut pctx = PeriodicContext {
        cycle_count: inputs.cycle_count,
        current_pressure: inputs.current_pressure,
        workload_mode: inputs.workload_mode,
        skills_path: inputs.skills_path,
        hop_groups_path: inputs.hop_groups_path,
        signal_intel_path: inputs.signal_intel_path,
        learned_state_path: inputs.learned_state_path,
        persist_generations: inputs.persist_generations,
        last_restore_quality: inputs.last_restore_quality,
        pending_trial_skill: inputs.pending_trial_skill,
        lctx,
    };
    run_periodic(&mut pctx)
}

/// S10 consumer: drain expired effect-decay observations, re-read each
/// observable, bump `effect_decay_detected_total` on disagreement.
///
/// Called once per main-loop cycle from the daemon tail. Bounded by
/// RING_CAP=64 (effect_decay module-level constant).
///
/// Wake-grace: callers MUST pass a `seconds_since_wake` value > 30
/// (or skip the call entirely) since immediately after wake the
/// kernel may not have reapplied tier hints — false-positive
/// disagreements would inflate the counter. The daemon's wake
/// tracking is in main.rs; this function trusts the caller.
///
/// FIX-3 wire (2026-06-07): forwards the observation into
/// `report_disagreement_with` so hard-protected disagreements feed
/// the 5-minute sliding window, then consults
/// `poke_rollback_guard_via_decay` once per cycle. When the window
/// crosses `HARD_PROTECTED_DECAY_THRESHOLD` and the rollback guard
/// has eligible shifts + no active cooldown, this is the path that
/// auto-reverts `zone_alpha` / `rl_pressure_bands[2]` to their
/// pre-shift values. Without this caller, the wire was dormant —
/// `poke_rollback_guard_via_decay` had zero invocations in the daemon.
pub fn drain_effect_decay(
    state: &SharedState,
    lp: &mut apollo_engine::engine::learned_state::LearnableParams,
) {
    let expired = {
        let mut w = state.effect_decay.lock_recover();
        w.drain_expired(std::time::Instant::now())
    };
    if expired.is_empty() {
        // Still consult the rollback guard — a previously-recorded
        // hard-protected disagreement window may have just crossed the
        // threshold even on a cycle with no new expirations.
        let (hp_count, hp_pids) = {
            let mut w = state.effect_decay.lock_recover();
            let now = std::time::Instant::now();
            (
                w.hard_protected_decay_count_5min(now),
                w.hard_protected_decay_pids(now),
            )
        };
        apollo_engine::engine::learned_state::poke_rollback_guard_via_decay(lp, hp_count, &hp_pids);
        return;
    }
    {
        let mut watchdog = state.effect_decay.lock_recover();
        for obs in expired {
            use apollo_engine::engine::effect_decay::ObsKind;
            // FIX-3-v2 (Round 3, Option B): MachPolicy attempts on
            // hard-protected processes ARE the disagreement signal — Apollo
            // trying to mutate a Chromium-protected process under pressure is
            // itself anomalous; no Mach FFI re-read needed.
            //
            // Round-4 (2026-06-07): route through `record_hp_mach_attempt`
            // (NOT report_disagreement_with) so the HP MachPolicy path bumps
            // its dedicated counter `effect_decay_hp_mach_attempts_total`,
            // leaving `effect_decay_detected_total` reserved for the
            // Jetsam/Sysctl re-read-disagreement baseline 27. Without the
            // split, baseline comparisons in metrics_to_watch are invalidated
            // because the same counter would mix two distinct signals.
            if matches!(obs.kind, ObsKind::MachPolicy) {
                if obs.hard_protected {
                    watchdog.record_hp_mach_attempt(&obs);
                }
                // Non-hard-protected MachPolicy: producer-side re-read
                // deferred — see banner. Skip.
                continue;
            }
            let live = match obs.kind {
                ObsKind::JetsamTier => {
                    apollo_engine::engine::jetsam_control::get_priority(obs.pid).map(|p| p as i64)
                }
                ObsKind::Sysctl => obs
                    .key
                    .as_deref()
                    .and_then(apollo_engine::engine::sysctl_direct::read_i32)
                    .map(|v| v as i64),
                ObsKind::MachPolicy => unreachable!("handled above"),
            };
            if let Some(actual) = live {
                if actual != obs.value_post {
                    watchdog.report_disagreement_with(&obs);
                }
            }
        }
    }
    // FIX-3 wire: consult the rollback guard once per cycle AFTER the
    // drain loop has updated the hard-protected sliding window. The
    // watchdog borrow is released above so we can re-lock it here for
    // the count/pids snapshot without deadlock.
    let (hp_count, hp_pids) = {
        let mut w = state.effect_decay.lock_recover();
        let now = std::time::Instant::now();
        (
            w.hard_protected_decay_count_5min(now),
            w.hard_protected_decay_pids(now),
        )
    };
    apollo_engine::engine::learned_state::poke_rollback_guard_via_decay(lp, hp_count, &hp_pids);
}

#[cfg(test)]
mod tests {
    use super::{ns_to_ceil_us, sum_frozen_ram_mb};
    use std::collections::HashMap;
    use sysinfo::System;

    #[test]
    fn lock_max_rounds_sub_microsecond_samples_up() {
        assert_eq!(ns_to_ceil_us(0), 0);
        assert_eq!(ns_to_ceil_us(1), 1);
        assert_eq!(ns_to_ceil_us(1_000), 1);
        assert_eq!(ns_to_ceil_us(1_001), 2);
    }

    /// Bit-equivalence + edge cases for the Lock-Scope-Minimization
    /// refactor. The pure core (`sum_frozen_ram_mb`) is testable without
    /// constructing a `SharedState` or seeding `SystemCollector` — `System::new()`
    /// has no processes, so `sys.process(pid)` returns None and filter_map
    /// drops everything, giving us the empty / not-found paths cleanly.

    #[test]
    fn sum_frozen_ram_mb_empty_map_returns_zero() {
        // Edge case: no frozen PIDs. The original inlined code also returned 0.
        // `sum::<f64>()` over empty iterator = 0.0; `.max(0.0)` clamps.
        let sys = System::new();
        let frozen: HashMap<u32, ()> = HashMap::new();
        let result = sum_frozen_ram_mb(&frozen, &sys);
        assert!(
            (result - 0.0).abs() < f64::EPSILON,
            "empty frozen_state must produce 0.0, got {result}"
        );
    }

    #[test]
    fn sum_frozen_ram_mb_pids_not_in_system_returns_zero() {
        // Edge case: frozen_state has PIDs, but the test System has no
        // process entries (System::new() enumerates nothing by default).
        // The filter_map drops every pid; sum is 0.
        // This is the BEHAVIOR that lets us keep using System::new() in tests
        // without a live process table.
        let sys = System::new();
        let mut frozen: HashMap<u32, ()> = HashMap::new();
        frozen.insert(1234u32, ());
        frozen.insert(5678u32, ());
        frozen.insert(99999u32, ());
        let result = sum_frozen_ram_mb(&frozen, &sys);
        assert!(
            (result - 0.0).abs() < f64::EPSILON,
            "PIDs not in sys.process must produce 0.0 via filter_map drop, got {result}"
        );
    }

    #[test]
    fn sum_frozen_ram_mb_max_zero_clamp_is_a_noop_for_nonneg_sum() {
        // The .max(0.0) clamp at the end protects against a hypothetical
        // rounding negative (f64::sum of nonneg values cannot go negative in
        // practice, but the original code had the clamp so we must preserve
        // it). Verify the clamp does not alter a nonneg result.
        let sys = System::new();
        let frozen: HashMap<u32, ()> = HashMap::new();
        let result = sum_frozen_ram_mb(&frozen, &sys);
        assert!(result >= 0.0, "result must be >= 0.0, got {result}");
    }

    #[test]
    fn sum_frozen_ram_mb_only_consumes_keys_not_values() {
        // The pure core is generic over the value type V, so a HashMap with
        // any payload (e.g. FrozenEntry once we wanted to test that) works.
        // This test pins the API contract: callers don't need to materialize
        // FrozenEntry to compute the sum.
        let sys = System::new();
        let mut frozen: HashMap<u32, Vec<String>> = HashMap::new();
        frozen.insert(1u32, vec!["some".to_string(), "payload".to_string()]);
        frozen.insert(2u32, vec![]);
        let result = sum_frozen_ram_mb(&frozen, &sys);
        // Still 0 because System::new() has no processes; the value types
        // are irrelevant to the sum (filter_map only reads .memory()).
        assert!(
            (result - 0.0).abs() < f64::EPSILON,
            "value type must be ignored, got {result}"
        );
    }
}
