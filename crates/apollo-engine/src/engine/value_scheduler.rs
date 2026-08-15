//! Fixed, deterministic admission for optional Apollo computation.
//!
//! The scheduler owns only optional compute permits. It has no closures,
//! models, process handles, effects, or action authority. All ranking math is
//! bounded integer arithmetic so a malformed cost or signal cannot create an
//! accidental budget surplus.

use std::cmp::Ordering;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::cycle_snapshot::{CycleContextSnapshot, SnapshotId, SnapshotIdentity};

pub const MAX_JOBS: usize = 64;
pub const MAX_SELECTED_PER_CYCLE: usize = 16;
pub const MAX_DEPENDENCIES_PER_JOB: usize = 4;
pub const MAX_IN_FLIGHT_OPTIONAL: usize = 4;
pub const MAX_COMPLETIONS_PER_CYCLE: usize = 64;
pub const MIN_COST_ESTIMATE_US: u64 = 50;
pub const MAX_JOB_SLICE_US: u64 = 60_000;
pub const NOMINAL_BUDGET_US: u64 = 150_000;
pub const GUARDED_BUDGET_US: u64 = 100_000;
pub const CONSTRAINED_BUDGET_US: u64 = 60_000;

const SCORE_SCALE: u64 = 1_000_000;
const Q_MAX: u64 = 10_000;
const JOB_COUNT: usize = 10;
const MAX_NON_TERMINAL_COMPLETIONS_PER_CYCLE: usize =
    MAX_COMPLETIONS_PER_CYCLE - MAX_IN_FLIGHT_OPTIONAL;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum JobId {
    GpuImagination,
    ReflexReasoningRefresh,
    WorldModelRefresh,
    AisRuntimeRefresh,
    HardwarePrediction,
    HoltWintersRefresh,
    PageReclaimRefresh,
    PlannerAdviceRefresh,
    PeriodicLearningMaintenance,
    TelemetryFlush,
}

impl JobId {
    pub const ALL: [Self; JOB_COUNT] = [
        Self::GpuImagination,
        Self::ReflexReasoningRefresh,
        Self::WorldModelRefresh,
        Self::AisRuntimeRefresh,
        Self::HardwarePrediction,
        Self::HoltWintersRefresh,
        Self::PageReclaimRefresh,
        Self::PlannerAdviceRefresh,
        Self::PeriodicLearningMaintenance,
        Self::TelemetryFlush,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::GpuImagination => 0,
            Self::ReflexReasoningRefresh => 1,
            Self::WorldModelRefresh => 2,
            Self::AisRuntimeRefresh => 3,
            Self::HardwarePrediction => 4,
            Self::HoltWintersRefresh => 5,
            Self::PageReclaimRefresh => 6,
            Self::PlannerAdviceRefresh => 7,
            Self::PeriodicLearningMaintenance => 8,
            Self::TelemetryFlush => 9,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuImagination => "gpu-imagination",
            Self::ReflexReasoningRefresh => "reflex-reasoning-refresh",
            Self::WorldModelRefresh => "world-model-refresh",
            Self::AisRuntimeRefresh => "ais-runtime-refresh",
            Self::HardwarePrediction => "hardware-prediction",
            Self::HoltWintersRefresh => "holt-winters-refresh",
            Self::PageReclaimRefresh => "page-reclaim-refresh",
            Self::PlannerAdviceRefresh => "planner-advice-refresh",
            Self::PeriodicLearningMaintenance => "periodic-learning-maintenance",
            Self::TelemetryFlush => "telemetry-flush",
        }
    }

    pub fn descriptor(self) -> JobDescriptor {
        JobDescriptor::ALL[self.index()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobDescriptor {
    pub id: JobId,
    pub base_value_q: u32,
    pub target_max_interval_us: u64,
    pub static_floor_us: u64,
    pub dependencies: &'static [JobId],
}

impl JobDescriptor {
    pub const ALL: [Self; JOB_COUNT] = [
        Self {
            id: JobId::GpuImagination,
            base_value_q: 8_500,
            target_max_interval_us: 10_000_000,
            static_floor_us: 4_000,
            dependencies: &[],
        },
        Self {
            id: JobId::ReflexReasoningRefresh,
            base_value_q: 9_800,
            target_max_interval_us: 2_000_000,
            static_floor_us: 2_000,
            dependencies: &[],
        },
        Self {
            id: JobId::WorldModelRefresh,
            base_value_q: 8_900,
            target_max_interval_us: 5_000_000,
            static_floor_us: 6_000,
            dependencies: &[],
        },
        Self {
            id: JobId::AisRuntimeRefresh,
            base_value_q: 7_000,
            target_max_interval_us: 30_000_000,
            static_floor_us: 1_500,
            dependencies: &[],
        },
        Self {
            id: JobId::HardwarePrediction,
            base_value_q: 7_600,
            target_max_interval_us: 5_000_000,
            static_floor_us: 3_000,
            dependencies: &[],
        },
        Self {
            id: JobId::HoltWintersRefresh,
            base_value_q: 7_800,
            target_max_interval_us: 5_000_000,
            static_floor_us: 1_000,
            dependencies: &[],
        },
        Self {
            id: JobId::PageReclaimRefresh,
            base_value_q: 8_100,
            target_max_interval_us: 2_000_000,
            static_floor_us: 2_500,
            dependencies: &[],
        },
        Self {
            id: JobId::PlannerAdviceRefresh,
            base_value_q: 8_000,
            target_max_interval_us: 5_000_000,
            static_floor_us: 2_500,
            dependencies: &[],
        },
        Self {
            id: JobId::PeriodicLearningMaintenance,
            base_value_q: 5_500,
            target_max_interval_us: 60_000_000,
            static_floor_us: 5_000,
            dependencies: &[],
        },
        Self {
            id: JobId::TelemetryFlush,
            base_value_q: 6_500,
            target_max_interval_us: 10_000_000,
            static_floor_us: 750,
            dependencies: &[],
        },
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SchedulerLevel {
    Nominal,
    Guarded,
    Constrained,
}

impl Default for SchedulerLevel {
    fn default() -> Self {
        Self::Nominal
    }
}

impl SchedulerLevel {
    pub const fn budget_us(self) -> u64 {
        match self {
            Self::Nominal => NOMINAL_BUDGET_US,
            Self::Guarded => GUARDED_BUDGET_US,
            Self::Constrained => CONSTRAINED_BUDGET_US,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SchedulerPhase {
    Shadow,
    Active,
    Disabled,
}

impl Default for SchedulerPhase {
    fn default() -> Self {
        Self::Shadow
    }
}

impl SchedulerPhase {
    pub const fn should_execute(self) -> bool {
        matches!(self, Self::Active)
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerInputs {
    pub level: SchedulerLevel,
    pub due: [bool; MAX_JOBS],
    pub signal_q: [u16; MAX_JOBS],
    pub elapsed_since_success_us: [u64; MAX_JOBS],
    pub consecutive_budget_skips: [u32; MAX_JOBS],
    pub cost_estimate_us: [Option<f64>; MAX_JOBS],
    pub in_flight: usize,
    pub kill_switch: bool,
    pub sleeping: bool,
}

impl Default for SchedulerInputs {
    fn default() -> Self {
        Self {
            level: SchedulerLevel::Nominal,
            due: [false; MAX_JOBS],
            signal_q: [0; MAX_JOBS],
            elapsed_since_success_us: [0; MAX_JOBS],
            consecutive_budget_skips: [0; MAX_JOBS],
            cost_estimate_us: [None; MAX_JOBS],
            in_flight: 0,
            kill_switch: false,
            sleeping: false,
        }
    }
}

impl SchedulerInputs {
    pub fn nominal_all_due() -> Self {
        Self::all_due(SchedulerLevel::Nominal)
    }

    pub fn guarded_all_due() -> Self {
        Self::all_due(SchedulerLevel::Guarded)
    }

    pub fn constrained_all_due() -> Self {
        Self::all_due(SchedulerLevel::Constrained)
    }

    pub fn all_due(level: SchedulerLevel) -> Self {
        let mut inputs = Self {
            level,
            ..Self::default()
        };
        for job in JobId::ALL {
            inputs.due[job.index()] = true;
            inputs.elapsed_since_success_us[job.index()] = job.descriptor().target_max_interval_us;
        }
        inputs
    }

    pub fn with_due_jobs(level: SchedulerLevel, jobs: &[JobId]) -> Self {
        let mut inputs = Self {
            level,
            ..Self::default()
        };
        for &job in jobs.iter().take(MAX_JOBS) {
            inputs.due[job.index()] = true;
            inputs.elapsed_since_success_us[job.index()] = job.descriptor().target_max_interval_us;
        }
        inputs
    }

    pub fn set_due(&mut self, job: JobId, due: bool) {
        self.due[job.index()] = due;
        if due && self.elapsed_since_success_us[job.index()] == 0 {
            self.elapsed_since_success_us[job.index()] = job.descriptor().target_max_interval_us;
        }
    }

    pub fn set_signal_q(&mut self, job: JobId, signal_q: u16) {
        self.signal_q[job.index()] = u64::from(signal_q).min(Q_MAX) as u16;
    }

    pub fn set_signal(&mut self, job: JobId, signal: f64) {
        self.signal_q[job.index()] = sanitize_signal(signal) as u16;
    }

    pub fn set_elapsed_since_success_us(&mut self, job: JobId, elapsed_us: u64) {
        self.elapsed_since_success_us[job.index()] = elapsed_us;
    }

    pub fn set_consecutive_budget_skips(&mut self, job: JobId, skips: u32) {
        self.consecutive_budget_skips[job.index()] = skips;
    }

    pub fn set_cost_estimate_us(&mut self, job: JobId, estimate_us: f64) {
        self.cost_estimate_us[job.index()] = Some(estimate_us);
    }

    pub fn with_in_flight(mut self, in_flight: usize) -> Self {
        self.in_flight = in_flight;
        self
    }

    pub fn with_kill_switch(mut self, kill_switch: bool) -> Self {
        self.kill_switch = kill_switch;
        self
    }

    pub fn with_sleeping(mut self, sleeping: bool) -> Self {
        self.sleeping = sleeping;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobPermit {
    pub job: JobId,
    pub snapshot_id: SnapshotId,
    pub workload_id: u64,
    pub capability_revision: u64,
    pub thermal_power_revision: u64,
    pub process_identity_revision: u64,
    pub generation: u64,
    pub submitted_mono_us: u64,
    pub deadline_mono_us: u64,
    pub predicted_us: u64,
    pub value_q: u64,
    pub score_q: u64,
    pub dependencies: &'static [JobId],
    pub should_execute: bool,
}

impl JobPermit {
    pub fn identity(&self) -> SnapshotIdentity {
        SnapshotIdentity {
            snapshot_id: self.snapshot_id,
            workload_id: self.workload_id,
            capability_revision: self.capability_revision,
            thermal_power_revision: self.thermal_power_revision,
            process_identity_revision: self.process_identity_revision,
        }
    }

    pub fn job_id(&self) -> JobId {
        self.job
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum JobCompletionStatus {
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobCompletion {
    pub job: JobId,
    pub snapshot_id: SnapshotId,
    pub workload_id: u64,
    pub capability_revision: u64,
    pub thermal_power_revision: u64,
    pub process_identity_revision: u64,
    pub generation: u64,
    pub deadline_mono_us: u64,
    pub completed_mono_us: u64,
    pub elapsed_us: u32,
    pub status: JobCompletionStatus,
}

impl JobCompletion {
    pub fn from_permit(
        permit: &JobPermit,
        completed_mono_us: u64,
        elapsed_us: u32,
        status: JobCompletionStatus,
    ) -> Self {
        Self {
            job: permit.job,
            snapshot_id: permit.snapshot_id,
            workload_id: permit.workload_id,
            capability_revision: permit.capability_revision,
            thermal_power_revision: permit.thermal_power_revision,
            process_identity_revision: permit.process_identity_revision,
            generation: permit.generation,
            deadline_mono_us: permit.deadline_mono_us,
            completed_mono_us,
            elapsed_us,
            status,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionDisposition {
    Accepted,
    Stale,
    Capacity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerPlan {
    pub level: SchedulerLevel,
    pub budget_us: u64,
    pub predicted_us: u64,
    pub eligible_jobs: usize,
    pub selected_jobs: usize,
    pub in_flight_jobs: usize,
    pub should_execute: bool,
    pub selection_latency_us: u64,
    pub permits: Vec<JobPermit>,
}

impl Default for SchedulerPlan {
    fn default() -> Self {
        Self {
            level: SchedulerLevel::Nominal,
            budget_us: NOMINAL_BUDGET_US,
            predicted_us: 0,
            eligible_jobs: 0,
            selected_jobs: 0,
            in_flight_jobs: 0,
            should_execute: false,
            selection_latency_us: 0,
            permits: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueSchedulerMetrics {
    pub phase: SchedulerPhase,
    pub registered_jobs: u64,
    pub eligible_jobs: u64,
    pub selected_jobs: u64,
    pub selected_total: u64,
    pub in_flight_jobs: u64,
    pub budget_us: u64,
    pub predicted_us: u64,
    pub actual_us: u64,
    pub selection_latency_us: u64,
    pub max_selection_latency_us: u64,
    pub completed_total: u64,
    pub success_total: u64,
    pub failed_total: u64,
    pub cancelled_total: u64,
    pub timed_out_total: u64,
    pub expired_active_total: u64,
    pub stale_total: u64,
    pub invalid_cost_total: u64,
    pub budget_skipped_total: u64,
    pub capacity_skipped_total: u64,
    pub running_skipped_total: u64,
    pub generation_overflow_total: u64,
    pub deadline_overflow_total: u64,
    pub foreign_snapshot_total: u64,
    pub regressive_snapshot_total: u64,
    pub completion_capacity_dropped_total: u64,
    pub completions_seen: u64,
}

impl Default for ValueSchedulerMetrics {
    fn default() -> Self {
        Self {
            phase: SchedulerPhase::Shadow,
            registered_jobs: JOB_COUNT as u64,
            eligible_jobs: 0,
            selected_jobs: 0,
            selected_total: 0,
            in_flight_jobs: 0,
            budget_us: NOMINAL_BUDGET_US,
            predicted_us: 0,
            actual_us: 0,
            selection_latency_us: 0,
            max_selection_latency_us: 0,
            completed_total: 0,
            success_total: 0,
            failed_total: 0,
            cancelled_total: 0,
            timed_out_total: 0,
            expired_active_total: 0,
            stale_total: 0,
            invalid_cost_total: 0,
            budget_skipped_total: 0,
            capacity_skipped_total: 0,
            running_skipped_total: 0,
            generation_overflow_total: 0,
            deadline_overflow_total: 0,
            foreign_snapshot_total: 0,
            regressive_snapshot_total: 0,
            completion_capacity_dropped_total: 0,
            completions_seen: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PermitIdentity {
    identity: SnapshotIdentity,
    generation: u64,
    submitted_mono_us: u64,
    deadline_mono_us: u64,
    lifecycle: PermitLifecycle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermitLifecycle {
    Pending,
    Running,
}

#[derive(Debug, Clone, Copy)]
struct JobState {
    generation: u64,
    ewma_us: u64,
    last_observed_us: u64,
    last_success_sequence: u64,
    active: Option<PermitIdentity>,
}

impl Default for JobState {
    fn default() -> Self {
        Self {
            generation: 0,
            ewma_us: 0,
            last_observed_us: 0,
            last_success_sequence: 0,
            active: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    job: JobId,
    descriptor: JobDescriptor,
    freshness_q: u64,
    value_q: u64,
    cost_us: u64,
    score_q: u64,
    oldest_success: u64,
}

pub struct ValueScheduler {
    phase: SchedulerPhase,
    jobs: [JobState; MAX_JOBS],
    metrics: ValueSchedulerMetrics,
    current_snapshot: Option<SnapshotIdentity>,
}

impl ValueScheduler {
    pub fn new(phase: SchedulerPhase) -> Self {
        let mut metrics = ValueSchedulerMetrics::default();
        metrics.phase = phase;
        Self {
            phase,
            jobs: [JobState::default(); MAX_JOBS],
            metrics,
            current_snapshot: None,
        }
    }

    pub fn phase(&self) -> SchedulerPhase {
        self.phase
    }

    pub fn set_phase(&mut self, phase: SchedulerPhase) {
        self.phase = phase;
        self.metrics.phase = phase;
    }

    pub fn metrics(&self) -> &ValueSchedulerMetrics {
        &self.metrics
    }

    pub fn estimated_cost_us(&self, job: JobId) -> u64 {
        let state = self.jobs[job.index()];
        let descriptor = job.descriptor();
        state
            .last_observed_us
            .max(state.ewma_us)
            .max(descriptor.static_floor_us)
            .clamp(MIN_COST_ESTIMATE_US, MAX_JOB_SLICE_US)
    }

    pub fn last_success_sequence(&self, job: JobId) -> Option<u64> {
        match self.jobs[job.index()].last_success_sequence {
            0 => None,
            sequence => Some(sequence),
        }
    }

    pub fn plan(
        &mut self,
        snapshot: &CycleContextSnapshot,
        inputs: SchedulerInputs,
    ) -> SchedulerPlan {
        let started = Instant::now();
        let budget_us = inputs.level.budget_us();
        if let Some(current) = self.current_snapshot {
            if snapshot.id.daemon_epoch != current.snapshot_id.daemon_epoch {
                self.metrics.foreign_snapshot_total =
                    self.metrics.foreign_snapshot_total.saturating_add(1);
                return self.rejected_plan(inputs.level, budget_us, started);
            }
            if snapshot.id.sequence <= current.snapshot_id.sequence {
                self.metrics.regressive_snapshot_total =
                    self.metrics.regressive_snapshot_total.saturating_add(1);
                return self.rejected_plan(inputs.level, budget_us, started);
            }
        }
        self.current_snapshot = Some(snapshot.identity());
        self.metrics.completions_seen = 0;
        self.metrics.actual_us = 0;
        self.expire_stale_active(snapshot.identity());
        let blocked = self.phase == SchedulerPhase::Disabled
            || inputs.kill_switch
            || inputs.sleeping
            || snapshot.kill_switch
            || snapshot.sleeping;

        let mut candidates = Vec::with_capacity(JOB_COUNT);
        if !blocked {
            for job in JobId::ALL {
                let index = job.index();
                if !inputs.due[index] {
                    continue;
                }
                let descriptor = job.descriptor();
                let freshness_q = freshness_q(
                    inputs.elapsed_since_success_us[index],
                    descriptor.target_max_interval_us,
                );
                let signal_q = u64::from(inputs.signal_q[index]).min(Q_MAX);
                let starvation_q = u64::from(inputs.consecutive_budget_skips[index])
                    .saturating_mul(1_000)
                    .min(Q_MAX);
                let value_q = (4 * u64::from(descriptor.base_value_q)
                    + 3 * freshness_q
                    + 2 * signal_q
                    + starvation_q)
                    / 10;
                let (cost_us, invalid) = self.cost_for(job, inputs.cost_estimate_us[index]);
                if invalid {
                    self.metrics.invalid_cost_total =
                        self.metrics.invalid_cost_total.saturating_add(1);
                }
                let score_q = value_q.saturating_mul(SCORE_SCALE) / cost_us.max(1);
                candidates.push(Candidate {
                    job,
                    descriptor,
                    freshness_q,
                    value_q,
                    cost_us,
                    score_q,
                    oldest_success: self.jobs[index].last_success_sequence,
                });
            }
        }

        candidates.sort_by(compare_candidates);
        let eligible_jobs = candidates.len();
        let internal_in_flight = self
            .jobs
            .iter()
            .filter(|state| state.active.is_some())
            .count();
        let in_flight_jobs = inputs.in_flight.max(internal_in_flight);
        let mut predicted_us = 0_u64;
        let mut selected = Vec::with_capacity(MAX_SELECTED_PER_CYCLE.min(JOB_COUNT));
        let mut selected_flags = [false; MAX_JOBS];
        let mut effective_in_flight = in_flight_jobs;
        let executable = self.phase.should_execute();

        if !blocked {
            for candidate in candidates {
                if selected.len() >= MAX_SELECTED_PER_CYCLE {
                    break;
                }
                let lifecycle = self.jobs[candidate.job.index()]
                    .active
                    .map(|active| active.lifecycle);
                if executable && lifecycle == Some(PermitLifecycle::Running) {
                    self.metrics.running_skipped_total =
                        self.metrics.running_skipped_total.saturating_add(1);
                    continue;
                }
                let replacing = executable && lifecycle == Some(PermitLifecycle::Pending);
                if executable && effective_in_flight >= MAX_IN_FLIGHT_OPTIONAL && !replacing {
                    self.metrics.capacity_skipped_total =
                        self.metrics.capacity_skipped_total.saturating_add(1);
                    continue;
                }
                if candidate.cost_us > budget_us.saturating_sub(predicted_us) {
                    self.metrics.budget_skipped_total =
                        self.metrics.budget_skipped_total.saturating_add(1);
                    continue;
                }
                if candidate
                    .descriptor
                    .dependencies
                    .iter()
                    .any(|dependency| !selected_flags[dependency.index()])
                {
                    self.metrics.budget_skipped_total =
                        self.metrics.budget_skipped_total.saturating_add(1);
                    continue;
                }
                let Some(generation) =
                    checked_next_generation(self.jobs[candidate.job.index()].generation)
                else {
                    self.metrics.generation_overflow_total =
                        self.metrics.generation_overflow_total.saturating_add(1);
                    continue;
                };
                let submitted_mono_us = snapshot.cut_completed_mono_us;
                let Some(deadline_mono_us) = submitted_mono_us.checked_add(candidate.cost_us)
                else {
                    self.metrics.deadline_overflow_total =
                        self.metrics.deadline_overflow_total.saturating_add(1);
                    continue;
                };
                if executable {
                    let state = &mut self.jobs[candidate.job.index()];
                    state.generation = generation;
                    let identity = snapshot.identity();
                    state.active = Some(PermitIdentity {
                        identity,
                        generation,
                        submitted_mono_us,
                        deadline_mono_us,
                        lifecycle: PermitLifecycle::Pending,
                    });
                    if !replacing {
                        effective_in_flight = effective_in_flight.saturating_add(1);
                    }
                }
                selected_flags[candidate.job.index()] = true;
                predicted_us = predicted_us.saturating_add(candidate.cost_us);
                selected.push(JobPermit {
                    job: candidate.job,
                    snapshot_id: snapshot.id,
                    workload_id: snapshot.workload_id,
                    capability_revision: snapshot.capability_revision,
                    thermal_power_revision: snapshot.thermal_power_revision,
                    process_identity_revision: snapshot.process_identity_revision,
                    generation,
                    submitted_mono_us,
                    deadline_mono_us,
                    predicted_us: candidate.cost_us,
                    value_q: candidate.value_q,
                    score_q: candidate.score_q,
                    dependencies: candidate.descriptor.dependencies,
                    should_execute: executable,
                });
            }
        }

        let selection_latency_us = started.elapsed().as_micros() as u64;
        self.metrics.eligible_jobs = eligible_jobs as u64;
        self.metrics.selected_jobs = selected.len() as u64;
        self.metrics.selected_total = self
            .metrics
            .selected_total
            .saturating_add(selected.len() as u64);
        self.metrics.in_flight_jobs = effective_in_flight as u64;
        self.metrics.budget_us = budget_us;
        self.metrics.predicted_us = predicted_us;
        self.metrics.selection_latency_us = selection_latency_us;
        self.metrics.max_selection_latency_us = self
            .metrics
            .max_selection_latency_us
            .max(selection_latency_us);

        SchedulerPlan {
            level: inputs.level,
            budget_us,
            predicted_us,
            eligible_jobs,
            selected_jobs: selected.len(),
            in_flight_jobs: effective_in_flight,
            should_execute: self.phase.should_execute() && !blocked,
            selection_latency_us,
            permits: selected,
        }
    }

    pub fn mark_running(&mut self, permit: &JobPermit) -> bool {
        let Some(active) = self.jobs[permit.job.index()].active.as_mut() else {
            return false;
        };
        if active.lifecycle != PermitLifecycle::Pending
            || active.identity != permit.identity()
            || active.generation != permit.generation
            || active.submitted_mono_us != permit.submitted_mono_us
            || active.deadline_mono_us != permit.deadline_mono_us
        {
            return false;
        }
        active.lifecycle = PermitLifecycle::Running;
        true
    }

    pub fn complete(&mut self, completion: JobCompletion) -> CompletionDisposition {
        let active = self.jobs[completion.job.index()].active;
        let permit_matches = active.is_some_and(|active| {
            active.lifecycle == PermitLifecycle::Running
                && active.identity.snapshot_id == completion.snapshot_id
                && active.identity.workload_id == completion.workload_id
                && active.identity.capability_revision == completion.capability_revision
                && active.identity.thermal_power_revision == completion.thermal_power_revision
                && active.identity.process_identity_revision == completion.process_identity_revision
                && active.generation == completion.generation
                && active.deadline_mono_us == completion.deadline_mono_us
        });
        let revision_is_current = self.current_snapshot.is_some_and(|current| {
            current.snapshot_id.daemon_epoch == completion.snapshot_id.daemon_epoch
                && current.snapshot_id.sequence >= completion.snapshot_id.sequence
                && current
                    .snapshot_id
                    .sequence
                    .saturating_sub(completion.snapshot_id.sequence)
                    <= 2
        });
        let identity_matches = active.is_some_and(|active| {
            revision_is_current
                && permit_matches
                && completion.completed_mono_us >= active.submitted_mono_us
                && completion.completed_mono_us <= active.deadline_mono_us
        });
        if !identity_matches {
            // A terminal response for the exact running permit must release its
            // slot even when it missed the acceptance deadline. Its output is
            // still stale and therefore never reaches a consumer.
            if permit_matches {
                self.jobs[completion.job.index()].active = None;
                self.metrics.timed_out_total = self.metrics.timed_out_total.saturating_add(1);
                self.metrics.expired_active_total =
                    self.metrics.expired_active_total.saturating_add(1);
                self.metrics.stale_total = self.metrics.stale_total.saturating_add(1);
                return CompletionDisposition::Stale;
            }
            if self.metrics.completions_seen as usize >= MAX_NON_TERMINAL_COMPLETIONS_PER_CYCLE {
                self.metrics.completion_capacity_dropped_total = self
                    .metrics
                    .completion_capacity_dropped_total
                    .saturating_add(1);
                return CompletionDisposition::Capacity;
            }
            self.metrics.completions_seen = self.metrics.completions_seen.saturating_add(1);
            self.metrics.stale_total = self.metrics.stale_total.saturating_add(1);
            return CompletionDisposition::Stale;
        }
        if self.metrics.completions_seen as usize >= MAX_COMPLETIONS_PER_CYCLE {
            self.metrics.completion_capacity_dropped_total = self
                .metrics
                .completion_capacity_dropped_total
                .saturating_add(1);
            return CompletionDisposition::Capacity;
        }
        self.metrics.completions_seen = self.metrics.completions_seen.saturating_add(1);

        self.jobs[completion.job.index()].active = None;
        self.metrics.completed_total = self.metrics.completed_total.saturating_add(1);
        self.metrics.actual_us = self
            .metrics
            .actual_us
            .saturating_add(u64::from(completion.elapsed_us));
        self.update_cost(completion.job, completion.elapsed_us);
        match completion.status {
            JobCompletionStatus::Succeeded => {
                self.jobs[completion.job.index()].last_success_sequence =
                    completion.snapshot_id.sequence;
                self.metrics.success_total = self.metrics.success_total.saturating_add(1);
            }
            JobCompletionStatus::Failed => {
                self.metrics.failed_total = self.metrics.failed_total.saturating_add(1)
            }
            JobCompletionStatus::Cancelled => {
                self.metrics.cancelled_total = self.metrics.cancelled_total.saturating_add(1)
            }
            JobCompletionStatus::TimedOut => {
                self.metrics.timed_out_total = self.metrics.timed_out_total.saturating_add(1)
            }
        }
        CompletionDisposition::Accepted
    }

    fn expire_stale_active(&mut self, current: SnapshotIdentity) {
        for state in &mut self.jobs {
            let expired = state.active.is_some_and(|active| {
                active.identity.snapshot_id.daemon_epoch != current.snapshot_id.daemon_epoch
                    || active.identity.workload_id != current.workload_id
                    || active.identity.capability_revision != current.capability_revision
                    || active.identity.thermal_power_revision != current.thermal_power_revision
                    || active.identity.process_identity_revision
                        != current.process_identity_revision
                    || current
                        .snapshot_id
                        .sequence
                        .saturating_sub(active.identity.snapshot_id.sequence)
                        > 2
            });
            if expired {
                state.active = None;
                self.metrics.timed_out_total = self.metrics.timed_out_total.saturating_add(1);
                self.metrics.expired_active_total =
                    self.metrics.expired_active_total.saturating_add(1);
            }
        }
    }

    fn rejected_plan(
        &mut self,
        level: SchedulerLevel,
        budget_us: u64,
        started: Instant,
    ) -> SchedulerPlan {
        let selection_latency_us = started.elapsed().as_micros() as u64;
        let in_flight_jobs = self
            .jobs
            .iter()
            .filter(|state| state.active.is_some())
            .count();
        self.metrics.eligible_jobs = 0;
        self.metrics.selected_jobs = 0;
        self.metrics.in_flight_jobs = in_flight_jobs as u64;
        self.metrics.selection_latency_us = selection_latency_us;
        self.metrics.max_selection_latency_us = self
            .metrics
            .max_selection_latency_us
            .max(selection_latency_us);
        SchedulerPlan {
            level,
            budget_us,
            in_flight_jobs,
            selection_latency_us,
            ..SchedulerPlan::default()
        }
    }

    pub fn observe_legacy_run(
        &mut self,
        job: JobId,
        snapshot_id: SnapshotId,
        elapsed_us: u32,
        succeeded: bool,
    ) {
        self.update_cost(job, elapsed_us);
        if succeeded {
            self.jobs[job.index()].last_success_sequence = snapshot_id.sequence;
        }
    }

    fn cost_for(&self, job: JobId, override_cost: Option<f64>) -> (u64, bool) {
        let state = self.jobs[job.index()];
        let descriptor = job.descriptor();
        if let Some(cost) = override_cost {
            if !cost.is_finite() || cost <= 0.0 {
                return (MAX_JOB_SLICE_US, true);
            }
            let observed = cost.round() as u64;
            let old_ewma = state.ewma_us.max(MIN_COST_ESTIMATE_US);
            let ewma = ((7 * old_ewma).saturating_add(observed)) / 8;
            return (
                descriptor
                    .static_floor_us
                    .max(ewma)
                    .max(observed)
                    .clamp(MIN_COST_ESTIMATE_US, MAX_JOB_SLICE_US),
                false,
            );
        }
        (
            state
                .last_observed_us
                .max(state.ewma_us)
                .max(descriptor.static_floor_us)
                .clamp(MIN_COST_ESTIMATE_US, MAX_JOB_SLICE_US),
            false,
        )
    }

    fn update_cost(&mut self, job: JobId, elapsed_us: u32) {
        let observed = sanitize_elapsed(elapsed_us);
        let state = &mut self.jobs[job.index()];
        let old_ewma = state
            .ewma_us
            .max(job.descriptor().static_floor_us)
            .max(MIN_COST_ESTIMATE_US);
        state.ewma_us = ((7 * old_ewma).saturating_add(observed)) / 8;
        state.ewma_us = state.ewma_us.clamp(MIN_COST_ESTIMATE_US, MAX_JOB_SLICE_US);
        state.last_observed_us = observed;
    }
}

fn freshness_q(elapsed_us: u64, target_max_interval_us: u64) -> u64 {
    if target_max_interval_us == 0 {
        return Q_MAX;
    }
    elapsed_us
        .saturating_mul(Q_MAX)
        .checked_div(target_max_interval_us)
        .unwrap_or(Q_MAX)
        .min(Q_MAX)
}

fn sanitize_signal(signal: f64) -> u64 {
    if !signal.is_finite() {
        return 0;
    }
    signal.round().clamp(0.0, Q_MAX as f64) as u64
}

fn sanitize_elapsed(elapsed_us: u32) -> u64 {
    if elapsed_us == 0 {
        MAX_JOB_SLICE_US
    } else {
        u64::from(elapsed_us).clamp(MIN_COST_ESTIMATE_US, MAX_JOB_SLICE_US)
    }
}

const fn checked_next_generation(current: u64) -> Option<u64> {
    current.checked_add(1)
}

fn compare_candidates(left: &Candidate, right: &Candidate) -> Ordering {
    right
        .score_q
        .cmp(&left.score_q)
        .then_with(|| right.freshness_q.cmp(&left.freshness_q))
        .then_with(|| left.oldest_success.cmp(&right.oldest_success))
        .then_with(|| left.job.cmp(&right.job))
}

#[cfg(test)]
mod tests {
    use super::checked_next_generation;

    #[test]
    fn checked_generation_rejects_overflow() {
        assert_eq!(checked_next_generation(0), Some(1));
        assert_eq!(checked_next_generation(u64::MAX - 1), Some(u64::MAX));
        assert_eq!(checked_next_generation(u64::MAX), None);
    }
}
