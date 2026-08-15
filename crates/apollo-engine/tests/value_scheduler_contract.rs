use std::collections::HashSet;
use std::sync::{Arc, Barrier};
use std::thread;

use apollo_engine::engine::cycle_snapshot::{
    CycleContextSnapshot, ObservationStatus, SnapshotId, SnapshotPublishError, SnapshotPublisher,
    SourceObservation,
};
use apollo_engine::engine::value_scheduler::{
    CompletionDisposition, JobCompletion, JobCompletionStatus, JobId, SchedulerInputs,
    SchedulerLevel, SchedulerPhase, ValueScheduler, CONSTRAINED_BUDGET_US, GUARDED_BUDGET_US,
    MAX_COMPLETIONS_PER_CYCLE, MAX_DEPENDENCIES_PER_JOB, MAX_IN_FLIGHT_OPTIONAL, MAX_JOBS,
    MAX_JOB_SLICE_US, MAX_SELECTED_PER_CYCLE, MIN_COST_ESTIMATE_US, NOMINAL_BUDGET_US,
};

macro_rules! assert_not_impl {
    ($type:ty: $trait:path) => {
        const _: fn() = || {
            trait AmbiguousIfImpl<A> {
                fn marker() {}
            }

            impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
            impl<T: ?Sized + $trait> AmbiguousIfImpl<u8> for T {}

            let _ = <$type as AmbiguousIfImpl<_>>::marker;
        };
    };
}

assert_not_impl!(SnapshotPublisher: Clone);
assert_not_impl!(ValueScheduler: Clone);

fn fixture_snapshot(sequence: u64) -> CycleContextSnapshot {
    CycleContextSnapshot::new(
        SnapshotId {
            daemon_epoch: 7,
            sequence,
        },
        11,
        13,
        17,
    )
    .with_cut_times(
        sequence.saturating_mul(100_000),
        sequence.saturating_mul(100_000).saturating_add(100),
    )
}

fn completion_for(permit: &apollo_engine::engine::value_scheduler::JobPermit) -> JobCompletion {
    completion_with_status(permit, 500, JobCompletionStatus::Succeeded)
}

fn completion_with_status(
    permit: &apollo_engine::engine::value_scheduler::JobPermit,
    elapsed_us: u32,
    status: JobCompletionStatus,
) -> JobCompletion {
    JobCompletion::from_permit(
        permit,
        permit
            .submitted_mono_us
            .saturating_add(u64::from(elapsed_us)),
        elapsed_us,
        status,
    )
}

#[test]
fn fixed_registry_has_exact_bounds_and_no_duplicate_jobs() {
    assert_eq!(MAX_JOBS, 64);
    assert_eq!(MAX_SELECTED_PER_CYCLE, 16);
    assert_eq!(MAX_DEPENDENCIES_PER_JOB, 4);
    assert_eq!(MAX_IN_FLIGHT_OPTIONAL, 4);
    assert_eq!(MAX_COMPLETIONS_PER_CYCLE, 64);
    assert_eq!(MIN_COST_ESTIMATE_US, 50);

    assert_eq!(
        JobId::ALL,
        [
            JobId::GpuImagination,
            JobId::ReflexReasoningRefresh,
            JobId::WorldModelRefresh,
            JobId::AisRuntimeRefresh,
            JobId::HardwarePrediction,
            JobId::HoltWintersRefresh,
            JobId::PageReclaimRefresh,
            JobId::PlannerAdviceRefresh,
            JobId::PeriodicLearningMaintenance,
            JobId::TelemetryFlush,
        ]
    );
    let ids: HashSet<_> = JobId::ALL.into_iter().collect();
    assert_eq!(ids.len(), JobId::ALL.len());
    assert_eq!(JobId::ALL.len(), 10);

    fn visit(job: JobId, visiting: &mut [bool; MAX_JOBS], visited: &mut [bool; MAX_JOBS]) {
        let index = job.index();
        assert!(!visiting[index], "dependency cycle at {}", job.as_str());
        if visited[index] {
            return;
        }
        visiting[index] = true;
        for &dependency in job.descriptor().dependencies {
            visit(dependency, visiting, visited);
        }
        visiting[index] = false;
        visited[index] = true;
    }

    let mut visiting = [false; MAX_JOBS];
    let mut visited = [false; MAX_JOBS];
    for descriptor in JobId::ALL.iter().map(|job| job.descriptor()) {
        assert_eq!(descriptor.id.descriptor(), descriptor);
        assert!((1..=10_000).contains(&descriptor.base_value_q));
        assert!(descriptor.target_max_interval_us > 0);
        assert!((MIN_COST_ESTIMATE_US..=MAX_JOB_SLICE_US).contains(&descriptor.static_floor_us));
        assert!(descriptor.dependencies.len() <= MAX_DEPENDENCIES_PER_JOB);
        assert!(!descriptor.dependencies.contains(&descriptor.id));
        visit(descriptor.id, &mut visiting, &mut visited);
    }
}

#[test]
fn value_and_score_follow_the_exact_integer_formula() {
    let job = JobId::GpuImagination;
    let mut inputs = SchedulerInputs::with_due_jobs(SchedulerLevel::Nominal, &[job]);
    inputs.set_signal_q(job, 2_500);
    inputs.set_consecutive_budget_skips(job, 3);

    let mut scheduler = ValueScheduler::new(SchedulerPhase::Shadow);
    let permit = scheduler
        .plan(&fixture_snapshot(1), inputs)
        .permits
        .into_iter()
        .next()
        .expect("permit");

    assert_eq!(permit.value_q, 7_200);
    assert_eq!(permit.predicted_us, 4_000);
    assert_eq!(permit.score_q, 1_800_000);
}

#[test]
fn deterministic_ranking_is_independent_of_due_input_order() {
    let snapshot = fixture_snapshot(7);
    let first_inputs = SchedulerInputs::with_due_jobs(
        apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
        &[
            JobId::TelemetryFlush,
            JobId::GpuImagination,
            JobId::WorldModelRefresh,
            JobId::HoltWintersRefresh,
        ],
    );
    let second_inputs = SchedulerInputs::with_due_jobs(
        apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
        &[
            JobId::HoltWintersRefresh,
            JobId::WorldModelRefresh,
            JobId::GpuImagination,
            JobId::TelemetryFlush,
        ],
    );

    let mut first_scheduler = ValueScheduler::new(SchedulerPhase::Shadow);
    let mut second_scheduler = ValueScheduler::new(SchedulerPhase::Shadow);
    let first = first_scheduler.plan(&snapshot, first_inputs);
    let second = second_scheduler.plan(&snapshot, second_inputs);

    let first_jobs: Vec<_> = first.permits.iter().map(|permit| permit.job).collect();
    let second_jobs: Vec<_> = second.permits.iter().map(|permit| permit.job).collect();
    assert_eq!(first_jobs, second_jobs);
    assert_eq!(
        first
            .permits
            .iter()
            .map(|permit| (permit.job, permit.score_q))
            .collect::<Vec<_>>(),
        second
            .permits
            .iter()
            .map(|permit| (permit.job, permit.score_q))
            .collect::<Vec<_>>()
    );
}

#[test]
fn predicted_plan_never_exceeds_budget_or_sixteen_jobs() {
    for (sequence, level, budget) in [
        (1, SchedulerLevel::Nominal, NOMINAL_BUDGET_US),
        (2, SchedulerLevel::Guarded, GUARDED_BUDGET_US),
        (3, SchedulerLevel::Constrained, CONSTRAINED_BUDGET_US),
    ] {
        let mut scheduler = ValueScheduler::new(SchedulerPhase::Shadow);
        let plan = scheduler.plan(&fixture_snapshot(sequence), SchedulerInputs::all_due(level));
        assert_eq!(plan.level, level);
        assert_eq!(plan.budget_us, budget);
        assert!(plan.permits.len() <= MAX_SELECTED_PER_CYCLE);
        assert_eq!(plan.permits.len(), JobId::ALL.len());
        assert!(plan.predicted_us <= budget);
        assert!(!plan.should_execute);
        assert!(plan.permits.iter().all(|permit| !permit.should_execute));
    }

    let mut active = ValueScheduler::new(SchedulerPhase::Active);
    let plan = active.plan(&fixture_snapshot(4), SchedulerInputs::nominal_all_due());
    assert_eq!(plan.permits.len(), MAX_IN_FLIGHT_OPTIONAL);
    assert_eq!(plan.in_flight_jobs, MAX_IN_FLIGHT_OPTIONAL);
    assert!(plan.permits.iter().all(|permit| permit.should_execute));
}

#[test]
fn every_kill_or_sleep_gate_admits_zero_and_wake_collapses_missed_intervals() {
    for phase in [SchedulerPhase::Shadow, SchedulerPhase::Active] {
        let mut scheduler = ValueScheduler::new(phase);

        let mut snapshot_kill = fixture_snapshot(1);
        snapshot_kill.kill_switch = true;
        assert!(scheduler
            .plan(&snapshot_kill, SchedulerInputs::nominal_all_due())
            .permits
            .is_empty());

        let mut snapshot_sleep = fixture_snapshot(2);
        snapshot_sleep.sleeping = true;
        assert!(scheduler
            .plan(&snapshot_sleep, SchedulerInputs::nominal_all_due())
            .permits
            .is_empty());

        assert!(scheduler
            .plan(
                &fixture_snapshot(3),
                SchedulerInputs::nominal_all_due().with_kill_switch(true),
            )
            .permits
            .is_empty());
        assert!(scheduler
            .plan(
                &fixture_snapshot(4),
                SchedulerInputs::nominal_all_due().with_sleeping(true),
            )
            .permits
            .is_empty());

        let wake = scheduler.plan(&fixture_snapshot(5), SchedulerInputs::nominal_all_due());
        let expected = if phase == SchedulerPhase::Shadow {
            JobId::ALL.len()
        } else {
            MAX_IN_FLIGHT_OPTIONAL
        };
        assert_eq!(wake.permits.len(), expected);
    }
}

#[test]
fn shadow_recommendations_never_occupy_executable_capacity() {
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Shadow);

    for sequence in 1..=3 {
        let plan = scheduler.plan(
            &fixture_snapshot(sequence),
            SchedulerInputs::nominal_all_due(),
        );
        assert_eq!(plan.permits.len(), JobId::ALL.len());
        assert_eq!(plan.in_flight_jobs, 0);
        assert!(!plan.should_execute);
        assert!(plan.permits.iter().all(|permit| !permit.should_execute));
    }

    assert_eq!(scheduler.metrics().in_flight_jobs, 0);
    assert_eq!(scheduler.metrics().capacity_skipped_total, 0);
}

#[test]
fn invalid_costs_are_finite_bounded_and_conservative() {
    let mut inputs = SchedulerInputs::nominal_all_due();
    inputs.set_cost_estimate_us(JobId::GpuImagination, f64::NAN);
    inputs.set_cost_estimate_us(JobId::WorldModelRefresh, f64::INFINITY);
    inputs.set_cost_estimate_us(JobId::AisRuntimeRefresh, 0.0);
    inputs.set_cost_estimate_us(JobId::HardwarePrediction, -1.0);

    let mut scheduler = ValueScheduler::new(SchedulerPhase::Shadow);
    let plan = scheduler.plan(&fixture_snapshot(7), inputs);
    assert!(plan.predicted_us <= NOMINAL_BUDGET_US);
    assert!(plan
        .permits
        .iter()
        .all(|permit| permit.predicted_us >= MIN_COST_ESTIMATE_US));
    assert!(plan
        .permits
        .iter()
        .all(|permit| permit.predicted_us <= MAX_JOB_SLICE_US));
    assert!(scheduler.metrics().invalid_cost_total >= 4);
}

#[test]
fn latest_generation_replaces_older_pending_work() {
    let job = JobId::GpuImagination;
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);
    let first_snapshot = fixture_snapshot(1);
    let first_inputs = SchedulerInputs::with_due_jobs(
        apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
        &[job],
    );
    let first = scheduler
        .plan(&first_snapshot, first_inputs)
        .permits
        .into_iter()
        .next()
        .expect("first permit");

    let second = scheduler
        .plan(
            &fixture_snapshot(2),
            SchedulerInputs::with_due_jobs(
                apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
                &[job],
            ),
        )
        .permits
        .into_iter()
        .next()
        .expect("replacement permit");

    assert_eq!(second.job, job);
    assert_eq!(second.generation, first.generation + 1);
    assert_eq!(
        scheduler.complete(completion_for(&first)),
        CompletionDisposition::Stale
    );
    assert!(scheduler.mark_running(&second));
    assert_eq!(
        scheduler.complete(completion_for(&second)),
        CompletionDisposition::Accepted
    );
}

#[test]
fn running_generation_is_never_replaced_or_over_issued() {
    let job = JobId::GpuImagination;
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);
    let running = scheduler
        .plan(
            &fixture_snapshot(1),
            SchedulerInputs::with_due_jobs(
                apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
                &[job],
            ),
        )
        .permits
        .into_iter()
        .next()
        .expect("pending permit");

    assert!(scheduler.mark_running(&running));
    assert!(!scheduler.mark_running(&running));

    let next = scheduler.plan(
        &fixture_snapshot(2),
        SchedulerInputs::with_due_jobs(
            apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
            &[job],
        ),
    );
    assert!(next.permits.is_empty());
    assert_eq!(next.in_flight_jobs, 1);
    assert_eq!(
        scheduler.complete(completion_for(&running)),
        CompletionDisposition::Accepted
    );
}

#[test]
fn completion_requires_a_running_generation() {
    let job = JobId::GpuImagination;
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);
    let pending = scheduler
        .plan(
            &fixture_snapshot(1),
            SchedulerInputs::with_due_jobs(
                apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
                &[job],
            ),
        )
        .permits
        .into_iter()
        .next()
        .expect("pending permit");

    assert_eq!(
        scheduler.complete(completion_for(&pending)),
        CompletionDisposition::Stale
    );
    assert!(scheduler.mark_running(&pending));
    assert_eq!(
        scheduler.complete(completion_for(&pending)),
        CompletionDisposition::Accepted
    );
}

#[test]
fn stale_wrong_epoch_wrong_revision_and_wrong_identity_completions_drop() {
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);
    let permit = scheduler
        .plan(&fixture_snapshot(9), SchedulerInputs::nominal_all_due())
        .permits
        .into_iter()
        .next()
        .expect("permit");
    assert!(scheduler.mark_running(&permit));

    let mut wrong_epoch = completion_for(&permit);
    wrong_epoch.snapshot_id = SnapshotId {
        daemon_epoch: 8,
        sequence: 9,
    };
    assert_eq!(
        scheduler.complete(wrong_epoch),
        CompletionDisposition::Stale
    );

    let mut wrong_workload = completion_for(&permit);
    wrong_workload.workload_id += 1;
    assert_eq!(
        scheduler.complete(wrong_workload),
        CompletionDisposition::Stale
    );

    let mut wrong_capability = completion_for(&permit);
    wrong_capability.capability_revision += 1;
    assert_eq!(
        scheduler.complete(wrong_capability),
        CompletionDisposition::Stale
    );

    let mut wrong_thermal = completion_for(&permit);
    wrong_thermal.thermal_power_revision += 1;
    assert_eq!(
        scheduler.complete(wrong_thermal),
        CompletionDisposition::Stale
    );

    let mut wrong_process = completion_for(&permit);
    wrong_process.process_identity_revision += 1;
    assert_eq!(
        scheduler.complete(wrong_process),
        CompletionDisposition::Stale
    );

    let mut wrong_generation = completion_for(&permit);
    wrong_generation.generation += 1;
    assert_eq!(
        scheduler.complete(wrong_generation),
        CompletionDisposition::Stale
    );
}

#[test]
fn completion_requires_matching_unexpired_monotonic_deadline() {
    let job = JobId::GpuImagination;
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);
    let permit = scheduler
        .plan(
            &fixture_snapshot(1),
            SchedulerInputs::with_due_jobs(
                apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
                &[job],
            ),
        )
        .permits
        .into_iter()
        .next()
        .expect("permit");
    assert!(scheduler.mark_running(&permit));

    let mut wrong_deadline = completion_for(&permit);
    wrong_deadline.deadline_mono_us -= 1;
    assert_eq!(
        scheduler.complete(wrong_deadline),
        CompletionDisposition::Stale
    );

    let mut before_submission = completion_for(&permit);
    before_submission.completed_mono_us = permit.submitted_mono_us.saturating_sub(1);
    assert_eq!(
        scheduler.complete(before_submission),
        CompletionDisposition::Stale
    );

    let mut expired = completion_for(&permit);
    expired.completed_mono_us = permit.deadline_mono_us.saturating_add(1);
    assert_eq!(scheduler.complete(expired), CompletionDisposition::Stale);

    assert_eq!(
        scheduler.complete(completion_for(&permit)),
        CompletionDisposition::Stale
    );
}

#[test]
fn exact_late_completion_releases_the_running_slot() {
    let job = JobId::GpuImagination;
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);
    let permit = scheduler
        .plan(
            &fixture_snapshot(1),
            SchedulerInputs::with_due_jobs(SchedulerLevel::Nominal, &[job]),
        )
        .permits
        .into_iter()
        .next()
        .expect("permit");
    assert!(scheduler.mark_running(&permit));

    let mut late = completion_with_status(&permit, 500, JobCompletionStatus::TimedOut);
    late.completed_mono_us = permit.deadline_mono_us.saturating_add(1);
    assert_eq!(scheduler.complete(late), CompletionDisposition::Stale);

    let next = scheduler.plan(
        &fixture_snapshot(2),
        SchedulerInputs::with_due_jobs(SchedulerLevel::Nominal, &[job]),
    );
    assert!(next.permits.iter().any(|candidate| candidate.job == job));
    assert_eq!(scheduler.metrics().timed_out_total, 1);
}

#[test]
fn next_snapshot_expires_a_lost_running_job() {
    let job = JobId::GpuImagination;
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);
    let permit = scheduler
        .plan(
            &fixture_snapshot(1),
            SchedulerInputs::with_due_jobs(SchedulerLevel::Nominal, &[job]),
        )
        .permits
        .into_iter()
        .next()
        .expect("permit");
    assert!(scheduler.mark_running(&permit));

    scheduler.plan(&fixture_snapshot(2), SchedulerInputs::default());
    scheduler.plan(&fixture_snapshot(3), SchedulerInputs::default());
    let next = scheduler.plan(
        &fixture_snapshot(4),
        SchedulerInputs::with_due_jobs(SchedulerLevel::Nominal, &[job]),
    );
    assert!(next.permits.iter().any(|candidate| candidate.job == job));
    assert_eq!(scheduler.metrics().timed_out_total, 1);
}

#[test]
fn completion_accepts_age_two_and_rejects_age_three_even_when_cycles_are_blocked() {
    let job = JobId::GpuImagination;

    let mut age_two = ValueScheduler::new(SchedulerPhase::Active);
    let permit = age_two
        .plan(
            &fixture_snapshot(10),
            SchedulerInputs::with_due_jobs(
                apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
                &[job],
            ),
        )
        .permits
        .into_iter()
        .next()
        .expect("permit");
    assert!(age_two.mark_running(&permit));
    age_two.plan(
        &fixture_snapshot(11),
        SchedulerInputs::default().with_kill_switch(true),
    );
    age_two.plan(
        &fixture_snapshot(12),
        SchedulerInputs::default().with_sleeping(true),
    );
    let mut completion = completion_for(&permit);
    completion.completed_mono_us = permit.deadline_mono_us;
    assert_eq!(
        age_two.complete(completion),
        CompletionDisposition::Accepted
    );

    let mut age_three = ValueScheduler::new(SchedulerPhase::Active);
    let permit = age_three
        .plan(
            &fixture_snapshot(20),
            SchedulerInputs::with_due_jobs(
                apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
                &[job],
            ),
        )
        .permits
        .into_iter()
        .next()
        .expect("permit");
    assert!(age_three.mark_running(&permit));
    for sequence in 21..=23 {
        age_three.plan(&fixture_snapshot(sequence), SchedulerInputs::default());
    }
    assert_eq!(
        age_three.complete(completion_for(&permit)),
        CompletionDisposition::Stale
    );
}

#[test]
fn foreign_duplicate_and_regressive_plans_are_rejected_without_state_reset() {
    let job = JobId::GpuImagination;
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);
    let permit = scheduler
        .plan(
            &fixture_snapshot(10),
            SchedulerInputs::with_due_jobs(
                apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
                &[job],
            ),
        )
        .permits
        .into_iter()
        .next()
        .expect("permit");

    assert!(scheduler
        .plan(&fixture_snapshot(10), SchedulerInputs::nominal_all_due())
        .permits
        .is_empty());
    assert!(scheduler
        .plan(&fixture_snapshot(9), SchedulerInputs::nominal_all_due())
        .permits
        .is_empty());
    let mut foreign = fixture_snapshot(11);
    foreign.id.daemon_epoch += 1;
    assert!(scheduler
        .plan(&foreign, SchedulerInputs::nominal_all_due())
        .permits
        .is_empty());

    assert_eq!(scheduler.metrics().regressive_snapshot_total, 2);
    assert_eq!(scheduler.metrics().foreign_snapshot_total, 1);
    assert!(scheduler.mark_running(&permit));
    assert_eq!(
        scheduler.complete(completion_for(&permit)),
        CompletionDisposition::Accepted
    );
}

#[test]
fn monotonic_deadline_overflow_rejects_the_job_conservatively() {
    let mut snapshot = fixture_snapshot(1);
    snapshot.cut_started_mono_us = u64::MAX - 10;
    snapshot.cut_completed_mono_us = u64::MAX - 10;
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);

    let plan = scheduler.plan(&snapshot, SchedulerInputs::nominal_all_due());

    assert!(plan.permits.is_empty());
    assert_eq!(
        scheduler.metrics().deadline_overflow_total,
        JobId::ALL.len() as u64
    );
}

#[test]
fn completion_accounting_is_bounded_and_terminal() {
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);
    let snapshot = fixture_snapshot(9);
    let plan = scheduler.plan(&snapshot, SchedulerInputs::nominal_all_due());
    let permit = plan.permits[0].clone();
    assert!(scheduler.mark_running(&permit));
    assert_eq!(
        scheduler.complete(completion_with_status(
            &permit,
            700,
            JobCompletionStatus::Failed,
        )),
        CompletionDisposition::Accepted
    );
    assert_eq!(
        scheduler.complete(completion_with_status(
            &permit,
            700,
            JobCompletionStatus::Failed,
        )),
        CompletionDisposition::Stale
    );
    assert_eq!(scheduler.metrics().failed_total, 1);
    assert_eq!(scheduler.metrics().completed_total, 1);
    assert!(scheduler.metrics().completions_seen <= MAX_COMPLETIONS_PER_CYCLE as u64);
}

#[test]
fn every_terminal_status_is_accounted_exactly_once() {
    let statuses = [
        JobCompletionStatus::Succeeded,
        JobCompletionStatus::Failed,
        JobCompletionStatus::Cancelled,
        JobCompletionStatus::TimedOut,
    ];
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);

    for (offset, status) in statuses.into_iter().enumerate() {
        let sequence = offset as u64 + 1;
        let permit = scheduler
            .plan(
                &fixture_snapshot(sequence),
                SchedulerInputs::with_due_jobs(SchedulerLevel::Nominal, &[JobId::TelemetryFlush]),
            )
            .permits
            .into_iter()
            .next()
            .expect("permit");
        assert!(scheduler.mark_running(&permit));
        assert_eq!(
            scheduler.complete(completion_with_status(&permit, 500, status)),
            CompletionDisposition::Accepted
        );
    }

    assert_eq!(scheduler.metrics().completed_total, 4);
    assert_eq!(scheduler.metrics().success_total, 1);
    assert_eq!(scheduler.metrics().failed_total, 1);
    assert_eq!(scheduler.metrics().cancelled_total, 1);
    assert_eq!(scheduler.metrics().timed_out_total, 1);
}

#[test]
fn stale_flood_is_bounded_without_suppressing_a_valid_terminal() {
    let job = JobId::GpuImagination;
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);
    let permit = scheduler
        .plan(
            &fixture_snapshot(1),
            SchedulerInputs::with_due_jobs(
                apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
                &[job],
            ),
        )
        .permits
        .into_iter()
        .next()
        .expect("permit");
    assert!(scheduler.mark_running(&permit));

    let mut stale_count = 0;
    let mut capacity_count = 0;
    for _ in 0..MAX_COMPLETIONS_PER_CYCLE {
        let mut stale = completion_for(&permit);
        stale.snapshot_id.daemon_epoch += 1;
        match scheduler.complete(stale) {
            CompletionDisposition::Stale => stale_count += 1,
            CompletionDisposition::Capacity => capacity_count += 1,
            CompletionDisposition::Accepted => panic!("foreign completion accepted"),
        }
    }

    assert_eq!(
        stale_count,
        MAX_COMPLETIONS_PER_CYCLE - MAX_IN_FLIGHT_OPTIONAL
    );
    assert_eq!(capacity_count, MAX_IN_FLIGHT_OPTIONAL);
    assert_eq!(
        scheduler.metrics().completion_capacity_dropped_total,
        MAX_IN_FLIGHT_OPTIONAL as u64
    );
    assert_eq!(
        scheduler.complete(completion_for(&permit)),
        CompletionDisposition::Accepted
    );
    assert!(scheduler.metrics().completions_seen <= MAX_COMPLETIONS_PER_CYCLE as u64);
}

#[test]
fn successful_completion_records_snapshot_sequence_for_oldest_success_ranking() {
    let older = JobId::WorldModelRefresh;
    let newer = JobId::GpuImagination;
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Active);

    let older_permit = scheduler
        .plan(
            &fixture_snapshot(10),
            SchedulerInputs::with_due_jobs(
                apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
                &[older],
            ),
        )
        .permits
        .into_iter()
        .next()
        .expect("older permit");
    assert!(scheduler.mark_running(&older_permit));
    assert_eq!(
        scheduler.complete(completion_for(&older_permit)),
        CompletionDisposition::Accepted
    );

    let newer_permit = scheduler
        .plan(
            &fixture_snapshot(100),
            SchedulerInputs::with_due_jobs(
                apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
                &[newer],
            ),
        )
        .permits
        .into_iter()
        .next()
        .expect("newer permit");
    assert!(scheduler.mark_running(&newer_permit));
    assert_eq!(
        scheduler.complete(completion_for(&newer_permit)),
        CompletionDisposition::Accepted
    );

    assert_eq!(scheduler.last_success_sequence(older), Some(10));
    assert_eq!(scheduler.last_success_sequence(newer), Some(100));

    let mut tied = SchedulerInputs::with_due_jobs(
        apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
        &[newer, older],
    );
    tied.set_signal_q(newer, 800);
    tied.set_cost_estimate_us(newer, 6_000.0);
    tied.set_cost_estimate_us(older, 6_000.0);
    let ranked = scheduler.plan(&fixture_snapshot(101), tied);
    assert_eq!(ranked.permits[0].job, older);
    assert_eq!(ranked.permits[0].score_q, ranked.permits[1].score_q);
    assert_eq!(ranked.permits[0].value_q, ranked.permits[1].value_q);
}

#[test]
fn legacy_observations_update_cost_without_granting_action_authority() {
    let snapshot = fixture_snapshot(3);
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Shadow);
    scheduler.observe_legacy_run(JobId::TelemetryFlush, snapshot.id, 2_000, true);
    scheduler.observe_legacy_run(JobId::TelemetryFlush, snapshot.id, 0, false);

    let plan = scheduler.plan(
        &snapshot,
        SchedulerInputs::with_due_jobs(
            apollo_engine::engine::value_scheduler::SchedulerLevel::Nominal,
            &[JobId::TelemetryFlush],
        ),
    );
    assert!(!plan.should_execute);
    assert_eq!(plan.permits.len(), 1);
    assert!(plan.permits[0].predicted_us >= MIN_COST_ESTIMATE_US);
}

#[test]
fn snapshot_publication_is_immutable_and_revision_overflow_is_checked() {
    let mut publisher = SnapshotPublisher::new(7);
    let first = publisher
        .publish(CycleContextSnapshot::new(
            SnapshotId {
                daemon_epoch: 0,
                sequence: 0,
            },
            11,
            13,
            17,
        ))
        .expect("first publication");
    let second = publisher
        .publish(CycleContextSnapshot::new(
            SnapshotId {
                daemon_epoch: 0,
                sequence: 0,
            },
            22,
            23,
            29,
        ))
        .expect("second publication");

    assert_eq!(
        first.id,
        SnapshotId {
            daemon_epoch: 7,
            sequence: 1
        }
    );
    assert_eq!(
        second.id,
        SnapshotId {
            daemon_epoch: 7,
            sequence: 2
        }
    );
    assert_eq!(first.workload_id, 11);
    assert_eq!(second.workload_id, 22);
    assert_eq!(publisher.latest().expect("latest").id, second.id);

    let mut exhausted = SnapshotPublisher::with_next_sequence(7, u64::MAX);
    let result = exhausted.publish(CycleContextSnapshot::default());
    assert_eq!(result, Err(SnapshotPublishError::SequenceExhausted));
}

#[test]
fn source_values_are_available_only_while_fresh() {
    let fresh = fixture_snapshot(1)
        .with_pressure(SourceObservation::fresh(7_000, 2, 3, 4))
        .with_thermal(SourceObservation::fresh(5_000, 5, 6, 7))
        .with_interaction(SourceObservation::fresh(3_000, 8, 9, 10));
    assert_eq!(fresh.pressure_q(), Some(7_000));
    assert_eq!(fresh.thermal_q(), Some(5_000));
    assert_eq!(fresh.interaction_q(), Some(3_000));
    assert_eq!(fresh.pressure().status(), ObservationStatus::Fresh);
    assert_eq!(fresh.pressure().generation(), 2);
    assert_eq!(fresh.pressure().revision(), 3);
    assert_eq!(fresh.pressure().age_us_at_cut(), 4);

    let stale =
        fixture_snapshot(2).with_pressure(SourceObservation::stale(Some(9_000), 12, 13, 11));
    assert_eq!(stale.pressure_q(), None);
    assert_eq!(stale.pressure().fresh_value(), None);
    assert_eq!(stale.pressure().status(), ObservationStatus::Stale);

    for observation in [
        SourceObservation::unavailable(1, 2),
        SourceObservation::invalid(1, 2),
        SourceObservation::truncated(1, 2),
    ] {
        let snapshot = fixture_snapshot(3).with_pressure(observation);
        assert_eq!(snapshot.pressure_q(), None);
        assert_eq!(snapshot.pressure().fresh_value(), None);
    }
}

#[test]
fn deserialization_cannot_make_stale_or_invalid_sources_look_fresh() {
    let mut encoded = serde_json::to_value(
        fixture_snapshot(1).with_pressure(SourceObservation::fresh(7_000, 2, 3, 4)),
    )
    .expect("serialize snapshot");
    encoded["pressure"]["status"] = serde_json::json!("stale");
    encoded["pressure"]["value"] = serde_json::json!(9_000);

    let stale: CycleContextSnapshot =
        serde_json::from_value(encoded).expect("deserialize stale source");
    assert_eq!(stale.pressure().status(), ObservationStatus::Stale);
    assert_eq!(stale.pressure_q(), None);

    let mut encoded = serde_json::to_value(
        fixture_snapshot(2).with_pressure(SourceObservation::fresh(7_000, 2, 3, 4)),
    )
    .expect("serialize snapshot");
    encoded["pressure"]["status"] = serde_json::json!("fresh");
    encoded["pressure"]["value"] = serde_json::Value::Null;

    let missing: CycleContextSnapshot =
        serde_json::from_value(encoded).expect("deserialize missing fresh value");
    assert_eq!(missing.pressure().status(), ObservationStatus::Invalid);
    assert_eq!(missing.pressure_q(), None);
}

#[test]
fn concurrent_readers_observe_whole_immutable_publications() {
    let mut publisher = SnapshotPublisher::new(7);
    let first = publisher
        .publish(fixture_snapshot(0).with_pressure(SourceObservation::fresh(1_111, 1, 1, 0)))
        .expect("first publication");
    let barrier = Arc::new(Barrier::new(2));
    let reader_barrier = Arc::clone(&barrier);
    let old_reader = thread::spawn(move || {
        reader_barrier.wait();
        (first.id, first.workload_id, first.pressure_q())
    });

    let mut second_snapshot =
        fixture_snapshot(0).with_pressure(SourceObservation::fresh(9_999, 9, 9, 0));
    second_snapshot.workload_id = 99;
    let second = publisher
        .publish(second_snapshot)
        .expect("second publication");
    barrier.wait();

    assert_eq!(
        old_reader.join().expect("reader joined"),
        (SnapshotId::new(7, 1), 11, Some(1_111))
    );
    assert_eq!(
        (second.id, second.workload_id, second.pressure_q()),
        (SnapshotId::new(7, 2), 99, Some(9_999))
    );
    assert_eq!(
        publisher.latest().map(|snapshot| (
            snapshot.id,
            snapshot.workload_id,
            snapshot.pressure_q()
        )),
        Some((SnapshotId::new(7, 2), 99, Some(9_999)))
    );
}

#[test]
fn repeated_full_registry_selection_meets_p95_and_max_latency_bounds() {
    let mut scheduler = ValueScheduler::new(SchedulerPhase::Shadow);
    let mut samples = Vec::with_capacity(1_024);

    for sequence in 1..=1_024 {
        let plan = scheduler.plan(
            &fixture_snapshot(sequence),
            SchedulerInputs::nominal_all_due(),
        );
        assert_eq!(plan.permits.len(), JobId::ALL.len());
        assert!(plan.predicted_us <= NOMINAL_BUDGET_US);
        samples.push(plan.selection_latency_us);
    }

    samples.sort_unstable();
    let p95 = samples[(samples.len() * 95 / 100).saturating_sub(1)];
    let max = *samples.last().expect("latency sample");

    assert!(p95 <= 250, "selection p95 was {p95}us");
    assert!(max <= 1_000, "selection max was {max}us");
}
