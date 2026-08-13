use apollo_engine::engine::compute_fabric::{
    BackendOutcome, CircuitState, CompletionStatus, ComputeBackendId, ComputeFabric,
    ComputeFabricConfig, ComputeJob, ComputePayload, EvaluationSample, FabricError, JobClassId,
    RolloutPhase, MAX_JOB_CLASSES, MAX_RUNNING_JOBS, MAX_VECTOR_VALUES, SHADOW_MIN_JOBS,
    SHADOW_MIN_WINDOW_US,
};
use apollo_engine::engine::world_state::WorldIdentity;

fn identity() -> WorldIdentity {
    WorldIdentity {
        daemon_epoch: 7,
        revision: 11,
        workload_id: 13,
        capability_revision: 17,
        thermal_revision: 19,
        process_revision: 23,
        session_revision: 29,
        kill_switch: false,
        sleeping: false,
    }
}

fn fabric() -> ComputeFabric {
    ComputeFabric::with_config(identity(), 0, ComputeFabricConfig::default())
}

fn job(
    id: u64,
    class: u16,
    backend: ComputeBackendId,
    submitted_at_us: u64,
    queue_deadline_us: u64,
    runtime_deadline_us: u64,
) -> ComputeJob {
    ComputeJob::new(
        id.into(),
        JobClassId::new(class),
        backend,
        identity(),
        ComputePayload::try_vector(vec![1.0, 2.0, 3.0]).unwrap(),
        submitted_at_us,
        queue_deadline_us,
        runtime_deadline_us,
    )
}

fn passing_sample(at_us: u64) -> EvaluationSample {
    EvaluationSample {
        at_us,
        deadline_met: true,
        oracle_error: false,
        baseline_latency_us: 100,
        candidate_latency_us: 90,
        baseline_energy_uj: 100,
        candidate_energy_uj: 100,
    }
}

fn move_to_canary(fabric: &mut ComputeFabric, backend: ComputeBackendId) {
    for _ in 0..SHADOW_MIN_JOBS {
        fabric.record_evaluation(backend, passing_sample(SHADOW_MIN_WINDOW_US)).unwrap();
    }
    assert_eq!(fabric.rollout_phase(backend), RolloutPhase::Canary);
}

#[test]
fn backend_catalog_contains_exactly_the_four_bounded_backends() {
    assert_eq!(
        ComputeBackendId::ALL,
        [
            ComputeBackendId::CpuLatency,
            ComputeBackendId::CpuUtility,
            ComputeBackendId::Metal,
            ComputeBackendId::CoreMl,
        ]
    );
}

#[test]
fn jobs_carry_the_exact_world_identity_without_a_second_identity_type() {
    let expected = identity();
    let request = job(1, 1, ComputeBackendId::CpuLatency, 0, 100, 100);
    assert_eq!(request.world_identity, expected);

    let outcome = BackendOutcome::new(
        request.id,
        expected,
        ComputePayload::try_candidate_ids(vec![7.into()]).unwrap(),
    );
    assert_eq!(outcome.world_identity, expected);
}

#[test]
fn registration_stops_at_thirty_two_job_classes() {
    let mut fabric = fabric();
    for index in 0..MAX_JOB_CLASSES {
        fabric.register_job_class(JobClassId::new(index as u16)).unwrap();
    }
    assert_eq!(fabric.registered_job_classes(), MAX_JOB_CLASSES);
    assert_eq!(
        fabric.register_job_class(JobClassId::new(MAX_JOB_CLASSES as u16)),
        Err(FabricError::TooManyJobClasses)
    );
}

#[test]
fn pending_work_is_latest_wins_per_job_class_and_backend() {
    let mut fabric = fabric();
    fabric.submit(job(1, 1, ComputeBackendId::Metal, 0, 1_000, 1_000)).unwrap();
    fabric.poll(1);

    fabric.submit(job(2, 1, ComputeBackendId::Metal, 2, 1_000, 1_000)).unwrap();
    fabric.submit(job(3, 1, ComputeBackendId::Metal, 3, 1_000, 1_000)).unwrap();

    assert_eq!(fabric.running_len(), 1);
    assert_eq!(fabric.pending_job_ids(), vec![3.into()]);
}

#[test]
fn no_more_than_four_jobs_run_at_once() {
    let mut fabric = fabric();
    for id in 1..=5 {
        fabric
            .submit(job(id, id as u16, ComputeBackendId::Metal, 0, 1_000, 1_000))
            .unwrap();
    }

    fabric.poll(1);
    assert_eq!(fabric.running_len(), MAX_RUNNING_JOBS);
    assert_eq!(fabric.pending_len(), 1);
}

#[test]
fn queue_age_and_runtime_deadlines_have_separate_outcomes() {
    let mut queue_expiring = fabric();
    queue_expiring
        .submit(job(1, 1, ComputeBackendId::Metal, 0, 10, 1_000))
        .unwrap();
    let queue_events = queue_expiring.poll(11);
    assert_eq!(queue_events[0].status, CompletionStatus::QueueExpired);

    let mut runtime_expiring = fabric();
    runtime_expiring
        .submit(job(2, 2, ComputeBackendId::Metal, 0, 1_000, 10))
        .unwrap();
    assert!(runtime_expiring.poll(1).is_empty());
    let runtime_events = runtime_expiring.poll(12);
    assert_eq!(runtime_events[0].status, CompletionStatus::RuntimeExpired);
}

#[test]
fn cpu_latency_and_utility_execute_vector_jobs_synchronously_on_poll() {
    for (id, backend) in [
        (1, ComputeBackendId::CpuLatency),
        (2, ComputeBackendId::CpuUtility),
    ] {
        let mut fabric = fabric();
        fabric
            .submit(job(id, id as u16, backend, 0, 1_000, 1_000))
            .unwrap();
        let events = fabric.poll(1);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].status, CompletionStatus::Valid);
        assert_eq!(fabric.running_len(), 0);
    }
}

#[test]
fn outcomes_accept_valid_results_and_reject_late_or_wrong_identity_results() {
    let mut valid = fabric();
    let request = job(1, 1, ComputeBackendId::Metal, 0, 1_000, 100);
    valid.submit(request.clone()).unwrap();
    assert!(valid.poll(1).is_empty());
    let accepted = valid.complete(
        10,
        BackendOutcome::new(
            request.id,
            identity(),
            ComputePayload::try_vector(vec![4.0]).unwrap(),
        ),
    );
    assert_eq!(accepted.status, CompletionStatus::Valid);

    let mut late = fabric();
    let late_request = job(2, 2, ComputeBackendId::Metal, 0, 1_000, 10);
    late.submit(late_request.clone()).unwrap();
    late.poll(1);
    let late_result = late.complete(
        12,
        BackendOutcome::new(
            late_request.id,
            identity(),
            ComputePayload::try_vector(vec![4.0]).unwrap(),
        ),
    );
    assert_eq!(late_result.status, CompletionStatus::Late);

    let mut wrong_identity = fabric();
    let identity_request = job(3, 3, ComputeBackendId::Metal, 0, 1_000, 100);
    wrong_identity.submit(identity_request.clone()).unwrap();
    wrong_identity.poll(1);
    let mut stale = identity();
    stale.revision += 1;
    let stale_result = wrong_identity.complete(
        10,
        BackendOutcome::new(
            identity_request.id,
            stale,
            ComputePayload::try_vector(vec![4.0]).unwrap(),
        ),
    );
    assert_eq!(stale_result.status, CompletionStatus::WrongIdentity);
}

#[test]
fn outcomes_reject_out_of_order_nonfinite_and_oversized_results() {
    let mut out_of_order = fabric();
    let unknown = out_of_order.complete(
        1,
        BackendOutcome::new(
            99.into(),
            identity(),
            ComputePayload::try_vector(vec![1.0]).unwrap(),
        ),
    );
    assert_eq!(unknown.status, CompletionStatus::OutOfOrder);

    let mut nonfinite = fabric();
    let nonfinite_request = job(1, 1, ComputeBackendId::Metal, 0, 1_000, 100);
    nonfinite.submit(nonfinite_request.clone()).unwrap();
    nonfinite.poll(1);
    let nonfinite_result = nonfinite.complete(
        10,
        BackendOutcome::new(
            nonfinite_request.id,
            identity(),
            ComputePayload::Vector(vec![f32::NAN]),
        ),
    );
    assert_eq!(nonfinite_result.status, CompletionStatus::NonFinite);

    let mut oversized = fabric();
    let oversized_request = job(2, 2, ComputeBackendId::Metal, 0, 1_000, 100);
    oversized.submit(oversized_request.clone()).unwrap();
    oversized.poll(1);
    let oversized_result = oversized.complete(
        10,
        BackendOutcome::new(
            oversized_request.id,
            identity(),
            ComputePayload::Vector(vec![0.0; MAX_VECTOR_VALUES + 1]),
        ),
    );
    assert_eq!(oversized_result.status, CompletionStatus::Oversized);
}

#[test]
fn each_backend_has_an_independent_closed_open_half_open_circuit() {
    let mut fabric = fabric();
    for _ in 0..3 {
        fabric.record_backend_failure(ComputeBackendId::Metal, 0);
    }
    assert_eq!(fabric.circuit_state(ComputeBackendId::Metal, 0), CircuitState::Open);
    assert_eq!(fabric.circuit_state(ComputeBackendId::CoreMl, 0), CircuitState::Closed);

    let half_open_at = ComputeFabricConfig::default().circuit_cooldown_us;
    assert_eq!(
        fabric.circuit_state(ComputeBackendId::Metal, half_open_at),
        CircuitState::HalfOpen
    );
    fabric.record_backend_success(ComputeBackendId::Metal, half_open_at);
    assert_eq!(
        fabric.circuit_state(ComputeBackendId::Metal, half_open_at),
        CircuitState::Closed
    );
}

#[test]
fn canary_traffic_is_exactly_ten_percent() {
    let mut fabric = fabric();
    move_to_canary(&mut fabric, ComputeBackendId::Metal);
    let admitted = (0..100)
        .filter(|id| fabric.canary_admitted(ComputeBackendId::Metal, (*id).into()))
        .count();
    assert_eq!(admitted, 10);
}

#[test]
fn promotion_requires_shadow_window_canary_sample_and_all_quality_gates() {
    let mut fabric = fabric();
    move_to_canary(&mut fabric, ComputeBackendId::Metal);
    for index in 0..500 {
        let mut sample = passing_sample(SHADOW_MIN_WINDOW_US);
        sample.deadline_met = index >= 5;
        sample.oracle_error = index < 5;
        fabric
            .record_evaluation(ComputeBackendId::Metal, sample)
            .unwrap();
    }
    assert_eq!(fabric.rollout_phase(ComputeBackendId::Metal), RolloutPhase::Active);

    let mut failing = fabric();
    move_to_canary(&mut failing, ComputeBackendId::CoreMl);
    for index in 0..500 {
        let mut sample = passing_sample(SHADOW_MIN_WINDOW_US);
        sample.deadline_met = index >= 6;
        sample.oracle_error = index < 6;
        sample.candidate_latency_us = 100;
        failing
            .record_evaluation(ComputeBackendId::CoreMl, sample)
            .unwrap();
    }
    assert_eq!(failing.rollout_phase(ComputeBackendId::CoreMl), RolloutPhase::Canary);
}

#[test]
fn promotion_accepts_the_fifteen_percent_energy_branch() {
    let mut fabric = fabric();
    move_to_canary(&mut fabric, ComputeBackendId::CoreMl);
    for _ in 0..500 {
        let mut sample = passing_sample(SHADOW_MIN_WINDOW_US);
        sample.candidate_latency_us = sample.baseline_latency_us;
        sample.candidate_energy_uj = 85;
        fabric.record_evaluation(ComputeBackendId::CoreMl, sample).unwrap();
    }
    assert_eq!(fabric.rollout_phase(ComputeBackendId::CoreMl), RolloutPhase::Active);
}

#[test]
fn shadow_requires_both_five_hundred_jobs_and_fifteen_minutes() {
    let mut fabric = fabric();
    for _ in 0..SHADOW_MIN_JOBS {
        fabric.record_evaluation(ComputeBackendId::Metal, passing_sample(1)).unwrap();
    }
    assert_eq!(fabric.rollout_phase(ComputeBackendId::Metal), RolloutPhase::Shadow);

    fabric.advance_rollouts(SHADOW_MIN_WINDOW_US);
    assert_eq!(fabric.rollout_phase(ComputeBackendId::Metal), RolloutPhase::Canary);
}

#[test]
fn rollback_is_scoped_to_one_backend() {
    let mut fabric = fabric();
    move_to_canary(&mut fabric, ComputeBackendId::Metal);
    for _ in 0..500 {
        fabric
            .record_evaluation(ComputeBackendId::Metal, passing_sample(SHADOW_MIN_WINDOW_US))
            .unwrap();
    }
    assert_eq!(fabric.rollout_phase(ComputeBackendId::Metal), RolloutPhase::Active);

    fabric.rollback_backend(ComputeBackendId::Metal, 123).unwrap();
    assert_eq!(fabric.rollout_phase(ComputeBackendId::Metal), RolloutPhase::RolledBack);
    assert_eq!(fabric.rollout_phase(ComputeBackendId::CoreMl), RolloutPhase::Shadow);
    assert_eq!(fabric.circuit_state(ComputeBackendId::Metal, 123), CircuitState::Open);
}
