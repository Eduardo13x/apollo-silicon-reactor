use std::thread;
use std::time::{Duration, Instant};

use apollo_engine::engine::compute_fabric::{
    ComputeBackendId, ComputeJob, ComputePayload, JobClassId,
};
use apollo_engine::engine::heterogeneous_executor::{ExecutorSubmitOutcome, HeterogeneousExecutor};
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

fn job(id: u64, backend: ComputeBackendId) -> ComputeJob {
    ComputeJob::new(
        id.into(),
        JobClassId::new(1),
        backend,
        identity(),
        ComputePayload::try_vector(vec![0.1; 16]).unwrap(),
        0,
        1_000_000,
        1_000_000,
    )
}

fn wait_until_ready(executor: &HeterogeneousExecutor) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while executor.status().workers_active < 3 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(executor.status().workers_active, 3);
}

#[test]
fn cpu_and_coreml_lanes_execute_off_thread_with_bounded_outputs() {
    let executor = HeterogeneousExecutor::spawn();
    wait_until_ready(&executor);
    for (id, backend) in [
        (1, ComputeBackendId::CpuLatency),
        (2, ComputeBackendId::CpuUtility),
        (3, ComputeBackendId::CoreMl),
    ] {
        assert_eq!(
            executor.try_submit(job(id, backend)),
            ExecutorSubmitOutcome::Submitted
        );
    }

    let deadline = Instant::now() + Duration::from_secs(2);
    let mut results = Vec::new();
    while results.len() < 3 && Instant::now() < deadline {
        results.extend(executor.drain(4));
        thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|result| {
        matches!(&result.outcome.payload, ComputePayload::Vector(values) if values.len() == 4 && values.iter().all(|value| value.is_finite()))
    }));
    let coreml_result = results
        .iter()
        .find(|result| result.backend == ComputeBackendId::CoreMl)
        .unwrap();
    assert_eq!(
        coreml_result.accelerator_effective,
        executor.status().coreml.accelerator_backend_available()
    );
}

#[test]
fn metal_remains_an_external_advisory_lane() {
    let executor = HeterogeneousExecutor::spawn();
    assert_eq!(
        executor.try_submit(job(1, ComputeBackendId::Metal)),
        ExecutorSubmitOutcome::Unavailable
    );
}

#[test]
fn coreml_result_queue_drops_are_attributed_to_the_backend() {
    let executor = HeterogeneousExecutor::spawn();
    wait_until_ready(&executor);
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut id = 100_u64;
    while executor.status().coreml_result_drops == 0 && Instant::now() < deadline {
        if executor.try_submit(job(id, ComputeBackendId::CoreMl))
            == ExecutorSubmitOutcome::Submitted
        {
            id += 1;
        }
        thread::sleep(Duration::from_millis(1));
    }

    let status = executor.status();
    assert!(status.coreml_result_drops > 0);
    assert!(status.coreml_result_drops <= status.result_drops);
}
