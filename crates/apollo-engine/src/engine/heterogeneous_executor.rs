//! Non-blocking bounded workers for optional heterogeneous compute.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use super::compute_fabric::{
    BackendOutcome, ComputeBackendId, ComputeJob, ComputePayload, MAX_RUNNING_JOBS,
};
use super::coreml_predictor::{
    cpu_oracle_predict, CoreMlPredictor, PredictorStatus, TemporalFeatureVector,
};

const LANE_QUEUE_CAPACITY: usize = 1;
const RESULT_QUEUE_CAPACITY: usize = MAX_RUNNING_JOBS * 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorSubmitOutcome {
    Submitted,
    Busy,
    Unavailable,
}

#[derive(Debug, Clone)]
pub struct ExecutorResult {
    pub outcome: BackendOutcome,
    pub backend: ComputeBackendId,
    pub elapsed_us: u64,
    pub completed_at_us: u64,
    /// True only when the requested accelerator produced this result. A
    /// deterministic CPU fallback remains useful but cannot earn accelerator
    /// rollout evidence.
    pub accelerator_effective: bool,
}

#[derive(Debug, Clone)]
pub struct ExecutorStatus {
    pub workers_active: u64,
    pub qos_failures: u64,
    pub result_drops: u64,
    pub coreml_result_drops: u64,
    pub worker_runtime_us: u64,
    pub coreml: PredictorStatus,
}

struct Lane {
    tx: SyncSender<ComputeJob>,
    available: Arc<AtomicBool>,
}

struct SharedStatus {
    workers_active: AtomicU64,
    qos_failures: AtomicU64,
    result_drops: AtomicU64,
    coreml_result_drops: AtomicU64,
    worker_runtime_us: AtomicU64,
    coreml: Mutex<PredictorStatus>,
}

pub struct HeterogeneousExecutor {
    cpu_latency: Lane,
    cpu_utility: Lane,
    coreml: Lane,
    result_rx: Receiver<ExecutorResult>,
    status: Arc<SharedStatus>,
}

impl HeterogeneousExecutor {
    pub fn spawn() -> Self {
        let (result_tx, result_rx) = mpsc::sync_channel(RESULT_QUEUE_CAPACITY);
        let status = Arc::new(SharedStatus {
            workers_active: AtomicU64::new(0),
            qos_failures: AtomicU64::new(0),
            result_drops: AtomicU64::new(0),
            coreml_result_drops: AtomicU64::new(0),
            worker_runtime_us: AtomicU64::new(0),
            coreml: Mutex::new(CoreMlPredictor::cpu_oracle().status()),
        });
        Self {
            cpu_latency: spawn_lane(ComputeBackendId::CpuLatency, &result_tx, &status),
            cpu_utility: spawn_lane(ComputeBackendId::CpuUtility, &result_tx, &status),
            coreml: spawn_lane(ComputeBackendId::CoreMl, &result_tx, &status),
            result_rx,
            status,
        }
    }

    pub fn try_submit(&self, job: ComputeJob) -> ExecutorSubmitOutcome {
        let lane = match job.backend {
            ComputeBackendId::CpuLatency => &self.cpu_latency,
            ComputeBackendId::CpuUtility => &self.cpu_utility,
            ComputeBackendId::CoreMl => &self.coreml,
            ComputeBackendId::Metal => return ExecutorSubmitOutcome::Unavailable,
        };
        if !lane.available.load(Ordering::Acquire) {
            return ExecutorSubmitOutcome::Unavailable;
        }
        match lane.tx.try_send(job) {
            Ok(()) => ExecutorSubmitOutcome::Submitted,
            Err(TrySendError::Full(_)) => ExecutorSubmitOutcome::Busy,
            Err(TrySendError::Disconnected(_)) => ExecutorSubmitOutcome::Unavailable,
        }
    }

    pub fn drain(&self, maximum: usize) -> Vec<ExecutorResult> {
        let mut results = Vec::with_capacity(maximum.min(RESULT_QUEUE_CAPACITY));
        while results.len() < maximum {
            match self.result_rx.try_recv() {
                Ok(result) => results.push(result),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        results
    }

    pub fn status(&self) -> ExecutorStatus {
        ExecutorStatus {
            workers_active: self.status.workers_active.load(Ordering::Acquire),
            qos_failures: self.status.qos_failures.load(Ordering::Relaxed),
            result_drops: self.status.result_drops.load(Ordering::Relaxed),
            coreml_result_drops: self.status.coreml_result_drops.load(Ordering::Relaxed),
            worker_runtime_us: self.status.worker_runtime_us.load(Ordering::Relaxed),
            coreml: self
                .status
                .coreml
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone(),
        }
    }
}

impl Default for HeterogeneousExecutor {
    fn default() -> Self {
        Self::spawn()
    }
}

fn spawn_lane(
    backend: ComputeBackendId,
    result_tx: &SyncSender<ExecutorResult>,
    status: &Arc<SharedStatus>,
) -> Lane {
    let (tx, rx) = mpsc::sync_channel(LANE_QUEUE_CAPACITY);
    let available = Arc::new(AtomicBool::new(false));
    let worker_available = Arc::clone(&available);
    let result_tx = result_tx.clone();
    let status = Arc::clone(status);
    let name = match backend {
        ComputeBackendId::CpuLatency => "apollo-compute-interactive",
        ComputeBackendId::CpuUtility => "apollo-compute-utility",
        ComputeBackendId::CoreMl => "apollo-compute-coreml",
        ComputeBackendId::Metal => "apollo-compute-metal",
    };
    let spawned = thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            if !set_worker_qos(backend) {
                status.qos_failures.fetch_add(1, Ordering::Relaxed);
            }
            let predictor = (backend == ComputeBackendId::CoreMl).then(CoreMlPredictor::new);
            if let Some(predictor) = predictor.as_ref() {
                *status
                    .coreml
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = predictor.status();
            }
            worker_available.store(true, Ordering::Release);
            status.workers_active.fetch_add(1, Ordering::AcqRel);
            while let Ok(job) = rx.recv() {
                let started = Instant::now();
                let outcome = execute(job, predictor.as_ref());
                let accelerator_effective = if let Some(predictor) = predictor.as_ref() {
                    let predictor_status = predictor.status();
                    *status
                        .coreml
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                        predictor_status.clone();
                    predictor_status.accelerator_backend_available()
                } else {
                    false
                };
                let elapsed_us = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
                status
                    .worker_runtime_us
                    .fetch_add(elapsed_us, Ordering::Relaxed);
                let result = ExecutorResult {
                    backend,
                    outcome,
                    elapsed_us,
                    completed_at_us: monotonic_us(),
                    accelerator_effective,
                };
                if result_tx.try_send(result).is_err() {
                    status.result_drops.fetch_add(1, Ordering::Relaxed);
                    if backend == ComputeBackendId::CoreMl {
                        status.coreml_result_drops.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            worker_available.store(false, Ordering::Release);
            status.workers_active.fetch_sub(1, Ordering::AcqRel);
        });
    if spawned.is_err() {
        available.store(false, Ordering::Release);
    }
    Lane { tx, available }
}

pub fn monotonic_us() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let status = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
    if status != 0 || time.tv_sec < 0 || time.tv_nsec < 0 {
        return 0;
    }
    (time.tv_sec as u64)
        .saturating_mul(1_000_000)
        .saturating_add((time.tv_nsec as u64) / 1_000)
}

fn execute(job: ComputeJob, predictor: Option<&CoreMlPredictor>) -> BackendOutcome {
    let output = match &job.payload {
        ComputePayload::Vector(values) => TemporalFeatureVector::try_from_slice(values)
            .map(|features| {
                predictor.map_or_else(
                    || cpu_oracle_predict(&features),
                    |predictor| predictor.predict(&features),
                )
            })
            .map(|prediction| ComputePayload::Vector(prediction.as_array().to_vec()))
            .unwrap_or_else(|_| ComputePayload::Vector(Vec::new())),
        ComputePayload::CandidateIds(values) => ComputePayload::CandidateIds(values.clone()),
    };
    BackendOutcome::new(job.id, job.world_identity, output)
}

fn set_worker_qos(backend: ComputeBackendId) -> bool {
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(
                qos_class: libc::c_uint,
                relative_priority: libc::c_int,
            ) -> libc::c_int;
        }
        let qos_class = match backend {
            ComputeBackendId::CpuLatency => 0x19,
            ComputeBackendId::CpuUtility | ComputeBackendId::CoreMl => 0x11,
            ComputeBackendId::Metal => 0x09,
        };
        return pthread_set_qos_class_self_np(qos_class, 0) == 0;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = backend;
        true
    }
}
