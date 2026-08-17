use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use apollo_engine::engine::compute_fabric::{
    AccuracyRequirement, CircuitState, CompletionStatus, ComputeBackendId, ComputeFabric,
    ComputeFabricConfig, ComputeJob, ComputePayload, EvaluationSample, JobClassId, RolloutPhase,
};
use apollo_engine::engine::coreml_predictor::{TemporalFeatureVector, TemporalObservation};
use apollo_engine::engine::heterogeneous_executor::{
    monotonic_us, ExecutorSubmitOutcome, HeterogeneousExecutor,
};
use apollo_engine::engine::world_state::WorldStateSnapshot;

const TEMPORAL_JOB_CLASS: JobClassId = JobClassId::new(1);
const INTERACTIVE_JOB_CLASS: JobClassId = JobClassId::new(2);
const QUEUE_DEADLINE_US: u64 = 50_000;
const RUNTIME_DEADLINE_US: u64 = 75_000;
const MAX_RESULTS_PER_TICK: usize = 8;
const MAX_CONTROL_P95_MS: f64 = 75.0;
const MAX_P95_REGRESSION: f64 = 1.10;
const MIN_RSS_DELTA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_RSS_DELTA_BYTES: u64 = 192 * 1024 * 1024;
const RSS_MEMORY_DIVISOR: u64 = 128;
const MAX_IDLE_CPU_PERCENT: f64 = 1.0;
const SUSTAINED_REGRESSION_CYCLES: u32 = 30;
const OPTIONAL_RECOVERY_CYCLES: u32 = 30;
/// Idle control samples retained for the rolling baseline median. The window
/// is the entire baseline state, so an expensive boot ages out on its own
/// rather than being latched for the life of the daemon. Bounded by
/// construction: 120 f64 ≈ 1 KiB, sorted only on idle cycles.
const BASELINE_WINDOW_SAMPLES: usize = 120;
/// Samples required before the baseline arms. Until then the fabric stays
/// blocked, which is the conservative direction.
const BASELINE_MIN_SAMPLES: usize = 60;
/// Consecutive idle cycles required before the baseline may track upward, so
/// work draining from an earlier cycle cannot inflate the control measurement.
const BASELINE_IDLE_CYCLES: u32 = 5;

#[derive(Debug, Clone)]
pub struct HeterogeneousTickInput {
    pub world: Arc<WorldStateSnapshot>,
    pub cpu_utilization: f64,
    pub pressure: f64,
    pub p95_cycle_ms: f64,
    pub transition: f64,
    pub interaction: f64,
    /// Normalized network throughput. `io_pressure` is a weighted input of the
    /// temporal model and is named in TEMPORAL_SCHEMA_HASH, but was fed a
    /// hardcoded 0.0 until the network observation was routed here.
    pub io_pressure: f64,
    pub optional_allowed: bool,
    /// True when the adaptive overhead governor itself permits speculation.
    /// Distinguishes "the budget said no" from "this job was not due this
    /// cycle", which `optional_allowed` alone conflates.
    pub budget_allows_speculation: bool,
    pub optional_recovery_healthy: bool,
    pub thermal_nominal: bool,
}

#[derive(Debug, Clone, Default)]
pub struct HeterogeneousTickMetrics {
    pub phase: String,
    pub blocker: String,
    pub workers_active: u64,
    pub qos_failures: u64,
    pub result_drops: u64,
    pub submitted_total: u64,
    pub completed_total: u64,
    pub cancelled_total: u64,
    pub stale_total: u64,
    pub deadline_misses_total: u64,
    pub dispatch_skips_total: u64,
    pub eligible_total: u64,
    pub evaluation_total: u64,
    pub last_latency_us: u64,
    pub coreml_model_available: bool,
    pub coreml_requested: String,
    pub coreml_effective: String,
    pub ane_execution_measured: bool,
    pub coreml_circuit: String,
    pub prediction_backend: String,
    pub prediction_load: f64,
    pub prediction_transition: f64,
    pub prediction_pressure: f64,
    pub prediction_p95: f64,
    pub prediction_authoritative: bool,
    pub control_p95_baseline_ms: f64,
    pub fabric_cpu_percent: f64,
    pub rss_delta_bytes: u64,
}

#[derive(Debug, Default)]
struct PairSample {
    cpu: Option<([f32; 4], u64)>,
    coreml: Option<([f32; 4], u64)>,
    evaluated: bool,
}

pub struct HeterogeneousRuntime {
    fabric: ComputeFabric,
    executor: HeterogeneousExecutor,
    next_job_id: u64,
    previous_observation: Option<TemporalObservation>,
    pairs: BTreeMap<u64, PairSample>,
    metrics: HeterogeneousTickMetrics,
    last_accelerator_prediction:
        Option<(apollo_engine::engine::world_state::WorldIdentity, [f32; 4])>,
    baseline_p95_ms: Option<f64>,
    baseline_window: VecDeque<f64>,
    baseline_last_submitted: u64,
    baseline_idle_streak: u32,
    baseline_rss_bytes: u64,
    rss_budget_bytes: u64,
    last_worker_runtime_us: u64,
    last_coreml_result_drops: u64,
    last_health_sample_us: u64,
    unhealthy_cycles: u32,
    optional_recovery_streak: u32,
}

impl HeterogeneousRuntime {
    pub fn new(initial: &WorldStateSnapshot) -> Self {
        let baseline_rss_bytes = peak_rss_bytes();
        let physical_memory_bytes = apollo_engine::engine::sysctl_direct::read_u64("hw.memsize")
            .unwrap_or(8 * 1024 * 1024 * 1024);
        let last_health_sample_us = monotonic_us();
        Self {
            fabric: ComputeFabric::with_config(
                initial.identity,
                last_health_sample_us,
                ComputeFabricConfig::default(),
            ),
            executor: HeterogeneousExecutor::spawn(),
            next_job_id: 1,
            previous_observation: None,
            pairs: BTreeMap::new(),
            metrics: HeterogeneousTickMetrics::default(),
            last_accelerator_prediction: None,
            baseline_p95_ms: None,
            baseline_window: VecDeque::with_capacity(BASELINE_WINDOW_SAMPLES),
            baseline_last_submitted: 0,
            baseline_idle_streak: 0,
            baseline_rss_bytes,
            rss_budget_bytes: rss_budget_bytes(physical_memory_bytes),
            last_worker_runtime_us: 0,
            last_coreml_result_drops: 0,
            last_health_sample_us,
            unhealthy_cycles: 0,
            optional_recovery_streak: 0,
        }
    }

    pub fn submitted_total(&self) -> u64 {
        self.metrics.submitted_total
    }

    /// Maintains the control p95 the promotion guard compares against.
    ///
    /// The baseline answers "what is control-loop p95 when the fabric is NOT
    /// doing optional work". Sampling it once, from whatever the first tick
    /// happened to see, made it a startup lottery: a daemon that booted during
    /// a quiet moment latched a p95 far below steady state and then blocked its
    /// own fabric with `control-p95` for the rest of its life. Observed in
    /// production at baseline=22ms against a steady-state 37ms, which put the
    /// 1.10x threshold at 24.2ms and held submissions at zero.
    ///
    /// Fixing it from a warmup window instead traded that for the opposite
    /// latch. Boot is expensive — state restore, model load, first Metal warmup
    /// — so the window measured ~107ms against a 39ms steady state and pinned
    /// the baseline at `MAX_CONTROL_P95_MS`. That ceiling is also the health
    /// limit checked independently below, so the guard became inert: every p95
    /// it would have rejected was already rejected by the ceiling.
    ///
    /// So the baseline is a rolling median over a bounded window of idle
    /// samples. One mechanism, both directions: an expensive boot ages out of
    /// the window on its own, a median cannot be dragged by a single quiet
    /// cycle, and a sustained regression still moves it only as fast as the
    /// window fills. Restricting samples to idle cycles is what keeps the
    /// fabric's own work out of the measurement it is judged against.
    fn update_control_baseline(&mut self, p95_cycle_ms: f64) {
        if !p95_cycle_ms.is_finite() || p95_cycle_ms <= 0.0 {
            return;
        }
        let fabric_idle = self.metrics.submitted_total == self.baseline_last_submitted;
        self.baseline_idle_streak = if fabric_idle {
            self.baseline_idle_streak.saturating_add(1)
        } else {
            0
        };
        self.baseline_last_submitted = self.metrics.submitted_total;

        // The baseline is defined as control-loop p95 with no optional work in
        // flight, so only a sustained idle streak yields an admissible sample:
        // work draining from an earlier cycle would otherwise be attributed to
        // the control loop.
        if self.baseline_idle_streak < BASELINE_IDLE_CYCLES {
            return;
        }
        if self.baseline_window.len() == BASELINE_WINDOW_SAMPLES {
            self.baseline_window.pop_front();
        }
        self.baseline_window.push_back(p95_cycle_ms);
        if self.baseline_window.len() < BASELINE_MIN_SAMPLES {
            return;
        }
        let mut sorted: Vec<f64> = self.baseline_window.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.baseline_p95_ms = Some(sorted[sorted.len() / 2].min(MAX_CONTROL_P95_MS));
    }

    pub fn tick(&mut self, input: HeterogeneousTickInput) -> HeterogeneousTickMetrics {
        let now_us = monotonic_us();
        let executor_status = self.executor.status();
        let health_elapsed_us = now_us.saturating_sub(self.last_health_sample_us).max(1);
        let worker_delta_us = executor_status
            .worker_runtime_us
            .saturating_sub(self.last_worker_runtime_us);
        self.last_health_sample_us = now_us;
        self.last_worker_runtime_us = executor_status.worker_runtime_us;
        self.metrics.fabric_cpu_percent = worker_delta_us as f64 / health_elapsed_us as f64 * 100.0;
        self.update_control_baseline(input.p95_cycle_ms);
        self.metrics.control_p95_baseline_ms = self.baseline_p95_ms.unwrap_or(0.0);
        self.metrics.rss_delta_bytes = peak_rss_bytes().saturating_sub(self.baseline_rss_bytes);
        let p95_healthy = input.p95_cycle_ms.is_finite()
            && input.p95_cycle_ms <= MAX_CONTROL_P95_MS
            && self.baseline_p95_ms.is_some_and(|baseline| {
                input.p95_cycle_ms <= (baseline * MAX_P95_REGRESSION).max(1.0)
            });
        let rss_healthy = self.metrics.rss_delta_bytes <= self.rss_budget_bytes;
        let idle_cpu_healthy = self.metrics.fabric_cpu_percent <= MAX_IDLE_CPU_PERCENT;
        let promotion_healthy = p95_healthy && rss_healthy && idle_cpu_healthy;
        if input.optional_allowed {
            self.optional_recovery_streak = OPTIONAL_RECOVERY_CYCLES;
        } else if input.optional_recovery_healthy
            && promotion_healthy
            && input.thermal_nominal
            && !input.world.identity.kill_switch
            && !input.world.identity.sleeping
            && input.pressure < 0.55
        {
            self.optional_recovery_streak = self.optional_recovery_streak.saturating_add(1);
        } else {
            self.optional_recovery_streak = 0;
        }
        let optional_allowed =
            input.optional_allowed || self.optional_recovery_streak >= OPTIONAL_RECOVERY_CYCLES;
        self.consume_results(now_us, promotion_healthy);
        let cancelled = self.fabric.update_world_identity(input.world.identity);
        self.account_completions(&cancelled, now_us);

        self.metrics.workers_active = executor_status.workers_active;
        self.metrics.qos_failures = executor_status.qos_failures;
        self.metrics.result_drops = executor_status.result_drops;
        let new_coreml_drops = executor_status
            .coreml_result_drops
            .saturating_sub(self.last_coreml_result_drops);
        self.last_coreml_result_drops = executor_status.coreml_result_drops;
        for _ in 0..new_coreml_drops {
            let _ = self
                .fabric
                .record_deadline_miss(ComputeBackendId::CoreMl, now_us);
            self.fabric
                .record_backend_failure(ComputeBackendId::CoreMl, now_us);
        }
        self.metrics.coreml_model_available = executor_status.coreml.model_available;
        self.metrics.coreml_requested = executor_status
            .coreml
            .requested_backend
            .as_str()
            .to_string();
        self.metrics.coreml_effective = executor_status.coreml.effective_backend.map_or_else(
            || "cpu-oracle".to_string(),
            |backend| backend.as_str().to_string(),
        );
        self.metrics.ane_execution_measured = executor_status.coreml.ane_execution_measured;

        let blocked = !optional_allowed
            || !input.thermal_nominal
            || input.world.identity.kill_switch
            || input.world.identity.sleeping
            || input.pressure >= 0.55
            || !promotion_healthy;
        self.metrics.blocker = if !optional_allowed {
            // Steady state on a healthy host is "this job was not due this
            // cycle", not a budget denial. Reporting both as overhead-budget
            // made a working fabric look permanently throttled.
            if input.budget_allows_speculation {
                "not-scheduled"
            } else {
                "overhead-budget"
            }
        } else if !input.thermal_nominal {
            "thermal"
        } else if input.world.identity.kill_switch {
            "kill-switch"
        } else if input.world.identity.sleeping {
            "sleep"
        } else if input.pressure >= 0.55 {
            "memory-pressure"
        } else if !p95_healthy {
            "control-p95"
        } else if !rss_healthy {
            "rss-budget"
        } else if !idle_cpu_healthy {
            "idle-cpu-budget"
        } else if !executor_status.coreml.accelerator_backend_available() {
            "coreml-accelerator-unavailable"
        } else {
            ""
        }
        .to_string();

        if promotion_healthy {
            self.unhealthy_cycles = 0;
        } else {
            self.unhealthy_cycles = self.unhealthy_cycles.saturating_add(1);
            if self.unhealthy_cycles >= SUSTAINED_REGRESSION_CYCLES
                && self.fabric.rollout_phase(ComputeBackendId::CoreMl) == RolloutPhase::Active
            {
                let _ = self
                    .fabric
                    .rollback_backend(ComputeBackendId::CoreMl, now_us);
            }
        }

        if !blocked {
            let observation = TemporalObservation {
                load: unit(input.cpu_utilization),
                transition: unit(input.transition),
                pressure: unit(input.pressure),
                p95: unit(input.p95_cycle_ms / 75.0),
                cpu_utilization: unit(input.cpu_utilization),
                memory_pressure: unit(input.pressure),
                io_pressure: unit(input.io_pressure),
                thermal_pressure: if input.thermal_nominal { 0.0 } else { 1.0 },
                run_queue: unit(input.cpu_utilization),
                active_work: unit(input.interaction),
                sample_age: 0.0,
            };
            let features =
                TemporalFeatureVector::from_observations(observation, self.previous_observation);
            self.previous_observation = Some(observation);
            self.submit_temporal_jobs(
                now_us,
                input.world.identity,
                features,
                executor_status.coreml.accelerator_backend_available(),
                input.transition >= 0.20 || input.p95_cycle_ms >= 60.0,
            );
        }

        let (completions, ready) = self.fabric.take_ready(now_us);
        self.account_completions(&completions, now_us);
        for job in ready {
            let job_id = job.id;
            match self.executor.try_submit(job) {
                ExecutorSubmitOutcome::Submitted => {}
                ExecutorSubmitOutcome::Busy | ExecutorSubmitOutcome::Unavailable => {
                    self.metrics.dispatch_skips_total =
                        self.metrics.dispatch_skips_total.saturating_add(1);
                    let completion = self.fabric.cancel_started(job_id, now_us);
                    if completion.backend == Some(ComputeBackendId::CoreMl) {
                        let _ = self
                            .fabric
                            .record_deadline_miss(ComputeBackendId::CoreMl, now_us);
                    }
                    self.account_completions(std::slice::from_ref(&completion), now_us);
                }
            }
        }

        let phase = self.fabric.rollout_phase(ComputeBackendId::CoreMl);
        self.metrics.phase = rollout_name(phase);
        self.metrics.prediction_authoritative = phase == RolloutPhase::Active
            && self
                .last_accelerator_prediction
                .is_some_and(|(identity, _)| {
                    identity.accepts_recent_result_for(input.world.identity, 2)
                });
        if self.metrics.prediction_authoritative {
            if let Some((_, values)) = self.last_accelerator_prediction {
                self.metrics.prediction_backend = "coreml-active".to_string();
                self.metrics.prediction_load = f64::from(values[0]);
                self.metrics.prediction_transition = f64::from(values[1]);
                self.metrics.prediction_pressure = f64::from(values[2]);
                self.metrics.prediction_p95 = f64::from(values[3]);
            }
        }
        self.metrics.coreml_circuit =
            circuit_name(self.fabric.circuit_state(ComputeBackendId::CoreMl, now_us));
        self.metrics.clone()
    }

    fn submit_temporal_jobs(
        &mut self,
        now_us: u64,
        identity: apollo_engine::engine::world_state::WorldIdentity,
        features: TemporalFeatureVector,
        coreml_available: bool,
        interactive_due: bool,
    ) {
        let payload = features.as_slice().to_vec();
        self.submit(
            now_us,
            identity,
            TEMPORAL_JOB_CLASS,
            ComputeBackendId::CpuUtility,
            &payload,
        );
        if interactive_due {
            self.submit(
                now_us,
                identity,
                INTERACTIVE_JOB_CLASS,
                ComputeBackendId::CpuLatency,
                &payload,
            );
        }
        if coreml_available {
            let phase = self.fabric.rollout_phase(ComputeBackendId::CoreMl);
            let eligible_ticket = self.metrics.eligible_total;
            self.metrics.eligible_total = self.metrics.eligible_total.saturating_add(1);
            if self
                .fabric
                .record_eligible(ComputeBackendId::CoreMl, now_us)
                .is_err()
            {
                return;
            }
            if phase != RolloutPhase::Canary
                || self
                    .fabric
                    .canary_admitted(ComputeBackendId::CoreMl, eligible_ticket.into())
            {
                self.submit(
                    now_us,
                    identity,
                    TEMPORAL_JOB_CLASS,
                    ComputeBackendId::CoreMl,
                    &payload,
                );
            }
        }
    }

    fn submit(
        &mut self,
        now_us: u64,
        identity: apollo_engine::engine::world_state::WorldIdentity,
        class: JobClassId,
        backend: ComputeBackendId,
        payload: &[f32],
    ) {
        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.saturating_add(1).max(1);
        let estimated_cost = payload.len().min(u32::MAX as usize) as u32;
        let Ok(payload) = ComputePayload::try_vector(payload.to_vec()) else {
            return;
        };
        let accuracy = if backend == ComputeBackendId::CoreMl {
            AccuracyRequirement::OracleWithinOnePercent
        } else {
            AccuracyRequirement::Deterministic
        };
        let job = ComputeJob::new(
            id.into(),
            class,
            backend,
            identity,
            payload,
            now_us,
            QUEUE_DEADLINE_US,
            RUNTIME_DEADLINE_US,
        )
        .with_contract(accuracy, estimated_cost);
        if self.fabric.submit(job).is_ok() {
            self.metrics.submitted_total = self.metrics.submitted_total.saturating_add(1);
        }
    }

    fn consume_results(&mut self, now_us: u64, promotion_healthy: bool) {
        for result in self.executor.drain(MAX_RESULTS_PER_TICK) {
            let backend = result.backend;
            let accelerator_effective = result.accelerator_effective;
            let world_identity = result.outcome.world_identity;
            let revision = world_identity.revision;
            let elapsed_us = result.elapsed_us;
            let completion = self.fabric.complete(result.completed_at_us, result.outcome);
            if completion.status == CompletionStatus::Valid {
                if let Some(values) = completion.payload.as_ref().and_then(prediction_array) {
                    let pair = self.pairs.entry(revision).or_default();
                    match backend {
                        ComputeBackendId::CpuUtility => pair.cpu = Some((values, elapsed_us)),
                        ComputeBackendId::CoreMl if accelerator_effective => {
                            pair.coreml = Some((values, elapsed_us));
                            self.last_accelerator_prediction = Some((world_identity, values));
                        }
                        ComputeBackendId::CoreMl => {}
                        ComputeBackendId::CpuLatency | ComputeBackendId::Metal => {}
                    }
                    self.metrics.prediction_backend = match backend {
                        ComputeBackendId::CoreMl if accelerator_effective => "coreml",
                        ComputeBackendId::CoreMl => "cpu-oracle-fallback",
                        ComputeBackendId::CpuLatency => "cpu-interactive",
                        ComputeBackendId::CpuUtility => "cpu-utility",
                        ComputeBackendId::Metal => "metal",
                    }
                    .to_string();
                    self.metrics.prediction_load = f64::from(values[0]);
                    self.metrics.prediction_transition = f64::from(values[1]);
                    self.metrics.prediction_pressure = f64::from(values[2]);
                    self.metrics.prediction_p95 = f64::from(values[3]);
                    self.metrics.last_latency_us = elapsed_us;
                }
            }
            self.account_completions(std::slice::from_ref(&completion), result.completed_at_us);
        }
        self.evaluate_pairs(now_us, promotion_healthy);
        while self.pairs.len() > 3 {
            if let Some(oldest) = self.pairs.keys().next().copied() {
                self.pairs.remove(&oldest);
            }
        }
    }

    fn evaluate_pairs(&mut self, now_us: u64, promotion_healthy: bool) {
        for pair in self.pairs.values_mut() {
            let (Some((cpu, cpu_us)), Some((coreml, coreml_us))) = (pair.cpu, pair.coreml) else {
                continue;
            };
            if pair.evaluated {
                continue;
            }
            if !promotion_healthy {
                pair.evaluated = true;
                continue;
            }
            let maximum_error = cpu
                .iter()
                .zip(coreml)
                .map(|(left, right)| (*left - right).abs())
                .fold(0.0_f32, f32::max);
            let _ = self.fabric.record_evaluation(
                ComputeBackendId::CoreMl,
                EvaluationSample {
                    at_us: now_us,
                    deadline_met: coreml_us <= RUNTIME_DEADLINE_US,
                    oracle_error: maximum_error > 0.01,
                    baseline_latency_us: cpu_us.max(1),
                    candidate_latency_us: coreml_us.max(1),
                    baseline_energy_uj: 1,
                    candidate_energy_uj: 1,
                    energy_measured: false,
                    safety_failure: false,
                },
            );
            pair.evaluated = true;
            self.metrics.evaluation_total = self.metrics.evaluation_total.saturating_add(1);
        }
    }

    fn account_completions(
        &mut self,
        completions: &[apollo_engine::engine::compute_fabric::ComputeCompletion],
        now_us: u64,
    ) {
        for completion in completions {
            match completion.status {
                CompletionStatus::Valid => {
                    self.metrics.completed_total = self.metrics.completed_total.saturating_add(1)
                }
                CompletionStatus::Cancelled => {
                    self.metrics.cancelled_total = self.metrics.cancelled_total.saturating_add(1)
                }
                CompletionStatus::QueueExpired
                | CompletionStatus::RuntimeExpired
                | CompletionStatus::Late => {
                    self.metrics.deadline_misses_total =
                        self.metrics.deadline_misses_total.saturating_add(1);
                    if completion.backend == Some(ComputeBackendId::CoreMl) {
                        let _ = self
                            .fabric
                            .record_deadline_miss(ComputeBackendId::CoreMl, now_us);
                    }
                }
                CompletionStatus::WrongIdentity | CompletionStatus::OutOfOrder => {
                    self.metrics.stale_total = self.metrics.stale_total.saturating_add(1)
                }
                CompletionStatus::NonFinite | CompletionStatus::Oversized => {
                    self.metrics.dispatch_skips_total =
                        self.metrics.dispatch_skips_total.saturating_add(1);
                    if completion.backend == Some(ComputeBackendId::CoreMl) {
                        let _ = self
                            .fabric
                            .record_safety_failure(ComputeBackendId::CoreMl, now_us);
                    }
                }
                CompletionStatus::CircuitOpen => {
                    self.metrics.dispatch_skips_total =
                        self.metrics.dispatch_skips_total.saturating_add(1)
                }
            }
        }
    }
}

fn peak_rss_bytes() -> u64 {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0;
    }
    #[cfg(target_os = "macos")]
    {
        usage.ru_maxrss.max(0) as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        (usage.ru_maxrss.max(0) as u64).saturating_mul(1024)
    }
}

fn rss_budget_bytes(physical_memory_bytes: u64) -> u64 {
    (physical_memory_bytes / RSS_MEMORY_DIVISOR).clamp(MIN_RSS_DELTA_BYTES, MAX_RSS_DELTA_BYTES)
}

fn prediction_array(payload: &ComputePayload) -> Option<[f32; 4]> {
    let ComputePayload::Vector(values) = payload else {
        return None;
    };
    values.as_slice().try_into().ok()
}

fn unit(value: f64) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
}

fn rollout_name(phase: RolloutPhase) -> String {
    match phase {
        RolloutPhase::Shadow => "shadow",
        RolloutPhase::Canary => "canary",
        RolloutPhase::Active => "active",
        RolloutPhase::RolledBack => "rolled-back",
    }
    .to_string()
}

fn circuit_name(state: CircuitState) -> String {
    match state {
        CircuitState::Closed => "closed",
        CircuitState::Open => "open",
        CircuitState::HalfOpen => "half-open",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::world_state::{FeatureStore, WorldIdentity};

    fn world(revision: u64) -> Arc<WorldStateSnapshot> {
        Arc::new(
            WorldStateSnapshot::new(
                WorldIdentity {
                    daemon_epoch: 7,
                    revision,
                    workload_id: 11,
                    capability_revision: 13,
                    thermal_revision: 17,
                    process_revision: 19,
                    session_revision: 23,
                    kill_switch: false,
                    sleeping: false,
                },
                revision,
                FeatureStore::try_new(1, vec![0.0; 14]).unwrap(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn optional_compute_remains_shadow_and_nonblocking_without_a_model() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        warm_baseline(&mut runtime, 35.0);
        let metrics = runtime.tick(HeterogeneousTickInput {
            world: world(2),
            cpu_utilization: 0.2,
            pressure: 0.3,
            p95_cycle_ms: 35.0,
            transition: 0.0,
            interaction: 0.5,
            io_pressure: 0.0,
            optional_allowed: true,
            budget_allows_speculation: true,
            optional_recovery_healthy: true,
            thermal_nominal: true,
        });
        assert_eq!(metrics.phase, "shadow");
        assert!(metrics.submitted_total >= 1);
        assert_eq!(runtime.submitted_total(), metrics.submitted_total);
        assert!(!metrics.ane_execution_measured);
    }

    fn tick_input(
        io_pressure: f64,
        optional_allowed: bool,
        budget: bool,
    ) -> HeterogeneousTickInput {
        HeterogeneousTickInput {
            world: world(2),
            cpu_utilization: 0.2,
            pressure: 0.3,
            p95_cycle_ms: 35.0,
            transition: 0.0,
            interaction: 0.5,
            io_pressure,
            optional_allowed,
            budget_allows_speculation: budget,
            optional_recovery_healthy: true,
            thermal_nominal: true,
        }
    }

    /// Cycles needed to arm the baseline: the leading idle streak is consumed
    /// establishing that the fabric is quiet and yields no samples.
    const CYCLES_TO_ARM: usize = BASELINE_MIN_SAMPLES + BASELINE_IDLE_CYCLES as usize + 1;

    /// Arms the control baseline so a test can exercise steady-state behaviour.
    /// Runs with the fabric blocked so the baseline is collected from genuinely
    /// idle cycles, exactly as in production.
    fn warm_baseline(runtime: &mut HeterogeneousRuntime, p95_cycle_ms: f64) {
        feed_control(runtime, p95_cycle_ms, CYCLES_TO_ARM);
    }

    /// Feeds `cycles` control-loop observations at `p95_cycle_ms` with the
    /// fabric held idle, which is the only regime the baseline is defined over.
    fn feed_control(runtime: &mut HeterogeneousRuntime, p95_cycle_ms: f64, cycles: usize) {
        for _ in 0..cycles {
            let mut sample = tick_input(0.0, false, false);
            sample.p95_cycle_ms = p95_cycle_ms;
            sample.optional_recovery_healthy = false;
            runtime.tick(sample);
        }
    }

    /// Boot is expensive — state restore, model load, first Metal warmup — and
    /// production measured 107ms there against a 39ms steady state. Fixing the
    /// baseline from that window pinned it at `MAX_CONTROL_P95_MS`, which is
    /// also the health ceiling on line 213: the guard then cannot express a
    /// regression at all, because every value it would reject is already above
    /// the ceiling that rejects independently.
    #[test]
    fn the_control_baseline_converges_from_an_expensive_warmup_to_the_steady_regime() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);

        feed_control(&mut runtime, 110.0, CYCLES_TO_ARM);
        feed_control(&mut runtime, 39.0, 240);

        let baseline = runtime.baseline_p95_ms.expect("baseline is armed");
        assert!(
            (30.0..=50.0).contains(&baseline),
            "baseline must converge toward the 39ms steady regime, got {baseline}"
        );
        assert!(
            baseline < MAX_CONTROL_P95_MS,
            "a baseline sitting on the ceiling makes the regression guard inert"
        );
    }

    /// Convergence must not become jitter: the baseline is a reference level,
    /// so one unusually quiet cycle cannot be allowed to drag it down and start
    /// rejecting healthy work on the next cycle.
    #[test]
    fn a_single_quiet_cycle_cannot_drag_the_control_baseline_down() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        feed_control(&mut runtime, 40.0, CYCLES_TO_ARM + 200);
        let settled = runtime.baseline_p95_ms.expect("baseline is armed");

        feed_control(&mut runtime, 1.0, 1);

        let after = runtime.baseline_p95_ms.expect("baseline stays armed");
        assert!(
            (settled - after).abs() < 1.0,
            "one 1ms sample moved the baseline {settled} -> {after}"
        );
    }

    /// The point of the guard: once the baseline describes the steady regime, a
    /// sustained control-loop regression must still stop optional work.
    #[test]
    fn a_sustained_regression_still_blocks_the_fabric_after_convergence() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        feed_control(&mut runtime, 39.0, CYCLES_TO_ARM + 200);
        let baseline = runtime.baseline_p95_ms.expect("baseline is armed");
        assert!(baseline < 50.0, "precondition: converged, got {baseline}");

        let mut regressed = tick_input(0.0, true, true);
        regressed.p95_cycle_ms = baseline * 1.5;
        let metrics = runtime.tick(regressed);

        assert_eq!(
            metrics.blocker, "control-p95",
            "a doubled control p95 must still be refused"
        );
    }

    #[test]
    fn network_throughput_reaches_the_temporal_model_instead_of_a_hardcoded_zero() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        warm_baseline(&mut runtime, 35.0);

        runtime.tick(tick_input(0.75, true, true));

        let observed = runtime
            .previous_observation
            .expect("an admitted cycle records its observation");
        assert!(
            (observed.io_pressure - 0.75).abs() < f32::EPSILON,
            "io_pressure is a weighted temporal feature and must carry the network signal, got {}",
            observed.io_pressure
        );
    }

    #[test]
    fn an_idle_network_still_reports_zero_io_pressure() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        warm_baseline(&mut runtime, 35.0);

        runtime.tick(tick_input(0.0, true, true));

        assert_eq!(runtime.previous_observation.unwrap().io_pressure, 0.0);
    }

    #[test]
    fn io_pressure_is_clamped_so_a_traffic_spike_cannot_escape_the_unit_range() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        warm_baseline(&mut runtime, 35.0);

        runtime.tick(tick_input(42.0, true, true));

        assert_eq!(runtime.previous_observation.unwrap().io_pressure, 1.0);
    }

    #[test]
    fn a_quiet_first_cycle_does_not_latch_a_baseline_that_blocks_the_fabric_forever() {
        // The production failure: daemon booted quiet, latched 22ms, steady
        // state settled at 37ms, and the 1.10x threshold (24.2ms) held
        // submissions at 0 for the whole daemon lifetime.
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);

        let mut quiet = tick_input(0.0, true, true);
        quiet.p95_cycle_ms = 22.0;
        runtime.tick(quiet);

        // One quiet sample must not be the baseline.
        assert_eq!(
            runtime.baseline_p95_ms, None,
            "baseline must not latch on a single startup sample"
        );

        // Steady state arrives before the warmup window closes.
        for _ in 0..CYCLES_TO_ARM {
            let mut steady = tick_input(0.0, true, true);
            steady.p95_cycle_ms = 37.0;
            runtime.tick(steady);
        }

        let baseline = runtime.baseline_p95_ms.expect("warmup window closed");
        assert!(
            37.0 <= baseline * MAX_P95_REGRESSION,
            "steady-state p95 37ms must stay inside the guard, baseline was {baseline}"
        );
    }

    /// Replaces `an_idle_fabric_lets_the_baseline_track_upward_but_never_downward`.
    ///
    /// That test pinned a ratchet: upward instantly, downward never. The
    /// "never downward" half is what pinned the baseline to the ceiling after
    /// an expensive boot and made the guard inert, so the rule is now a rolling
    /// median that moves in both directions. What must survive is the property
    /// the old test actually protected — a quiet spell cannot lower the bar and
    /// re-lock the fabric — which the median delivers by robustness rather than
    /// by refusing to move. The upward half is deliberately weakened too: a
    /// brief rise no longer moves the baseline at all.
    #[test]
    fn the_baseline_follows_a_sustained_control_shift_and_ignores_brief_ones() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        feed_control(&mut runtime, 20.0, CYCLES_TO_ARM);
        assert_eq!(runtime.baseline_p95_ms, Some(20.0));

        // A brief excursion is not a change in the control condition.
        feed_control(&mut runtime, 40.0, BASELINE_IDLE_CYCLES as usize + 2);
        assert_eq!(
            runtime.baseline_p95_ms,
            Some(20.0),
            "a handful of slow cycles must not redefine the control regime"
        );

        // Sustained, it is: the control loop genuinely got slower while the
        // fabric did nothing, so the reference level must follow.
        feed_control(&mut runtime, 40.0, BASELINE_WINDOW_SAMPLES);
        assert_eq!(runtime.baseline_p95_ms, Some(40.0));

        // And a quiet spell must still not lower the bar and re-lock the fabric.
        feed_control(&mut runtime, 12.0, 20);
        assert_eq!(
            runtime.baseline_p95_ms,
            Some(40.0),
            "a quiet period must not lower the baseline and re-lock the fabric"
        );
    }

    #[test]
    fn the_baseline_never_exceeds_the_absolute_control_ceiling() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        for _ in 0..CYCLES_TO_ARM {
            let mut awful = tick_input(0.0, false, false);
            awful.p95_cycle_ms = 5_000.0;
            runtime.tick(awful);
        }
        assert_eq!(runtime.baseline_p95_ms, Some(MAX_CONTROL_P95_MS));
    }

    #[test]
    fn an_unscheduled_cycle_is_not_reported_as_a_budget_denial() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);

        // No permit this cycle, but the overhead governor still allows
        // speculation: that is scheduling, not throttling.
        let metrics = runtime.tick(tick_input(0.0, false, true));
        assert_eq!(metrics.blocker, "not-scheduled");

        // The governor itself denying speculation keeps the budget label.
        let mut runtime = HeterogeneousRuntime::new(&initial);
        let metrics = runtime.tick(tick_input(0.0, false, false));
        assert_eq!(metrics.blocker, "overhead-budget");
    }

    #[test]
    fn pressure_cancels_optional_submission_without_touching_reflexes() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        let metrics = runtime.tick(HeterogeneousTickInput {
            world: world(2),
            cpu_utilization: 0.2,
            pressure: 0.7,
            p95_cycle_ms: 35.0,
            transition: 0.0,
            interaction: 0.5,
            io_pressure: 0.0,
            optional_allowed: true,
            budget_allows_speculation: true,
            optional_recovery_healthy: true,
            thermal_nominal: true,
        });
        assert_eq!(metrics.blocker, "memory-pressure");
        assert_eq!(metrics.submitted_total, 0);
    }

    #[test]
    fn control_p95_regression_blocks_optional_work_and_promotion_evidence() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        let mut input = HeterogeneousTickInput {
            world: Arc::clone(&initial),
            cpu_utilization: 0.2,
            pressure: 0.3,
            p95_cycle_ms: 35.0,
            transition: 0.0,
            interaction: 0.5,
            io_pressure: 0.0,
            optional_allowed: true,
            budget_allows_speculation: true,
            optional_recovery_healthy: true,
            thermal_nominal: true,
        };
        warm_baseline(&mut runtime, 35.0);
        let _ = runtime.tick(input.clone());
        std::thread::sleep(std::time::Duration::from_millis(5));
        input.p95_cycle_ms = 40.0;
        let metrics = runtime.tick(input);

        assert_eq!(metrics.blocker, "control-p95");
        assert_eq!(metrics.control_p95_baseline_ms, 35.0);
        assert_eq!(metrics.phase, "shadow");
    }

    #[test]
    fn runtime_shadow_window_starts_at_runtime_not_monotonic_epoch_zero() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        for _ in 0..500 {
            runtime
                .fabric
                .record_evaluation(
                    ComputeBackendId::CoreMl,
                    EvaluationSample {
                        at_us: runtime.last_health_sample_us.saturating_add(1),
                        deadline_met: true,
                        oracle_error: false,
                        baseline_latency_us: 100,
                        candidate_latency_us: 80,
                        baseline_energy_uj: 100,
                        candidate_energy_uj: 100,
                        energy_measured: false,
                        safety_failure: false,
                    },
                )
                .unwrap();
        }
        assert_eq!(
            runtime.fabric.rollout_phase(ComputeBackendId::CoreMl),
            RolloutPhase::Shadow
        );
    }

    #[test]
    fn rss_budget_scales_with_physical_memory_but_stays_bounded() {
        assert_eq!(rss_budget_bytes(8 * 1024 * 1024 * 1024), 64 * 1024 * 1024);
        assert_eq!(rss_budget_bytes(16 * 1024 * 1024 * 1024), 128 * 1024 * 1024);
        assert_eq!(rss_budget_bytes(64 * 1024 * 1024 * 1024), 192 * 1024 * 1024);
    }

    #[test]
    fn sustained_healthy_cycles_recover_only_the_optional_fabric_lane() {
        let initial = world(1);
        let mut runtime = HeterogeneousRuntime::new(&initial);
        warm_baseline(&mut runtime, 35.0);
        let mut last = HeterogeneousTickMetrics::default();
        for revision in 2..=(OPTIONAL_RECOVERY_CYCLES as u64) {
            last = runtime.tick(HeterogeneousTickInput {
                world: world(revision),
                cpu_utilization: 0.2,
                pressure: 0.3,
                p95_cycle_ms: 35.0,
                transition: 0.0,
                interaction: 0.5,
                io_pressure: 0.0,
                optional_allowed: false,
                budget_allows_speculation: false,
                optional_recovery_healthy: true,
                thermal_nominal: true,
            });
        }
        assert_eq!(last.blocker, "overhead-budget");

        let recovered = runtime.tick(HeterogeneousTickInput {
            world: world(OPTIONAL_RECOVERY_CYCLES as u64 + 1),
            cpu_utilization: 0.2,
            pressure: 0.3,
            p95_cycle_ms: 35.0,
            transition: 0.0,
            interaction: 0.5,
            io_pressure: 0.0,
            optional_allowed: false,
            budget_allows_speculation: false,
            optional_recovery_healthy: true,
            thermal_nominal: true,
        });
        assert_ne!(recovered.blocker, "overhead-budget");
        assert!(recovered.submitted_total > 0);
    }
}
