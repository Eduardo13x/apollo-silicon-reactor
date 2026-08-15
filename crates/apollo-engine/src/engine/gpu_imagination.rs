//! Asynchronous Metal Monte Carlo advisor for World Model candidates.
//!
//! Apollo's control plane remains on the CPU. This module batches thousands
//! of uncertainty perturbations for actions already proposed by specialists,
//! then returns a small ranking-only support signal. It cannot manufacture,
//! veto, or execute an action.

use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::thread;
use std::time::{Duration, Instant};

const MAX_CANDIDATES: usize = 24;
const DEFAULT_SAMPLES_PER_CANDIDATE: u32 = 4_096;
const MAX_SAMPLES_PER_CANDIDATE: u32 = 4_096;
const METAL_CROSSOVER_SAMPLES: u64 = 4_096;
const MAX_WORKLOAD_BYTES: usize = 128;
const MAX_ACTION_KEY_BYTES: usize = 256;
const UNCERTAINTY_TO_SIGMA: f32 = 0.08;
const MAX_RANK_SUPPORT: f64 = 0.005;
const MAX_CONTEXT_SCORE: f64 = 0.08;
const MAX_RESULT_AGE_CYCLES: u64 = 30;
const SUBMIT_COOLDOWN: Duration = Duration::from_secs(10);
const GPU_IMAGINATION_DEADLINE: Duration = Duration::from_millis(250);
const GPU_CIRCUIT_COOLDOWN: Duration = Duration::from_secs(30);
const GPU_FAILURE_THRESHOLD: u32 = 3;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GpuImaginationBackend {
    #[default]
    Initializing,
    Metal,
    CpuReference,
    Unavailable,
}

impl GpuImaginationBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Initializing => "initializing",
            Self::Metal => "metal",
            Self::CpuReference => "cpu-reference",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuImaginationGate {
    pub speculation_allowed: bool,
    pub memory_pressure: f64,
    /// GPU utilization in [0, 1] when available.
    pub gpu_load: f64,
    /// `None` means the public sensors did not report power. Unknown is not
    /// rewritten to zero; the remaining public load/thermal gates still apply.
    pub gpu_watts: Option<f64>,
    pub thermal_nominal: bool,
    pub app_launching: bool,
    pub fluidity_degraded: bool,
}

impl GpuImaginationGate {
    pub fn blocker(self) -> Option<&'static str> {
        if !self.speculation_allowed {
            Some("overhead-budget")
        } else if !self.memory_pressure.is_finite() || self.memory_pressure >= 0.55 {
            Some("memory-pressure")
        } else if !self.gpu_load.is_finite() || self.gpu_load >= 0.20 {
            Some("gpu-busy")
        } else if self
            .gpu_watts
            .is_some_and(|watts| !watts.is_finite() || watts >= 2.5)
        {
            Some("gpu-power")
        } else if !self.thermal_nominal {
            Some("thermal")
        } else if self.app_launching {
            Some("app-launch")
        } else if self.fluidity_degraded {
            Some("fluidity")
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpuImaginationCandidate {
    pub action_key: String,
    pub expected_gain: f64,
    pub uncertainty: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpuImaginationRequest {
    pub generation: u64,
    pub workload: String,
    pub context_revision: u64,
    pub candidates: Vec<GpuImaginationCandidate>,
    pub samples_per_candidate: u32,
    pub seed: u32,
}

impl GpuImaginationRequest {
    pub fn new(
        generation: u64,
        workload: &str,
        candidates: impl IntoIterator<Item = GpuImaginationCandidate>,
    ) -> Option<Self> {
        let mut candidates: Vec<_> = candidates
            .into_iter()
            .filter(|candidate| {
                !candidate.action_key.is_empty()
                    && candidate.expected_gain.is_finite()
                    && candidate.uncertainty.is_finite()
            })
            .collect();
        candidates.sort_by(|left, right| {
            left.action_key
                .cmp(&right.action_key)
                .then_with(|| left.uncertainty.total_cmp(&right.uncertainty))
                .then_with(|| right.expected_gain.total_cmp(&left.expected_gain))
        });
        candidates.dedup_by(|left, right| left.action_key == right.action_key);
        candidates.truncate(MAX_CANDIDATES);
        if candidates.is_empty() {
            return None;
        }
        for candidate in &mut candidates {
            candidate.expected_gain = candidate.expected_gain.clamp(-1.0, 1.0);
            candidate.uncertainty = candidate.uncertainty.clamp(0.0, 1.0);
        }
        let seed = candidates
            .iter()
            .fold(generation as u32, |hash, candidate| {
                fnv1a32(hash, candidate.action_key.as_bytes())
            });
        Some(Self {
            generation,
            workload: workload.to_string(),
            context_revision: 0,
            candidates,
            samples_per_candidate: DEFAULT_SAMPLES_PER_CANDIDATE,
            seed,
        })
    }

    pub fn total_samples(&self) -> u64 {
        self.candidates.len() as u64 * self.samples_per_candidate as u64
    }

    pub fn with_context_revision(mut self, context_revision: u64) -> Self {
        self.context_revision = context_revision;
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuCandidateAdvice {
    pub action_key: String,
    pub expected_gain: f64,
    pub uncertainty: f64,
    pub mean_gain: f64,
    pub p10_gain: f64,
    pub positive_probability: f64,
    /// Ranking-only contribution bounded to +/-0.005 utility.
    pub rank_support: f64,
    /// Contextual confidence contribution for external specialist lanes.
    /// It remains ranking-only and cannot authorize an action.
    pub context_score: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuImaginationResult {
    pub generation: u64,
    pub workload: String,
    pub context_revision: u64,
    pub backend: GpuImaginationBackend,
    pub device_name: String,
    pub samples: u64,
    pub gpu_time_ns: u64,
    pub wall_time_ns: u64,
    pub candidates: Vec<GpuCandidateAdvice>,
    pub error: Option<String>,
}

impl GpuImaginationResult {
    pub fn is_fresh_for(&self, cycle: u64, workload: &str) -> bool {
        self.error.is_none()
            && self.workload == workload
            && cycle >= self.generation
            && cycle - self.generation <= MAX_RESULT_AGE_CYCLES
    }

    pub fn is_fresh_for_context(&self, cycle: u64, workload: &str, context_revision: u64) -> bool {
        self.context_revision == context_revision && self.is_fresh_for(cycle, workload)
    }

    pub fn support_for(&self, action_key: &str) -> Option<f64> {
        self.candidates
            .iter()
            .find(|candidate| candidate.action_key == action_key)
            .map(|candidate| candidate.rank_support)
    }

    pub fn best(&self) -> Option<&GpuCandidateAdvice> {
        self.candidates.iter().max_by(|left, right| {
            left.rank_support
                .total_cmp(&right.rank_support)
                .then_with(|| left.action_key.cmp(&right.action_key))
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSubmitOutcome {
    Submitted,
    Busy,
    Cooldown,
    Gated(&'static str),
    Unavailable,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GpuCircuitState {
    #[default]
    Closed,
    Open,
    HalfOpen,
}

impl GpuCircuitState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Closed => "closed",
            Self::Open => "open",
            Self::HalfOpen => "half-open",
        }
    }
}

impl GpuSubmitOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Submitted => "submitted",
            Self::Busy => "busy",
            Self::Cooldown => "cooldown",
            Self::Gated(reason) => reason,
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone)]
struct BackendStatus {
    backend: GpuImaginationBackend,
    device_name: String,
    initialization_error: Option<String>,
}

#[derive(Debug, Clone)]
struct InFlightJob {
    generation: u64,
    workload: String,
    context_revision: u64,
    submitted_at: Instant,
}

#[derive(Debug, Clone, Copy)]
enum BackendPreference {
    MetalOnly,
    #[cfg(test)]
    CpuReference,
}

/// Non-blocking handle. Metal pipeline creation and command-buffer waits live
/// exclusively on the worker thread.
pub struct GpuImaginationWorker {
    request_tx: SyncSender<GpuImaginationRequest>,
    result_rx: Receiver<GpuImaginationResult>,
    status_rx: Receiver<BackendStatus>,
    backend: GpuImaginationBackend,
    device_name: String,
    initialization_error: Option<String>,
    last_submit: Option<Instant>,
    latest: Option<GpuImaginationResult>,
    last_consumed_generation: Option<u64>,
    in_flight: Option<InFlightJob>,
    draining: Option<InFlightJob>,
    circuit_state: GpuCircuitState,
    circuit_opened_at: Option<Instant>,
    circuit_cooldown: Duration,
    consecutive_failures: u32,
    quarantine_reason: Option<String>,
}

impl std::fmt::Debug for GpuImaginationWorker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GpuImaginationWorker")
            .field("backend", &self.backend)
            .field("device_name", &self.device_name)
            .finish_non_exhaustive()
    }
}

impl GpuImaginationWorker {
    pub fn spawn_metal_only() -> Self {
        Self::spawn(
            BackendPreference::MetalOnly,
            Duration::ZERO,
            GPU_CIRCUIT_COOLDOWN,
            false,
        )
    }

    #[cfg(test)]
    fn spawn_cpu_reference() -> Self {
        Self::spawn(
            BackendPreference::CpuReference,
            Duration::ZERO,
            GPU_CIRCUIT_COOLDOWN,
            false,
        )
    }

    #[cfg(test)]
    fn spawn_cpu_reference_with_delay(delay: Duration) -> Self {
        Self::spawn(
            BackendPreference::CpuReference,
            delay,
            GPU_CIRCUIT_COOLDOWN,
            false,
        )
    }

    #[cfg(test)]
    fn spawn_cpu_reference_with_timing(delay: Duration, circuit_cooldown: Duration) -> Self {
        Self::spawn(
            BackendPreference::CpuReference,
            delay,
            circuit_cooldown,
            false,
        )
    }

    #[cfg(test)]
    fn spawn_cpu_reference_with_recovery_timing(
        first_delay: Duration,
        circuit_cooldown: Duration,
    ) -> Self {
        Self::spawn(
            BackendPreference::CpuReference,
            first_delay,
            circuit_cooldown,
            true,
        )
    }

    fn spawn(
        preference: BackendPreference,
        test_delay: Duration,
        circuit_cooldown: Duration,
        test_delay_once: bool,
    ) -> Self {
        let (request_tx, request_rx) = mpsc::sync_channel::<GpuImaginationRequest>(1);
        let (result_tx, result_rx) = mpsc::sync_channel::<GpuImaginationResult>(1);
        let (status_tx, status_rx) = mpsc::sync_channel::<BackendStatus>(1);
        let spawn_result = thread::Builder::new()
            .name("apollo-gpu-imagination".to_string())
            .spawn(move || {
                run_worker(
                    preference,
                    request_rx,
                    result_tx,
                    status_tx,
                    test_delay,
                    test_delay_once,
                )
            });
        let (backend, initialization_error) = match spawn_result {
            Ok(_) => (GpuImaginationBackend::Initializing, None),
            Err(error) => (
                GpuImaginationBackend::Unavailable,
                Some(format!("GPU imagination worker spawn failed: {error}")),
            ),
        };
        Self {
            request_tx,
            result_rx,
            status_rx,
            backend,
            device_name: String::new(),
            initialization_error,
            last_submit: None,
            latest: None,
            last_consumed_generation: None,
            in_flight: None,
            draining: None,
            circuit_state: GpuCircuitState::Closed,
            circuit_opened_at: None,
            circuit_cooldown,
            consecutive_failures: 0,
            quarantine_reason: None,
        }
    }

    pub fn backend(&mut self) -> GpuImaginationBackend {
        self.poll();
        self.backend
    }

    pub fn device_name(&mut self) -> &str {
        self.poll();
        &self.device_name
    }

    pub fn initialization_error(&mut self) -> Option<&str> {
        self.poll();
        self.initialization_error.as_deref()
    }

    pub fn latest(&mut self) -> Option<&GpuImaginationResult> {
        self.poll();
        self.latest.as_ref()
    }

    /// Returns each completed job once for cumulative telemetry while keeping
    /// the latest result available as short-lived ranking context.
    pub fn take_completed(&mut self) -> Option<GpuImaginationResult> {
        self.poll();
        let result = self.latest.as_ref()?;
        if self.last_consumed_generation == Some(result.generation) {
            return None;
        }
        self.last_consumed_generation = Some(result.generation);
        Some(result.clone())
    }

    pub fn quarantine_reason(&mut self) -> Option<&str> {
        self.poll();
        self.quarantine_reason.as_deref()
    }

    pub fn circuit_state(&mut self) -> GpuCircuitState {
        self.poll();
        self.circuit_state
    }

    /// Results submitted before sleep/wake are no longer comparable with the
    /// resumed hardware context, so they are dropped rather than consumed.
    pub fn invalidate_for_wake(&mut self) {
        if let Some(in_flight) = self.in_flight.take() {
            self.draining = Some(in_flight);
            self.open_circuit("wake-invalidation");
        }
        self.latest = None;
        self.last_consumed_generation = None;
        self.last_submit = None;
    }

    pub fn try_submit(
        &mut self,
        request: GpuImaginationRequest,
        gate: GpuImaginationGate,
    ) -> GpuSubmitOutcome {
        self.poll();
        if request.candidates.is_empty()
            || request.candidates.len() > MAX_CANDIDATES
            || request.samples_per_candidate == 0
            || request.samples_per_candidate > MAX_SAMPLES_PER_CANDIDATE
            || request.workload.len() > MAX_WORKLOAD_BYTES
            || request.candidates.iter().any(|candidate| {
                candidate.action_key.is_empty()
                    || candidate.action_key.len() > MAX_ACTION_KEY_BYTES
                    || !candidate.expected_gain.is_finite()
                    || !candidate.uncertainty.is_finite()
            })
        {
            return GpuSubmitOutcome::Gated("invalid-request");
        }
        if request.total_samples() < METAL_CROSSOVER_SAMPLES {
            return GpuSubmitOutcome::Gated("below-crossover");
        }
        if let Some(blocker) = gate.blocker() {
            return GpuSubmitOutcome::Gated(blocker);
        }
        if self.backend == GpuImaginationBackend::Unavailable {
            return GpuSubmitOutcome::Unavailable;
        }
        self.maybe_enter_half_open();
        if self.circuit_state == GpuCircuitState::Open {
            return GpuSubmitOutcome::Unavailable;
        }
        if self.circuit_state == GpuCircuitState::Closed
            && self
                .last_submit
                .is_some_and(|last| last.elapsed() < SUBMIT_COOLDOWN)
        {
            return GpuSubmitOutcome::Cooldown;
        }
        let generation = request.generation;
        let workload = request.workload.clone();
        let context_revision = request.context_revision;
        match self.request_tx.try_send(request) {
            Ok(()) => {
                self.last_submit = Some(Instant::now());
                self.in_flight = Some(InFlightJob {
                    generation,
                    workload,
                    context_revision,
                    submitted_at: Instant::now(),
                });
                GpuSubmitOutcome::Submitted
            }
            Err(TrySendError::Full(_)) => GpuSubmitOutcome::Busy,
            Err(TrySendError::Disconnected(_)) => {
                self.backend = GpuImaginationBackend::Unavailable;
                GpuSubmitOutcome::Unavailable
            }
        }
    }

    fn poll(&mut self) {
        while let Ok(status) = self.status_rx.try_recv() {
            self.backend = status.backend;
            self.device_name = status.device_name;
            self.initialization_error = status.initialization_error;
        }
        loop {
            match self.result_rx.try_recv() {
                Ok(result) => {
                    let matches_in_flight = self.in_flight.as_ref().is_some_and(|job| {
                        job.generation == result.generation
                            && job.workload == result.workload
                            && job.context_revision == result.context_revision
                    });
                    let matches_draining = self.draining.as_ref().is_some_and(|job| {
                        job.generation == result.generation
                            && job.workload == result.workload
                            && job.context_revision == result.context_revision
                    });
                    if matches_draining {
                        self.draining = None;
                        continue;
                    }
                    if !matches_in_flight {
                        continue;
                    }
                    self.in_flight = None;
                    if Duration::from_nanos(result.wall_time_ns) > GPU_IMAGINATION_DEADLINE {
                        self.open_circuit("deadline");
                        self.initialization_error =
                            Some("GPU imagination deadline exceeded".to_string());
                        continue;
                    }
                    if result.error.is_some() {
                        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
                        if self.circuit_state == GpuCircuitState::HalfOpen
                            || self.consecutive_failures >= GPU_FAILURE_THRESHOLD
                        {
                            self.open_circuit("backend-error");
                        }
                    } else {
                        self.consecutive_failures = 0;
                        self.circuit_state = GpuCircuitState::Closed;
                        self.circuit_opened_at = None;
                        self.quarantine_reason = None;
                        self.initialization_error = None;
                        self.backend = result.backend;
                        if !result.device_name.is_empty() {
                            self.device_name.clone_from(&result.device_name);
                        }
                    }
                    self.latest = Some(result);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.backend = GpuImaginationBackend::Unavailable;
                    break;
                }
            }
        }
        // The control loop may collect less frequently than the GPU deadline.
        // Drain queued completions first and judge them by worker wall time;
        // elapsed collection time only expires a job that has no result yet.
        if self
            .in_flight
            .as_ref()
            .is_some_and(|job| job.submitted_at.elapsed() > GPU_IMAGINATION_DEADLINE)
        {
            self.draining = self.in_flight.take();
            self.open_circuit("deadline");
            self.initialization_error = Some("GPU imagination deadline exceeded".to_string());
        }
    }

    fn open_circuit(&mut self, reason: &str) {
        self.circuit_state = GpuCircuitState::Open;
        self.circuit_opened_at = Some(Instant::now());
        self.quarantine_reason = Some(reason.to_string());
    }

    fn maybe_enter_half_open(&mut self) {
        if self.circuit_state == GpuCircuitState::Open
            && self.draining.is_none()
            && self
                .circuit_opened_at
                .is_some_and(|opened| opened.elapsed() >= self.circuit_cooldown)
        {
            self.circuit_state = GpuCircuitState::HalfOpen;
        }
    }
}

fn run_worker(
    preference: BackendPreference,
    request_rx: Receiver<GpuImaginationRequest>,
    result_tx: SyncSender<GpuImaginationResult>,
    status_tx: SyncSender<BackendStatus>,
    test_delay: Duration,
    test_delay_once: bool,
) {
    #[cfg(target_os = "macos")]
    let (mut metal, mut initialization_error) =
        if matches!(preference, BackendPreference::MetalOnly) {
            match metal::MetalKernel::new() {
                Ok(kernel) => (Some(kernel), None),
                Err(error) => (None, Some(error)),
            }
        } else {
            (None, None)
        };
    #[cfg(target_os = "macos")]
    if let Some(kernel) = metal.as_mut() {
        // Pipeline creation and this one minimal dispatch stay on the worker,
        // so the daemon's control cycle never pays Metal warmup latency.
        let prewarm = GpuImaginationRequest {
            generation: 0,
            workload: "prewarm".to_string(),
            context_revision: 0,
            candidates: vec![GpuImaginationCandidate {
                action_key: "prewarm".to_string(),
                expected_gain: 0.0,
                uncertainty: 0.0,
            }],
            samples_per_candidate: METAL_CROSSOVER_SAMPLES as u32,
            seed: 0,
        };
        if let Err(error) = kernel.run(&prewarm) {
            initialization_error = Some(format!("Metal prewarm failed: {error}"));
            metal = None;
        }
    }
    #[cfg(not(target_os = "macos"))]
    let metal: Option<()> = None;
    #[cfg(not(target_os = "macos"))]
    let initialization_error = Some("Metal unavailable on this platform".to_string());

    let backend = match preference {
        BackendPreference::MetalOnly if metal.is_some() => GpuImaginationBackend::Metal,
        BackendPreference::MetalOnly => GpuImaginationBackend::Unavailable,
        #[cfg(test)]
        BackendPreference::CpuReference => GpuImaginationBackend::CpuReference,
    };
    #[cfg(target_os = "macos")]
    let device_name = metal
        .as_ref()
        .map(|kernel| kernel.device_name().to_string())
        .unwrap_or_default();
    #[cfg(not(target_os = "macos"))]
    let device_name = String::new();
    let _ = status_tx.send(BackendStatus {
        backend,
        device_name: device_name.clone(),
        initialization_error: initialization_error.clone(),
    });

    #[cfg(test)]
    let mut first_request = true;
    while let Ok(request) = request_rx.recv() {
        let started = Instant::now();
        #[cfg(test)]
        if !test_delay.is_zero() && (!test_delay_once || first_request) {
            thread::sleep(test_delay);
        }
        #[cfg(not(test))]
        let _ = (test_delay, test_delay_once);
        #[cfg(test)]
        {
            first_request = false;
        }
        let execution = match preference {
            BackendPreference::MetalOnly => {
                #[cfg(target_os = "macos")]
                {
                    metal
                        .as_mut()
                        .ok_or_else(|| "Metal unavailable".to_string())
                        .and_then(|kernel| kernel.run(&request))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Err(initialization_error
                        .clone()
                        .unwrap_or_else(|| "Metal unavailable".to_string()))
                }
            }
            #[cfg(test)]
            BackendPreference::CpuReference => Ok((simulate_samples_cpu(&request), 0)),
        };
        let wall_time_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let result = match execution {
            Ok((samples, gpu_time_ns)) => summarize(
                &request,
                samples,
                backend,
                &device_name,
                gpu_time_ns,
                wall_time_ns,
            ),
            Err(error) => GpuImaginationResult {
                generation: request.generation,
                workload: request.workload,
                context_revision: request.context_revision,
                backend: GpuImaginationBackend::Unavailable,
                device_name: device_name.clone(),
                wall_time_ns,
                error: Some(error),
                ..GpuImaginationResult::default()
            },
        };
        let _ = result_tx.try_send(result);
    }
}

fn summarize(
    request: &GpuImaginationRequest,
    samples: Vec<f32>,
    backend: GpuImaginationBackend,
    device_name: &str,
    gpu_time_ns: u64,
    wall_time_ns: u64,
) -> GpuImaginationResult {
    let samples_per_candidate = request.samples_per_candidate as usize;
    if samples.len() != samples_per_candidate.saturating_mul(request.candidates.len()) {
        return GpuImaginationResult {
            generation: request.generation,
            workload: request.workload.clone(),
            context_revision: request.context_revision,
            backend: GpuImaginationBackend::Unavailable,
            device_name: device_name.to_string(),
            wall_time_ns,
            error: Some("GPU output shape mismatch".to_string()),
            ..GpuImaginationResult::default()
        };
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return GpuImaginationResult {
            generation: request.generation,
            workload: request.workload.clone(),
            context_revision: request.context_revision,
            backend: GpuImaginationBackend::Unavailable,
            device_name: device_name.to_string(),
            wall_time_ns,
            error: Some("GPU output is non-finite".to_string()),
            ..GpuImaginationResult::default()
        };
    }
    let mut advice = Vec::with_capacity(request.candidates.len());
    for (index, candidate) in request.candidates.iter().enumerate() {
        let start = index * samples_per_candidate;
        let end = start + samples_per_candidate;
        let mut values = samples[start..end].to_vec();
        let mean = values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64;
        let positive_probability =
            values.iter().filter(|value| **value > 0.0).count() as f64 / values.len() as f64;
        let p10_index = (values.len() - 1) / 10;
        let (_, p10, _) = values.select_nth_unstable_by(p10_index, f32::total_cmp);
        let p10 = *p10 as f64;
        let confidence_support = (positive_probability - 0.5) * 0.008;
        let downside_support = p10.clamp(-0.10, 0.10) * 0.02;
        let context_score = ((positive_probability - 0.5) * 0.12
            + mean.clamp(-0.20, 0.20) * 0.10
            + p10.clamp(-0.20, 0.20) * 0.10)
            .clamp(-MAX_CONTEXT_SCORE, MAX_CONTEXT_SCORE);
        advice.push(GpuCandidateAdvice {
            action_key: candidate.action_key.clone(),
            expected_gain: candidate.expected_gain,
            uncertainty: candidate.uncertainty,
            mean_gain: mean,
            p10_gain: p10,
            positive_probability,
            rank_support: (confidence_support + downside_support)
                .clamp(-MAX_RANK_SUPPORT, MAX_RANK_SUPPORT),
            context_score,
        });
    }
    GpuImaginationResult {
        generation: request.generation,
        workload: request.workload.clone(),
        context_revision: request.context_revision,
        backend,
        device_name: device_name.to_string(),
        samples: request.total_samples(),
        gpu_time_ns,
        wall_time_ns,
        candidates: advice,
        error: None,
    }
}

#[cfg(test)]
fn simulate_samples_cpu(request: &GpuImaginationRequest) -> Vec<f32> {
    let mut output = vec![0.0_f32; request.total_samples() as usize];
    for (candidate_index, candidate) in request.candidates.iter().enumerate() {
        let expected_gain = candidate.expected_gain as f32;
        let sigma = candidate.uncertainty as f32 * UNCERTAINTY_TO_SIGMA;
        let candidate_seed = fnv1a32(0x811c_9dc5, candidate.action_key.as_bytes());
        for sample_index in 0..request.samples_per_candidate {
            let state = request.seed
                ^ candidate_seed
                ^ sample_index.wrapping_mul(0x9e37_79b9)
                ^ (candidate_index as u32).wrapping_mul(0x85eb_ca6b);
            let mut noise = -3.0_f32;
            for lane in 0..6_u32 {
                noise += uniform01(state.wrapping_add(lane.wrapping_mul(0x27d4_eb2d)));
            }
            output[candidate_index * request.samples_per_candidate as usize
                + sample_index as usize] = (expected_gain + sigma * noise).clamp(-1.0, 1.0);
        }
    }
    output
}

fn fnv1a32(mut hash: u32, bytes: &[u8]) -> u32 {
    for byte in bytes {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

#[cfg(test)]
fn mix_bits(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value
}

#[cfg(test)]
fn uniform01(value: u32) -> f32 {
    (mix_bits(value) & 0x00ff_ffff) as f32 * (1.0 / 16_777_216.0)
}

#[cfg(target_os = "macos")]
mod metal {
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr::NonNull;

    use super::{fnv1a32, GpuImaginationRequest, UNCERTAINTY_TO_SIGMA};

    #[repr(C)]
    struct MetalCandidate {
        expected_gain: f32,
        sigma: f32,
        candidate_seed: u32,
        reserved: u32,
    }

    unsafe extern "C" {
        fn apollo_gpu_imagination_create(error_out: *mut c_char, capacity: usize) -> *mut c_void;
        fn apollo_gpu_imagination_destroy(context: *mut c_void);
        fn apollo_gpu_imagination_device_name(
            context: *mut c_void,
            out: *mut c_char,
            capacity: usize,
        ) -> i32;
        fn apollo_gpu_imagination_run(
            context: *mut c_void,
            candidates: *const MetalCandidate,
            candidate_count: u32,
            samples_per_candidate: u32,
            seed: u32,
            out_gain: *mut f32,
            gpu_time_ns: *mut u64,
        ) -> i32;
    }

    pub(super) struct MetalKernel {
        context: NonNull<c_void>,
        device_name: String,
    }

    impl MetalKernel {
        pub(super) fn new() -> Result<Self, String> {
            let mut error = [0_i8; 512];
            let context = NonNull::new(unsafe {
                apollo_gpu_imagination_create(error.as_mut_ptr(), error.len())
            })
            .ok_or_else(|| {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                if message.is_empty() {
                    "Metal initialization failed".to_string()
                } else {
                    message
                }
            })?;
            let mut name = [0_i8; 128];
            let status = unsafe {
                apollo_gpu_imagination_device_name(context.as_ptr(), name.as_mut_ptr(), name.len())
            };
            let device_name = if status == 0 {
                unsafe { CStr::from_ptr(name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Apple GPU".to_string()
            };
            Ok(Self {
                context,
                device_name,
            })
        }

        pub(super) fn device_name(&self) -> &str {
            &self.device_name
        }

        pub(super) fn run(
            &mut self,
            request: &GpuImaginationRequest,
        ) -> Result<(Vec<f32>, u64), String> {
            let candidates: Vec<_> = request
                .candidates
                .iter()
                .map(|candidate| MetalCandidate {
                    expected_gain: candidate.expected_gain as f32,
                    sigma: candidate.uncertainty as f32 * UNCERTAINTY_TO_SIGMA,
                    candidate_seed: fnv1a32(0x811c_9dc5, candidate.action_key.as_bytes()),
                    reserved: 0,
                })
                .collect();
            let mut output = vec![0.0_f32; request.total_samples() as usize];
            let mut gpu_time_ns = 0_u64;
            let status = unsafe {
                apollo_gpu_imagination_run(
                    self.context.as_ptr(),
                    candidates.as_ptr(),
                    candidates.len() as u32,
                    request.samples_per_candidate,
                    request.seed,
                    output.as_mut_ptr(),
                    &mut gpu_time_ns,
                )
            };
            if status == 0 {
                Ok((output, gpu_time_ns))
            } else {
                Err(format!("Metal imagination failed with status {status}"))
            }
        }
    }

    impl Drop for MetalKernel {
        fn drop(&mut self) {
            unsafe { apollo_gpu_imagination_destroy(self.context.as_ptr()) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> GpuImaginationRequest {
        GpuImaginationRequest::new(
            42,
            "coding",
            [
                GpuImaginationCandidate {
                    action_key: "boost:Editor".to_string(),
                    expected_gain: 0.04,
                    uncertainty: 0.10,
                },
                GpuImaginationCandidate {
                    action_key: "interaction_qos:foreground".to_string(),
                    expected_gain: 0.01,
                    uncertainty: 0.45,
                },
            ],
        )
        .expect("two valid candidates")
    }

    #[test]
    fn request_is_bounded_sorted_and_deterministic() {
        let candidate_request = request();
        assert_eq!(candidate_request.candidates.len(), 2);
        assert_eq!(candidate_request.candidates[0].action_key, "boost:Editor");
        assert_eq!(candidate_request.total_samples(), 8_192);
        assert_eq!(candidate_request.seed, request().seed);
    }

    #[test]
    fn submission_rejects_invalid_shapes_and_work_below_the_metal_crossover() {
        let mut worker = GpuImaginationWorker::spawn_cpu_reference();
        let gate = GpuImaginationGate {
            speculation_allowed: true,
            memory_pressure: 0.20,
            gpu_load: 0.0,
            gpu_watts: None,
            thermal_nominal: true,
            app_launching: false,
            fluidity_degraded: false,
        };
        let mut too_small = request();
        too_small.samples_per_candidate = 1;
        assert_eq!(
            worker.try_submit(too_small, gate),
            GpuSubmitOutcome::Gated("below-crossover")
        );
        let mut oversized = request();
        oversized.samples_per_candidate = MAX_SAMPLES_PER_CANDIDATE + 1;
        assert_eq!(
            worker.try_submit(oversized, gate),
            GpuSubmitOutcome::Gated("invalid-request")
        );
    }

    #[test]
    fn gate_yields_to_user_gpu_and_system_stress() {
        let safe = GpuImaginationGate {
            speculation_allowed: true,
            memory_pressure: 0.30,
            gpu_load: 0.05,
            gpu_watts: Some(0.4),
            thermal_nominal: true,
            app_launching: false,
            fluidity_degraded: false,
        };
        assert_eq!(safe.blocker(), None);
        assert_eq!(
            GpuImaginationGate {
                gpu_load: 0.80,
                ..safe
            }
            .blocker(),
            Some("gpu-busy")
        );
        assert_eq!(
            GpuImaginationGate {
                memory_pressure: 0.70,
                ..safe
            }
            .blocker(),
            Some("memory-pressure")
        );
    }

    #[test]
    fn missing_gpu_watts_stays_unknown_without_blocking_public_safety_gates() {
        let unknown = GpuImaginationGate {
            speculation_allowed: true,
            memory_pressure: 0.20,
            gpu_load: 0.05,
            gpu_watts: None,
            thermal_nominal: true,
            app_launching: false,
            fluidity_degraded: false,
        };
        assert_eq!(unknown.blocker(), None);
        assert_eq!(
            GpuImaginationGate {
                gpu_watts: Some(3.0),
                ..unknown
            }
            .blocker(),
            Some("gpu-power")
        );
        assert_eq!(
            GpuImaginationGate {
                gpu_watts: Some(f64::NAN),
                ..unknown
            }
            .blocker(),
            Some("gpu-power")
        );
    }

    #[test]
    fn cpu_reference_is_deterministic_and_support_is_bounded() {
        let request = request();
        let first = simulate_samples_cpu(&request);
        let second = simulate_samples_cpu(&request);
        assert_eq!(first, second);
        let result = summarize(
            &request,
            first,
            GpuImaginationBackend::CpuReference,
            "cpu",
            0,
            1,
        );
        assert_eq!(result.candidates.len(), 2);
        assert!(result
            .candidates
            .iter()
            .all(|candidate| candidate.rank_support.abs() <= MAX_RANK_SUPPORT));
        assert!(result
            .candidates
            .iter()
            .all(|candidate| candidate.context_score.abs() <= MAX_CONTEXT_SCORE));
        assert_eq!(
            result.best().map(|candidate| candidate.action_key.as_str()),
            Some("boost:Editor")
        );
    }

    #[test]
    fn stale_or_cross_workload_result_never_contributes() {
        let request = request();
        let result = summarize(
            &request,
            simulate_samples_cpu(&request),
            GpuImaginationBackend::CpuReference,
            "cpu",
            0,
            1,
        );
        assert!(result.is_fresh_for(60, "coding"));
        assert!(!result.is_fresh_for(100, "coding"));
        assert!(!result.is_fresh_for(60, "browsing"));
    }

    #[test]
    fn result_never_crosses_capability_or_thermal_revision() {
        let request = request().with_context_revision(17);
        let result = summarize(
            &request,
            simulate_samples_cpu(&request),
            GpuImaginationBackend::CpuReference,
            "cpu",
            0,
            1,
        );
        assert!(result.is_fresh_for_context(60, "coding", 17));
        assert!(!result.is_fresh_for_context(60, "coding", 18));
    }

    #[test]
    fn worker_keeps_compute_off_the_calling_thread() {
        let mut worker = GpuImaginationWorker::spawn_cpu_reference();
        let gate = GpuImaginationGate {
            speculation_allowed: true,
            memory_pressure: 0.20,
            gpu_load: 0.0,
            gpu_watts: Some(0.0),
            thermal_nominal: true,
            app_launching: false,
            fluidity_degraded: false,
        };
        assert_eq!(
            worker.try_submit(request(), gate),
            GpuSubmitOutcome::Submitted
        );
        let deadline = Instant::now() + Duration::from_secs(2);
        while worker.latest().is_none() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(worker.backend(), GpuImaginationBackend::CpuReference);
        assert!(worker.latest().is_some_and(|result| result.error.is_none()));
        assert!(worker.take_completed().is_some());
        assert!(worker.take_completed().is_none());
    }

    #[test]
    fn non_finite_gpu_output_is_rejected() {
        let request = request();
        let mut samples = simulate_samples_cpu(&request);
        samples[0] = f32::NAN;
        let result = summarize(
            &request,
            samples,
            GpuImaginationBackend::CpuReference,
            "cpu",
            0,
            1,
        );
        assert_eq!(result.error.as_deref(), Some("GPU output is non-finite"));
    }

    #[test]
    fn overdue_gpu_job_is_quarantined_and_late_result_is_ignored() {
        let mut worker = GpuImaginationWorker::spawn_cpu_reference_with_delay(
            GPU_IMAGINATION_DEADLINE + Duration::from_millis(40),
        );
        let gate = GpuImaginationGate {
            speculation_allowed: true,
            memory_pressure: 0.20,
            gpu_load: 0.0,
            gpu_watts: None,
            thermal_nominal: true,
            app_launching: false,
            fluidity_degraded: false,
        };
        assert_eq!(
            worker.try_submit(request(), gate),
            GpuSubmitOutcome::Submitted
        );
        thread::sleep(GPU_IMAGINATION_DEADLINE + Duration::from_millis(10));
        assert_eq!(worker.quarantine_reason(), Some("deadline"));
        thread::sleep(Duration::from_millis(50));
        assert!(worker.latest().is_none());
        assert_eq!(
            worker.try_submit(request(), gate),
            GpuSubmitOutcome::Unavailable
        );
    }

    #[test]
    fn completed_but_overdue_result_is_rejected_before_receive() {
        let mut worker = GpuImaginationWorker::spawn_cpu_reference_with_delay(
            GPU_IMAGINATION_DEADLINE + Duration::from_millis(10),
        );
        let gate = GpuImaginationGate {
            speculation_allowed: true,
            memory_pressure: 0.20,
            gpu_load: 0.0,
            gpu_watts: None,
            thermal_nominal: true,
            app_launching: false,
            fluidity_degraded: false,
        };
        assert_eq!(
            worker.try_submit(request(), gate),
            GpuSubmitOutcome::Submitted
        );
        thread::sleep(GPU_IMAGINATION_DEADLINE + Duration::from_millis(40));
        assert_eq!(worker.quarantine_reason(), Some("deadline"));
        assert!(worker.latest().is_none());
    }

    #[test]
    fn result_completed_within_deadline_survives_a_slow_control_poll() {
        let mut worker =
            GpuImaginationWorker::spawn_cpu_reference_with_delay(Duration::from_millis(10));
        let gate = GpuImaginationGate {
            speculation_allowed: true,
            memory_pressure: 0.20,
            gpu_load: 0.0,
            gpu_watts: None,
            thermal_nominal: true,
            app_launching: false,
            fluidity_degraded: false,
        };
        assert_eq!(
            worker.try_submit(request(), gate),
            GpuSubmitOutcome::Submitted
        );

        // Production often polls this worker on a cadence longer than the GPU
        // deadline. Completion time, not collection time, decides freshness.
        thread::sleep(GPU_IMAGINATION_DEADLINE + Duration::from_millis(40));

        let completed = worker.take_completed().expect("on-time worker result");
        assert!(completed.error.is_none());
        assert_eq!(worker.circuit_state(), GpuCircuitState::Closed);
        assert_eq!(worker.quarantine_reason(), None);
    }

    #[test]
    fn wake_invalidation_discards_pre_wake_result_and_reopens_lane() {
        let mut worker = GpuImaginationWorker::spawn_cpu_reference_with_timing(
            Duration::from_millis(40),
            Duration::from_millis(10),
        );
        let gate = GpuImaginationGate {
            speculation_allowed: true,
            memory_pressure: 0.20,
            gpu_load: 0.0,
            gpu_watts: None,
            thermal_nominal: true,
            app_launching: false,
            fluidity_degraded: false,
        };
        assert_eq!(
            worker.try_submit(request(), gate),
            GpuSubmitOutcome::Submitted
        );
        worker.invalidate_for_wake();
        assert_eq!(worker.circuit_state(), GpuCircuitState::Open);
        assert_eq!(
            worker.try_submit(request(), gate),
            GpuSubmitOutcome::Unavailable
        );
        thread::sleep(Duration::from_millis(60));
        assert!(worker.latest().is_none());
        assert_eq!(
            worker.try_submit(request(), gate),
            GpuSubmitOutcome::Submitted
        );
        assert_eq!(worker.circuit_state(), GpuCircuitState::HalfOpen);

        let deadline = Instant::now() + Duration::from_secs(1);
        while worker.latest().is_none() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(worker.circuit_state(), GpuCircuitState::Closed);
        assert_eq!(worker.quarantine_reason(), None);
    }

    #[test]
    fn timed_out_worker_recovers_only_after_late_result_drains_and_probe_succeeds() {
        let mut worker = GpuImaginationWorker::spawn_cpu_reference_with_recovery_timing(
            GPU_IMAGINATION_DEADLINE + Duration::from_millis(30),
            Duration::from_millis(10),
        );
        let gate = GpuImaginationGate {
            speculation_allowed: true,
            memory_pressure: 0.20,
            gpu_load: 0.0,
            gpu_watts: None,
            thermal_nominal: true,
            app_launching: false,
            fluidity_degraded: false,
        };
        assert_eq!(
            worker.try_submit(request(), gate),
            GpuSubmitOutcome::Submitted
        );
        thread::sleep(GPU_IMAGINATION_DEADLINE + Duration::from_millis(5));
        assert_eq!(worker.circuit_state(), GpuCircuitState::Open);
        assert_eq!(
            worker.try_submit(request(), gate),
            GpuSubmitOutcome::Unavailable
        );

        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            worker.try_submit(request(), gate),
            GpuSubmitOutcome::Submitted
        );
        assert_eq!(worker.circuit_state(), GpuCircuitState::HalfOpen);
        let deadline = Instant::now() + Duration::from_secs(1);
        while worker.latest().is_none() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(worker.circuit_state(), GpuCircuitState::Closed);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matches_cpu_reference_when_available() {
        let Ok(mut metal) = metal::MetalKernel::new() else {
            return;
        };
        let request = request();
        let cpu = simulate_samples_cpu(&request);
        let (gpu, gpu_time_ns) = metal.run(&request).expect("Metal dispatch");
        eprintln!(
            "metal_device={} samples={} gpu_time_us={:.1}",
            metal.device_name(),
            request.total_samples(),
            gpu_time_ns as f64 / 1_000.0
        );
        assert_eq!(cpu.len(), gpu.len());
        let max_error = cpu
            .iter()
            .zip(&gpu)
            .map(|(cpu, gpu)| (cpu - gpu).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error <= 1e-6, "CPU/Metal divergence {max_error}");
    }
}
