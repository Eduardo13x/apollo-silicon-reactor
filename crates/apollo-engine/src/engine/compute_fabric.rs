//! Bounded optional compute routing with identity-bound completions.
//!
//! This module owns scheduling and evidence only. Its payload is deliberately
//! limited to numeric vectors and candidate identifiers, so it cannot carry
//! an action or acquire action authority by construction.

pub use super::world_state::WorldIdentity;

pub const MAX_JOB_CLASSES: usize = 32;
pub const MAX_RUNNING_JOBS: usize = 4;
pub const BACKEND_COUNT: usize = 4;
pub const MAX_PENDING_JOBS: usize = MAX_JOB_CLASSES * BACKEND_COUNT;
pub const MAX_RESULT_BYTES: usize = 64 * 1024;
pub const MAX_VECTOR_VALUES: usize = MAX_RESULT_BYTES / std::mem::size_of::<f32>();
pub const MAX_CANDIDATE_IDS: usize = MAX_RESULT_BYTES / std::mem::size_of::<CandidateId>();
pub const SHADOW_MIN_JOBS: u64 = 500;
pub const SHADOW_MIN_WINDOW_US: u64 = 15 * 60 * 1_000_000;
pub const CANARY_PERCENT: u8 = 10;
pub const CANARY_MIN_JOBS: u64 = 500;
pub const CIRCUIT_FAILURE_THRESHOLD: u8 = 3;
pub const CIRCUIT_COOLDOWN_US: u64 = 60 * 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct JobClassId(pub u16);

impl JobClassId {
    pub const fn new(value: u16) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ComputeJobId(pub u64);

impl From<u64> for ComputeJobId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct CandidateId(pub u64);

impl From<u64> for CandidateId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum ComputeBackendId {
    CpuLatency,
    CpuUtility,
    Metal,
    CoreMl,
}

impl ComputeBackendId {
    pub const ALL: [Self; BACKEND_COUNT] = [
        Self::CpuLatency,
        Self::CpuUtility,
        Self::Metal,
        Self::CoreMl,
    ];

    const fn index(self) -> usize {
        match self {
            Self::CpuLatency => 0,
            Self::CpuUtility => 1,
            Self::Metal => 2,
            Self::CoreMl => 3,
        }
    }

    const fn is_deterministic_cpu(self) -> bool {
        matches!(self, Self::CpuLatency | Self::CpuUtility)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComputePayload {
    Vector(Vec<f32>),
    CandidateIds(Vec<CandidateId>),
}

impl ComputePayload {
    pub fn try_vector(values: Vec<f32>) -> Result<Self, PayloadError> {
        let payload = Self::Vector(values);
        payload.validate()?;
        Ok(payload)
    }

    pub fn try_candidate_ids(values: Vec<CandidateId>) -> Result<Self, PayloadError> {
        let payload = Self::CandidateIds(values);
        payload.validate()?;
        Ok(payload)
    }

    pub fn byte_len(&self) -> usize {
        match self {
            Self::Vector(values) => values.len() * std::mem::size_of::<f32>(),
            Self::CandidateIds(values) => values.len() * std::mem::size_of::<CandidateId>(),
        }
    }

    fn validate(&self) -> Result<(), PayloadError> {
        if self.byte_len() > MAX_RESULT_BYTES {
            return Err(PayloadError::Oversized);
        }
        match self {
            Self::Vector(values) if values.iter().any(|value| !value.is_finite()) => {
                Err(PayloadError::NonFinite)
            }
            Self::Vector(values) if values.len() > MAX_VECTOR_VALUES => {
                Err(PayloadError::Oversized)
            }
            Self::CandidateIds(values) if values.len() > MAX_CANDIDATE_IDS => {
                Err(PayloadError::Oversized)
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadError {
    NonFinite,
    Oversized,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComputeJob {
    pub id: ComputeJobId,
    pub job_class: JobClassId,
    pub backend: ComputeBackendId,
    pub world_identity: WorldIdentity,
    pub payload: ComputePayload,
    pub submitted_at_us: u64,
    pub queue_deadline_us: u64,
    pub runtime_deadline_us: u64,
}

impl ComputeJob {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ComputeJobId,
        job_class: JobClassId,
        backend: ComputeBackendId,
        world_identity: WorldIdentity,
        payload: ComputePayload,
        submitted_at_us: u64,
        queue_deadline_us: u64,
        runtime_deadline_us: u64,
    ) -> Self {
        Self {
            id,
            job_class,
            backend,
            world_identity,
            payload,
            submitted_at_us,
            queue_deadline_us,
            runtime_deadline_us,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendOutcome {
    pub job_id: ComputeJobId,
    pub world_identity: WorldIdentity,
    pub payload: ComputePayload,
}

impl BackendOutcome {
    pub fn new(
        job_id: ComputeJobId,
        world_identity: WorldIdentity,
        payload: ComputePayload,
    ) -> Self {
        Self {
            job_id,
            world_identity,
            payload,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionStatus {
    Valid,
    QueueExpired,
    RuntimeExpired,
    Late,
    WrongIdentity,
    OutOfOrder,
    NonFinite,
    Oversized,
    CircuitOpen,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComputeCompletion {
    pub job_id: ComputeJobId,
    pub backend: Option<ComputeBackendId>,
    pub status: CompletionStatus,
    pub payload: Option<ComputePayload>,
}

impl ComputeCompletion {
    fn new(
        job_id: ComputeJobId,
        backend: Option<ComputeBackendId>,
        status: CompletionStatus,
        payload: Option<ComputePayload>,
    ) -> Self {
        Self {
            job_id,
            backend,
            status,
            payload,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FabricError {
    TooManyJobClasses,
    DuplicateJobClass,
    DuplicateJobId,
    PendingQueueFull,
    InvalidDeadline,
    InvalidPayload(PayloadError),
    WrongWorldIdentity,
    BackendRolledBack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitOutcome {
    Queued,
    ReplacedPending { replaced_job_id: ComputeJobId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RolloutPhase {
    Shadow,
    Canary,
    Active,
    RolledBack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeFabricConfig {
    pub circuit_failure_threshold: u8,
    pub circuit_cooldown_us: u64,
    pub shadow_min_jobs: u64,
    pub shadow_window_us: u64,
    pub canary_percent: u8,
    pub canary_min_jobs: u64,
}

impl Default for ComputeFabricConfig {
    fn default() -> Self {
        Self {
            circuit_failure_threshold: CIRCUIT_FAILURE_THRESHOLD,
            circuit_cooldown_us: CIRCUIT_COOLDOWN_US,
            shadow_min_jobs: SHADOW_MIN_JOBS,
            shadow_window_us: SHADOW_MIN_WINDOW_US,
            canary_percent: CANARY_PERCENT,
            canary_min_jobs: CANARY_MIN_JOBS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationSample {
    pub at_us: u64,
    pub deadline_met: bool,
    pub oracle_error: bool,
    pub baseline_latency_us: u64,
    pub candidate_latency_us: u64,
    pub baseline_energy_uj: u64,
    pub candidate_energy_uj: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct RolloutStats {
    jobs: u64,
    deadlines_met: u64,
    oracle_errors: u64,
    baseline_latency_us: u128,
    candidate_latency_us: u128,
    baseline_energy_uj: u128,
    candidate_energy_uj: u128,
}

impl RolloutStats {
    fn record(&mut self, sample: EvaluationSample) {
        self.jobs = self.jobs.saturating_add(1);
        if sample.deadline_met {
            self.deadlines_met = self.deadlines_met.saturating_add(1);
        }
        if sample.oracle_error {
            self.oracle_errors = self.oracle_errors.saturating_add(1);
        }
        self.baseline_latency_us = self
            .baseline_latency_us
            .saturating_add(u128::from(sample.baseline_latency_us));
        self.candidate_latency_us = self
            .candidate_latency_us
            .saturating_add(u128::from(sample.candidate_latency_us));
        self.baseline_energy_uj = self
            .baseline_energy_uj
            .saturating_add(u128::from(sample.baseline_energy_uj));
        self.candidate_energy_uj = self
            .candidate_energy_uj
            .saturating_add(u128::from(sample.candidate_energy_uj));
    }

    fn promotable(self) -> bool {
        self.jobs > 0
            && self.deadlines_met.saturating_mul(100) >= self.jobs.saturating_mul(99)
            && self.oracle_errors.saturating_mul(100) <= self.jobs
            && (self.candidate_latency_us.saturating_mul(110)
                <= self.baseline_latency_us.saturating_mul(100)
                || self.candidate_energy_uj.saturating_mul(115)
                    <= self.baseline_energy_uj.saturating_mul(100))
    }
}

#[derive(Debug, Clone, Copy)]
struct Circuit {
    state: CircuitState,
    failures: u8,
    opened_at_us: u64,
    probe_in_flight: bool,
}

impl Default for Circuit {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            failures: 0,
            opened_at_us: 0,
            probe_in_flight: false,
        }
    }
}

impl Circuit {
    fn refresh(&mut self, now_us: u64, cooldown_us: u64) {
        if self.state == CircuitState::Open
            && now_us.saturating_sub(self.opened_at_us) >= cooldown_us
        {
            self.state = CircuitState::HalfOpen;
            self.probe_in_flight = false;
        }
    }

    fn try_start(&mut self, now_us: u64, config: ComputeFabricConfig) -> bool {
        self.refresh(now_us, config.circuit_cooldown_us);
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen if self.probe_in_flight => false,
            CircuitState::HalfOpen => {
                self.probe_in_flight = true;
                true
            }
        }
    }

    fn success(&mut self) {
        self.state = CircuitState::Closed;
        self.failures = 0;
        self.probe_in_flight = false;
    }

    fn failure(&mut self, now_us: u64, config: ComputeFabricConfig) {
        self.probe_in_flight = false;
        if self.state == CircuitState::HalfOpen {
            self.state = CircuitState::Open;
            self.opened_at_us = now_us;
            return;
        }
        self.failures = self.failures.saturating_add(1);
        if self.failures >= config.circuit_failure_threshold.max(1) {
            self.state = CircuitState::Open;
            self.opened_at_us = now_us;
        }
    }

    fn force_open(&mut self, now_us: u64) {
        self.state = CircuitState::Open;
        self.opened_at_us = now_us;
        self.probe_in_flight = false;
    }
}

#[derive(Debug, Clone, Copy)]
struct BackendState {
    circuit: Circuit,
    rollout: RolloutPhase,
    shadow: RolloutStats,
    canary: RolloutStats,
}

impl Default for BackendState {
    fn default() -> Self {
        Self {
            circuit: Circuit::default(),
            rollout: RolloutPhase::Shadow,
            shadow: RolloutStats::default(),
            canary: RolloutStats::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct RunningJob {
    job: ComputeJob,
    started_at_us: u64,
}

pub struct ComputeFabric {
    identity: WorldIdentity,
    rollout_started_at_us: u64,
    config: ComputeFabricConfig,
    registered_job_classes: Vec<JobClassId>,
    pending: Vec<ComputeJob>,
    running: Vec<RunningJob>,
    backends: [BackendState; BACKEND_COUNT],
}

impl Default for ComputeFabric {
    fn default() -> Self {
        Self::new(WorldIdentity::default())
    }
}

impl ComputeFabric {
    pub fn new(identity: WorldIdentity) -> Self {
        Self::with_config(identity, 0, ComputeFabricConfig::default())
    }

    pub fn with_config(
        identity: WorldIdentity,
        rollout_started_at_us: u64,
        mut config: ComputeFabricConfig,
    ) -> Self {
        config.circuit_failure_threshold = config.circuit_failure_threshold.max(1);
        config.canary_percent = config.canary_percent.min(100);
        Self {
            identity,
            rollout_started_at_us,
            config,
            registered_job_classes: Vec::with_capacity(MAX_JOB_CLASSES),
            pending: Vec::with_capacity(MAX_PENDING_JOBS),
            running: Vec::with_capacity(MAX_RUNNING_JOBS),
            backends: std::array::from_fn(|_| BackendState::default()),
        }
    }

    pub fn world_identity(&self) -> WorldIdentity {
        self.identity
    }

    pub fn register_job_class(&mut self, class: JobClassId) -> Result<(), FabricError> {
        if self.registered_job_classes.contains(&class) {
            return Ok(());
        }
        if self.registered_job_classes.len() >= MAX_JOB_CLASSES {
            return Err(FabricError::TooManyJobClasses);
        }
        self.registered_job_classes.push(class);
        Ok(())
    }

    pub fn registered_job_classes(&self) -> usize {
        self.registered_job_classes.len()
    }

    pub fn submit(&mut self, job: ComputeJob) -> Result<SubmitOutcome, FabricError> {
        if job.world_identity != self.identity {
            return Err(FabricError::WrongWorldIdentity);
        }
        if job.queue_deadline_us == 0 || job.runtime_deadline_us == 0 {
            return Err(FabricError::InvalidDeadline);
        }
        job.payload
            .validate()
            .map_err(FabricError::InvalidPayload)?;
        self.register_job_class(job.job_class)?;
        if self.job_is_known(job.id) {
            return Err(FabricError::DuplicateJobId);
        }

        if let Some(index) = self
            .pending
            .iter()
            .position(|pending| pending.job_class == job.job_class && pending.backend == job.backend)
        {
            let replaced_job_id = self.pending[index].id;
            self.pending[index] = job;
            return Ok(SubmitOutcome::ReplacedPending { replaced_job_id });
        }
        if self.pending.len() >= MAX_PENDING_JOBS {
            return Err(FabricError::PendingQueueFull);
        }
        self.pending.push(job);
        Ok(SubmitOutcome::Queued)
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn pending_job_ids(&self) -> Vec<ComputeJobId> {
        self.pending.iter().map(|job| job.id).collect()
    }

    pub fn poll(&mut self, now_us: u64) -> Vec<ComputeCompletion> {
        self.advance_rollouts(now_us);
        let mut completions = Vec::with_capacity(MAX_PENDING_JOBS + MAX_RUNNING_JOBS);

        let mut index = 0;
        while index < self.running.len() {
            let running = &self.running[index];
            if now_us.saturating_sub(running.started_at_us) > running.job.runtime_deadline_us {
                let expired = self.running.remove(index);
                self.record_backend_failure(expired.job.backend, now_us);
                completions.push(ComputeCompletion::new(
                    expired.job.id,
                    Some(expired.job.backend),
                    CompletionStatus::RuntimeExpired,
                    None,
                ));
            } else {
                index += 1;
            }
        }

        index = 0;
        while index < self.pending.len() {
            let pending = &self.pending[index];
            if now_us.saturating_sub(pending.submitted_at_us) > pending.queue_deadline_us {
                let expired = self.pending.remove(index);
                completions.push(ComputeCompletion::new(
                    expired.id,
                    Some(expired.backend),
                    CompletionStatus::QueueExpired,
                    None,
                ));
            } else {
                index += 1;
            }
        }

        while self.running.len() < MAX_RUNNING_JOBS && !self.pending.is_empty() {
            let job = self.pending.remove(0);
            let backend = job.backend;
            if self.rollout_phase(backend) == RolloutPhase::RolledBack {
                completions.push(ComputeCompletion::new(
                    job.id,
                    Some(backend),
                    CompletionStatus::CircuitOpen,
                    None,
                ));
                continue;
            }
            let circuit_ready = self.backends[backend.index()]
                .circuit
                .try_start(now_us, self.config);
            if !circuit_ready {
                completions.push(ComputeCompletion::new(
                    job.id,
                    Some(backend),
                    CompletionStatus::CircuitOpen,
                    None,
                ));
                continue;
            }

            let synchronous = backend.is_deterministic_cpu();
            let outcome = synchronous.then(|| {
                BackendOutcome::new(job.id, job.world_identity, job.payload.clone())
            });
            self.running.push(RunningJob {
                job,
                started_at_us: now_us,
            });
            if let Some(outcome) = outcome {
                completions.push(self.complete(now_us, outcome));
            }
        }

        completions
    }

    pub fn complete(&mut self, now_us: u64, outcome: BackendOutcome) -> ComputeCompletion {
        let Some(index) = self
            .running
            .iter()
            .position(|running| running.job.id == outcome.job_id)
        else {
            return ComputeCompletion::new(
                outcome.job_id,
                None,
                CompletionStatus::OutOfOrder,
                None,
            );
        };

        let running = self.running.remove(index);
        let backend = running.job.backend;
        if now_us < running.started_at_us {
            return self.failed_completion(
                running.job.id,
                backend,
                CompletionStatus::OutOfOrder,
                now_us,
            );
        }
        if now_us.saturating_sub(running.started_at_us) > running.job.runtime_deadline_us {
            return self.failed_completion(running.job.id, backend, CompletionStatus::Late, now_us);
        }
        if outcome.world_identity != self.identity
            || outcome.world_identity != running.job.world_identity
        {
            return self.failed_completion(
                running.job.id,
                backend,
                CompletionStatus::WrongIdentity,
                now_us,
            );
        }
        if self.rollout_phase(backend) == RolloutPhase::RolledBack {
            return self.failed_completion(
                running.job.id,
                backend,
                CompletionStatus::CircuitOpen,
                now_us,
            );
        }
        match outcome.payload.validate() {
            Ok(()) => {
                self.record_backend_success(backend, now_us);
                ComputeCompletion::new(
                    running.job.id,
                    Some(backend),
                    CompletionStatus::Valid,
                    Some(outcome.payload),
                )
            }
            Err(PayloadError::NonFinite) => {
                self.failed_completion(running.job.id, backend, CompletionStatus::NonFinite, now_us)
            }
            Err(PayloadError::Oversized) => {
                self.failed_completion(running.job.id, backend, CompletionStatus::Oversized, now_us)
            }
        }
    }

    pub fn record_backend_failure(&mut self, backend: ComputeBackendId, now_us: u64) {
        self.backends[backend.index()]
            .circuit
            .refresh(now_us, self.config.circuit_cooldown_us);
        self.backends[backend.index()]
            .circuit
            .failure(now_us, self.config);
    }

    pub fn record_backend_success(&mut self, backend: ComputeBackendId, now_us: u64) {
        if self.rollout_phase(backend) == RolloutPhase::RolledBack {
            return;
        }
        self.backends[backend.index()]
            .circuit
            .refresh(now_us, self.config.circuit_cooldown_us);
        self.backends[backend.index()].circuit.success();
    }

    pub fn circuit_state(&mut self, backend: ComputeBackendId, now_us: u64) -> CircuitState {
        if self.rollout_phase(backend) == RolloutPhase::RolledBack {
            return CircuitState::Open;
        }
        let circuit = &mut self.backends[backend.index()].circuit;
        circuit.refresh(now_us, self.config.circuit_cooldown_us);
        circuit.state
    }

    pub fn rollout_phase(&self, backend: ComputeBackendId) -> RolloutPhase {
        self.backends[backend.index()].rollout
    }

    pub fn canary_admitted(&self, backend: ComputeBackendId, job_id: ComputeJobId) -> bool {
        if self.rollout_phase(backend) != RolloutPhase::Canary {
            return false;
        }
        (job_id.0 % 100) < u64::from(self.config.canary_percent)
    }

    pub fn record_evaluation(
        &mut self,
        backend: ComputeBackendId,
        sample: EvaluationSample,
    ) -> Result<RolloutPhase, FabricError> {
        let state = &mut self.backends[backend.index()];
        match state.rollout {
            RolloutPhase::Shadow => state.shadow.record(sample),
            RolloutPhase::Canary => state.canary.record(sample),
            RolloutPhase::Active => return Ok(RolloutPhase::Active),
            RolloutPhase::RolledBack => return Err(FabricError::BackendRolledBack),
        }
        self.advance_rollouts(sample.at_us);
        Ok(self.rollout_phase(backend))
    }

    pub fn advance_rollouts(&mut self, now_us: u64) {
        for backend in ComputeBackendId::ALL {
            let state = &mut self.backends[backend.index()];
            match state.rollout {
                RolloutPhase::Shadow
                    if state.shadow.jobs >= self.config.shadow_min_jobs
                        && now_us.saturating_sub(self.rollout_started_at_us)
                            >= self.config.shadow_window_us =>
                {
                    state.rollout = RolloutPhase::Canary;
                    state.canary = RolloutStats::default();
                }
                RolloutPhase::Canary
                    if state.canary.jobs >= self.config.canary_min_jobs
                        && state.canary.promotable() =>
                {
                    state.rollout = RolloutPhase::Active;
                }
                _ => {}
            }
        }
    }

    pub fn rollback_backend(
        &mut self,
        backend: ComputeBackendId,
        now_us: u64,
    ) -> Result<(), FabricError> {
        let state = &mut self.backends[backend.index()];
        state.rollout = RolloutPhase::RolledBack;
        state.circuit.force_open(now_us);
        Ok(())
    }

    fn failed_completion(
        &mut self,
        job_id: ComputeJobId,
        backend: ComputeBackendId,
        status: CompletionStatus,
        now_us: u64,
    ) -> ComputeCompletion {
        self.record_backend_failure(backend, now_us);
        ComputeCompletion::new(job_id, Some(backend), status, None)
    }

    fn job_is_known(&self, job_id: ComputeJobId) -> bool {
        self.pending.iter().any(|job| job.id == job_id)
            || self.running.iter().any(|running| running.job.id == job_id)
    }
}
