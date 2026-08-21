//! Workload-local, bounded early-warning envelope for stutter precursors.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::adaptive_overhead::OverheadInput;

const SIGNAL_COUNT: usize = 9;
const STATE_VERSION: u32 = 1;
const THRESHOLD_REFRESH_SAMPLES: u64 = 8;
const ADAPTIVE_MARGIN: f32 = 1.35;
const GUARDED_RATIO: f32 = 1.50;
const CONSTRAINED_RATIO: f32 = 3.00;
const MAX_NORMALIZED_SAMPLE: f32 = 4.0;

pub(crate) const BASELINE_WINDOW_CAPACITY: usize = 120;
pub(crate) const BASELINE_MIN_SAMPLES: usize = 60;
pub(crate) const SHADOW_MIN_OBSERVATIONS: usize = 120;
pub(crate) const MAX_LEARNED_THRESHOLD: f32 = 0.90;
pub(crate) const HARD_REFAULT_GUARD_BPS: f64 = 384.0 * 1024.0 * 1024.0;

const NORMALIZERS: [f64; SIGNAL_COUNT] = [
    HARD_REFAULT_GUARD_BPS,
    0.15,
    512.0 * 1024.0,
    1_500.0,
    0.80,
    0.60,
    35.0,
    60.0,
    0.35,
];
const THRESHOLD_FLOORS: [f32; SIGNAL_COUNT] =
    [0.10, 0.10, 0.10, 0.10, 0.50, 0.35, 0.25, 0.55, 0.20];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadRegime {
    Quiet,
    Interactive,
    Media,
    Compute,
}

impl WorkloadRegime {
    fn index(self) -> usize {
        match self {
            Self::Quiet => 0,
            Self::Interactive => 1,
            Self::Media => 2,
            Self::Compute => 3,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quiet => "quiet",
            Self::Interactive => "interactive",
            Self::Media => "media",
            Self::Compute => "compute",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopePhase {
    ColdStart,
    Shadow,
    Active,
}

impl EnvelopePhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ColdStart => "cold",
            Self::Shadow => "shadow",
            Self::Active => "active",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeRisk {
    Nominal,
    Guarded,
    Constrained,
}

impl EnvelopeRisk {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::Guarded => "guarded",
            Self::Constrained => "constrained",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StutterObservation {
    pub p95_cycle_ms: f64,
    pub refault_bytes_per_sec: f64,
    pub stall_fraction: f64,
    pub swap_growth_bytes_per_sec: f64,
    pub thrashing_score: f64,
    pub cpu_max_busy: f64,
    pub gpu_render_load: f64,
    pub compositor_cpu_pct: f64,
    pub predicted_fluidity_3s: f64,
    pub realtime_media_active: bool,
    pub media_output_active: bool,
    pub interaction_q: f64,
    pub fluidity_degraded: bool,
    pub pressure_sample_stale: bool,
}

impl StutterObservation {
    pub fn from_overhead(input: OverheadInput) -> Self {
        let page_size = input.vm_page_size_bytes.clamp(4 * 1024, 64 * 1024) as f64;
        let refault_bytes_per_sec = if input.refault_delta_per_sec.is_finite() {
            input.refault_delta_per_sec.max(0.0) * page_size
        } else {
            0.0
        };
        Self {
            p95_cycle_ms: input.p95_cycle_ms,
            refault_bytes_per_sec,
            stall_fraction: input.stall_fraction,
            swap_growth_bytes_per_sec: input.swap_delta_bps.max(0.0),
            thrashing_score: input.thrashing_score,
            cpu_max_busy: input.cpu_max_busy,
            gpu_render_load: if input.hardware_sample_stale {
                0.0
            } else {
                input.gpu_render_load
            },
            compositor_cpu_pct: input.compositor_cpu_pct,
            predicted_fluidity_3s: input.predicted_fluidity_3s,
            realtime_media_active: input.realtime_media_active,
            media_output_active: input.media_output_active,
            interaction_q: input.interaction_q,
            fluidity_degraded: input.fluidity_degraded,
            pressure_sample_stale: input.pressure_sample_stale,
        }
    }

    fn regime(self) -> WorkloadRegime {
        if self.realtime_media_active || self.media_output_active {
            WorkloadRegime::Media
        } else if self.interaction_q >= 0.05 {
            WorkloadRegime::Interactive
        } else if self.cpu_max_busy >= 0.50 || self.gpu_render_load >= 0.35 {
            WorkloadRegime::Compute
        } else {
            WorkloadRegime::Quiet
        }
    }

    fn normalized(self) -> Option<[f32; SIGNAL_COUNT]> {
        if self.pressure_sample_stale {
            return None;
        }
        let predicted_fluidity_risk = if self.predicted_fluidity_3s > 0.0 {
            (0.65 - self.predicted_fluidity_3s).max(0.0)
        } else {
            0.0
        };
        let raw = [
            self.refault_bytes_per_sec,
            self.stall_fraction,
            self.swap_growth_bytes_per_sec,
            self.thrashing_score,
            self.cpu_max_busy,
            self.gpu_render_load,
            self.compositor_cpu_pct,
            self.p95_cycle_ms,
            predicted_fluidity_risk,
        ];
        if raw.iter().any(|value| !value.is_finite()) {
            return None;
        }
        Some(std::array::from_fn(|index| {
            (raw[index].max(0.0) / NORMALIZERS[index]).clamp(0.0, f64::from(MAX_NORMALIZED_SAMPLE))
                as f32
        }))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EnvelopeDecision {
    pub phase: EnvelopePhase,
    pub risk: EnvelopeRisk,
    pub regime: WorkloadRegime,
    pub would_guard: bool,
    pub score: f64,
    pub samples: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct RegimeEnvelope {
    samples: VecDeque<[f32; SIGNAL_COUNT]>,
    thresholds: [f32; SIGNAL_COUNT],
    previous: Option<[f32; SIGNAL_COUNT]>,
    accepted_total: u64,
    shadow_observations: u64,
}

impl Default for RegimeEnvelope {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(BASELINE_WINDOW_CAPACITY),
            thresholds: THRESHOLD_FLOORS,
            previous: None,
            accepted_total: 0,
            shadow_observations: 0,
        }
    }
}

impl RegimeEnvelope {
    fn phase(&self) -> EnvelopePhase {
        if self.samples.len() < BASELINE_MIN_SAMPLES {
            EnvelopePhase::ColdStart
        } else if self.shadow_observations < SHADOW_MIN_OBSERVATIONS as u64 {
            EnvelopePhase::Shadow
        } else {
            EnvelopePhase::Active
        }
    }

    fn classify(&self, current: [f32; SIGNAL_COUNT]) -> (EnvelopeRisk, f32) {
        let mut breaches = 0usize;
        let mut max_ratio = 0.0_f32;
        let mut rapid_rises = 0usize;
        for index in 0..SIGNAL_COUNT {
            let threshold = self.thresholds[index].max(0.001);
            let ratio = current[index] / threshold;
            max_ratio = max_ratio.max(ratio);
            breaches += usize::from(ratio > 1.0);
            if let Some(previous) = self.previous {
                let rise_floor = (threshold * 0.35).max(0.10);
                rapid_rises += usize::from(
                    current[index] > threshold * 0.80
                        && current[index] - previous[index] > rise_floor,
                );
            }
        }

        let risk = if max_ratio >= CONSTRAINED_RATIO || breaches >= 3 {
            EnvelopeRisk::Constrained
        } else if max_ratio >= GUARDED_RATIO || breaches >= 2 || rapid_rises >= 2 {
            EnvelopeRisk::Guarded
        } else {
            EnvelopeRisk::Nominal
        };
        (risk, max_ratio)
    }

    fn admit_baseline(
        &self,
        observation: StutterObservation,
        current: [f32; SIGNAL_COUNT],
        candidate: EnvelopeRisk,
    ) -> bool {
        let cold_start_safe = [0, 1, 2, 3, 8]
            .into_iter()
            .all(|index| current[index] <= THRESHOLD_FLOORS[index] * CONSTRAINED_RATIO);
        !observation.fluidity_degraded
            && observation.p95_cycle_ms > 0.0
            && observation.p95_cycle_ms < 60.0
            && current.iter().all(|value| *value <= 1.0)
            && match self.phase() {
                EnvelopePhase::ColdStart => cold_start_safe,
                EnvelopePhase::Shadow | EnvelopePhase::Active => candidate == EnvelopeRisk::Nominal,
            }
    }

    fn push(&mut self, sample: [f32; SIGNAL_COUNT]) {
        if self.samples.len() == BASELINE_WINDOW_CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        self.accepted_total = self.accepted_total.saturating_add(1);
        if self.samples.len() == BASELINE_MIN_SAMPLES
            || self
                .accepted_total
                .is_multiple_of(THRESHOLD_REFRESH_SAMPLES)
        {
            self.recompute_thresholds();
        }
    }

    fn recompute_thresholds(&mut self) {
        if self.samples.is_empty() {
            return;
        }
        let percentile_index = (self.samples.len() - 1) * 95 / 100;
        for signal in 0..SIGNAL_COUNT {
            let mut values: Vec<f32> = self.samples.iter().map(|sample| sample[signal]).collect();
            let (_, percentile, _) = values
                .select_nth_unstable_by(percentile_index, |left, right| {
                    left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
                });
            self.thresholds[signal] = (*percentile * ADAPTIVE_MARGIN)
                .clamp(THRESHOLD_FLOORS[signal], MAX_LEARNED_THRESHOLD);
        }
    }

    fn sanitize(&mut self) {
        while self.samples.len() > BASELINE_WINDOW_CAPACITY {
            self.samples.pop_front();
        }
        self.samples
            .retain(|sample| sample.iter().all(|value| value.is_finite()));
        for (index, threshold) in self.thresholds.iter_mut().enumerate() {
            if !threshold.is_finite() {
                *threshold = THRESHOLD_FLOORS[index];
            }
            *threshold = threshold.clamp(THRESHOLD_FLOORS[index], MAX_LEARNED_THRESHOLD);
        }
        if self
            .previous
            .is_some_and(|sample| sample.iter().any(|value| !value.is_finite()))
        {
            self.previous = None;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveStutterEnvelope {
    version: u32,
    regimes: [RegimeEnvelope; 4],
}

impl Default for AdaptiveStutterEnvelope {
    fn default() -> Self {
        Self {
            version: STATE_VERSION,
            regimes: std::array::from_fn(|_| RegimeEnvelope::default()),
        }
    }
}

impl AdaptiveStutterEnvelope {
    pub fn load(path: &Path) -> Self {
        let Some(mut restored) =
            apollo_engine::engine::policy_store::read_json::<AdaptiveStutterEnvelope>(path)
        else {
            return Self::default();
        };
        if restored.version != STATE_VERSION {
            return Self::default();
        }
        for regime in &mut restored.regimes {
            regime.sanitize();
        }
        restored
    }

    pub fn observe(&mut self, observation: StutterObservation) -> EnvelopeDecision {
        let regime = observation.regime();
        let state = &mut self.regimes[regime.index()];
        let phase = state.phase();
        let Some(current) = observation.normalized() else {
            return EnvelopeDecision {
                phase,
                risk: EnvelopeRisk::Nominal,
                regime,
                would_guard: false,
                score: 0.0,
                samples: state.samples.len() as u64,
            };
        };

        let (candidate, score) = if phase == EnvelopePhase::ColdStart {
            (EnvelopeRisk::Nominal, 0.0)
        } else {
            state.classify(current)
        };
        if phase == EnvelopePhase::Shadow {
            state.shadow_observations = state.shadow_observations.saturating_add(1);
        }
        if state.admit_baseline(observation, current, candidate) {
            state.push(current);
        }
        state.previous = Some(current);

        EnvelopeDecision {
            phase,
            risk: if phase == EnvelopePhase::Active {
                candidate
            } else {
                EnvelopeRisk::Nominal
            },
            regime,
            would_guard: candidate != EnvelopeRisk::Nominal,
            score: f64::from(score),
            samples: state.samples.len() as u64,
        }
    }

    pub fn sample_count(&self, regime: WorkloadRegime) -> usize {
        self.regimes[regime.index()].samples.len()
    }

    pub fn thresholds(&self, regime: WorkloadRegime) -> [f32; SIGNAL_COUNT] {
        self.regimes[regime.index()].thresholds
    }
}

enum PersistCommand {
    Snapshot(PathBuf, AdaptiveStutterEnvelope),
    Flush(PathBuf, AdaptiveStutterEnvelope, mpsc::Sender<bool>),
}

fn persist(path: &Path, state: &AdaptiveStutterEnvelope) -> bool {
    apollo_engine::engine::policy_store::write_json_transactional(path, state, Some(0o600)).is_ok()
}

fn try_send_until<T>(tx: &SyncSender<T>, mut value: T, deadline: Instant) -> Result<(), T> {
    loop {
        match tx.try_send(value) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Disconnected(value)) => return Err(value),
            Err(TrySendError::Full(returned)) => {
                value = returned;
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(value);
                }
                std::thread::sleep(remaining.min(Duration::from_millis(1)));
            }
        }
    }
}

pub struct AdaptiveStutterWriter {
    tx: SyncSender<PersistCommand>,
    failures: Arc<AtomicU64>,
}

impl AdaptiveStutterWriter {
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::sync_channel(1);
        let failures = Arc::new(AtomicU64::new(0));
        let worker_failures = failures.clone();
        std::thread::Builder::new()
            .name("apollo-stutter-writer".to_string())
            .spawn(move || {
                while let Ok(command) = rx.recv() {
                    match command {
                        PersistCommand::Snapshot(path, state) => {
                            if !persist(&path, &state) {
                                worker_failures.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        PersistCommand::Flush(path, state, ack) => {
                            let persisted = persist(&path, &state);
                            if !persisted {
                                worker_failures.fetch_add(1, Ordering::Relaxed);
                            }
                            let _ = ack.send(persisted);
                        }
                    }
                }
            })
            .expect("failed to spawn adaptive-stutter writer");
        Self { tx, failures }
    }

    pub fn submit(&self, path: PathBuf, state: AdaptiveStutterEnvelope) -> bool {
        match self.tx.try_send(PersistCommand::Snapshot(path, state)) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        }
    }

    pub fn flush(&self, path: PathBuf, state: AdaptiveStutterEnvelope, timeout: Duration) -> bool {
        let (ack_tx, ack_rx) = mpsc::channel();
        let deadline = Instant::now() + timeout;
        if try_send_until(
            &self.tx,
            PersistCommand::Flush(path, state, ack_tx),
            deadline,
        )
        .is_err()
        {
            return false;
        }
        ack_rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .is_ok_and(|persisted| persisted)
    }

    pub fn failures_total(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stable(regime: WorkloadRegime) -> StutterObservation {
        let mut sample = StutterObservation {
            p95_cycle_ms: 38.0,
            refault_bytes_per_sec: 24.0 * 1024.0 * 1024.0,
            stall_fraction: 0.02,
            swap_growth_bytes_per_sec: 0.0,
            thrashing_score: 120.0,
            cpu_max_busy: 0.25,
            gpu_render_load: 0.10,
            compositor_cpu_pct: 8.0,
            predicted_fluidity_3s: 0.95,
            realtime_media_active: false,
            media_output_active: false,
            interaction_q: 0.0,
            fluidity_degraded: false,
            pressure_sample_stale: false,
        };
        match regime {
            WorkloadRegime::Quiet => {}
            WorkloadRegime::Interactive => sample.interaction_q = 0.40,
            WorkloadRegime::Media => sample.media_output_active = true,
            WorkloadRegime::Compute => sample.cpu_max_busy = 0.65,
        }
        sample
    }

    fn warm_to_active(envelope: &mut AdaptiveStutterEnvelope, sample: StutterObservation) {
        for _ in 0..(BASELINE_MIN_SAMPLES + SHADOW_MIN_OBSERVATIONS + 1) {
            envelope.observe(sample);
        }
        assert_eq!(envelope.observe(sample).phase, EnvelopePhase::Active);
    }

    #[test]
    fn cold_start_is_observational_only() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let mut spike = stable(WorkloadRegime::Quiet);
        spike.refault_bytes_per_sec = 300.0 * 1024.0 * 1024.0;

        let decision = envelope.observe(spike);

        assert_eq!(decision.phase, EnvelopePhase::ColdStart);
        assert_eq!(decision.risk, EnvelopeRisk::Nominal);
    }

    #[test]
    fn cold_start_refault_burst_never_enters_the_baseline() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let mut spike = stable(WorkloadRegime::Quiet);
        spike.refault_bytes_per_sec = HARD_REFAULT_GUARD_BPS * 0.60;

        for _ in 0..4 {
            envelope.observe(spike);
        }

        assert_eq!(envelope.sample_count(WorkloadRegime::Quiet), 0);
        envelope.observe(stable(WorkloadRegime::Quiet));
        assert_eq!(envelope.sample_count(WorkloadRegime::Quiet), 1);
    }

    #[test]
    fn stale_hardware_sample_neutralizes_cached_gpu_power() {
        let observation = StutterObservation::from_overhead(OverheadInput {
            hardware_sample_stale: true,
            gpu_render_load: 1.0,
            ..OverheadInput::default()
        });

        assert_eq!(observation.gpu_render_load, 0.0);
    }

    #[test]
    fn shadow_reports_anomaly_without_enforcing_it() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let sample = stable(WorkloadRegime::Quiet);
        for _ in 0..BASELINE_MIN_SAMPLES {
            envelope.observe(sample);
        }
        let mut spike = sample;
        spike.refault_bytes_per_sec = 300.0 * 1024.0 * 1024.0;

        let decision = envelope.observe(spike);

        assert_eq!(decision.phase, EnvelopePhase::Shadow);
        assert!(decision.would_guard);
        assert_eq!(decision.risk, EnvelopeRisk::Nominal);
    }

    #[test]
    fn active_envelope_detects_a_local_refault_storm_before_hard_limit() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let sample = stable(WorkloadRegime::Quiet);
        warm_to_active(&mut envelope, sample);
        let mut spike = sample;
        spike.refault_bytes_per_sec = 300.0 * 1024.0 * 1024.0;

        let decision = envelope.observe(spike);

        assert_eq!(decision.phase, EnvelopePhase::Active);
        assert!(decision.would_guard);
        assert_ne!(decision.risk, EnvelopeRisk::Nominal);
        assert!(spike.refault_bytes_per_sec < HARD_REFAULT_GUARD_BPS);
    }

    #[test]
    fn workload_regimes_learn_independently() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        warm_to_active(&mut envelope, stable(WorkloadRegime::Quiet));

        let compute = envelope.observe(stable(WorkloadRegime::Compute));

        assert_eq!(compute.regime, WorkloadRegime::Compute);
        assert_eq!(compute.phase, EnvelopePhase::ColdStart);
        assert_eq!(compute.risk, EnvelopeRisk::Nominal);
    }

    #[test]
    fn saturated_gpu_signal_can_still_build_a_local_compute_baseline() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let mut sample = stable(WorkloadRegime::Compute);
        sample.gpu_render_load = 0.60;

        for _ in 0..BASELINE_MIN_SAMPLES {
            envelope.observe(sample);
        }

        assert_eq!(
            envelope.sample_count(WorkloadRegime::Compute),
            BASELINE_MIN_SAMPLES
        );
    }

    #[test]
    fn fluidity_forecast_can_guard_before_measured_latency_regresses() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let sample = stable(WorkloadRegime::Quiet);
        warm_to_active(&mut envelope, sample);
        let mut predicted_drop = sample;
        predicted_drop.predicted_fluidity_3s = 0.45;

        let decision = envelope.observe(predicted_drop);

        assert_ne!(decision.risk, EnvelopeRisk::Nominal);
        assert_eq!(predicted_drop.p95_cycle_ms, sample.p95_cycle_ms);
    }

    #[test]
    fn first_healthy_fluidity_forecast_after_warmup_is_neutral() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let mut unavailable = stable(WorkloadRegime::Quiet);
        unavailable.predicted_fluidity_3s = 0.0;
        warm_to_active(&mut envelope, unavailable);
        let mut healthy_prediction = unavailable;
        healthy_prediction.predicted_fluidity_3s = 0.80;

        let decision = envelope.observe(healthy_prediction);

        assert_eq!(decision.risk, EnvelopeRisk::Nominal);
    }

    #[test]
    fn incidents_never_train_the_baseline_upward() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let sample = stable(WorkloadRegime::Quiet);
        warm_to_active(&mut envelope, sample);
        let before = envelope.thresholds(WorkloadRegime::Quiet);
        let mut spike = sample;
        spike.refault_bytes_per_sec = 300.0 * 1024.0 * 1024.0;

        for _ in 0..20 {
            assert_ne!(envelope.observe(spike).risk, EnvelopeRisk::Nominal);
        }

        assert_eq!(envelope.thresholds(WorkloadRegime::Quiet), before);
    }

    #[test]
    fn learned_thresholds_never_exceed_hard_safety_ceiling() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let mut elevated = stable(WorkloadRegime::Quiet);
        elevated.refault_bytes_per_sec = HARD_REFAULT_GUARD_BPS * 0.85;
        for _ in 0..BASELINE_MIN_SAMPLES {
            envelope.observe(elevated);
        }

        let thresholds = envelope.thresholds(WorkloadRegime::Quiet);

        assert!(thresholds[0] <= MAX_LEARNED_THRESHOLD);
    }

    #[test]
    fn rolling_state_is_bounded_and_serializable() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let sample = stable(WorkloadRegime::Quiet);
        for _ in 0..(BASELINE_WINDOW_CAPACITY + 100) {
            envelope.observe(sample);
        }
        assert_eq!(
            envelope.sample_count(WorkloadRegime::Quiet),
            BASELINE_WINDOW_CAPACITY
        );

        let json = serde_json::to_string(&envelope).unwrap();
        let restored: AdaptiveStutterEnvelope = serde_json::from_str(&json).unwrap();

        assert_eq!(
            restored.sample_count(WorkloadRegime::Quiet),
            BASELINE_WINDOW_CAPACITY
        );
        assert_eq!(
            restored.thresholds(WorkloadRegime::Quiet),
            envelope.thresholds(WorkloadRegime::Quiet)
        );
    }

    #[test]
    fn invalid_samples_do_not_poison_learning() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let mut invalid = stable(WorkloadRegime::Quiet);
        invalid.stall_fraction = f64::NAN;

        let decision = envelope.observe(invalid);

        assert_eq!(decision.risk, EnvelopeRisk::Nominal);
        assert_eq!(envelope.sample_count(WorkloadRegime::Quiet), 0);
    }

    #[test]
    fn stale_pressure_sample_does_not_train_the_envelope() {
        let mut envelope = AdaptiveStutterEnvelope::default();
        let observation = StutterObservation::from_overhead(OverheadInput {
            pressure_sample_stale: true,
            p95_cycle_ms: 38.0,
            ..OverheadInput::default()
        });

        let decision = envelope.observe(observation);

        assert_eq!(decision.risk, EnvelopeRisk::Nominal);
        assert_eq!(envelope.sample_count(WorkloadRegime::Quiet), 0);
    }

    #[test]
    fn background_persistence_never_drops_shutdown_flush() {
        let path = std::env::temp_dir().join(format!(
            "apollo-stutter-writer-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let writer = AdaptiveStutterWriter::spawn();
        let mut envelope = AdaptiveStutterEnvelope::default();
        envelope.observe(stable(WorkloadRegime::Quiet));
        assert!(writer.submit(path.clone(), envelope.clone()));

        assert!(writer.flush(path.clone(), envelope, Duration::from_millis(500)));
        assert!(AdaptiveStutterEnvelope::load(&path).sample_count(WorkloadRegime::Quiet) > 0);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bounded_send_honors_deadline_when_queue_is_full() {
        let (tx, _rx) = mpsc::sync_channel(1);
        tx.send(1_u8).unwrap();
        let started = std::time::Instant::now();

        assert!(try_send_until(&tx, 2_u8, started + Duration::from_millis(20)).is_err());
        assert!(started.elapsed() < Duration::from_millis(500));
    }

    #[test]
    fn shutdown_flush_reports_persistence_failure() {
        let writer = AdaptiveStutterWriter::spawn();

        assert!(!writer.flush(
            std::env::temp_dir(),
            AdaptiveStutterEnvelope::default(),
            Duration::from_millis(500),
        ));
        assert_eq!(writer.failures_total(), 1);
    }
}
