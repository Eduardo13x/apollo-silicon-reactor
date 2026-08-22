//! Capacity-normalized Apple Silicon memory evidence.
//!
//! This module is pure: it performs no I/O and emits no actions. Callers feed
//! one observation per background-collector generation and reuse the result on
//! faster daemon cycles.

use std::time::{Duration, Instant};

const GIB: u64 = 1024 * 1024 * 1024;
const MIN_PHYSICAL_MEMORY_BYTES: u64 = GIB;
const MAX_PHYSICAL_MEMORY_BYTES: u64 = 1024 * GIB;
const MIN_PAGE_SIZE_BYTES: u64 = 4 * 1024;
const MAX_PAGE_SIZE_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityConfidence {
    Observed,
    Fallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryCapabilities {
    pub physical_memory_bytes: u64,
    pub page_size_bytes: u64,
    pub memory_confidence: CapabilityConfidence,
    pub page_size_confidence: CapabilityConfidence,
}

impl MemoryCapabilities {
    pub fn new(physical_memory_bytes: u64, page_size_bytes: u64) -> Option<Self> {
        if !(MIN_PHYSICAL_MEMORY_BYTES..=MAX_PHYSICAL_MEMORY_BYTES).contains(&physical_memory_bytes)
            || !(MIN_PAGE_SIZE_BYTES..=MAX_PAGE_SIZE_BYTES).contains(&page_size_bytes)
            || !page_size_bytes.is_power_of_two()
        {
            return None;
        }
        Some(Self {
            physical_memory_bytes,
            page_size_bytes,
            memory_confidence: CapabilityConfidence::Observed,
            page_size_confidence: CapabilityConfidence::Observed,
        })
    }

    pub fn apple_silicon_fallback() -> Self {
        Self {
            physical_memory_bytes: 8 * GIB,
            page_size_bytes: 16 * 1024,
            memory_confidence: CapabilityConfidence::Fallback,
            page_size_confidence: CapabilityConfidence::Fallback,
        }
    }

    pub fn from_partial(physical_memory_bytes: Option<u64>, page_size_bytes: Option<u64>) -> Self {
        let fallback = Self::apple_silicon_fallback();
        let (physical_memory_bytes, memory_confidence) = physical_memory_bytes
            .filter(|value| (MIN_PHYSICAL_MEMORY_BYTES..=MAX_PHYSICAL_MEMORY_BYTES).contains(value))
            .map(|value| (value, CapabilityConfidence::Observed))
            .unwrap_or((
                fallback.physical_memory_bytes,
                CapabilityConfidence::Fallback,
            ));
        let (page_size_bytes, page_size_confidence) = page_size_bytes
            .filter(|value| {
                (MIN_PAGE_SIZE_BYTES..=MAX_PAGE_SIZE_BYTES).contains(value)
                    && value.is_power_of_two()
            })
            .map(|value| (value, CapabilityConfidence::Observed))
            .unwrap_or((fallback.page_size_bytes, CapabilityConfidence::Fallback));

        Self {
            physical_memory_bytes,
            page_size_bytes,
            memory_confidence,
            page_size_confidence,
        }
    }

    pub fn with_observed_page_size(self, page_size_bytes: u64) -> Self {
        Self::new(self.physical_memory_bytes, page_size_bytes)
            .map(|mut capabilities| {
                capabilities.memory_confidence = self.memory_confidence;
                capabilities
            })
            .unwrap_or(self)
    }
}

#[derive(Debug, Clone)]
pub struct MemoryObservation {
    pub generation: u64,
    pub observed_at: Instant,
    pub pressure: f64,
    pub pressure_velocity_per_second: f64,
    pub swap_used_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_source_valid: bool,
    pub swap_delta_bytes_per_second: f64,
    pub compressions_per_second: f64,
    pub decompressions_per_second: f64,
    pub purges_per_second: f64,
    pub swapouts_per_second: f64,
    pub page_size_bytes: u64,
    pub vm_source_valid: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedMemoryState {
    pub generation: u64,
    pub observed_at: Instant,
    pub sample_age: Duration,
    pub valid: bool,
    pub confidence: f64,
    pub pressure: f64,
    pub pressure_velocity_per_second: f64,
    pub swap_fraction_of_ram: f64,
    pub dynamic_swap_fraction: Option<f64>,
    pub swap_growth_fraction_per_minute: f64,
    pub compression_fraction_per_second: f64,
    pub reclaim_fraction_per_second: f64,
    pub swapout_fraction_per_second: f64,
}

pub struct MemoryNormalizer;

impl MemoryNormalizer {
    pub const MAX_SAMPLE_AGE: Duration = Duration::from_secs(2);

    pub fn normalize(
        capabilities: &MemoryCapabilities,
        observation: &MemoryObservation,
        now: Instant,
    ) -> NormalizedMemoryState {
        let timestamp_ordered = now
            .checked_duration_since(observation.observed_at)
            .is_some();
        let sample_age = now.saturating_duration_since(observation.observed_at);
        let memory_capacity_valid = (MIN_PHYSICAL_MEMORY_BYTES..=MAX_PHYSICAL_MEMORY_BYTES)
            .contains(&capabilities.physical_memory_bytes);
        let page_size_valid = (MIN_PAGE_SIZE_BYTES..=MAX_PAGE_SIZE_BYTES)
            .contains(&observation.page_size_bytes)
            && observation.page_size_bytes.is_power_of_two();
        let finite = [
            observation.pressure,
            observation.pressure_velocity_per_second,
            observation.swap_delta_bytes_per_second,
            observation.compressions_per_second,
            observation.decompressions_per_second,
            observation.purges_per_second,
            observation.swapouts_per_second,
        ]
        .into_iter()
        .all(f64::is_finite);
        let non_negative_rates = observation.compressions_per_second >= 0.0
            && observation.decompressions_per_second >= 0.0
            && observation.purges_per_second >= 0.0
            && observation.swapouts_per_second >= 0.0;
        let ram = if memory_capacity_valid {
            capabilities.physical_memory_bytes as f64
        } else {
            1.0
        };
        let page = if page_size_valid {
            observation.page_size_bytes as f64
        } else {
            1.0
        };
        let swap_fraction_of_ram = observation.swap_used_bytes as f64 / ram;
        let dynamic_swap_fraction = (observation.swap_total_bytes > 0)
            .then(|| observation.swap_used_bytes as f64 / observation.swap_total_bytes as f64);
        let swap_growth_fraction_per_minute = observation.swap_delta_bytes_per_second / ram * 60.0;
        let compression_fraction_per_second = observation.compressions_per_second * page / ram;
        let reclaim_fraction_per_second =
            (observation.decompressions_per_second + observation.purges_per_second) * page / ram;
        let swapout_fraction_per_second = observation.swapouts_per_second * page / ram;
        let derived_finite = [
            swap_fraction_of_ram,
            swap_growth_fraction_per_minute,
            compression_fraction_per_second,
            reclaim_fraction_per_second,
            swapout_fraction_per_second,
        ]
        .into_iter()
        .all(f64::is_finite)
            && dynamic_swap_fraction.is_none_or(f64::is_finite);
        let valid = finite
            && derived_finite
            && non_negative_rates
            && observation.vm_source_valid
            && observation.swap_source_valid
            && memory_capacity_valid
            && page_size_valid
            && timestamp_ordered
            && observation.pressure >= 0.0
            && observation.pressure <= 1.0
            && sample_age <= Self::MAX_SAMPLE_AGE;

        let confidence = if capabilities.memory_confidence == CapabilityConfidence::Observed
            && capabilities.page_size_confidence == CapabilityConfidence::Observed
            && observation.page_size_bytes == capabilities.page_size_bytes
        {
            1.0
        } else {
            0.5
        };

        NormalizedMemoryState {
            generation: observation.generation,
            observed_at: observation.observed_at,
            sample_age,
            valid,
            confidence: if valid { confidence } else { 0.0 },
            pressure: if observation.pressure.is_finite() {
                observation.pressure.clamp(0.0, 1.0)
            } else {
                0.0
            },
            pressure_velocity_per_second: finite_or_zero(observation.pressure_velocity_per_second),
            swap_fraction_of_ram: finite_or_zero(swap_fraction_of_ram).max(0.0),
            dynamic_swap_fraction: dynamic_swap_fraction
                .map(|value| finite_or_zero(value).max(0.0)),
            swap_growth_fraction_per_minute: finite_or_zero(swap_growth_fraction_per_minute),
            compression_fraction_per_second: finite_or_zero(compression_fraction_per_second),
            reclaim_fraction_per_second: finite_or_zero(reclaim_fraction_per_second),
            swapout_fraction_per_second: finite_or_zero(swapout_fraction_per_second),
        }
    }
}

#[inline]
fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegime {
    Unknown,
    Calm,
    Building,
    Contended,
    Crisis,
    Recovering,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRegimeEvidence {
    pub regime: MemoryRegime,
    pub generation: u64,
    pub confidence: f64,
    pub normalized_score: f64,
    pub swap_fraction_of_ram: f64,
    pub swap_growth_fraction_per_minute: f64,
    pub adverse_flow: bool,
    pub sustained: bool,
}

impl Default for MemoryRegimeEvidence {
    fn default() -> Self {
        Self {
            regime: MemoryRegime::Unknown,
            generation: 0,
            confidence: 0.0,
            normalized_score: 0.0,
            swap_fraction_of_ram: 0.0,
            swap_growth_fraction_per_minute: 0.0,
            adverse_flow: false,
            sustained: false,
        }
    }
}

impl MemoryRegimeEvidence {
    pub fn physically_correlated_crisis(&self) -> bool {
        self.regime == MemoryRegime::Crisis
            && self.sustained
            && self.adverse_flow
            && self.confidence > 0.0
    }
}

#[derive(Debug, Clone)]
pub struct MemoryRegimePolicy {
    pub building_growth_fraction_per_minute: f64,
    pub strong_growth_fraction_per_minute: f64,
    pub draining_fraction_per_minute: f64,
    pub contended_pressure: f64,
    pub crisis_pressure: f64,
    pub recovery_pressure: f64,
    pub swap_envelope_fraction_of_ram: f64,
    pub compression_flow_fraction_per_second: f64,
    pub building_dwell: Duration,
    pub contended_dwell: Duration,
    pub crisis_dwell: Duration,
    pub recovery_dwell: Duration,
}

impl Default for MemoryRegimePolicy {
    fn default() -> Self {
        Self {
            building_growth_fraction_per_minute: 0.001,
            strong_growth_fraction_per_minute: 0.005,
            draining_fraction_per_minute: -0.001,
            contended_pressure: 0.65,
            crisis_pressure: 0.82,
            recovery_pressure: 0.60,
            swap_envelope_fraction_of_ram: 0.50,
            compression_flow_fraction_per_second: 0.000_01,
            building_dwell: Duration::from_secs(2),
            contended_dwell: Duration::from_secs(2),
            crisis_dwell: Duration::from_secs(3),
            recovery_dwell: Duration::from_secs(2),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryRegimeDetector {
    policy: MemoryRegimePolicy,
    regime: MemoryRegime,
    candidate: MemoryRegime,
    candidate_since: Option<Instant>,
    last_generation: Option<u64>,
    last_evidence: MemoryRegimeEvidence,
    accepted_samples: u64,
}

impl Default for MemoryRegimeDetector {
    fn default() -> Self {
        Self::new(MemoryRegimePolicy::default())
    }
}

impl MemoryRegimeDetector {
    pub fn new(policy: MemoryRegimePolicy) -> Self {
        Self {
            policy,
            regime: MemoryRegime::Unknown,
            candidate: MemoryRegime::Unknown,
            candidate_since: None,
            last_generation: None,
            last_evidence: MemoryRegimeEvidence::default(),
            accepted_samples: 0,
        }
    }

    pub fn accepted_samples(&self) -> u64 {
        self.accepted_samples
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.policy.clone());
    }

    pub fn update(&mut self, state: &NormalizedMemoryState, now: Instant) -> MemoryRegimeEvidence {
        if !state.valid {
            self.last_generation = Some(state.generation);
            self.regime = MemoryRegime::Unknown;
            self.candidate = MemoryRegime::Unknown;
            self.candidate_since = None;
            self.last_evidence = MemoryRegimeEvidence {
                generation: state.generation,
                ..MemoryRegimeEvidence::default()
            };
            return self.last_evidence.clone();
        }
        if self.last_generation == Some(state.generation) {
            return self.last_evidence.clone();
        }
        self.last_generation = Some(state.generation);
        self.accepted_samples = self.accepted_samples.saturating_add(1);

        let compression_net =
            state.compression_fraction_per_second - state.reclaim_fraction_per_second;
        let adverse_flow = state.swap_growth_fraction_per_minute
            >= self.policy.building_growth_fraction_per_minute
            || state.swapout_fraction_per_second > 0.0
            || compression_net >= self.policy.compression_flow_fraction_per_second;
        let strong_flow = state.swap_growth_fraction_per_minute
            >= self.policy.strong_growth_fraction_per_minute
            || state.swapout_fraction_per_second > 0.0;
        let exhausted_envelope = state.swap_fraction_of_ram
            >= self.policy.swap_envelope_fraction_of_ram
            && state.pressure >= self.policy.recovery_pressure;

        let target = if (state.pressure >= self.policy.crisis_pressure && strong_flow)
            || exhausted_envelope
        {
            MemoryRegime::Crisis
        } else if state.pressure >= self.policy.contended_pressure && adverse_flow {
            MemoryRegime::Contended
        } else if adverse_flow || state.pressure_velocity_per_second > 0.005 {
            MemoryRegime::Building
        } else if matches!(
            self.regime,
            MemoryRegime::Building | MemoryRegime::Contended | MemoryRegime::Crisis
        ) && state.swap_growth_fraction_per_minute
            <= self.policy.draining_fraction_per_minute
        {
            MemoryRegime::Recovering
        } else {
            MemoryRegime::Calm
        };

        if target != self.candidate {
            self.candidate = target;
            self.candidate_since = Some(now);
        }
        let dwell = match target {
            MemoryRegime::Unknown | MemoryRegime::Calm => Duration::ZERO,
            MemoryRegime::Building => self.policy.building_dwell,
            MemoryRegime::Contended => self.policy.contended_dwell,
            MemoryRegime::Crisis => self.policy.crisis_dwell,
            MemoryRegime::Recovering => self.policy.recovery_dwell,
        };
        let sustained = self
            .candidate_since
            .is_some_and(|since| now.saturating_duration_since(since) >= dwell);
        if sustained {
            self.regime = target;
        }

        let flow_score = (state.swap_growth_fraction_per_minute
            / self.policy.strong_growth_fraction_per_minute)
            .clamp(0.0, 1.0);
        let normalized_score =
            (0.65 * state.pressure + 0.20 * flow_score + 0.15 * state.swap_fraction_of_ram)
                .clamp(0.0, 1.0);
        self.last_evidence = MemoryRegimeEvidence {
            regime: self.regime,
            generation: state.generation,
            confidence: state.confidence,
            normalized_score,
            swap_fraction_of_ram: state.swap_fraction_of_ram,
            swap_growth_fraction_per_minute: state.swap_growth_fraction_per_minute,
            adverse_flow,
            sustained: self.regime == target && sustained,
        };
        self.last_evidence.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn capabilities(ram_gib: u64, page_size_bytes: u64) -> MemoryCapabilities {
        MemoryCapabilities::new(ram_gib * GIB, page_size_bytes).expect("valid capabilities")
    }

    fn observation(
        generation: u64,
        observed_at: Instant,
        ram_gib: u64,
        swap_gib: f64,
        swap_growth_fraction_per_minute: f64,
        pressure: f64,
    ) -> MemoryObservation {
        let ram = ram_gib as f64 * GIB as f64;
        MemoryObservation {
            generation,
            observed_at,
            pressure,
            pressure_velocity_per_second: 0.0,
            swap_used_bytes: (swap_gib * GIB as f64) as u64,
            swap_total_bytes: 2 * GIB,
            swap_source_valid: true,
            swap_delta_bytes_per_second: swap_growth_fraction_per_minute * ram / 60.0,
            compressions_per_second: 0.0,
            decompressions_per_second: 0.0,
            purges_per_second: 0.0,
            swapouts_per_second: 0.0,
            page_size_bytes: 16 * 1024,
            vm_source_valid: true,
        }
    }

    #[test]
    fn capabilities_reject_invalid_ram_and_page_sizes() {
        assert!(MemoryCapabilities::new(0, 16 * 1024).is_none());
        assert!(MemoryCapabilities::new(8 * GIB, 0).is_none());
        assert!(MemoryCapabilities::new(8 * GIB, 12 * 1024).is_none());
        assert!(MemoryCapabilities::new(8 * GIB, 16 * 1024).is_some());
    }

    #[test]
    fn normalization_is_equivalent_across_ram_capacities() {
        let now = Instant::now();
        let m1 = MemoryNormalizer::normalize(
            &capabilities(8, 16 * 1024),
            &observation(1, now, 8, 2.0, 0.01, 0.70),
            now,
        );
        let m4 = MemoryNormalizer::normalize(
            &capabilities(16, 16 * 1024),
            &observation(1, now, 16, 4.0, 0.01, 0.70),
            now,
        );

        assert!(m1.valid && m4.valid);
        assert!((m1.swap_fraction_of_ram - m4.swap_fraction_of_ram).abs() < 1e-12);
        assert!(
            (m1.swap_growth_fraction_per_minute - m4.swap_growth_fraction_per_minute).abs() < 1e-12
        );
    }

    #[test]
    fn page_flow_uses_the_observed_native_page_size() {
        let now = Instant::now();
        let mut four_k = observation(1, now, 8, 0.0, 0.0, 0.50);
        four_k.page_size_bytes = 4 * 1024;
        four_k.compressions_per_second = 1_000.0;
        let mut sixteen_k = four_k.clone();
        sixteen_k.page_size_bytes = 16 * 1024;

        let a = MemoryNormalizer::normalize(&capabilities(8, 4 * 1024), &four_k, now);
        let b = MemoryNormalizer::normalize(&capabilities(8, 16 * 1024), &sixteen_k, now);

        assert!(
            (b.compression_fraction_per_second / a.compression_fraction_per_second - 4.0).abs()
                < 1e-12
        );
    }

    #[test]
    fn stale_and_non_finite_observations_fail_closed() {
        let now = Instant::now();
        let stale = observation(1, now - Duration::from_secs(3), 8, 1.0, 0.0, 0.70);
        let mut invalid = observation(2, now, 8, 1.0, 0.0, 0.70);
        invalid.swap_delta_bytes_per_second = f64::NAN;

        let stale_state = MemoryNormalizer::normalize(&capabilities(8, 16 * 1024), &stale, now);
        let invalid_state = MemoryNormalizer::normalize(&capabilities(8, 16 * 1024), &invalid, now);

        assert!(!stale_state.valid);
        assert!(!invalid_state.valid);
        assert_eq!(
            MemoryRegimeDetector::default()
                .update(&invalid_state, now)
                .regime,
            MemoryRegime::Unknown
        );
    }

    #[test]
    fn a_single_burst_cannot_establish_crisis() {
        let now = Instant::now();
        let state = MemoryNormalizer::normalize(
            &capabilities(8, 16 * 1024),
            &observation(1, now, 8, 4.5, 0.20, 0.90),
            now,
        );
        let mut detector = MemoryRegimeDetector::default();

        let evidence = detector.update(&state, now);

        assert_ne!(evidence.regime, MemoryRegime::Crisis);
        assert!(!evidence.physically_correlated_crisis());
    }

    #[test]
    fn sustained_evidence_uses_wall_clock_not_update_count() {
        let start = Instant::now();
        let caps = capabilities(8, 16 * 1024);
        let mut detector = MemoryRegimeDetector::default();

        for (generation, millis) in [(1, 0), (2, 500), (3, 1_000), (4, 2_500), (5, 3_500)] {
            let at = start + Duration::from_millis(millis);
            let state = MemoryNormalizer::normalize(
                &caps,
                &observation(generation, at, 8, 4.5, 0.02, 0.88),
                at,
            );
            let evidence = detector.update(&state, at);
            if millis < 3_000 {
                assert_ne!(evidence.regime, MemoryRegime::Crisis);
            } else {
                assert_eq!(evidence.regime, MemoryRegime::Crisis);
                assert!(evidence.physically_correlated_crisis());
            }
        }
    }

    #[test]
    fn duplicate_generation_returns_cached_evidence() {
        let now = Instant::now();
        let caps = capabilities(8, 16 * 1024);
        let state =
            MemoryNormalizer::normalize(&caps, &observation(7, now, 8, 2.0, 0.01, 0.70), now);
        let mut detector = MemoryRegimeDetector::default();
        let first = detector.update(&state, now);
        let duplicate = detector.update(&state, now + Duration::from_millis(100));

        assert_eq!(first, duplicate);
        assert_eq!(detector.accepted_samples(), 1);
    }

    #[test]
    fn stale_duplicate_generation_invalidates_cached_crisis() {
        let start = Instant::now();
        let caps = capabilities(8, 16 * 1024);
        let mut detector = MemoryRegimeDetector::default();
        let mut latest = MemoryRegimeEvidence::default();

        for (generation, seconds) in [(1, 0), (2, 1), (3, 2), (4, 4)] {
            let at = start + Duration::from_secs(seconds);
            let state = MemoryNormalizer::normalize(
                &caps,
                &observation(generation, at, 8, 4.5, 0.02, 0.90),
                at,
            );
            latest = detector.update(&state, at);
        }
        assert_eq!(latest.regime, MemoryRegime::Crisis);

        let stale = MemoryNormalizer::normalize(
            &caps,
            &observation(4, start + Duration::from_secs(4), 8, 4.5, 0.02, 0.90),
            start + Duration::from_secs(7),
        );
        let evidence = detector.update(&stale, start + Duration::from_secs(7));

        assert_eq!(evidence.regime, MemoryRegime::Unknown);
        assert!(!evidence.sustained);
    }

    #[test]
    fn derived_overflow_fails_closed_and_stays_finite() {
        let now = Instant::now();
        let mut extreme = observation(1, now, 8, 1.0, 0.0, 0.70);
        extreme.compressions_per_second = f64::MAX;

        let state = MemoryNormalizer::normalize(&capabilities(8, 64 * 1024), &extreme, now);

        assert!(!state.valid);
        assert!(state.compression_fraction_per_second.is_finite());
        assert_eq!(
            MemoryRegimeDetector::default().update(&state, now).regime,
            MemoryRegime::Unknown
        );
    }

    #[test]
    fn partial_kernel_sources_never_become_fresh_regime_evidence() {
        let now = Instant::now();
        let caps = capabilities(8, 16 * 1024);
        let mut partial = observation(1, now, 8, 4.5, 0.02, 0.90);
        partial.swap_source_valid = false;

        let state = MemoryNormalizer::normalize(&caps, &partial, now);

        assert!(!state.valid);
        assert_eq!(
            MemoryRegimeDetector::default().update(&state, now).regime,
            MemoryRegime::Unknown
        );
    }

    #[test]
    #[ignore = "run explicitly in release mode for the latency gate"]
    fn memory_pipeline_latency_probe() {
        use crate::engine::swap_predictor::SwapPredictor;
        use crate::engine::swap_reclaim::{SwapReclaimModel, VmFlowSample};

        fn percentile(mut values: Vec<u128>, percentile: usize) -> u128 {
            values.sort_unstable();
            values[(values.len() - 1) * percentile / 100]
        }

        let start = Instant::now();
        let caps = capabilities(8, 16 * 1024);
        let mut detector = MemoryRegimeDetector::default();
        let mut predictor = SwapPredictor::new();
        let mut reclaim = SwapReclaimModel::new();
        let mut fresh_ns = Vec::with_capacity(20_000);
        let mut duplicate_ns = Vec::with_capacity(20_000);

        for index in 1..=20_000_u64 {
            let at = start + Duration::from_millis(index * 500);
            let obs = observation(index, at, 8, 1.0, 0.002, 0.68);
            let flow = VmFlowSample {
                compressions_per_sec: 100.0,
                decompressions_per_sec: 80.0,
                purges_per_sec: 5.0,
                swapouts_per_sec: 2.0,
                swap_used_bytes: obs.swap_used_bytes,
                swap_total_bytes: obs.swap_total_bytes,
            };

            let timer = Instant::now();
            let normalized = MemoryNormalizer::normalize(&caps, &obs, at);
            std::hint::black_box(detector.update(&normalized, at));
            std::hint::black_box(predictor.update_at(
                index,
                at,
                obs.swap_used_bytes,
                obs.swap_total_bytes,
                caps.physical_memory_bytes,
            ));
            std::hint::black_box(reclaim.update_at(
                &flow,
                index,
                at,
                caps.page_size_bytes,
                caps.physical_memory_bytes,
            ));
            fresh_ns.push(timer.elapsed().as_nanos());

            let timer = Instant::now();
            std::hint::black_box(detector.update(&normalized, at));
            std::hint::black_box(predictor.update_at(
                index,
                at,
                obs.swap_used_bytes,
                obs.swap_total_bytes,
                caps.physical_memory_bytes,
            ));
            std::hint::black_box(reclaim.update_at(
                &flow,
                index,
                at,
                caps.page_size_bytes,
                caps.physical_memory_bytes,
            ));
            duplicate_ns.push(timer.elapsed().as_nanos());
        }

        let fresh_p50 = percentile(fresh_ns.clone(), 50);
        let fresh_p95 = percentile(fresh_ns.clone(), 95);
        let fresh_max = fresh_ns.into_iter().max().unwrap_or(0);
        let duplicate_p50 = percentile(duplicate_ns.clone(), 50);
        let duplicate_p95 = percentile(duplicate_ns.clone(), 95);
        let duplicate_max = duplicate_ns.into_iter().max().unwrap_or(0);
        eprintln!(
            "memory-pipeline-ns fresh[p50={fresh_p50} p95={fresh_p95} max={fresh_max}] duplicate[p50={duplicate_p50} p95={duplicate_p95} max={duplicate_max}]"
        );

        assert!(fresh_p95 <= 50_000, "fresh p95 {fresh_p95}ns exceeds 50us");
    }
}
