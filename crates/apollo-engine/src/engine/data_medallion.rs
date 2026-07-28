//! Medallion curation for resolved learning outcomes.
//!
//! Bronze counts every raw outcome, Silver contains structurally valid data,
//! and Gold contains stable, contextual, plausible, unique observations that
//! are safe to fan out to Apollo's long-lived learners. The curator performs
//! no I/O and keeps a fixed-size duplicate window so its cost stays bounded.

use serde::{Deserialize, Serialize};

const RECENT_FINGERPRINTS: usize = 64;
const MAX_IDENTITY_BYTES: usize = 256;
const MAX_PLAUSIBLE_PRESSURE_DELTA: f64 = 0.35;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MedallionTier {
    Bronze,
    Silver,
    Gold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationSource {
    ResolvedActionOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurationRejection {
    PipelineDisabled,
    NonFiniteMeasurement,
    OutOfRangeMeasurement,
    InvalidIdentity,
    EphemeralIdentity,
    MissingWorkloadContext,
    ImplausibleDelta,
    Duplicate,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QualityTags {
    pub finite_measurements: bool,
    pub bounded_measurements: bool,
    pub valid_identity: bool,
    pub stable_identity: bool,
    pub workload_context: bool,
    pub plausible_delta: bool,
    pub unique: bool,
}

impl QualityTags {
    fn score(self) -> f64 {
        0.20 * self.finite_measurements as u8 as f64
            + 0.20 * self.bounded_measurements as u8 as f64
            + 0.15 * self.valid_identity as u8 as f64
            + 0.15 * self.stable_identity as u8 as f64
            + 0.10 * self.workload_context as u8 as f64
            + 0.10 * self.plausible_delta as u8 as f64
            + 0.10 * self.unique as u8 as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CuratedLabel {
    pub tier: MedallionTier,
    pub source: ObservationSource,
    pub quality_score: f64,
    pub tags: QualityTags,
    pub rejection: Option<CurationRejection>,
}

impl CuratedLabel {
    pub fn is_gold(self) -> bool {
        self.tier == MedallionTier::Gold
    }

    pub fn pipeline_disabled() -> Self {
        Self {
            tier: MedallionTier::Bronze,
            source: ObservationSource::ResolvedActionOutcome,
            quality_score: 0.0,
            tags: QualityTags::default(),
            rejection: Some(CurationRejection::PipelineDisabled),
        }
    }

    /// Compatibility label for direct engine APIs that predate medallion
    /// curation. Production resolves through `DataMedallion::curate`.
    pub fn trusted_legacy() -> Self {
        Self {
            tier: MedallionTier::Gold,
            source: ObservationSource::ResolvedActionOutcome,
            quality_score: 1.0,
            tags: QualityTags {
                finite_measurements: true,
                bounded_measurements: true,
                valid_identity: true,
                stable_identity: true,
                workload_context: true,
                plausible_delta: true,
                unique: true,
            },
            rejection: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MedallionMetrics {
    pub bronze_total: u64,
    pub silver_total: u64,
    pub gold_total: u64,
    pub rejected_total: u64,
    pub invalid_total: u64,
    pub duplicate_total: u64,
    pub mean_quality: f64,
    pub gold_rate: f64,
}

/// Crash-safe snapshot of the medallion trust boundary. Keeping the duplicate
/// window preserves the guarantee that a daemon restart cannot admit the same
/// resolved outcome twice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataMedallionPersisted {
    #[serde(default)]
    pub recent_fingerprints: Vec<u64>,
    #[serde(default)]
    pub recent_cursor: usize,
    #[serde(default)]
    pub bronze_total: u64,
    #[serde(default)]
    pub silver_total: u64,
    #[serde(default)]
    pub gold_total: u64,
    #[serde(default)]
    pub invalid_total: u64,
    #[serde(default)]
    pub duplicate_total: u64,
    #[serde(default)]
    pub quality_sum: f64,
}

pub struct MedallionObservation<'a> {
    pub process_name: &'a str,
    pub workload: &'a str,
    pub pre_pressure: f64,
    pub post_pressure: f64,
    pub cycle: u64,
    pub action_code: u8,
}

#[derive(Debug)]
pub struct DataMedallion {
    recent_fingerprints: [u64; RECENT_FINGERPRINTS],
    recent_cursor: usize,
    bronze_total: u64,
    silver_total: u64,
    gold_total: u64,
    invalid_total: u64,
    duplicate_total: u64,
    quality_sum: f64,
}

impl DataMedallion {
    pub fn new() -> Self {
        Self {
            recent_fingerprints: [0; RECENT_FINGERPRINTS],
            recent_cursor: 0,
            bronze_total: 0,
            silver_total: 0,
            gold_total: 0,
            invalid_total: 0,
            duplicate_total: 0,
            quality_sum: 0.0,
        }
    }

    pub fn curate(&mut self, observation: MedallionObservation<'_>) -> CuratedLabel {
        self.bronze_total = self.bronze_total.saturating_add(1);

        let finite_measurements =
            observation.pre_pressure.is_finite() && observation.post_pressure.is_finite();
        let bounded_measurements = finite_measurements
            && (0.0..=1.0).contains(&observation.pre_pressure)
            && (0.0..=1.0).contains(&observation.post_pressure);
        let valid_identity = is_valid_identity(observation.process_name);
        let stable_identity = valid_identity && is_stable_identity(observation.process_name);
        let workload_context = matches!(
            observation.workload,
            "idle" | "browsing" | "build" | "llm-inference" | "any"
        );
        let plausible_delta = finite_measurements
            && (observation.pre_pressure - observation.post_pressure).abs()
                <= MAX_PLAUSIBLE_PRESSURE_DELTA;
        let fingerprint = observation_fingerprint(&observation);
        let unique = !self.recent_fingerprints.contains(&fingerprint);

        let tags = QualityTags {
            finite_measurements,
            bounded_measurements,
            valid_identity,
            stable_identity,
            workload_context,
            plausible_delta,
            unique,
        };
        let quality_score = tags.score();
        self.quality_sum += quality_score;

        let rejection = if !finite_measurements {
            Some(CurationRejection::NonFiniteMeasurement)
        } else if !bounded_measurements {
            Some(CurationRejection::OutOfRangeMeasurement)
        } else if !valid_identity {
            Some(CurationRejection::InvalidIdentity)
        } else {
            None
        };

        if let Some(rejection) = rejection {
            self.invalid_total = self.invalid_total.saturating_add(1);
            return CuratedLabel {
                tier: MedallionTier::Bronze,
                source: ObservationSource::ResolvedActionOutcome,
                quality_score,
                tags,
                rejection: Some(rejection),
            };
        }

        self.silver_total = self.silver_total.saturating_add(1);
        let rejection = if !stable_identity {
            Some(CurationRejection::EphemeralIdentity)
        } else if !workload_context {
            Some(CurationRejection::MissingWorkloadContext)
        } else if !plausible_delta {
            Some(CurationRejection::ImplausibleDelta)
        } else if !unique {
            self.duplicate_total = self.duplicate_total.saturating_add(1);
            Some(CurationRejection::Duplicate)
        } else {
            None
        };

        if let Some(rejection) = rejection {
            return CuratedLabel {
                tier: MedallionTier::Silver,
                source: ObservationSource::ResolvedActionOutcome,
                quality_score,
                tags,
                rejection: Some(rejection),
            };
        }

        self.recent_fingerprints[self.recent_cursor] = fingerprint;
        self.recent_cursor = (self.recent_cursor + 1) % RECENT_FINGERPRINTS;
        self.gold_total = self.gold_total.saturating_add(1);
        CuratedLabel {
            tier: MedallionTier::Gold,
            source: ObservationSource::ResolvedActionOutcome,
            quality_score,
            tags,
            rejection: None,
        }
    }

    pub fn metrics(&self) -> MedallionMetrics {
        let rejected_total = self.bronze_total.saturating_sub(self.gold_total);
        MedallionMetrics {
            bronze_total: self.bronze_total,
            silver_total: self.silver_total,
            gold_total: self.gold_total,
            rejected_total,
            invalid_total: self.invalid_total,
            duplicate_total: self.duplicate_total,
            mean_quality: if self.bronze_total == 0 {
                0.0
            } else {
                (self.quality_sum / self.bronze_total as f64).clamp(0.0, 1.0)
            },
            gold_rate: if self.bronze_total == 0 {
                0.0
            } else {
                self.gold_total as f64 / self.bronze_total as f64
            },
        }
    }

    pub fn snapshot(&self) -> DataMedallionPersisted {
        DataMedallionPersisted {
            recent_fingerprints: self.recent_fingerprints.to_vec(),
            recent_cursor: self.recent_cursor,
            bronze_total: self.bronze_total,
            silver_total: self.silver_total,
            gold_total: self.gold_total,
            invalid_total: self.invalid_total,
            duplicate_total: self.duplicate_total,
            quality_sum: self.quality_sum,
        }
    }

    /// Restore a validated snapshot. Malformed or stale fields are bounded so
    /// persistence can never manufacture a Gold outcome or poison curation.
    pub fn restore(&mut self, state: DataMedallionPersisted) {
        self.recent_fingerprints = [0; RECENT_FINGERPRINTS];
        for (slot, fingerprint) in state
            .recent_fingerprints
            .into_iter()
            .take(RECENT_FINGERPRINTS)
            .enumerate()
        {
            self.recent_fingerprints[slot] = fingerprint;
        }
        self.recent_cursor = state.recent_cursor % RECENT_FINGERPRINTS;
        self.bronze_total = state.bronze_total;
        self.silver_total = state.silver_total.min(self.bronze_total);
        self.gold_total = state.gold_total.min(self.silver_total);
        self.invalid_total = state.invalid_total.min(self.bronze_total);
        self.duplicate_total = state
            .duplicate_total
            .min(self.silver_total.saturating_sub(self.gold_total));
        self.quality_sum = if state.quality_sum.is_finite() {
            state.quality_sum.clamp(0.0, self.bronze_total as f64)
        } else {
            0.0
        };
    }
}

impl Default for DataMedallion {
    fn default() -> Self {
        Self::new()
    }
}

fn is_valid_identity(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_IDENTITY_BYTES
        && name.trim() == name
        && !name.bytes().any(|byte| byte.is_ascii_control())
}

fn is_stable_identity(name: &str) -> bool {
    !name
        .strip_prefix("pid:")
        .is_some_and(|pid| !pid.is_empty() && pid.bytes().all(|byte| byte.is_ascii_digit()))
}

fn observation_fingerprint(observation: &MedallionObservation<'_>) -> u64 {
    // FNV-1a is deterministic and avoids RandomState initialization on this
    // bounded, non-adversarial in-process deduplication key.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in observation.process_name.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^= observation.action_code as u64;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    for byte in observation.cycle.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Zero is the empty-slot sentinel in the fixed ring.
    hash.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn observation<'a>(name: &'a str, workload: &'a str, cycle: u64) -> MedallionObservation<'a> {
        MedallionObservation {
            process_name: name,
            workload,
            pre_pressure: 0.72,
            post_pressure: 0.68,
            cycle,
            action_code: 0,
        }
    }

    #[test]
    fn clean_resolved_outcome_reaches_gold() {
        let mut medallion = DataMedallion::new();
        let label = medallion.curate(observation("Safari", "browsing", 10));

        assert!(label.is_gold());
        assert_eq!(label.quality_score, 1.0);
        assert_eq!(medallion.metrics().gold_total, 1);
    }

    #[test]
    fn non_finite_measurement_stays_bronze() {
        let mut medallion = DataMedallion::new();
        let mut raw = observation("Safari", "browsing", 10);
        raw.post_pressure = f64::NAN;
        let label = medallion.curate(raw);

        assert_eq!(label.tier, MedallionTier::Bronze);
        assert_eq!(
            label.rejection,
            Some(CurationRejection::NonFiniteMeasurement)
        );
        assert_eq!(medallion.metrics().invalid_total, 1);
    }

    #[test]
    fn ephemeral_pid_identity_stops_at_silver() {
        let mut medallion = DataMedallion::new();
        let label = medallion.curate(observation("pid:321", "idle", 10));

        assert_eq!(label.tier, MedallionTier::Silver);
        assert_eq!(label.rejection, Some(CurationRejection::EphemeralIdentity));
    }

    #[test]
    fn duplicate_outcome_does_not_reach_learners_twice() {
        let mut medallion = DataMedallion::new();
        assert!(medallion
            .curate(observation("Safari", "browsing", 10))
            .is_gold());
        let duplicate = medallion.curate(observation("Safari", "browsing", 10));

        assert_eq!(duplicate.tier, MedallionTier::Silver);
        assert_eq!(duplicate.rejection, Some(CurationRejection::Duplicate));
        let metrics = medallion.metrics();
        assert_eq!(metrics.bronze_total, 2);
        assert_eq!(metrics.gold_total, 1);
        assert_eq!(metrics.duplicate_total, 1);
    }

    #[test]
    fn snapshot_restore_preserves_metrics_and_duplicate_window() {
        let mut original = DataMedallion::new();
        assert!(original
            .curate(observation("Safari", "browsing", 10))
            .is_gold());
        let expected = original.metrics();
        let encoded = serde_json::to_string(&original.snapshot()).expect("snapshot serializes");
        let persisted = serde_json::from_str(&encoded).expect("snapshot deserializes");

        let mut restored = DataMedallion::new();
        restored.restore(persisted);
        assert_eq!(restored.metrics(), expected);

        let duplicate = restored.curate(observation("Safari", "browsing", 10));
        assert_eq!(duplicate.rejection, Some(CurationRejection::Duplicate));
    }

    #[test]
    fn restore_never_inflates_invalid_persisted_counts() {
        let mut medallion = DataMedallion::new();
        medallion.restore(DataMedallionPersisted {
            recent_fingerprints: vec![1; RECENT_FINGERPRINTS + 1],
            recent_cursor: RECENT_FINGERPRINTS + 5,
            bronze_total: 3,
            silver_total: 99,
            gold_total: 99,
            invalid_total: 99,
            duplicate_total: 99,
            quality_sum: f64::NAN,
        });

        let metrics = medallion.metrics();
        assert_eq!(metrics.bronze_total, 3);
        assert_eq!(metrics.silver_total, 3);
        assert_eq!(metrics.gold_total, 3);
        assert_eq!(metrics.invalid_total, 3);
        assert_eq!(metrics.duplicate_total, 0);
        assert_eq!(metrics.mean_quality, 0.0);
    }

    #[test]
    fn implausibly_large_delta_stops_at_silver() {
        let mut medallion = DataMedallion::new();
        let mut raw = observation("Safari", "browsing", 10);
        raw.post_pressure = 0.10;
        let label = medallion.curate(raw);

        assert_eq!(label.tier, MedallionTier::Silver);
        assert_eq!(label.rejection, Some(CurationRejection::ImplausibleDelta));
    }

    #[test]
    fn curation_benchmark_10k_stays_bounded() {
        let mut medallion = DataMedallion::new();
        let started = Instant::now();
        for cycle in 1..=10_000 {
            assert!(medallion
                .curate(observation("Safari", "browsing", cycle))
                .is_gold());
        }
        let elapsed = started.elapsed();

        println!(
            "medallion_benchmark: 10000 outcomes in {:?} ({:.0} ns/outcome)",
            elapsed,
            elapsed.as_nanos() as f64 / 10_000.0
        );
        assert!(
            elapsed < Duration::from_millis(100),
            "curation exceeded 10us/outcome budget: {elapsed:?}"
        );
    }
}
