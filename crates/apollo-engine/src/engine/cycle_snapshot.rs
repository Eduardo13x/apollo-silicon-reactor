//! Immutable, identity-bound input for one Apollo control cycle.
//!
//! This module contains only compact values and source metadata. It does not
//! retain live process objects, synchronization primitives, caches, actions,
//! or model state. A publisher assigns the daemon epoch and monotonically
//! increasing sequence before exposing an `Arc` to readers.

use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct SnapshotId {
    pub daemon_epoch: u64,
    pub sequence: u64,
}

impl SnapshotId {
    pub const fn new(daemon_epoch: u64, sequence: u64) -> Self {
        Self {
            daemon_epoch,
            sequence,
        }
    }

    pub const fn next_sequence(self) -> Option<Self> {
        match self.sequence.checked_add(1) {
            Some(sequence) => Some(Self {
                daemon_epoch: self.daemon_epoch,
                sequence,
            }),
            None => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ObservationStatus {
    Fresh,
    Stale,
    Unavailable,
    Invalid,
    Truncated,
}

impl Default for ObservationStatus {
    fn default() -> Self {
        Self::Unavailable
    }
}

impl ObservationStatus {
    pub const fn is_usable(self) -> bool {
        self.is_fresh()
    }

    pub const fn is_fresh(self) -> bool {
        matches!(self, Self::Fresh)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceObservation<T> {
    value: Option<T>,
    generation: u64,
    revision: u64,
    age_us_at_cut: u64,
    status: ObservationStatus,
}

impl<T> SourceObservation<T> {
    pub fn fresh(value: T, generation: u64, revision: u64, age_us_at_cut: u64) -> Self {
        Self {
            value: Some(value),
            generation,
            revision,
            age_us_at_cut,
            status: ObservationStatus::Fresh,
        }
    }

    pub fn unavailable(generation: u64, revision: u64) -> Self {
        Self {
            value: None,
            generation,
            revision,
            age_us_at_cut: 0,
            status: ObservationStatus::Unavailable,
        }
    }

    pub fn stale(value: Option<T>, generation: u64, revision: u64, age_us_at_cut: u64) -> Self {
        Self {
            value,
            generation,
            revision,
            age_us_at_cut,
            status: ObservationStatus::Stale,
        }
    }

    pub fn invalid(generation: u64, revision: u64) -> Self {
        Self {
            value: None,
            generation,
            revision,
            age_us_at_cut: 0,
            status: ObservationStatus::Invalid,
        }
    }

    pub fn truncated(generation: u64, revision: u64) -> Self {
        Self {
            value: None,
            generation,
            revision,
            age_us_at_cut: 0,
            status: ObservationStatus::Truncated,
        }
    }

    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> SourceObservation<U> {
        SourceObservation {
            value: self.value.map(map),
            generation: self.generation,
            revision: self.revision,
            age_us_at_cut: self.age_us_at_cut,
            status: self.status,
        }
    }

    pub fn fresh_value(&self) -> Option<&T> {
        self.status
            .is_fresh()
            .then(|| self.value.as_ref())
            .flatten()
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn revision(&self) -> u64 {
        self.revision
    }

    pub const fn age_us_at_cut(&self) -> u64 {
        self.age_us_at_cut
    }

    pub const fn status(&self) -> ObservationStatus {
        self.status
    }
}

impl<'de, T> Deserialize<'de> for SourceObservation<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(bound(deserialize = "T: Deserialize<'de>"))]
        struct Wire<T> {
            value: Option<T>,
            #[serde(default)]
            generation: u64,
            #[serde(default)]
            revision: u64,
            #[serde(default)]
            age_us_at_cut: u64,
            #[serde(default)]
            status: ObservationStatus,
        }

        let wire = Wire::<T>::deserialize(deserializer)?;
        let (value, status) = match (wire.status, wire.value) {
            (ObservationStatus::Fresh, Some(value)) => (Some(value), ObservationStatus::Fresh),
            (ObservationStatus::Fresh, None) => (None, ObservationStatus::Invalid),
            (ObservationStatus::Stale, value) => (value, ObservationStatus::Stale),
            (status, _) => (None, status),
        };
        Ok(Self {
            value,
            generation: wire.generation,
            revision: wire.revision,
            age_us_at_cut: wire.age_us_at_cut,
            status,
        })
    }
}

impl<T> Default for SourceObservation<T> {
    fn default() -> Self {
        Self::unavailable(0, 0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CycleContextSnapshot {
    pub id: SnapshotId,
    pub cycle: u64,
    pub cut_started_mono_us: u64,
    pub cut_completed_mono_us: u64,
    pub workload_id: u64,
    pub capability_revision: u64,
    pub thermal_power_revision: u64,
    pub process_identity_revision: u64,
    pub low_power: bool,
    pub kill_switch: bool,
    pub sleeping: bool,
    pressure: SourceObservation<u16>,
    thermal: SourceObservation<u16>,
    interaction: SourceObservation<u16>,
}

impl Default for CycleContextSnapshot {
    fn default() -> Self {
        Self {
            id: SnapshotId::default(),
            cycle: 0,
            cut_started_mono_us: 0,
            cut_completed_mono_us: 0,
            workload_id: 0,
            capability_revision: 0,
            thermal_power_revision: 0,
            process_identity_revision: 0,
            low_power: false,
            kill_switch: false,
            sleeping: false,
            pressure: SourceObservation::default(),
            thermal: SourceObservation::default(),
            interaction: SourceObservation::default(),
        }
    }
}

impl CycleContextSnapshot {
    pub fn new(
        id: SnapshotId,
        workload_id: u64,
        capability_revision: u64,
        thermal_power_revision: u64,
    ) -> Self {
        Self {
            id,
            cycle: id.sequence,
            workload_id,
            capability_revision,
            thermal_power_revision,
            ..Self::default()
        }
    }

    pub fn with_cut_times(mut self, started_mono_us: u64, completed_mono_us: u64) -> Self {
        self.cut_started_mono_us = started_mono_us;
        self.cut_completed_mono_us = completed_mono_us.max(started_mono_us);
        self
    }

    pub fn with_pressure(mut self, observation: SourceObservation<u16>) -> Self {
        self.pressure = observation;
        self
    }

    pub fn with_thermal(mut self, observation: SourceObservation<u16>) -> Self {
        self.thermal = observation;
        self
    }

    pub fn with_interaction(mut self, observation: SourceObservation<u16>) -> Self {
        self.interaction = observation;
        self
    }

    pub fn pressure(&self) -> &SourceObservation<u16> {
        &self.pressure
    }

    pub fn thermal(&self) -> &SourceObservation<u16> {
        &self.thermal
    }

    pub fn interaction(&self) -> &SourceObservation<u16> {
        &self.interaction
    }

    pub fn pressure_q(&self) -> Option<u16> {
        self.pressure.fresh_value().copied()
    }

    pub fn thermal_q(&self) -> Option<u16> {
        self.thermal.fresh_value().copied()
    }

    pub fn interaction_q(&self) -> Option<u16> {
        self.interaction.fresh_value().copied()
    }

    pub fn identity(&self) -> SnapshotIdentity {
        SnapshotIdentity {
            snapshot_id: self.id,
            workload_id: self.workload_id,
            capability_revision: self.capability_revision,
            thermal_power_revision: self.thermal_power_revision,
            process_identity_revision: self.process_identity_revision,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SnapshotIdentity {
    pub snapshot_id: SnapshotId,
    pub workload_id: u64,
    pub capability_revision: u64,
    pub thermal_power_revision: u64,
    pub process_identity_revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotPublishError {
    SequenceExhausted,
}

#[derive(Debug)]
pub struct SnapshotPublisher {
    daemon_epoch: u64,
    next_sequence: u64,
    latest: Option<Arc<CycleContextSnapshot>>,
}

impl SnapshotPublisher {
    pub fn new(daemon_epoch: u64) -> Self {
        Self {
            daemon_epoch,
            next_sequence: 0,
            latest: None,
        }
    }

    pub fn with_next_sequence(daemon_epoch: u64, next_sequence: u64) -> Self {
        Self {
            daemon_epoch,
            next_sequence,
            latest: None,
        }
    }

    pub fn daemon_epoch(&self) -> u64 {
        self.daemon_epoch
    }

    pub fn next_id(&mut self) -> Result<SnapshotId, SnapshotPublishError> {
        let sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(SnapshotPublishError::SequenceExhausted)?;
        self.next_sequence = sequence;
        Ok(SnapshotId::new(self.daemon_epoch, sequence))
    }

    pub fn publish(
        &mut self,
        mut snapshot: CycleContextSnapshot,
    ) -> Result<Arc<CycleContextSnapshot>, SnapshotPublishError> {
        let id = self.next_id()?;
        snapshot.id = id;
        snapshot.cycle = id.sequence;
        let published = Arc::new(snapshot);
        self.latest = Some(Arc::clone(&published));
        Ok(published)
    }

    pub fn latest(&self) -> Option<Arc<CycleContextSnapshot>> {
        self.latest.as_ref().map(Arc::clone)
    }

    pub fn revision(&self) -> u64 {
        self.latest
            .as_ref()
            .map_or(0, |snapshot| snapshot.id.sequence)
    }
}
