//! The seam between source-specific producers and the agnostic core.
//!
//! Everything a source knows that the core must not — per-frame timing entries,
//! a browser's own interaction identifiers, background-worker lifecycles,
//! surfaces, page loads, shell prompts, editor commands — lives behind this
//! trait.

use super::capabilities::PerceptualCapabilities;
use super::types::{
    MonotonicMillis, PerceptualEventEnvelope, PerceptualObservation, PerceptualSourceKind,
};
use super::validation::PerceptualValidationError;

/// Health of one adapter, for the doctor. Deliberately not a single boolean:
/// "no data" and "rejecting everything" need different remedies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdapterHealth {
    pub accepted_total: u64,
    pub rejected_total: u64,
    pub last_observation_at_ms: u64,
    pub legacy_contract_total: u64,
}

impl AdapterHealth {
    pub fn has_data(&self) -> bool {
        self.accepted_total > 0
    }

    /// Rejecting everything it receives is a distinct failure from silence.
    pub fn is_rejecting_everything(&self) -> bool {
        self.accepted_total == 0 && self.rejected_total > 0
    }
}

pub trait PerceptualAdapter {
    fn source_kind(&self) -> PerceptualSourceKind;

    fn capabilities(&self) -> PerceptualCapabilities;

    fn validate(&self, envelope: &PerceptualEventEnvelope)
        -> Result<(), PerceptualValidationError>;

    /// Turn one producer payload into zero or more core observations. Returning
    /// an empty vector is legitimate: not every payload describes an
    /// interaction, and inventing one would be worse than reporting none.
    fn normalize(
        &mut self,
        envelope: PerceptualEventEnvelope,
        now: MonotonicMillis,
    ) -> Vec<PerceptualObservation>;

    fn health(&self) -> AdapterHealth;
}

pub mod macos_observer;
pub mod synthetic;
