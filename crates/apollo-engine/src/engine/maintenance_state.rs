// crates/apollo-engine/src/engine/maintenance_state.rs
//! Maintenance Purge Gate state — opportunistic non-crisis purge orchestration.
//!
//! See docs/superpowers/specs/2026-05-10-maintenance-purge-design.md
//!
//! Asymmetric cooldown: survival_tick writes last_any_purge_at but does not
//! read it (survival is physical-crisis sovereign). maintenance_tick reads
//! and writes (yields to anything recent).

use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaintenanceState {
    #[serde(skip)]
    pub swap_delta_window: SwapDeltaWindow,

    #[serde(default)]
    pub last_any_purge_at: Option<SystemTime>,

    #[serde(default)]
    pub last_cli_purge_at: Option<SystemTime>,

    #[serde(skip)]
    pub last_wake_at: Option<Instant>,
}

impl MaintenanceState {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SwapDeltaWindow {
    samples: VecDeque<(SystemTime, f64)>,
}

impl SwapDeltaWindow {
    pub const CAP: usize = 45;

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
