//! apollo-optimizer — facade crate for binaries and integration tests.
//!
//! TEMPORARY (Commit 2 only): the `engine` module here is a re-export
//! of `apollo_engine::engine` so existing `crate::engine::*` paths in
//! src/bin/ keep compiling during the move. Commit 3 removes this
//! shim and bins migrate to `apollo_engine::engine::*`.

pub mod collector {
    pub use apollo_engine::collector::*;
}
pub mod dashboard;

pub mod engine {
    pub use apollo_engine::engine::*;
}

/// Convenient re-exports of the types most commonly needed by consumers
/// of this crate (`apollo-optimizerctl`, `apollo-menubar`, integration tests).
///
/// Import with: `use apollo_optimizer::prelude::*;`
pub mod prelude {
    pub use crate::engine::protocol::{DaemonRequest, DaemonResponse, PROTOCOL_VERSION};
    pub use crate::engine::types::{
        BlockerScore, CapabilityReport, DaemonStatus, FrozenEntry, LatencyTarget,
        OptimizationProfile, RuntimeMetrics, SafetyPolicy,
    };
}
