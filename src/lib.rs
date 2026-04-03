pub mod collector;
pub mod dashboard;
pub mod engine;

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
