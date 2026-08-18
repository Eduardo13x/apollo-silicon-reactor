//! Perceptual Interaction Layer — agnostic to application, sensor and chip.
//!
//! The core holds observations of differing precision without flattening them:
//! an instrumented browser episode, an inferred interaction and an aggregate
//! window are all evidence, and they are not the same evidence. Every
//! source-specific concept — tabs, navigations, `PerformanceEventTiming`,
//! service workers, shell prompts, editor commands — lives behind an adapter.
//!
//! Observational only. Nothing here emits an action, grants credit or promotes
//! anything to Gold.

pub mod adapters;
pub mod capabilities;
pub mod store;
pub mod types;
pub mod validation;

pub use capabilities::PerceptualCapabilities;
pub use store::{PerceptualObservationStore, PerceptualStoreMetrics, PerceptualStorePersisted};
pub use types::{
    CorrelationState, InferenceBasis, InferredInteractionEpisode, InstrumentedInteractionEpisode,
    InteractionScope, LatencyComponent, LatencyComponentKind, MeasurementMode, MonotonicMillis,
    ObservationHeader, PerceptualEventEnvelope, PerceptualId, PerceptualMeasurement,
    PerceptualObservation, PerceptualQuality, PerceptualSourceKind, PerceptualTransportTrace,
    PerceptualWindowObservation, ProducerKind,
};
pub use validation::{validate_envelope, PerceptualValidationError, PERCEPTUAL_SCHEMA_VERSION};

#[cfg(test)]
mod architecture_tests {
    //! Invariants that fail if the core stops being agnostic.

    const CORE_FILES: [(&str, &str); 5] = [
        ("types.rs", include_str!("types.rs")),
        ("capabilities.rs", include_str!("capabilities.rs")),
        ("validation.rs", include_str!("validation.rs")),
        ("store.rs", include_str!("store.rs")),
        ("adapters/mod.rs", include_str!("adapters/mod.rs")),
    ];

    /// Browser *mechanism* vocabulary. The core may categorise a source as
    /// `BrowserChromium` — a bounded label is not a coupling — but it must never
    /// know how a browser works. Written as a test because a comment saying
    /// "keep this generic" does not fail a build.
    const FORBIDDEN_IN_CORE: [&str; 8] = [
        "WebFlow",
        "webflow",
        "interactionId",
        "PerformanceEventTiming",
        "service_worker",
        "serviceWorker",
        "tab_session",
        "navigation_id",
    ];

    #[test]
    fn the_core_never_names_a_browser_mechanism() {
        for (name, source) in CORE_FILES {
            for forbidden in FORBIDDEN_IN_CORE {
                assert!(
                    !source.contains(forbidden),
                    "{name} contains browser-specific term {forbidden:?}; \
                     it belongs in an adapter"
                );
            }
        }
    }

    #[test]
    fn chromium_appears_only_as_a_bounded_source_category() {
        // `BrowserChromium` is a closed label the core sorts observations by.
        // Any other mention would mean the core had learned browser behaviour.
        for (name, source) in CORE_FILES {
            for (index, line) in source.lines().enumerate() {
                let stripped = line
                    .replace("BrowserChromium", "")
                    .replace("browser-chromium", "");
                assert!(
                    !stripped.to_lowercase().contains("chromium"),
                    "{name}:{} mentions Chromium outside the bounded category: {line:?}",
                    index + 1
                );
            }
        }
    }

    #[test]
    fn no_public_core_field_is_named_after_a_browser() {
        for (name, source) in CORE_FILES {
            for line in source.lines() {
                let trimmed = line.trim();
                if !trimmed.starts_with("pub ") {
                    continue;
                }
                assert!(
                    !trimmed.contains("browser_"),
                    "{name}: public item {trimmed:?} carries a browser-specific name"
                );
            }
        }
    }

    #[test]
    fn the_core_does_not_import_a_source_specific_module() {
        for (name, source) in CORE_FILES {
            for line in source.lines() {
                let trimmed = line.trim();
                if !trimmed.starts_with("use ") {
                    continue;
                }
                for module in ["webflow_types", "webflow_controller", "webflow_native"] {
                    assert!(
                        !trimmed.contains(module),
                        "{name} imports {module}: the dependency must point the other way"
                    );
                }
            }
        }
    }

    #[test]
    fn observational_types_carry_no_reward_or_gold_vocabulary() {
        // A guard against the layer quietly acquiring authority it was built
        // without: observation feeds analysis, never credit.
        for (name, source) in CORE_FILES {
            for forbidden in ["fn grant_credit", "PairGold", "reward(", "promote_to_gold"] {
                assert!(
                    !source.contains(forbidden),
                    "{name} references {forbidden:?}: the perceptual core must stay observational"
                );
            }
        }
    }
}
