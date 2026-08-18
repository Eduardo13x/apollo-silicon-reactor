//! What a producer can actually observe.
//!
//! Capabilities are **declared**, never inferred from the adapter's name. A
//! browser extension inside the page sees a paint boundary; an observer outside
//! an opaque application does not, and must not be asked to invent one. The
//! distinction this file exists to preserve: an absent field means *not
//! observed*, which is a different statement from zero.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PerceptualCapabilities {
    pub has_started_event: bool,
    pub has_completed_event: bool,
    pub has_total_duration: bool,
    pub has_latency_breakdown: bool,
    pub has_semantic_operation: bool,
    pub has_surface_identity: bool,
    pub has_client_monotonic_clock: bool,
    pub has_response_signal: bool,
    pub supports_transport_trace: bool,
}

impl PerceptualCapabilities {
    /// A producer instrumented inside the application it measures.
    pub const fn instrumented() -> Self {
        Self {
            has_started_event: false,
            has_completed_event: true,
            has_total_duration: true,
            has_latency_breakdown: true,
            has_semantic_operation: false,
            has_surface_identity: true,
            has_client_monotonic_clock: true,
            has_response_signal: true,
            supports_transport_trace: true,
        }
    }

    /// An observer outside the application: it sees that something happened and
    /// roughly how the machine responded, never the internal stages.
    pub const fn external_observer() -> Self {
        Self {
            has_started_event: false,
            has_completed_event: false,
            has_total_duration: false,
            has_latency_breakdown: false,
            has_semantic_operation: false,
            has_surface_identity: false,
            has_client_monotonic_clock: false,
            has_response_signal: true,
            supports_transport_trace: false,
        }
    }

    /// A declaration is incoherent when it claims a derived ability without the
    /// ability it derives from. Catching this at the boundary stops a producer
    /// from promising precision it cannot deliver.
    pub fn is_coherent(self) -> bool {
        if self.has_latency_breakdown && !self.has_total_duration {
            return false;
        }
        if self.has_total_duration && !(self.has_completed_event || self.has_response_signal) {
            return false;
        }
        true
    }

    /// Every capability this producer lacks, as stable labels. Used by the
    /// dashboard to print "unsupported" rather than a zero.
    pub fn missing(self) -> Vec<&'static str> {
        let mut out = Vec::new();
        for (present, label) in [
            (self.has_started_event, "started"),
            (self.has_completed_event, "completed"),
            (self.has_total_duration, "total"),
            (self.has_latency_breakdown, "breakdown"),
            (self.has_semantic_operation, "semantic"),
            (self.has_surface_identity, "surface"),
            (self.has_client_monotonic_clock, "client-clock"),
            (self.has_response_signal, "response"),
            (self.supports_transport_trace, "transport"),
        ] {
            if !present {
                out.push(label);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_breakdown_without_a_total_is_incoherent() {
        let claim = PerceptualCapabilities {
            has_latency_breakdown: true,
            has_total_duration: false,
            ..PerceptualCapabilities::default()
        };
        assert!(!claim.is_coherent());
    }

    #[test]
    fn a_total_needs_something_that_could_have_measured_it() {
        let claim = PerceptualCapabilities {
            has_total_duration: true,
            has_completed_event: false,
            has_response_signal: false,
            ..PerceptualCapabilities::default()
        };
        assert!(!claim.is_coherent());
    }

    #[test]
    fn the_two_presets_are_coherent_and_differ_where_it_matters() {
        assert!(PerceptualCapabilities::instrumented().is_coherent());
        assert!(PerceptualCapabilities::external_observer().is_coherent());
        assert!(PerceptualCapabilities::instrumented().has_latency_breakdown);
        assert!(
            !PerceptualCapabilities::external_observer().has_latency_breakdown,
            "an outside observer must never claim internal stages"
        );
    }

    #[test]
    fn missing_capabilities_are_reportable_so_a_zero_is_never_printed_instead() {
        let missing = PerceptualCapabilities::external_observer().missing();
        assert!(missing.contains(&"breakdown"));
        assert!(missing.contains(&"total"));
        assert!(!missing.contains(&"response"));
    }
}
