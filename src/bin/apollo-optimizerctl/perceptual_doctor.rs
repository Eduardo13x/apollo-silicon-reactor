//! Single diagnostic for the Perceptual Interaction Layer.
//!
//! Reports which hop is broken rather than a bare "0 samples": a stale
//! extension, an unreachable bridge and a genuinely idle browser all produce
//! the same zeros, and telling them apart is the whole point of this command.

use apollo_engine::engine::types::RuntimeMetrics;

/// Terminal verdict. Ordered by severity so the worst finding wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerceptualVerdict {
    ReadyFor0b,
    ObservationPartial,
    /// Lifecycle events arrive but the content script has produced no vitals.
    CollectorSilent,
    NoData,
    StaleExtension,
    SchemaMismatch,
    TransportBroken,
}

impl PerceptualVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadyFor0b => "READY_FOR_0B",
            Self::ObservationPartial => "OBSERVATION_PARTIAL",
            Self::CollectorSilent => "COLLECTOR_SILENT",
            Self::NoData => "NO_DATA",
            Self::StaleExtension => "STALE_EXTENSION",
            Self::SchemaMismatch => "SCHEMA_MISMATCH",
            Self::TransportBroken => "TRANSPORT_BROKEN",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub hop: &'static str,
    pub ok: bool,
    pub detail: String,
}

impl Check {
    fn new(hop: &'static str, ok: bool, detail: impl Into<String>) -> Self {
        Self {
            hop,
            ok,
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub checks: Vec<Check>,
    pub verdict: PerceptualVerdict,
}

/// Derive the report from published metrics alone, so it is testable without a
/// daemon and cannot disagree with what the dashboard shows.
pub fn diagnose(m: &RuntimeMetrics) -> Report {
    let mut checks = Vec::new();
    let status = m.webflow_extension_status.as_str();
    let any_event = m.webflow_accepted_v1_total > 0 || m.webflow_accepted_v2_total > 0;

    checks.push(Check::new(
        "extension",
        any_event,
        if status.is_empty() {
            "no producer has ever reported".to_string()
        } else {
            format!(
                "status={status} version={} v1={} v2={}",
                if m.webflow_extension_version.is_empty() {
                    "-"
                } else {
                    m.webflow_extension_version.as_str()
                },
                m.webflow_accepted_v1_total,
                m.webflow_accepted_v2_total
            )
        },
    ));
    checks.push(Check::new(
        "schema",
        m.webflow_schema_rejected_total == 0,
        format!("rejected={}", m.webflow_schema_rejected_total),
    ));
    checks.push(Check::new(
        "vitals",
        m.browser_latency_samples > 0,
        format!(
            "samples={} mode={}",
            m.browser_latency_samples,
            if m.webflow_mode.is_empty() {
                "-"
            } else {
                m.webflow_mode.as_str()
            }
        ),
    ));
    checks.push(Check::new(
        "transport",
        m.webflow_transport_samples > 0,
        format!(
            "samples={} client_p95={} sw_wake_p95={} cold_starts={}",
            m.webflow_transport_samples,
            m.webflow_transport_client_p95_ms
                .map_or_else(|| "-".to_string(), |v| format!("{v:.0}ms")),
            m.webflow_transport_sw_wake_p95_ms
                .map_or_else(|| "-".to_string(), |v| format!("{v:.0}ms")),
            m.webflow_transport_cold_starts
        ),
    ));
    checks.push(Check::new(
        "interactions",
        m.browser_interaction_samples > 0,
        format!(
            "samples={} inp={} dropped={}",
            m.browser_interaction_samples,
            m.browser_inp_estimate_ms
                .map_or_else(|| "-".to_string(), |v| format!("{v:.0}ms")),
            m.browser_interactions_dropped
        ),
    ));
    let component_total = m
        .browser_input_delay_total_ms
        .saturating_add(m.browser_processing_total_ms)
        .saturating_add(m.browser_presentation_total_ms);
    checks.push(Check::new(
        "components",
        component_total > 0,
        format!(
            "input={}ms processing={}ms presentation={}ms",
            m.browser_input_delay_total_ms,
            m.browser_processing_total_ms,
            m.browser_presentation_total_ms
        ),
    ));

    // Worst finding wins, and each verdict names the hop that produced it.
    let verdict = if m.webflow_schema_rejected_total > 0 && !any_event {
        PerceptualVerdict::SchemaMismatch
    } else if !any_event {
        PerceptualVerdict::NoData
    } else if m.webflow_accepted_v2_total == 0 {
        PerceptualVerdict::StaleExtension
    } else if m.browser_latency_samples == 0 {
        // Events reach the daemon, so the transport is intact end to end; the
        // content script simply has not reported. Pages loaded before the last
        // extension reload keep the injection list they were opened with.
        PerceptualVerdict::CollectorSilent
    } else if m.webflow_transport_samples == 0 {
        PerceptualVerdict::TransportBroken
    } else if m.browser_interaction_samples == 0 || component_total == 0 {
        PerceptualVerdict::ObservationPartial
    } else {
        PerceptualVerdict::ReadyFor0b
    };
    Report { checks, verdict }
}

pub fn render(report: &Report) -> String {
    let mut out = String::from("Apollo perceptual-doctor\n\n");
    for check in &report.checks {
        out.push_str(&format!(
            "  [{}] {:<13} {}\n",
            if check.ok { "ok" } else { "--" },
            check.hop,
            check.detail
        ));
    }
    out.push_str(&format!("\n  verdict: {}\n", report.verdict.as_str()));
    out.push_str(match report.verdict {
        PerceptualVerdict::ReadyFor0b => "  the observational circuit is complete end to end.\n",
        PerceptualVerdict::ObservationPartial => {
            "  events arrive but interaction evidence is incomplete; the browser\n  \
             may simply be idle. Interact with a page and re-run.\n"
        }
        PerceptualVerdict::NoData => {
            "  no producer has reported. Load the extension and grant it host\n  \
             permissions from its toolbar action.\n"
        }
        PerceptualVerdict::StaleExtension => {
            "  a v1 extension is reporting. Reload it at chrome://extensions so\n  \
             the corrected per-interaction collector takes over.\n"
        }
        PerceptualVerdict::SchemaMismatch => {
            "  payloads are refused as uninterpretable. Extension and daemon\n  \
             disagree on the wire schema; redeploy both from the same commit.\n"
        }
        PerceptualVerdict::CollectorSilent => {
            "  navigation events arrive, so the whole transport works, but the\n  \
             content script has reported no vitals. Tabs opened before the last\n  \
             extension reload keep their original injection: reload the open\n  \
             tabs (or open a new one) and interact with the page.\n"
        }
        PerceptualVerdict::TransportBroken => {
            "  events arrive without transport stamps. The bridge or the service\n  \
             worker is dropping them; check the native host manifest.\n"
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics() -> RuntimeMetrics {
        RuntimeMetrics::default()
    }

    #[test]
    fn a_silent_producer_is_no_data_not_transport_broken() {
        let report = diagnose(&metrics());
        assert_eq!(report.verdict, PerceptualVerdict::NoData);
        assert!(report.checks.iter().any(|c| c.hop == "extension" && !c.ok));
    }

    #[test]
    fn the_deployed_v1_extension_is_named_as_the_cause() {
        // The exact production case: events flow, nothing corrected can arrive.
        let mut m = metrics();
        m.webflow_extension_status = "v1-stale".to_string();
        m.webflow_accepted_v1_total = 120;
        let report = diagnose(&m);
        assert_eq!(report.verdict, PerceptualVerdict::StaleExtension);
        assert!(render(&report).contains("chrome://extensions"));
    }

    #[test]
    fn a_refused_schema_outranks_plain_absence() {
        let mut m = metrics();
        m.webflow_schema_rejected_total = 4;
        assert_eq!(diagnose(&m).verdict, PerceptualVerdict::SchemaMismatch);
    }

    #[test]
    fn lifecycle_only_traffic_is_a_silent_collector_not_a_broken_transport() {
        // The production case: nine navigation events landed, so every hop
        // works, yet the content script had reported nothing.
        let mut m = metrics();
        m.webflow_accepted_v2_total = 9;
        m.webflow_mode = "lifecycle".to_string();
        let report = diagnose(&m);
        assert_eq!(report.verdict, PerceptualVerdict::CollectorSilent);
        assert!(render(&report).contains("reload the open"));
    }

    #[test]
    fn vitals_without_transport_stamps_are_transport_broken() {
        let mut m = metrics();
        m.webflow_accepted_v2_total = 40;
        m.browser_latency_samples = 12;
        assert_eq!(diagnose(&m).verdict, PerceptualVerdict::TransportBroken);
    }

    #[test]
    fn an_idle_browser_is_partial_rather_than_broken() {
        let mut m = metrics();
        m.webflow_accepted_v2_total = 40;
        m.browser_latency_samples = 12;
        m.webflow_transport_samples = 40;
        let report = diagnose(&m);
        assert_eq!(report.verdict, PerceptualVerdict::ObservationPartial);
        assert!(render(&report).contains("idle"));
    }

    #[test]
    fn a_complete_circuit_is_ready_for_0b() {
        let mut m = metrics();
        m.webflow_accepted_v2_total = 40;
        m.browser_latency_samples = 64;
        m.webflow_transport_samples = 40;
        m.browser_interaction_samples = 312;
        m.browser_inp_estimate_ms = Some(184.0);
        m.browser_input_delay_total_ms = 1_200;
        m.browser_processing_total_ms = 8_400;
        m.browser_presentation_total_ms = 2_400;
        let report = diagnose(&m);
        assert_eq!(report.verdict, PerceptualVerdict::ReadyFor0b);
        assert!(report.checks.iter().all(|c| c.ok), "{:?}", report.checks);
    }

    #[test]
    fn every_check_names_a_hop_so_a_failure_is_locatable() {
        let report = diagnose(&metrics());
        let hops: Vec<_> = report.checks.iter().map(|c| c.hop).collect();
        assert_eq!(
            hops,
            vec![
                "extension",
                "schema",
                "vitals",
                "transport",
                "interactions",
                "components"
            ]
        );
    }
}
