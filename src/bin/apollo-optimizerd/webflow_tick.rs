use std::collections::VecDeque;

use apollo_engine::engine::webflow_controller::{
    WebFlowController, WebFlowCounters, WebFlowIntent, WebFlowRolloutPhase, WebFlowTickInput,
    WebWorldObservation, MAX_WEBFLOW_EVENT_AGE_MS,
};
use apollo_engine::engine::webflow_types::{
    OpaqueId, ReceivedWebFlowEvent, WebFlowEvent, WebFlowMetrics, WebFlowPhase, WebFlowSource,
    WebFlowTransport, WEBFLOW_SCHEMA_VERSION,
};

/// Bounded per-metric sample ring for browser Web Vitals. The extension is an
/// untrusted producer, so the window is fixed-size and never grows with input.
///
/// The `event_duration` ring holds individual `PerformanceEventTiming.duration`
/// values, **not** interactions: it cannot yield INP, only the tail of single
/// entries above the collector's threshold.
const MAX_VITALS_SAMPLES: usize = 64;

/// Nearest-rank p95 over a bounded sample window.
///
/// ponytail: nearest-rank on a 64-sample copy, not a streaming quantile
/// sketch. Upgrade path: swap for a t-digest if the window ever needs to grow
/// past a few hundred samples.
fn vitals_p95(samples: &[u32]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = ((sorted.len() as f64) * 0.95).ceil() as usize;
    let index = rank.clamp(1, sorted.len()) - 1;
    Some(f64::from(sorted[index]))
}

#[derive(Debug, Default)]
struct VitalsWindow {
    lcp_ms: VecDeque<u32>,
    event_duration_ms: VecDeque<u32>,
    /// Corrected per-interaction series. Kept apart from the entry-duration
    /// ring above so the two can never be folded into one reading.
    inp_estimate_ms: VecDeque<u32>,
    interaction_count: u64,
    interactions_dropped: u64,
    input_delay_total_ms: u64,
    processing_total_ms: u64,
    presentation_total_ms: u64,
    /// Browser-clock transport segments. Same-clock differences only.
    client_segment_ms: VecDeque<u32>,
    service_worker_wake_ms: VecDeque<u32>,
    cold_starts: u64,
    transport_samples: u64,
    max_tab_queue_depth: u32,
}

impl VitalsWindow {
    fn record(&mut self, metrics: &WebFlowMetrics) {
        for (value, ring) in [
            (metrics.lcp_ms, &mut self.lcp_ms),
            (metrics.event_duration_ms, &mut self.event_duration_ms),
            (metrics.inp_estimate_ms, &mut self.inp_estimate_ms),
        ] {
            let Some(value) = value else {
                continue;
            };
            if ring.len() >= MAX_VITALS_SAMPLES {
                ring.pop_front();
            }
            ring.push_back(value);
        }
        self.interaction_count = self
            .interaction_count
            .saturating_add(u64::from(metrics.interaction_count.unwrap_or(0)));
        self.interactions_dropped = self
            .interactions_dropped
            .saturating_add(u64::from(metrics.interactions_dropped.unwrap_or(0)));
        self.input_delay_total_ms = self
            .input_delay_total_ms
            .saturating_add(u64::from(metrics.input_delay_total_ms.unwrap_or(0)));
        self.processing_total_ms = self
            .processing_total_ms
            .saturating_add(u64::from(metrics.processing_total_ms.unwrap_or(0)));
        self.presentation_total_ms = self
            .presentation_total_ms
            .saturating_add(u64::from(metrics.presentation_total_ms.unwrap_or(0)));
    }

    fn clear(&mut self) {
        self.lcp_ms.clear();
        self.event_duration_ms.clear();
        self.inp_estimate_ms.clear();
        self.interaction_count = 0;
        self.interactions_dropped = 0;
        self.input_delay_total_ms = 0;
        self.processing_total_ms = 0;
        self.presentation_total_ms = 0;
        self.client_segment_ms.clear();
        self.service_worker_wake_ms.clear();
        self.cold_starts = 0;
        self.transport_samples = 0;
        self.max_tab_queue_depth = 0;
    }

    fn record_transport(&mut self, transport: &WebFlowTransport) {
        let mut observed = false;
        for (value, ring) in [
            (transport.client_segment_ms(), &mut self.client_segment_ms),
            (
                transport.service_worker_wake_ms(),
                &mut self.service_worker_wake_ms,
            ),
        ] {
            let Some(value) = value else { continue };
            observed = true;
            if ring.len() >= MAX_VITALS_SAMPLES {
                ring.pop_front();
            }
            ring.push_back(value.min(u64::from(u32::MAX)) as u32);
        }
        if transport.service_worker_cold_start == Some(true) {
            self.cold_starts = self.cold_starts.saturating_add(1);
        }
        self.max_tab_queue_depth = self
            .max_tab_queue_depth
            .max(transport.tab_queue_depth.unwrap_or(0));
        if observed {
            self.transport_samples = self.transport_samples.saturating_add(1);
        }
    }

    fn client_segment_p95_ms(&self) -> Option<f64> {
        vitals_p95(&self.client_segment_ms.iter().copied().collect::<Vec<_>>())
    }

    fn service_worker_wake_p95_ms(&self) -> Option<f64> {
        vitals_p95(
            &self
                .service_worker_wake_ms
                .iter()
                .copied()
                .collect::<Vec<_>>(),
        )
    }

    fn inp_estimate_p95_ms(&self) -> Option<f64> {
        vitals_p95(&self.inp_estimate_ms.iter().copied().collect::<Vec<_>>())
    }

    fn lcp_p95_ms(&self) -> Option<f64> {
        vitals_p95(&self.lcp_ms.iter().copied().collect::<Vec<_>>())
    }

    fn event_duration_tail_ms(&self) -> Option<f64> {
        vitals_p95(&self.event_duration_ms.iter().copied().collect::<Vec<_>>())
    }

    /// Widest sample count backing the published p95s, so a reader can tell a
    /// three-sample guess from a settled measurement.
    fn samples(&self) -> u64 {
        self.lcp_ms.len().max(self.event_duration_ms.len()) as u64
    }
}

pub fn is_supported_browser(name: &str) -> bool {
    const BROWSERS: &[&str] = &[
        "Brave Browser",
        "Google Chrome",
        "Microsoft Edge",
        "Chromium",
        "Arc",
        "Vivaldi",
        "Opera",
    ];
    let name = name.trim();
    BROWSERS.iter().any(|browser| {
        name == *browser
            || name
                .strip_prefix(browser)
                .is_some_and(|suffix| suffix.starts_with(' ') || suffix.starts_with(" Helper"))
    })
}

#[derive(Debug, Clone, Copy)]
pub struct WebFlowCycleInput<'a> {
    pub now_ms: u64,
    pub events: &'a [ReceivedWebFlowEvent],
    pub foreground_browser: bool,
    pub foreground_identity_available: bool,
    pub interaction_active: bool,
    pub browser_socket_active: bool,
    pub pressure_constrained: bool,
    pub thermal_constrained: bool,
    pub low_power: bool,
    pub sleeping: bool,
    pub kill_switch: bool,
    pub session_revision: u64,
    pub control_p95_ms: f64,
    pub profile_matches: bool,
    pub failures_total: u64,
    pub rollback_failures_total: u64,
    pub protected_actions_total: u64,
}

#[derive(Debug, Clone)]
pub struct WebFlowCycleOutput {
    pub observation: WebWorldObservation,
    pub intent: Option<WebFlowIntent>,
    pub admitted: bool,
    pub mesh_events: Vec<WebFlowEvent>,
    pub counters: WebFlowCounters,
    pub rollout: WebFlowRolloutPhase,
    pub valid_health_cycles: u64,
    pub rollout_blocker: &'static str,
    /// p95 over the bounded Web Vitals window, `None` until the extension has
    /// reported at least one sample of that metric.
    pub lcp_p95_ms: Option<f64>,
    pub event_duration_tail_ms: Option<f64>,
    pub inp_estimate_ms: Option<f64>,
    pub interaction_count: u64,
    pub interactions_dropped: u64,
    pub input_delay_total_ms: u64,
    pub processing_total_ms: u64,
    pub presentation_total_ms: u64,
    pub transport_client_segment_p95_ms: Option<f64>,
    pub transport_sw_wake_p95_ms: Option<f64>,
    pub transport_cold_starts: u64,
    pub transport_samples: u64,
    pub transport_max_queue_depth: u32,
    /// Samples behind those p95s.
    pub vitals_samples: u64,
}

pub struct WebFlowRuntime {
    controller: WebFlowController,
    inference_sequence: u64,
    inference_active: bool,
    last_exact_at_ms: Option<u64>,
    rollout: WebFlowRolloutPhase,
    valid_health_cycles: u64,
    baseline_failures: Option<(u64, u64, u64)>,
    rollout_blocker: &'static str,
    vitals: VitalsWindow,
    vitals_session_revision: u64,
}

impl WebFlowRuntime {
    pub fn new(rollout: WebFlowRolloutPhase) -> Self {
        Self {
            controller: WebFlowController::new(rollout),
            inference_sequence: 0,
            inference_active: false,
            last_exact_at_ms: None,
            rollout,
            valid_health_cycles: 0,
            baseline_failures: None,
            rollout_blocker: if rollout == WebFlowRolloutPhase::Shadow {
                "collecting-samples"
            } else {
                "configured"
            },
            vitals: VitalsWindow::default(),
            vitals_session_revision: 0,
        }
    }

    pub fn needs_daemon_inference_probe(&self, now_ms: u64) -> bool {
        self.last_exact_at_ms
            .is_none_or(|last| now_ms.saturating_sub(last) > 10_000)
    }

    pub fn tick(&mut self, input: WebFlowCycleInput<'_>) -> WebFlowCycleOutput {
        self.observe_health(input);
        let mut events = input.events.to_vec();
        let newest_exact_at_ms = events
            .iter()
            .filter(|received| {
                received.event.source != WebFlowSource::DaemonInference
                    && received.received_at_ms != 0
                    && received.received_at_ms <= input.now_ms
                    && input.now_ms.saturating_sub(received.received_at_ms)
                        <= MAX_WEBFLOW_EVENT_AGE_MS
                    && received.event.validate().is_ok()
            })
            .map(|received| received.received_at_ms)
            .max();
        let exact_present = newest_exact_at_ms.is_some();
        // Vitals are session- and liveness-scoped exactly like the observation:
        // a sleep, a kill-switch flip, or a new login session drops the window
        // rather than letting stale page timings age into the published p95.
        if input.sleeping
            || input.kill_switch
            || input.session_revision != self.vitals_session_revision
        {
            self.vitals.clear();
            self.vitals_session_revision = input.session_revision;
        }
        for received in &events {
            // Only fresh, schema-valid, first-party measurements. Daemon
            // inference carries no timings and must never fabricate a vital.
            if received.event.source == WebFlowSource::DaemonInference
                || received.received_at_ms == 0
                || received.received_at_ms > input.now_ms
                || input.now_ms.saturating_sub(received.received_at_ms) > MAX_WEBFLOW_EVENT_AGE_MS
                || received.event.validate().is_err()
            {
                continue;
            }
            self.vitals.record(&received.event.metrics);
            self.vitals.record_transport(&received.event.transport);
        }
        if let Some(received_at_ms) = newest_exact_at_ms {
            self.last_exact_at_ms = Some(
                self.last_exact_at_ms
                    .map_or(received_at_ms, |previous| previous.max(received_at_ms)),
            );
        }
        if input.sleeping || input.kill_switch {
            self.inference_active = false;
        }
        let exact_fresh = self
            .last_exact_at_ms
            .is_some_and(|last| input.now_ms.saturating_sub(last) <= 10_000);
        let infer_now = !exact_present
            && !exact_fresh
            && input.foreground_browser
            && input.foreground_identity_available
            && input.interaction_active
            && input.browser_socket_active
            && !input.sleeping
            && !input.kill_switch;
        if infer_now && !self.inference_active {
            self.inference_sequence = self.inference_sequence.saturating_add(1).max(1);
            let marker = self.inference_sequence.to_le_bytes()[0].max(1);
            events.push(ReceivedWebFlowEvent {
                event: WebFlowEvent {
                    schema_version: WEBFLOW_SCHEMA_VERSION,
                    browser_session_id: OpaqueId::new([0xD1; 16]).expect("static inference id"),
                    tab_session_id: OpaqueId::new([0xD2; 16]).expect("static inference id"),
                    navigation_id: OpaqueId::new([marker; 16]).expect("nonzero inference id"),
                    sequence: self.inference_sequence,
                    phase: WebFlowPhase::Started,
                    source: WebFlowSource::DaemonInference,
                    site_bucket: None,
                    metrics: WebFlowMetrics::default(),
                    // Daemon-synthesised: no transport hops to record.
                    transport: WebFlowTransport::default(),
                },
                received_at_ms: input.now_ms,
            });
        }
        self.inference_active = infer_now;

        let mesh_events = events
            .iter()
            .map(|received| received.event.clone())
            .collect();
        let output = self.controller.tick(
            WebFlowTickInput {
                now_ms: input.now_ms,
                foreground_browser: input.foreground_browser,
                identity_available: input.foreground_identity_available,
                pressure_constrained: input.pressure_constrained,
                thermal_constrained: input.thermal_constrained,
                low_power: input.low_power,
                sleeping: input.sleeping,
                kill_switch: input.kill_switch,
                session_revision: input.session_revision,
            },
            events,
        );
        WebFlowCycleOutput {
            observation: output.observation,
            intent: output.intent,
            admitted: output.admitted,
            mesh_events,
            counters: output.counters,
            rollout: self.rollout,
            valid_health_cycles: self.valid_health_cycles,
            rollout_blocker: self.rollout_blocker,
            lcp_p95_ms: self.vitals.lcp_p95_ms(),
            event_duration_tail_ms: self.vitals.event_duration_tail_ms(),
            inp_estimate_ms: self.vitals.inp_estimate_p95_ms(),
            interaction_count: self.vitals.interaction_count,
            interactions_dropped: self.vitals.interactions_dropped,
            input_delay_total_ms: self.vitals.input_delay_total_ms,
            processing_total_ms: self.vitals.processing_total_ms,
            presentation_total_ms: self.vitals.presentation_total_ms,
            transport_client_segment_p95_ms: self.vitals.client_segment_p95_ms(),
            transport_sw_wake_p95_ms: self.vitals.service_worker_wake_p95_ms(),
            transport_cold_starts: self.vitals.cold_starts,
            transport_samples: self.vitals.transport_samples,
            transport_max_queue_depth: self.vitals.max_tab_queue_depth,
            vitals_samples: self.vitals.samples(),
        }
    }

    fn observe_health(&mut self, input: WebFlowCycleInput<'_>) {
        if self.rollout != WebFlowRolloutPhase::Shadow {
            return;
        }
        let current_failures = (
            input.failures_total,
            input.rollback_failures_total,
            input.protected_actions_total,
        );
        let baseline = *self.baseline_failures.get_or_insert(current_failures);
        if current_failures.0 > baseline.0
            || current_failures.1 > baseline.1
            || current_failures.2 > baseline.2
        {
            self.valid_health_cycles = 0;
            self.baseline_failures = Some(current_failures);
            self.rollout_blocker = "safety-failure";
            return;
        }
        let healthy = input.profile_matches
            && input.control_p95_ms.is_finite()
            && input.control_p95_ms < 75.0
            && !input.pressure_constrained
            && !input.thermal_constrained
            && !input.low_power
            && !input.sleeping
            && !input.kill_switch;
        if !healthy {
            self.rollout_blocker = if !input.profile_matches {
                "profile-mismatch"
            } else if !input.control_p95_ms.is_finite() || input.control_p95_ms >= 75.0 {
                "cycle-p95"
            } else {
                "system-constrained"
            };
            return;
        }
        self.valid_health_cycles = self.valid_health_cycles.saturating_add(1);
        self.rollout_blocker = "collecting-samples";
        if self.valid_health_cycles >= 500 {
            self.rollout = WebFlowRolloutPhase::Active;
            self.controller.set_rollout(WebFlowRolloutPhase::Active);
            self.rollout_blocker = "ready";
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::webflow_controller::WebFlowRolloutPhase;
    use apollo_engine::engine::webflow_types::{
        OpaqueId, ReceivedWebFlowEvent, WebFlowEvent, WebFlowMetrics, WebFlowPhase, WebFlowSource,
        WEBFLOW_SCHEMA_VERSION,
    };

    fn exact(sequence: u64) -> ReceivedWebFlowEvent {
        ReceivedWebFlowEvent {
            event: WebFlowEvent {
                schema_version: WEBFLOW_SCHEMA_VERSION,
                browser_session_id: OpaqueId::new([1; 16]).unwrap(),
                tab_session_id: OpaqueId::new([2; 16]).unwrap(),
                navigation_id: OpaqueId::new([3; 16]).unwrap(),
                sequence,
                phase: WebFlowPhase::Started,
                source: WebFlowSource::ExtensionLifecycle,
                site_bucket: None,
                metrics: WebFlowMetrics::default(),
                transport: WebFlowTransport::default(),
            },
            received_at_ms: 100,
        }
    }

    fn input<'a>(events: &'a [ReceivedWebFlowEvent]) -> WebFlowCycleInput<'a> {
        WebFlowCycleInput {
            now_ms: 110,
            events,
            foreground_browser: true,
            foreground_identity_available: true,
            interaction_active: true,
            browser_socket_active: false,
            pressure_constrained: false,
            thermal_constrained: false,
            low_power: false,
            sleeping: false,
            kill_switch: false,
            session_revision: 1,
            control_p95_ms: 35.0,
            profile_matches: true,
            failures_total: 0,
            rollback_failures_total: 0,
            protected_actions_total: 0,
        }
    }

    #[test]
    fn exact_event_produces_one_snapshot_observation_and_intent() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        let events = [exact(1)];
        let output = runtime.tick(input(&events));
        assert_eq!(output.observation.active_navigations, 1);
        assert_eq!(
            output.observation.source,
            Some(WebFlowSource::ExtensionLifecycle)
        );
        assert!(output.intent.is_some());
        assert_eq!(output.mesh_events.len(), 1);
    }

    #[test]
    fn daemon_inference_requires_foreground_interaction_and_socket_activity() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        let mut inferred = input(&[]);
        inferred.browser_socket_active = true;
        let output = runtime.tick(inferred);
        assert_eq!(
            output.observation.source,
            Some(WebFlowSource::DaemonInference)
        );
        assert!(output.intent.is_some());

        let mut quiet = input(&[]);
        quiet.browser_socket_active = true;
        quiet.interaction_active = false;
        let quiet_output = runtime.tick(quiet);
        assert_eq!(quiet_output.observation.accepted_events, 0);
    }

    #[test]
    fn active_daemon_inference_keeps_sampling_until_exact_evidence_arrives() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        let mut inferred = input(&[]);
        inferred.browser_socket_active = true;
        runtime.tick(inferred);
        assert!(runtime.needs_daemon_inference_probe(200));
    }

    #[test]
    fn sleep_invalidates_active_webflow_episode() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Active);
        runtime.tick(input(&[exact(1)]));
        let mut sleeping = input(&[]);
        sleeping.sleeping = true;
        let output = runtime.tick(sleeping);
        assert!(output.intent.is_none());
        assert_eq!(output.observation.active_navigations, 0);
    }

    #[test]
    fn fresh_exact_episode_suppresses_daemon_inference_between_phases() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Active);
        runtime.tick(input(&[exact(1)]));
        let mut next = input(&[]);
        next.now_ms = 500;
        next.browser_socket_active = true;
        let output = runtime.tick(next);
        assert!(output.mesh_events.is_empty());
        assert_eq!(output.observation.active_navigations, 1);
        assert_eq!(
            output.observation.source,
            Some(WebFlowSource::ExtensionLifecycle)
        );
    }

    #[test]
    fn stale_exact_event_does_not_suppress_fresh_daemon_inference() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Active);
        let mut stale = exact(1);
        stale.received_at_ms = 100;
        let events = [stale];
        let mut cycle = input(&events);
        cycle.now_ms = 2_101;
        cycle.browser_socket_active = true;

        let output = runtime.tick(cycle);

        assert_eq!(output.counters.stale, 1);
        assert_eq!(
            output.observation.source,
            Some(WebFlowSource::DaemonInference)
        );
        assert!(output.admitted);
    }

    fn with_vitals(
        sequence: u64,
        lcp_ms: Option<u32>,
        event_duration_ms: Option<u32>,
    ) -> ReceivedWebFlowEvent {
        let mut received = exact(sequence);
        received.event.source = WebFlowSource::ExtensionVitals;
        received.event.metrics = WebFlowMetrics {
            lcp_ms,
            event_duration_ms,
            ..WebFlowMetrics::default()
        };
        received
    }

    #[test]
    fn reported_vitals_reach_the_published_p95_instead_of_being_dropped() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        let events = [with_vitals(1, Some(1_200), Some(180))];

        let output = runtime.tick(input(&events));

        assert_eq!(output.lcp_p95_ms, Some(1_200.0));
        assert_eq!(output.event_duration_tail_ms, Some(180.0));
    }

    #[test]
    fn the_sample_count_travels_with_the_p95_so_a_guess_is_distinguishable() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        assert_eq!(runtime.tick(input(&[])).vitals_samples, 0);

        let output = runtime.tick(input(&[with_vitals(1, Some(1_200), Some(180))]));
        assert_eq!(output.vitals_samples, 1);
        assert_eq!(output.lcp_p95_ms, Some(1_200.0));

        for sequence in 2..=10 {
            runtime.tick(input(&[with_vitals(sequence, Some(1_200), Some(180))]));
        }
        assert_eq!(runtime.vitals.samples(), 10);

        // Never exceeds the window, and drops with it on invalidation.
        for sequence in 11..=(MAX_VITALS_SAMPLES as u64 * 2) {
            runtime.tick(input(&[with_vitals(sequence, Some(1_200), None)]));
        }
        assert_eq!(runtime.vitals.samples(), MAX_VITALS_SAMPLES as u64);

        let mut sleeping = input(&[]);
        sleeping.sleeping = true;
        assert_eq!(runtime.tick(sleeping).vitals_samples, 0);
    }

    #[test]
    fn a_metric_absent_from_the_report_stays_none_rather_than_becoming_zero() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        let events = [with_vitals(1, Some(900), None)];

        let output = runtime.tick(input(&events));

        assert_eq!(output.lcp_p95_ms, Some(900.0));
        assert_eq!(output.event_duration_tail_ms, None);
    }

    #[test]
    fn daemon_inference_never_fabricates_a_vital() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        let mut inferred = input(&[]);
        inferred.browser_socket_active = true;

        let output = runtime.tick(inferred);

        assert_eq!(
            output.observation.source,
            Some(WebFlowSource::DaemonInference)
        );
        assert_eq!(output.lcp_p95_ms, None);
        assert_eq!(output.event_duration_tail_ms, None);
    }

    #[test]
    fn a_stale_report_is_excluded_from_the_vitals_window() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        let mut stale = with_vitals(1, Some(5_000), Some(900));
        stale.received_at_ms = 100;
        let events = [stale];
        let mut cycle = input(&events);
        cycle.now_ms = 2_101;

        let output = runtime.tick(cycle);

        assert_eq!(output.lcp_p95_ms, None);
        assert_eq!(output.event_duration_tail_ms, None);
    }

    #[test]
    fn the_vitals_window_is_bounded_and_evicts_oldest_first() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        // Feed far more samples than the window holds.
        for sequence in 1..=(MAX_VITALS_SAMPLES as u64 * 3) {
            let events = [with_vitals(sequence, Some(100), None)];
            runtime.tick(input(&events));
        }
        assert_eq!(runtime.vitals.lcp_ms.len(), MAX_VITALS_SAMPLES);

        // Refill the window entirely with slow samples: the fast ones must all
        // have been evicted, so p95 follows the new regime.
        for sequence in 1..=(MAX_VITALS_SAMPLES as u64) {
            let events = [with_vitals(sequence, Some(4_000), None)];
            runtime.tick(input(&events));
        }
        assert_eq!(runtime.vitals.lcp_ms.len(), MAX_VITALS_SAMPLES);
        assert_eq!(runtime.vitals.lcp_p95_ms(), Some(4_000.0));
    }

    #[test]
    fn p95_is_a_tail_statistic_that_one_slow_page_cannot_swing() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        for sequence in 1..=(MAX_VITALS_SAMPLES as u64 - 1) {
            runtime.tick(input(&[with_vitals(sequence, Some(100), None)]));
        }

        // A single outlier in a 64-sample window sits above the 95th
        // percentile rank and must not move the reported value.
        let output = runtime.tick(input(&[with_vitals(9_999, Some(4_000), None)]));
        assert_eq!(output.lcp_p95_ms, Some(100.0));

        // A sustained tail does move it.
        for sequence in 10_000..10_008 {
            runtime.tick(input(&[with_vitals(sequence, Some(4_000), None)]));
        }
        assert_eq!(runtime.vitals.lcp_p95_ms(), Some(4_000.0));
    }

    #[test]
    fn sleep_and_session_change_drop_the_vitals_window() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Active);
        runtime.tick(input(&[with_vitals(1, Some(1_000), Some(120))]));
        assert!(runtime.vitals.lcp_p95_ms().is_some());

        let mut sleeping = input(&[]);
        sleeping.sleeping = true;
        let output = runtime.tick(sleeping);
        assert_eq!(output.lcp_p95_ms, None);
        assert_eq!(output.event_duration_tail_ms, None);

        runtime.tick(input(&[with_vitals(2, Some(1_000), Some(120))]));
        let mut relogin = input(&[]);
        relogin.session_revision = 99;
        let output = runtime.tick(relogin);
        assert_eq!(output.lcp_p95_ms, None);
    }

    #[test]
    fn browser_detection_excludes_non_browser_electron_apps() {
        for name in [
            "Brave Browser",
            "Google Chrome",
            "Microsoft Edge",
            "Chromium",
            "Arc",
            "Vivaldi",
            "Opera",
        ] {
            assert!(is_supported_browser(name), "{name}");
        }
        for name in ["Slack", "Code", "Discord", "Notion", "Safari"] {
            assert!(!is_supported_browser(name), "{name}");
        }
    }

    #[test]
    fn five_hundred_healthy_cycles_promote_shadow_to_active() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        for _ in 0..499 {
            let output = runtime.tick(input(&[]));
            assert_eq!(output.rollout, WebFlowRolloutPhase::Shadow);
        }
        let events = [exact(1)];
        let output = runtime.tick(input(&events));
        assert_eq!(output.rollout, WebFlowRolloutPhase::Active);
        assert!(output.admitted);
    }

    #[test]
    fn unhealthy_cycles_do_not_advance_webflow_promotion() {
        let mut runtime = WebFlowRuntime::new(WebFlowRolloutPhase::Shadow);
        for _ in 0..500 {
            let mut unhealthy = input(&[]);
            unhealthy.control_p95_ms = 90.0;
            runtime.tick(unhealthy);
        }
        let events = [exact(1)];
        let output = runtime.tick(input(&events));
        assert_eq!(output.rollout, WebFlowRolloutPhase::Shadow);
        assert!(!output.admitted);
    }
}
