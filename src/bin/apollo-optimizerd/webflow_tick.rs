use apollo_engine::engine::webflow_controller::{
    WebFlowController, WebFlowCounters, WebFlowIntent, WebFlowRolloutPhase, WebFlowTickInput,
    WebWorldObservation, MAX_WEBFLOW_EVENT_AGE_MS,
};
use apollo_engine::engine::webflow_types::{
    OpaqueId, ReceivedWebFlowEvent, WebFlowEvent, WebFlowMetrics, WebFlowPhase, WebFlowSource,
    WEBFLOW_SCHEMA_VERSION,
};

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
