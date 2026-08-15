use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use apollo_engine::engine::cycle_snapshot::{CycleContextSnapshot, SnapshotId, SourceObservation};
use apollo_engine::engine::event_mesh::{EventEnvelope, EventMesh, EventSource, LifecycleEvent};
use apollo_engine::engine::network_flow::NetworkWorldObservation;
use apollo_engine::engine::value_scheduler::{
    JobId, SchedulerInputs, SchedulerLevel, SchedulerPhase, ValueScheduler,
};
use apollo_engine::engine::webflow_controller::WebWorldObservation;
use apollo_engine::engine::webflow_types::WebFlowEvent;
use apollo_engine::engine::world_state::{
    FeatureStore, WorldIdentity, WorldStatePublisher, WorldStateSnapshot,
};

use crate::adaptive_overhead::OverheadLevel;

const SHADOW_CYCLES_REQUIRED: u64 = 500;

#[derive(Debug, Clone)]
pub struct ValueSchedulerTickInput<'a> {
    pub cycle: u64,
    pub overhead_level: OverheadLevel,
    pub allow_speculation: bool,
    pub holt_cadence: u64,
    pub page_reclaim_cadence: u64,
    pub pressure: f64,
    pub pressure_age: Duration,
    pub thermal_level: &'a str,
    pub interaction_active: bool,
    pub context_epoch: Option<u64>,
    pub context_visual_q: Option<f64>,
    pub context_interaction_q: Option<f64>,
    pub context_audio_active: Option<bool>,
    pub context_permissions_bits: u16,
    pub workload: &'a str,
    pub capability_revision: u64,
    pub kill_switch: bool,
    pub sleeping: bool,
    pub profile_matches: bool,
    pub p95_cycle_ms: f64,
    pub holt_cost_ms: f64,
    pub page_reclaim_cost_ms: f64,
    pub webflow_events: &'a [WebFlowEvent],
    pub webflow_observation: Option<WebWorldObservation>,
    pub network_observation: Option<NetworkWorldObservation>,
}

#[derive(Debug, Clone, Default)]
pub struct ValueSchedulerTickMetrics {
    pub phase: String,
    pub blocker: String,
    pub valid_cycles: u64,
    pub shadow_cycles_required: u64,
    pub snapshot_epoch: u64,
    pub snapshot_revision: u64,
    pub registered_jobs: u64,
    pub eligible_jobs: u64,
    pub selected_jobs: u64,
    pub selected_total: u64,
    pub budget_us: u64,
    pub predicted_us: u64,
    pub selection_latency_us: u64,
    pub max_selection_latency_us: u64,
    pub budget_skips_total: u64,
    pub capacity_skips_total: u64,
    pub invalid_samples_total: u64,
    pub world_snapshot: Option<Arc<WorldStateSnapshot>>,
}

pub struct ValueSchedulerRuntime {
    world_publisher: WorldStatePublisher,
    event_mesh: EventMesh,
    scheduler: ValueScheduler,
    started_at: Instant,
    valid_cycles: u64,
    invalid_samples_total: u64,
    last_workload_id: u64,
    process_revision: u64,
    last_thermal_q: Option<u16>,
    thermal_revision: u64,
    session_revision: u64,
    context_capability_revision: u64,
    last_context_epoch: Option<u64>,
    last_context_permissions_bits: u16,
    last_sleeping: bool,
    webflow_source_sequence: u64,
}

impl ValueSchedulerRuntime {
    pub fn new(capability_revision: u64) -> Self {
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(1)
            ^ u64::from(std::process::id());
        Self::with_epoch_and_capability(epoch.max(1), capability_revision.max(1))
    }

    fn with_epoch(epoch: u64) -> Self {
        Self::with_epoch_and_capability(epoch, 1)
    }

    fn with_epoch_and_capability(epoch: u64, capability_revision: u64) -> Self {
        let initial = WorldStateSnapshot::new(
            WorldIdentity {
                daemon_epoch: epoch,
                revision: 1,
                capability_revision,
                thermal_revision: 1,
                process_revision: 1,
                session_revision: 1,
                ..WorldIdentity::default()
            },
            0,
            FeatureStore::try_new(1, Vec::new()).expect("empty bootstrap feature store"),
        )
        .expect("bootstrap world identity");
        Self {
            world_publisher: WorldStatePublisher::new(initial),
            event_mesh: EventMesh::new(epoch),
            scheduler: ValueScheduler::new(SchedulerPhase::Shadow),
            started_at: Instant::now(),
            valid_cycles: 0,
            invalid_samples_total: 0,
            last_workload_id: 0,
            process_revision: 1,
            last_thermal_q: None,
            thermal_revision: 1,
            session_revision: 1,
            context_capability_revision: 1,
            last_context_epoch: None,
            last_context_permissions_bits: 0,
            last_sleeping: false,
            webflow_source_sequence: 0,
        }
    }

    pub fn tick(&mut self, input: ValueSchedulerTickInput<'_>) -> ValueSchedulerTickMetrics {
        let cut_started_us = self.started_at.elapsed().as_micros() as u64;
        let pressure_q = finite_unit_q(input.pressure);
        let pressure = match pressure_q {
            Some(value) if input.pressure_age <= Duration::from_secs(10) => {
                SourceObservation::fresh(
                    value,
                    1,
                    input.cycle,
                    input.pressure_age.as_micros() as u64,
                )
            }
            Some(value) => SourceObservation::stale(
                Some(value),
                1,
                input.cycle,
                input.pressure_age.as_micros() as u64,
            ),
            None => SourceObservation::invalid(1, input.cycle),
        };
        let thermal_q = thermal_q(input.thermal_level);
        let thermal = thermal_q.map_or_else(
            || SourceObservation::unavailable(1, input.cycle),
            |value| SourceObservation::fresh(value, 1, input.cycle, 0),
        );
        let interaction = SourceObservation::fresh(
            if input.interaction_active { 10_000 } else { 0 },
            1,
            input.cycle,
            0,
        );
        let cut_completed_us = self.started_at.elapsed().as_micros() as u64;
        let workload_id = stable_hash(input.workload);
        if self.last_workload_id != 0 && self.last_workload_id != workload_id {
            self.process_revision = self.process_revision.saturating_add(1);
        }
        self.last_workload_id = workload_id;
        if self.last_thermal_q != thermal_q {
            self.thermal_revision = self.thermal_revision.saturating_add(1);
            self.last_thermal_q = thermal_q;
        }
        if self.last_context_epoch != input.context_epoch {
            self.session_revision = self.session_revision.saturating_add(1);
            self.last_context_epoch = input.context_epoch;
        }
        if self.last_context_permissions_bits != input.context_permissions_bits {
            self.context_capability_revision = self.context_capability_revision.saturating_add(1);
            self.last_context_permissions_bits = input.context_permissions_bits;
        }
        let capability_revision = input
            .capability_revision
            .max(1)
            .saturating_mul(1_000_000)
            .saturating_add(self.context_capability_revision);

        self.ingest_cycle_events(
            input.cycle,
            cut_completed_us,
            pressure_q,
            thermal_q,
            input.interaction_active,
            input.context_visual_q,
            input.context_audio_active,
            input.sleeping,
        );
        self.ingest_webflow_events(
            cut_completed_us,
            input.webflow_events,
            input.webflow_observation,
        );
        if let Some(network) = input.network_observation {
            let traffic_q =
                (network.traffic_bps as f64 / (8.0 * 1024.0 * 1024.0)).clamp(0.0, 1.0) as f32;
            if let Ok(event) = EventEnvelope::scalar(
                self.world_publisher.latest().identity.daemon_epoch,
                EventSource::Network,
                1,
                input.cycle,
                cut_completed_us.saturating_add(129),
                network.confidence_q,
                traffic_q,
            ) {
                let _ = self.event_mesh.ingest(event);
            }
        }
        let drained = self.event_mesh.drain(self.event_mesh.capacity());
        let event_watermark = drained.last().map_or_else(
            || self.world_publisher.latest().event_watermark,
            |event| event.ingest_sequence,
        );
        let current = self.world_publisher.latest();
        let Some(world_revision) = current.identity.revision.checked_add(1) else {
            self.invalid_samples_total = self.invalid_samples_total.saturating_add(1);
            return self.rejected_metrics("snapshot-sequence-exhausted");
        };
        let features = match FeatureStore::try_new(
            1,
            world_features(
                pressure_q,
                thermal_q,
                input.interaction_active,
                input.context_visual_q,
                input.context_interaction_q,
                input.context_audio_active,
                input.p95_cycle_ms,
                input.allow_speculation,
                input.overhead_level,
                input.webflow_observation,
                input.network_observation,
            ),
        ) {
            Ok(features) => features,
            Err(_) => {
                self.invalid_samples_total = self.invalid_samples_total.saturating_add(1);
                return self.rejected_metrics("invalid-feature-vector");
            }
        };
        let world = match WorldStateSnapshot::new(
            WorldIdentity {
                daemon_epoch: current.identity.daemon_epoch,
                revision: world_revision,
                workload_id,
                capability_revision,
                thermal_revision: self.thermal_revision,
                process_revision: self.process_revision,
                session_revision: self.session_revision,
                kill_switch: input.kill_switch,
                sleeping: input.sleeping,
            },
            event_watermark,
            features,
        )
        .map(|snapshot| {
            snapshot
                .with_web(input.webflow_observation)
                .with_network(input.network_observation)
        })
        .and_then(|snapshot| {
            self.world_publisher
                .publish(snapshot)
                .map_err(|_| "world publication failed")
        }) {
            Ok(world) => world,
            Err(_) => {
                self.invalid_samples_total = self.invalid_samples_total.saturating_add(1);
                return self.rejected_metrics("snapshot-publication-failed");
            }
        };

        let mut snapshot = CycleContextSnapshot::new(
            SnapshotId::new(world.identity.daemon_epoch, world.identity.revision),
            world.identity.workload_id,
            world.identity.capability_revision,
            world.identity.thermal_revision,
        )
        .with_cut_times(cut_started_us, cut_completed_us)
        .with_pressure(pressure)
        .with_thermal(thermal)
        .with_interaction(interaction);
        snapshot.cycle = input.cycle;
        snapshot.process_identity_revision = world.identity.process_revision;
        snapshot.kill_switch = input.kill_switch;
        snapshot.sleeping = input.sleeping;

        let mut scheduler_inputs = SchedulerInputs::default();
        scheduler_inputs.level = map_level(input.overhead_level);
        scheduler_inputs.kill_switch = input.kill_switch;
        scheduler_inputs.sleeping = input.sleeping;
        set_due_jobs(&mut scheduler_inputs, &input);
        set_cost(
            &mut scheduler_inputs,
            JobId::HoltWintersRefresh,
            input.holt_cost_ms,
        );
        set_cost(
            &mut scheduler_inputs,
            JobId::PageReclaimRefresh,
            input.page_reclaim_cost_ms,
        );
        scheduler_inputs.set_signal(
            JobId::GpuImagination,
            if input.allow_speculation { 0.8 } else { 0.0 },
        );
        scheduler_inputs.set_signal(
            JobId::ReflexReasoningRefresh,
            if input.interaction_active { 1.0 } else { 0.4 },
        );
        scheduler_inputs.set_signal(JobId::WorldModelRefresh, input.pressure.clamp(0.0, 1.0));

        let plan = self.scheduler.plan(&snapshot, scheduler_inputs);
        let sample_valid = snapshot.pressure_q().is_some()
            && snapshot.thermal_q().is_some()
            && input.profile_matches
            && input.p95_cycle_ms.is_finite()
            && input.p95_cycle_ms < 75.0
            && plan.selection_latency_us <= 1_000
            && !input.kill_switch
            && !input.sleeping;
        if sample_valid {
            self.valid_cycles = self.valid_cycles.saturating_add(1);
        } else {
            self.invalid_samples_total = self.invalid_samples_total.saturating_add(1);
        }

        let blocker = if !input.profile_matches {
            "profile-mismatch"
        } else if snapshot.pressure_q().is_none() {
            "pressure-stale"
        } else if snapshot.thermal_q().is_none() {
            "thermal-unavailable"
        } else if !input.p95_cycle_ms.is_finite() || input.p95_cycle_ms >= 75.0 {
            "cycle-p95"
        } else if plan.selection_latency_us > 1_000 {
            "selection-latency"
        } else if self.valid_cycles < SHADOW_CYCLES_REQUIRED {
            "collecting-samples"
        } else {
            "legacy-bypass"
        };
        let scheduler_metrics = self.scheduler.metrics();
        ValueSchedulerTickMetrics {
            phase: "shadow".to_string(),
            blocker: blocker.to_string(),
            valid_cycles: self.valid_cycles,
            shadow_cycles_required: SHADOW_CYCLES_REQUIRED,
            snapshot_epoch: world.identity.daemon_epoch,
            snapshot_revision: world.identity.revision,
            registered_jobs: scheduler_metrics.registered_jobs,
            eligible_jobs: scheduler_metrics.eligible_jobs,
            selected_jobs: scheduler_metrics.selected_jobs,
            selected_total: scheduler_metrics.selected_total,
            budget_us: scheduler_metrics.budget_us,
            predicted_us: scheduler_metrics.predicted_us,
            selection_latency_us: scheduler_metrics.selection_latency_us,
            max_selection_latency_us: scheduler_metrics.max_selection_latency_us,
            budget_skips_total: scheduler_metrics.budget_skipped_total,
            capacity_skips_total: scheduler_metrics.capacity_skipped_total,
            invalid_samples_total: self.invalid_samples_total,
            world_snapshot: Some(Arc::clone(&world)),
        }
    }

    fn rejected_metrics(&self, blocker: &str) -> ValueSchedulerTickMetrics {
        ValueSchedulerTickMetrics {
            phase: "shadow".to_string(),
            blocker: blocker.to_string(),
            shadow_cycles_required: SHADOW_CYCLES_REQUIRED,
            invalid_samples_total: self.invalid_samples_total,
            ..ValueSchedulerTickMetrics::default()
        }
    }

    fn ingest_cycle_events(
        &mut self,
        cycle: u64,
        monotonic_time_us: u64,
        pressure_q: Option<u16>,
        thermal_q: Option<u16>,
        interaction_active: bool,
        context_visual_q: Option<f64>,
        context_audio_active: Option<bool>,
        sleeping: bool,
    ) {
        let epoch = self.world_publisher.latest().identity.daemon_epoch;
        for (source, value) in [
            (
                EventSource::Pressure,
                pressure_q.map(|value| f32::from(value) / 10_000.0),
            ),
            (
                EventSource::Thermal,
                thermal_q.map(|value| f32::from(value) / 10_000.0),
            ),
            (
                EventSource::VisualActivity,
                Some(
                    context_visual_q
                        .filter(|value| value.is_finite())
                        .unwrap_or(if interaction_active { 1.0 } else { 0.0 })
                        .clamp(0.0, 1.0) as f32,
                ),
            ),
            (
                EventSource::AudioActivity,
                context_audio_active.map(|active| if active { 1.0 } else { 0.0 }),
            ),
        ] {
            if let Some(value) = value {
                if let Ok(event) =
                    EventEnvelope::scalar(epoch, source, 1, cycle, monotonic_time_us, 10_000, value)
                {
                    let _ = self.event_mesh.ingest(event);
                }
            }
        }
        if sleeping != self.last_sleeping {
            let event = EventEnvelope::lifecycle(
                epoch,
                1,
                cycle,
                monotonic_time_us,
                if sleeping {
                    LifecycleEvent::Sleep
                } else {
                    LifecycleEvent::Wake
                },
            );
            let _ = self.event_mesh.ingest(event);
            self.last_sleeping = sleeping;
        }
    }

    fn ingest_webflow_events(
        &mut self,
        monotonic_time_us: u64,
        events: &[WebFlowEvent],
        observation: Option<WebWorldObservation>,
    ) {
        let epoch = self.world_publisher.latest().identity.daemon_epoch;
        for event in events.iter().take(128) {
            let Some(next_sequence) = self.webflow_source_sequence.checked_add(1) else {
                break;
            };
            self.webflow_source_sequence = next_sequence;
            let active_navigations = observation.map_or(0, |web| web.active_navigations);
            if let Ok(envelope) = EventEnvelope::webflow(
                epoch,
                1,
                next_sequence,
                monotonic_time_us.saturating_add(next_sequence % 128),
                event.phase,
                event.source,
                active_navigations,
            ) {
                let _ = self.event_mesh.ingest(envelope);
            }
        }
    }
}

fn world_features(
    pressure_q: Option<u16>,
    thermal_q: Option<u16>,
    interaction_active: bool,
    context_visual_q: Option<f64>,
    context_interaction_q: Option<f64>,
    context_audio_active: Option<bool>,
    p95_cycle_ms: f64,
    allow_speculation: bool,
    overhead_level: OverheadLevel,
    webflow: Option<WebWorldObservation>,
    network: Option<NetworkWorldObservation>,
) -> Vec<f32> {
    vec![
        pressure_q.map_or(0.0, |value| f32::from(value) / 10_000.0),
        if pressure_q.is_some() { 1.0 } else { 0.0 },
        thermal_q.map_or(0.0, |value| f32::from(value) / 10_000.0),
        if thermal_q.is_some() { 1.0 } else { 0.0 },
        if interaction_active { 1.0 } else { 0.0 },
        context_visual_q
            .filter(|value| value.is_finite())
            .map_or(0.0, |value| value.clamp(0.0, 1.0) as f32),
        if context_visual_q.is_some() { 1.0 } else { 0.0 },
        context_interaction_q
            .filter(|value| value.is_finite())
            .map_or(0.0, |value| value.clamp(0.0, 1.0) as f32),
        if context_interaction_q.is_some() {
            1.0
        } else {
            0.0
        },
        context_audio_active.map_or(0.0, |active| if active { 1.0 } else { 0.0 }),
        if context_audio_active.is_some() {
            1.0
        } else {
            0.0
        },
        if p95_cycle_ms.is_finite() {
            (p95_cycle_ms / 75.0).clamp(0.0, 4.0) as f32
        } else {
            4.0
        },
        if allow_speculation { 1.0 } else { 0.0 },
        match overhead_level {
            OverheadLevel::Nominal => 0.0,
            OverheadLevel::Guarded => 0.5,
            OverheadLevel::Constrained => 1.0,
        },
        webflow.map_or(0.0, |web| f32::from(web.active_navigations) / 64.0),
        webflow.map_or(0.0, |web| f32::from(web.confidence_q) / 10_000.0),
        webflow.map_or(0.0, |web| if web.vitals_available { 1.0 } else { 0.0 }),
        network.map_or(0.0, |flow| if flow.active { 1.0 } else { 0.0 }),
        network.map_or(0.0, |flow| f32::from(flow.confidence_q) / 10_000.0),
        network.map_or(0.0, |flow| {
            (flow.traffic_bps as f64 / (8.0 * 1024.0 * 1024.0)).clamp(0.0, 1.0) as f32
        }),
        network.map_or(0.0, |flow| if flow.sample_fresh { 1.0 } else { 0.0 }),
    ]
}

fn set_due_jobs(inputs: &mut SchedulerInputs, tick: &ValueSchedulerTickInput<'_>) {
    inputs.set_due(JobId::ReflexReasoningRefresh, true);
    inputs.set_due(JobId::WorldModelRefresh, true);
    inputs.set_due(
        JobId::GpuImagination,
        tick.allow_speculation && tick.cycle.is_multiple_of(20),
    );
    inputs.set_due(JobId::AisRuntimeRefresh, tick.cycle.is_multiple_of(20));
    inputs.set_due(JobId::HardwarePrediction, tick.cycle.is_multiple_of(10));
    inputs.set_due(
        JobId::HoltWintersRefresh,
        tick.cycle.is_multiple_of(tick.holt_cadence.max(1)),
    );
    inputs.set_due(
        JobId::PageReclaimRefresh,
        tick.cycle.is_multiple_of(tick.page_reclaim_cadence.max(1)),
    );
    inputs.set_due(JobId::PlannerAdviceRefresh, tick.cycle.is_multiple_of(15));
    inputs.set_due(
        JobId::PeriodicLearningMaintenance,
        tick.cycle.is_multiple_of(100),
    );
    inputs.set_due(JobId::TelemetryFlush, tick.cycle.is_multiple_of(20));
}

fn set_cost(inputs: &mut SchedulerInputs, job: JobId, milliseconds: f64) {
    if milliseconds.is_finite() && milliseconds > 0.0 {
        inputs.set_cost_estimate_us(job, milliseconds * 1_000.0);
    }
}

fn map_level(level: OverheadLevel) -> SchedulerLevel {
    match level {
        OverheadLevel::Nominal => SchedulerLevel::Nominal,
        OverheadLevel::Guarded => SchedulerLevel::Guarded,
        OverheadLevel::Constrained => SchedulerLevel::Constrained,
    }
}

fn finite_unit_q(value: f64) -> Option<u16> {
    value
        .is_finite()
        .then(|| (value.clamp(0.0, 1.0) * 10_000.0).round() as u16)
}

fn thermal_q(level: &str) -> Option<u16> {
    match level.to_ascii_lowercase().as_str() {
        "nominal" | "normal" => Some(0),
        "fair" | "moderate" => Some(3_333),
        "serious" | "high" => Some(6_667),
        "critical" => Some(10_000),
        _ => None,
    }
}

fn stable_hash(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::webflow_types::{WebFlowPhase, WebFlowSource};

    fn input(cycle: u64) -> ValueSchedulerTickInput<'static> {
        ValueSchedulerTickInput {
            cycle,
            overhead_level: OverheadLevel::Nominal,
            allow_speculation: true,
            holt_cadence: 1,
            page_reclaim_cadence: 10,
            pressure: 0.39,
            pressure_age: Duration::from_secs(1),
            thermal_level: "nominal",
            interaction_active: true,
            context_epoch: None,
            context_visual_q: None,
            context_interaction_q: None,
            context_audio_active: None,
            context_permissions_bits: 0,
            workload: "interactive",
            capability_revision: 1,
            kill_switch: false,
            sleeping: false,
            profile_matches: true,
            p95_cycle_ms: 35.0,
            holt_cost_ms: 0.1,
            page_reclaim_cost_ms: 0.2,
            webflow_events: &[],
            webflow_observation: None,
            network_observation: None,
        }
    }

    #[test]
    fn shadow_observes_but_never_executes() {
        let mut runtime = ValueSchedulerRuntime::with_epoch(7);
        let metrics = runtime.tick(input(20));
        assert_eq!(metrics.phase, "shadow");
        assert!(metrics.selected_jobs > 0);
        assert_eq!(runtime.scheduler.phase(), SchedulerPhase::Shadow);
        assert_eq!(runtime.scheduler.metrics().in_flight_jobs, 0);
    }

    #[test]
    fn stale_pressure_is_reported_instead_of_nominal() {
        let mut runtime = ValueSchedulerRuntime::with_epoch(7);
        let mut stale = input(1);
        stale.pressure_age = Duration::from_secs(11);
        let metrics = runtime.tick(stale);
        assert_eq!(metrics.blocker, "pressure-stale");
        assert_eq!(metrics.valid_cycles, 0);
        assert_eq!(metrics.invalid_samples_total, 1);
    }

    #[test]
    fn five_hundred_observations_still_report_legacy_bypass() {
        let mut runtime = ValueSchedulerRuntime::with_epoch(7);
        let mut last = ValueSchedulerTickMetrics::default();
        for cycle in 1..=SHADOW_CYCLES_REQUIRED {
            last = runtime.tick(input(cycle));
        }
        assert_eq!(last.valid_cycles, SHADOW_CYCLES_REQUIRED);
        assert_eq!(last.blocker, "legacy-bypass");
        assert_eq!(last.phase, "shadow");
    }

    #[test]
    fn world_identity_changes_only_when_the_observed_context_changes() {
        let mut runtime = ValueSchedulerRuntime::with_epoch(7);
        runtime.tick(input(1));
        let first = runtime.world_publisher.latest();
        runtime.tick(input(2));
        let stable = runtime.world_publisher.latest();
        assert_eq!(
            first.identity.process_revision,
            stable.identity.process_revision
        );
        assert!(stable.event_watermark > first.event_watermark);

        let mut changed = input(3);
        changed.workload = "browser";
        runtime.tick(changed);
        let changed = runtime.world_publisher.latest();
        assert!(changed.identity.process_revision > stable.identity.process_revision);
        assert_eq!(runtime.world_publisher.retained_len(), 3);
    }

    #[test]
    fn missing_sensor_values_have_explicit_feature_masks() {
        let features = world_features(
            None,
            None,
            false,
            None,
            None,
            None,
            f64::NAN,
            false,
            OverheadLevel::Constrained,
            None,
            None,
        );
        assert_eq!(&features[..5], &[0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(features[11], 4.0);
        assert_eq!(features[13], 1.0);
    }

    #[test]
    fn webflow_observation_is_published_in_the_same_world_revision() {
        let mut runtime = ValueSchedulerRuntime::with_epoch(7);
        let mut tick = input(1);
        tick.webflow_observation = Some(WebWorldObservation {
            accepted_events: 1,
            active_navigations: 1,
            last_phase: Some(WebFlowPhase::Started),
            source: Some(WebFlowSource::ExtensionLifecycle),
            confidence_q: 9_000,
            last_event_age_ms: Some(2),
            vitals_available: false,
        });
        let metrics = runtime.tick(tick);
        assert_eq!(
            metrics
                .world_snapshot
                .expect("world")
                .web
                .expect("web observation")
                .active_navigations,
            1
        );
    }

    #[test]
    fn universal_network_observation_is_published_in_the_same_world_revision() {
        let mut runtime = ValueSchedulerRuntime::with_epoch(7);
        let mut tick = input(1);
        tick.network_observation = Some(NetworkWorldObservation {
            active: true,
            inferred: true,
            target_available: true,
            socket_active: true,
            sample_fresh: true,
            interaction_active: true,
            traffic_bps: 240_000,
            confidence_q: 5_000,
        });
        let metrics = runtime.tick(tick);
        assert!(
            metrics
                .world_snapshot
                .expect("world")
                .network
                .expect("network observation")
                .active
        );
    }
}
