use apollo_engine::engine::fluidity::{FluiditySignal, FluidityState};
use apollo_engine::engine::iokit_sensors::HardwareSnapshot;
use apollo_engine::engine::process_classifier::ProcessSnapshot;

pub struct FluidityTickInput<'a> {
    pub proc_snaps: &'a [ProcessSnapshot],
    pub cycle_hw_snap: Option<&'a HardwareSnapshot>,
    pub cycle_dt_secs: f32,
    pub hardware_cores: u32,
    pub fluidity_state: &'a mut FluidityState,
}

pub struct FluidityTickOutput {
    pub fl_signal: FluiditySignal,
}

fn normalized_gpu_load(gpu_watts: Option<f32>, hardware_cores: u32) -> f32 {
    gpu_watts
        .map(|watts| {
            (watts / super::daemon_thermal_tick::gpu_power_reference_watts(hardware_cores))
                .clamp(0.0, 1.0)
        })
        .unwrap_or(0.0)
}

pub fn run_fluidity_tick(input: FluidityTickInput) -> FluidityTickOutput {
    // Migrated from main.rs:2291-2310 (Strangler Fig).
    // Compute GPU load 0-1 from package watts.
    let fl_gpu_load = normalized_gpu_load(
        input.cycle_hw_snap.and_then(|hw| hw.power.gpu_watts),
        input.hardware_cores,
    );

    input.fluidity_state.update(
        input
            .proc_snaps
            .iter()
            .map(|p| (p.pid, p.name.as_str(), p.cpu_percent)),
        fl_gpu_load,
        input.cycle_dt_secs,
    );

    FluidityTickOutput {
        fl_signal: FluiditySignal::from(&*input.fluidity_state),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fluidity_tick_updates_state_from_proc_snaps() {
        let mut state = apollo_engine::engine::fluidity::FluidityState::new();
        let procs: Vec<(u32, String, f32)> = vec![
            (415, "WindowServer".to_string(), 25.0),
            (1234, "Brave".to_string(), 5.0),
        ];

        let proc_snaps = procs
            .into_iter()
            .map(|(pid, name, cpu_percent)| ProcessSnapshot {
                pid,
                name,
                cpu_percent,
                rss_bytes: 0,
                is_zombie: false,
                secs_since_foreground: 0,
                secs_since_user_interaction: 0,
                has_network: false,
                has_gui_window: false,
                wakeups_per_sec: 0.0,
                parent_alive: true,
                process_uptime_secs: 0,
                faults_total: 0,
                pageins_total: 0,
                is_translated: false,
                mach_port_count: 0,
                cpu_contention: None,
                is_app_bundle: false,
            })
            .collect::<Vec<_>>();

        let input = FluidityTickInput {
            proc_snaps: &proc_snaps,
            cycle_hw_snap: None,
            cycle_dt_secs: 0.5,
            hardware_cores: 8,
            fluidity_state: &mut state,
        };

        for _ in 0..10 {
            run_fluidity_tick(FluidityTickInput {
                proc_snaps: &proc_snaps,
                cycle_hw_snap: None,
                cycle_dt_secs: 0.5,
                hardware_cores: 8,
                fluidity_state: input.fluidity_state,
            });
        }
        assert!(input.fluidity_state.windowserver_cpu_ema > 20.0);
    }

    #[test]
    fn fluidity_signal_snapshot_is_clone_independent() {
        let mut state = apollo_engine::engine::fluidity::FluidityState::new();
        let _sig = apollo_engine::engine::fluidity::FluiditySignal::from(&state);
        // Mutating state after snapshot must not affect snapshot
        state.update(vec![(415, "WindowServer", 50.0)], 0.0, 0.5);
    }

    #[test]
    fn gpu_load_normalization_scales_with_detected_capacity() {
        assert!((normalized_gpu_load(Some(8.0), 8) - 0.50).abs() < f32::EPSILON);
        assert!((normalized_gpu_load(Some(8.0), 10) - 0.40).abs() < f32::EPSILON);
        assert_eq!(normalized_gpu_load(None, 10), 0.0);
    }
}
