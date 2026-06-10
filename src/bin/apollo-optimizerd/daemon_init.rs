//! Daemon initialization helpers — subsystem construction for apollo-optimizerd.
//!
//! `DaemonSubsystems` bundles the stateless, zero-dependency subsystems that are
//! constructed once at startup and then moved into the main loop.  Grouping them
//! here reduces the line count in `main.rs` and provides a single place to track
//! which subsystems exist.

use crate::daemon_memory_budget::MemoryBudgetState;
use apollo_engine::engine::action_queue::ActionQueue;
use apollo_engine::engine::analytics::AnalyticsEngine;
use apollo_engine::engine::causal_graph::CausalGraph;
use apollo_engine::engine::coalition::CoalitionTracker;
use apollo_engine::engine::daemon_helpers::{hop_groups_path, recently_applied_path, skills_path};
use apollo_engine::engine::effectiveness_tracker::EffectivenessTracker;
use apollo_engine::engine::energy::EnergyTracker;
use apollo_engine::engine::energy_pid::EnergyPidTracker;
use apollo_engine::engine::evolved_anomaly::EvolvedAnomalyDetector;
use apollo_engine::engine::ioreport::IOReportReader;
use apollo_engine::engine::learning_pipeline::LearningPipeline;
use apollo_engine::engine::memory_analyzer::MemoryAnalyzer;
use apollo_engine::engine::network_monitor::NetworkMonitor;
use apollo_engine::engine::neuromodulator::ApolloNeuromodulator;
use apollo_engine::engine::optimization_skills::SkillRegistry;
use apollo_engine::engine::outcome_tracker::OutcomeTracker;
use apollo_engine::engine::power_management::PowerManager;
use apollo_engine::engine::predictive_agent::SpecialistAccuracyTracker;
use apollo_engine::engine::process_recovery::ProcessRecoveryManager;
use apollo_engine::engine::swap_predictor::SwapPredictor;
use apollo_engine::engine::swap_reclaim::SwapReclaimModel;
use apollo_engine::engine::syscall_classifier::SyscallClassifier;
use apollo_engine::engine::thermal_bailout::ThermalBailout;
use apollo_engine::engine::thermal_manager::ThermalManager;
use apollo_engine::engine::thread_selfcounts::CycleIpcTracker;
use apollo_engine::engine::unfreeze_decay::UnfreezeDecayModel;
use apollo_engine::engine::wake_storm_detector::WakeStormDetector;

/// Subsystems constructed once at daemon startup with no shared-state dependencies.
///
/// Immediately destructure this into `let DaemonSubsystems { .. } = DaemonSubsystems::new()`
/// in `main.rs` so all fields become independent `mut` locals.
pub(super) struct DaemonSubsystems {
    pub analytics: AnalyticsEngine,
    pub mem_analyzer: MemoryAnalyzer,
    pub power_mgr: PowerManager,
    pub proc_recovery: ProcessRecoveryManager,
    pub swap_predictor: SwapPredictor,
    pub syscall_classifier: SyscallClassifier,
    pub network_monitor: NetworkMonitor,
    pub thermal_mgr: ThermalManager,
    pub wake_storm: WakeStormDetector,
    pub darwin_anomaly: EvolvedAnomalyDetector,
    pub energy_tracker: EnergyTracker,
    pub outcome_tracker: OutcomeTracker,
    pub causal_graph: CausalGraph,
    pub neuromod: ApolloNeuromodulator,
    pub skill_registry: SkillRegistry,
    pub specialist_accuracy: SpecialistAccuracyTracker,
    pub effectiveness_tracker: EffectivenessTracker,
    pub cache_warmer: apollo_engine::engine::cache_warmer::CacheWarmer,
    pub display_turbo: apollo_engine::engine::display_turbo::DisplayTurbo,
    pub io_shaper: apollo_engine::engine::io_tiering::IoShaper,
    pub thermal_bailout: ThermalBailout,
    pub coalition_tracker: CoalitionTracker,
    pub action_queue: ActionQueue,
    pub learning_pipeline: LearningPipeline,
    pub ioreport: IOReportReader,
    pub energy_pid_tracker: EnergyPidTracker,
    pub cycle_ipc_tracker: CycleIpcTracker,
    /// First-order ODE model of post-SIGCONT RSS re-accumulation.
    /// Learns per-app τ from observed thaws and predicts RSS for the next cycle.
    pub unfreeze_decay: UnfreezeDecayModel,
    /// ODE model for compressor/swap saturation dynamics.
    /// dS/dt = dirty_rate − reclaim_rate; predicts time-to-saturation each cycle.
    pub swap_reclaim: SwapReclaimModel,
    /// Persistent state for memory budget hysteresis and rate-limiting.
    pub memory_budget: MemoryBudgetState,
    /// Self-diagnosis meta-observer over known regression classes
    /// (dedup spam, sysinfo cadence drift, reactor saturation).
    /// [Hellerstein 2004 §9] detection-only meta-observer.
    pub self_diagnosis: apollo_engine::engine::self_diagnosis::SelfDiagnosis,
    /// Cross-cycle governor state memory (SuperPlan 2026-05-06).
    /// Suppresses re-emission of identical decisions for PIDs already in
    /// the target state. Closes 87.5% journal `success: false` rate.
    pub recently_applied: apollo_engine::engine::recently_applied::RecentlyApplied,
    pub recently_applied_restore_status: apollo_engine::engine::recently_applied::RestoreStatus,
    /// Identity validation cache lifecycle owner (Sprint 3 cost recovery +
    /// Sprint 4 Fase 2 manager consolidation).
    /// Memoizes proc_pidpath/csops syscalls per (pid, start_sec, start_usec)
    /// behind a single owner that concentrates verify/notify_exited/cleanup.
    pub identity_cache: apollo_engine::engine::identity_cache_manager::IdentityCacheManager,
    /// Maintenance Purge Gate state (2026-05-10) — opportunistic non-crisis
    /// purge orchestration with asymmetric cooldown vs survival_tick.
    pub maintenance_state: apollo_engine::engine::maintenance_state::MaintenanceState,
    /// Directional companion graph (Sprint C 2026-05-10) — `P(proc | fg_app)`
    /// with Lift normalization. Protects satellites of actively-used apps
    /// from ProactivePurge without a hardcoded list.
    pub companion_graph: apollo_engine::engine::companion_graph::CompanionGraph,
    /// Time-decayed envelope of recently-active app coalitions (Sprint C
    /// 2026-05-10). Closes the gap during rapid app switching: tabbing
    /// from Antigravity to Terminal for a 3-second `git status` no longer
    /// strips Antigravity's helpers of coalition protection.
    pub active_coalitions:
        apollo_engine::engine::active_coalition_envelope::ActiveCoalitionEnvelope,
}

/// Detect hardware capabilities (core count and RAM) once at startup.
///
/// Cost is ~1ms for the sysinfo queries; call once and reuse the result.
/// Returns `(hw_cores, hw_ram_gb)`.
pub(super) fn detect_hw_caps() -> (u32, f64) {
    let hw_cores: u32 = {
        let mut s = sysinfo::System::new();
        s.refresh_cpu();
        s.cpus().len().max(1) as u32
    };
    let hw_ram_gb: f64 = {
        let mut s = sysinfo::System::new();
        s.refresh_memory();
        s.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0)
    };
    (hw_cores, hw_ram_gb)
}

impl DaemonSubsystems {
    pub(super) fn new() -> Self {
        let mut outcome_tracker = OutcomeTracker::new();
        outcome_tracker.load_hop_groups(std::path::Path::new(hop_groups_path()));

        let mut skill_registry = SkillRegistry::new();
        skill_registry.load(std::path::Path::new(skills_path()));

        let (recently_applied_cache, restore_status) =
            apollo_engine::engine::recently_applied::RecentlyApplied::load_from_disk(
                std::path::Path::new(recently_applied_path()),
            );

        DaemonSubsystems {
            analytics: AnalyticsEngine::new(),
            mem_analyzer: MemoryAnalyzer::new(),
            power_mgr: PowerManager::new(),
            proc_recovery: ProcessRecoveryManager::new(),
            swap_predictor: SwapPredictor::new(),
            syscall_classifier: SyscallClassifier::new(),
            network_monitor: NetworkMonitor::new(),
            thermal_mgr: ThermalManager::new(),
            wake_storm: WakeStormDetector::new(),
            darwin_anomaly: EvolvedAnomalyDetector::new(),
            energy_tracker: EnergyTracker::new(),
            outcome_tracker,
            causal_graph: CausalGraph::new(),
            neuromod: ApolloNeuromodulator::new(),
            skill_registry,
            specialist_accuracy: SpecialistAccuracyTracker::new(),
            effectiveness_tracker: EffectivenessTracker::new(),
            cache_warmer: apollo_engine::engine::cache_warmer::CacheWarmer::new(),
            display_turbo: apollo_engine::engine::display_turbo::DisplayTurbo::new(),
            io_shaper: apollo_engine::engine::io_tiering::IoShaper::new(),
            thermal_bailout: ThermalBailout::new(),
            coalition_tracker: CoalitionTracker::new(),
            action_queue: ActionQueue::new(20, 100),
            learning_pipeline: LearningPipeline::new(),
            ioreport: IOReportReader::new(),
            energy_pid_tracker: EnergyPidTracker::new(),
            cycle_ipc_tracker: CycleIpcTracker::new(),
            unfreeze_decay: UnfreezeDecayModel::new(),
            swap_reclaim: SwapReclaimModel::new(),
            memory_budget: MemoryBudgetState::default(),
            self_diagnosis: apollo_engine::engine::self_diagnosis::SelfDiagnosis::new(
                if unsafe { libc::geteuid() } == 0 {
                    std::path::PathBuf::from("/var/lib/apollo/self_diagnosis.jsonl")
                } else {
                    std::path::PathBuf::from("/tmp/apollo_self_diagnosis.jsonl")
                },
            ),
            recently_applied: recently_applied_cache,
            recently_applied_restore_status: restore_status,
            identity_cache:
                apollo_engine::engine::identity_cache_manager::IdentityCacheManager::new(),
            maintenance_state: apollo_engine::engine::maintenance_state::MaintenanceState::new(),
            companion_graph: apollo_engine::engine::companion_graph::CompanionGraph::new(),
            active_coalitions:
                apollo_engine::engine::active_coalition_envelope::ActiveCoalitionEnvelope::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_hw_caps_sane() {
        let (cores, ram_gb) = detect_hw_caps();
        assert!(cores >= 1, "cores must be >= 1, got {cores}");
        assert!(ram_gb >= 1.0, "ram_gb must be >= 1.0, got {ram_gb}");
    }

    #[test]
    fn daemon_subsystems_constructs_without_panic() {
        // Characterization test: every field in DaemonSubsystems::new() must construct
        // without panicking regardless of filesystem state (missing hop_groups / skills
        // files are silently tolerated by the loaders). [Feathers 2004 §11]
        let _ = DaemonSubsystems::new();
    }
}
