//! Daemon initialization helpers — subsystem construction for apollo-optimizerd.
//!
//! `DaemonSubsystems` bundles the stateless, zero-dependency subsystems that are
//! constructed once at startup and then moved into the main loop.  Grouping them
//! here reduces the line count in `main.rs` and provides a single place to track
//! which subsystems exist.

use apollo_optimizer::engine::action_queue::ActionQueue;
use apollo_optimizer::engine::analytics::AnalyticsEngine;
use apollo_optimizer::engine::causal_graph::CausalGraph;
use apollo_optimizer::engine::coalition::CoalitionTracker;
use apollo_optimizer::engine::daemon_helpers::{hop_groups_path, skills_path, recently_applied_path};
use apollo_optimizer::engine::effectiveness_tracker::EffectivenessTracker;
use apollo_optimizer::engine::energy::EnergyTracker;
use apollo_optimizer::engine::energy_pid::EnergyPidTracker;
use apollo_optimizer::engine::evolved_anomaly::EvolvedAnomalyDetector;
use apollo_optimizer::engine::ioreport::IOReportReader;
use apollo_optimizer::engine::learning_pipeline::LearningPipeline;
use apollo_optimizer::engine::memory_analyzer::MemoryAnalyzer;
use apollo_optimizer::engine::network_monitor::NetworkMonitor;
use apollo_optimizer::engine::network_optimizer::NetworkOptimizer;
use apollo_optimizer::engine::neuromodulator::ApolloNeuromodulator;
use apollo_optimizer::engine::optimization_skills::SkillRegistry;
use apollo_optimizer::engine::outcome_tracker::OutcomeTracker;
use apollo_optimizer::engine::power_management::PowerManager;
use apollo_optimizer::engine::predictive_agent::SpecialistAccuracyTracker;
use apollo_optimizer::engine::process_recovery::ProcessRecoveryManager;
use apollo_optimizer::engine::swap_predictor::SwapPredictor;
use apollo_optimizer::engine::syscall_classifier::SyscallClassifier;
use apollo_optimizer::engine::thermal_bailout::ThermalBailout;
use apollo_optimizer::engine::thermal_manager::ThermalManager;
use apollo_optimizer::engine::thread_selfcounts::CycleIpcTracker;
use apollo_optimizer::engine::swap_reclaim::SwapReclaimModel;
use apollo_optimizer::engine::unfreeze_decay::UnfreezeDecayModel;
use apollo_optimizer::engine::wake_storm_detector::WakeStormDetector;
use crate::daemon_memory_budget::MemoryBudgetState;

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
    pub net_optimizer: NetworkOptimizer,
    pub energy_tracker: EnergyTracker,
    pub outcome_tracker: OutcomeTracker,
    pub causal_graph: CausalGraph,
    pub neuromod: ApolloNeuromodulator,
    pub skill_registry: SkillRegistry,
    pub specialist_accuracy: SpecialistAccuracyTracker,
    pub effectiveness_tracker: EffectivenessTracker,
    pub cache_warmer: apollo_optimizer::engine::cache_warmer::CacheWarmer,
    pub display_turbo: apollo_optimizer::engine::display_turbo::DisplayTurbo,
    pub io_shaper: apollo_optimizer::engine::io_tiering::IoShaper,
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
    pub self_diagnosis: apollo_optimizer::engine::self_diagnosis::SelfDiagnosis,
    /// Cross-cycle governor state memory (SuperPlan 2026-05-06).
    /// Suppresses re-emission of identical decisions for PIDs already in
    /// the target state. Closes 87.5% journal `success: false` rate.
    pub recently_applied: apollo_optimizer::engine::recently_applied::RecentlyApplied,
    pub recently_applied_restore_status: apollo_optimizer::engine::recently_applied::RestoreStatus,
    /// Identity validation cache lifecycle owner (Sprint 3 cost recovery +
    /// Sprint 4 Fase 2 manager consolidation).
    /// Memoizes proc_pidpath/csops syscalls per (pid, start_sec, start_usec)
    /// behind a single owner that concentrates verify/notify_exited/cleanup.
    pub identity_cache: apollo_optimizer::engine::identity_cache_manager::IdentityCacheManager,
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
            apollo_optimizer::engine::recently_applied::RecentlyApplied::load_from_disk(
                std::path::Path::new(recently_applied_path())
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
            net_optimizer: NetworkOptimizer::new(),
            energy_tracker: EnergyTracker::new(),
            outcome_tracker,
            causal_graph: CausalGraph::new(),
            neuromod: ApolloNeuromodulator::new(),
            skill_registry,
            specialist_accuracy: SpecialistAccuracyTracker::new(),
            effectiveness_tracker: EffectivenessTracker::new(),
            cache_warmer: apollo_optimizer::engine::cache_warmer::CacheWarmer::new(),
            display_turbo: apollo_optimizer::engine::display_turbo::DisplayTurbo::new(),
            io_shaper: apollo_optimizer::engine::io_tiering::IoShaper::new(),
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
            self_diagnosis: apollo_optimizer::engine::self_diagnosis::SelfDiagnosis::new(
                if unsafe { libc::geteuid() } == 0 {
                    std::path::PathBuf::from("/var/lib/apollo/self_diagnosis.jsonl")
                } else {
                    std::path::PathBuf::from("/tmp/apollo_self_diagnosis.jsonl")
                },
            ),
            recently_applied: recently_applied_cache,
            recently_applied_restore_status: restore_status,
            identity_cache: apollo_optimizer::engine::identity_cache_manager::IdentityCacheManager::new(),
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
