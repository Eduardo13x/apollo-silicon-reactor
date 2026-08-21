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
use apollo_engine::engine::memory_regime::MemoryCapabilities;
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
#[cfg(feature = "adaptive-multicore")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "adaptive-multicore")]
use std::sync::Arc;

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

pub(super) fn detect_memory_capabilities() -> MemoryCapabilities {
    let physical_memory_bytes =
        apollo_engine::engine::sysctl_direct::read_u64("hw.memsize").unwrap_or(0);
    let page_size_bytes = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    MemoryCapabilities::new(
        physical_memory_bytes,
        u64::try_from(page_size_bytes).unwrap_or(0),
    )
    .unwrap_or_else(MemoryCapabilities::apple_silicon_fallback)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BuildCapabilities {
    pub expected_profile: String,
    pub compiled_profile: String,
    pub effective_profile: String,
    pub adaptive_feature_compiled: bool,
    pub max_worker_threads: usize,
    pub disabled_reason: String,
    pub worker_qos_intent: String,
    pub worker_qos_status: String,
    pub worker_qos_failures: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ParallelRuntimeConfig {
    pub enabled: bool,
    pub worker_threads: usize,
    pub capability_revision: u64,
    pub build: BuildCapabilities,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParallelPoolResolution {
    enabled: bool,
    pool_available: bool,
    effective_workers: usize,
}

fn resolve_parallel_pool(
    requested_workers: usize,
    build_succeeded: bool,
    observed_workers: usize,
) -> ParallelPoolResolution {
    let observed_workers = observed_workers.max(1);
    let pool_available = build_succeeded || observed_workers > 1;
    let enabled = requested_workers > 1 && pool_available && observed_workers > 1;
    ParallelPoolResolution {
        enabled,
        pool_available,
        effective_workers: if enabled { observed_workers } else { 1 },
    }
}

fn choose_parallel_workers(total_cores: usize, p_cores: usize, e_cores: usize) -> usize {
    if total_cores < 10 || p_cores < 4 || e_cores < 4 {
        return 1;
    }
    // The live M4 A/B favored 4 workers over 6 and 10 for ~675 processes.
    // Cap the shared pool so sensing cannot occupy every CPU during contention.
    p_cores.clamp(2, 4)
}

fn build_capabilities_for(
    total_cores: usize,
    p_cores: usize,
    e_cores: usize,
    feature_compiled: bool,
    pool_built: bool,
    effective_workers: usize,
) -> BuildCapabilities {
    let max_worker_threads = choose_parallel_workers(total_cores, p_cores, e_cores);
    let expected_adaptive = max_worker_threads > 1;
    let effective_adaptive =
        expected_adaptive && feature_compiled && pool_built && effective_workers > 1;
    let disabled_reason = if !expected_adaptive {
        "hardware-sequential"
    } else if !feature_compiled {
        "feature-not-compiled"
    } else if !pool_built {
        "worker-pool-unavailable"
    } else if effective_workers <= 1 {
        "insufficient-workers"
    } else {
        ""
    };
    BuildCapabilities {
        expected_profile: if expected_adaptive {
            "adaptive-multicore"
        } else {
            "sequential"
        }
        .to_string(),
        compiled_profile: if feature_compiled {
            "adaptive-multicore"
        } else {
            "sequential"
        }
        .to_string(),
        effective_profile: if effective_adaptive {
            "adaptive-multicore"
        } else {
            "sequential"
        }
        .to_string(),
        adaptive_feature_compiled: feature_compiled,
        max_worker_threads,
        disabled_reason: disabled_reason.to_string(),
        worker_qos_intent: parallel_worker_qos_intent().to_string(),
        worker_qos_status: if effective_adaptive {
            "pending"
        } else {
            "disabled"
        }
        .to_string(),
        worker_qos_failures: 0,
    }
}

/// Configure sysinfo's persistent Rayon pool before the first process refresh.
pub(super) fn configure_parallel_runtime() -> ParallelRuntimeConfig {
    let capability_graph = apollo_engine::engine::capabilities::detect_capability_graph();
    let caps = capability_graph.legacy_report();
    let fallback_total = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let p_cores = caps.p_core_count.unwrap_or(0) as usize;
    let e_cores = caps.e_core_count.unwrap_or(0) as usize;
    let total_cores = (p_cores + e_cores).max(fallback_total);
    let worker_threads = choose_parallel_workers(total_cores, p_cores, e_cores)
        .min(capability_graph.recommended_cpu_workers().max(1));

    #[cfg(feature = "adaptive-multicore")]
    {
        let qos_failures = Arc::new(AtomicU64::new(0));
        let worker_qos_failures = Arc::clone(&qos_failures);
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_threads)
            .thread_name(|index| format!("apollo-sense-{index}"))
            .start_handler(move |_| {
                if set_parallel_worker_qos() != 0 {
                    worker_qos_failures.fetch_add(1, Ordering::Relaxed);
                }
            })
            .build_global()
            .is_ok();
        let pool = resolve_parallel_pool(worker_threads, built, rayon::current_num_threads());
        let mut build = build_capabilities_for(
            total_cores,
            p_cores,
            e_cores,
            true,
            pool.pool_available,
            pool.effective_workers,
        );
        build.worker_qos_failures = qos_failures.load(Ordering::Relaxed);
        let (intent, status) = parallel_worker_qos_report(built, build.worker_qos_failures);
        build.worker_qos_intent = intent.to_string();
        build.worker_qos_status = status.to_string();
        return ParallelRuntimeConfig {
            enabled: pool.enabled,
            worker_threads: pool.effective_workers,
            capability_revision: capability_graph.revision,
            build,
        };
    }

    #[cfg(not(feature = "adaptive-multicore"))]
    {
        let _ = worker_threads;
        ParallelRuntimeConfig {
            enabled: false,
            worker_threads: 1,
            capability_revision: capability_graph.revision,
            build: build_capabilities_for(total_cores, p_cores, e_cores, false, false, 1),
        }
    }
}

fn parallel_worker_qos_intent() -> &'static str {
    "utility"
}

fn parallel_worker_qos_report(
    apollo_owned_pool: bool,
    failures: u64,
) -> (&'static str, &'static str) {
    if !apollo_owned_pool {
        ("inherited", "not-observed")
    } else if failures > 0 {
        ("utility", "failed")
    } else {
        ("utility", "ok")
    }
}

fn parallel_worker_qos_class() -> libc::c_uint {
    0x11
}

#[cfg(all(feature = "adaptive-multicore", target_os = "macos"))]
fn set_parallel_worker_qos() -> libc::c_int {
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(
                qos_class: libc::c_uint,
                relative_priority: libc::c_int,
            ) -> libc::c_int;
        }
        pthread_set_qos_class_self_np(parallel_worker_qos_class(), 0)
    }
}

#[cfg(all(feature = "adaptive-multicore", not(target_os = "macos")))]
fn set_parallel_worker_qos() -> libc::c_int {
    0
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
    fn detected_memory_capabilities_are_valid() {
        let capabilities = detect_memory_capabilities();
        assert!(capabilities.physical_memory_bytes >= 1024 * 1024 * 1024);
        assert!(capabilities.page_size_bytes.is_power_of_two());
    }

    #[test]
    fn parallel_workers_follow_heterogeneous_capacity() {
        assert_eq!(choose_parallel_workers(8, 4, 4), 1);
        assert_eq!(choose_parallel_workers(10, 4, 6), 4);
        assert_eq!(choose_parallel_workers(14, 10, 4), 4);
        assert_eq!(choose_parallel_workers(10, 0, 0), 1);
    }

    #[test]
    fn parallel_sensing_uses_efficiency_intent_not_foreground_priority() {
        assert_eq!(parallel_worker_qos_intent(), "utility");
        assert_eq!(parallel_worker_qos_class(), 0x11);
    }

    #[test]
    fn qos_report_distinguishes_apollo_owned_and_inherited_global_pools() {
        assert_eq!(parallel_worker_qos_report(true, 0), ("utility", "ok"));
        assert_eq!(
            parallel_worker_qos_report(false, 0),
            ("inherited", "not-observed")
        );
        assert_eq!(parallel_worker_qos_report(true, 2), ("utility", "failed"));
    }

    #[test]
    fn build_capabilities_distinguish_expected_compiled_and_effective_profiles() {
        let mismatch = build_capabilities_for(10, 4, 6, false, false, 1);
        assert_eq!(mismatch.expected_profile, "adaptive-multicore");
        assert_eq!(mismatch.compiled_profile, "sequential");
        assert_eq!(mismatch.effective_profile, "sequential");
        assert_eq!(mismatch.disabled_reason, "feature-not-compiled");
        assert_eq!(mismatch.max_worker_threads, 4);

        let m4 = build_capabilities_for(10, 4, 6, true, true, 4);
        assert_eq!(m4.expected_profile, "adaptive-multicore");
        assert_eq!(m4.compiled_profile, "adaptive-multicore");
        assert_eq!(m4.effective_profile, "adaptive-multicore");
        assert!(m4.disabled_reason.is_empty());

        let m1 = build_capabilities_for(8, 4, 4, false, false, 1);
        assert_eq!(m1.expected_profile, "sequential");
        assert_eq!(m1.compiled_profile, "sequential");
        assert_eq!(m1.effective_profile, "sequential");
        assert_eq!(m1.disabled_reason, "hardware-sequential");
    }

    #[test]
    fn existing_parallel_pool_is_usable_when_global_build_was_already_claimed() {
        let resolved = resolve_parallel_pool(4, false, 4);
        assert!(resolved.enabled);
        assert!(resolved.pool_available);
        assert_eq!(resolved.effective_workers, 4);
    }

    #[test]
    fn daemon_subsystems_constructs_without_panic() {
        // Characterization test: every field in DaemonSubsystems::new() must construct
        // without panicking regardless of filesystem state (missing hop_groups / skills
        // files are silently tolerated by the loaders). [Feathers 2004 §11]
        let _ = DaemonSubsystems::new();
    }
}
