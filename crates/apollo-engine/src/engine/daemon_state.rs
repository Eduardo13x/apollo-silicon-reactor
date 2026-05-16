//! Daemon State — grouped sub-structs for SharedState.
//!
//! Consolidates 20+ individual Mutex fields into 6 domain-specific groups.
//! Each group is behind a single Mutex, reducing lock operations by ~40%.
//!
//! Domain groups:
//! - MetricsState: runtime metrics, thermal, reactor counters
//! - PolicyState: optimization profile, governor, learned policy
//! - ProcessState: frozen processes, blockers, wake state
//! - HardwareState: hardware snapshots, QoS, sysctl governor
//! - LlmDomainState: LLM config/state and associated paths
//! - UsageDomainState: usage model and tracker

use std::collections::{HashMap, VecDeque};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use chrono::{DateTime, Utc};

use crate::engine::adaptive_governor::AdaptiveGovernor;
use crate::engine::circuit_breaker::CircuitBreaker;
use crate::engine::daemon_helpers::WakeRuntimeState;
use crate::engine::degradation::DegradationController;
use crate::engine::iokit_sensors::HardwareSnapshot;
use crate::engine::llm::{LearnedPolicy, LlmConfig, LlmState};
use crate::engine::mach_qos::MachQoSManager;
use crate::engine::profile_governor::ProfileGovernor;
use crate::engine::sysctl_governor::SysctlGovernorStatus;
use crate::engine::thermal_interrupt::ResourceInterruptState;
use crate::engine::types::{
    BlockerScore, FrozenEntry, LatencyTarget, OptimizationProfile, ProfileTransition,
    RuntimeMetrics,
};
use crate::engine::usage_model::UsageModel;

// ── Metrics Domain ──────────────────────────────────────────────────────────

/// Runtime metrics, thermal state, reactor counters — the "dashboard" data.
/// Highest contention group (~32 accesses), mitigated by try_lock in socket handler.
///
/// Cross-crate visibility: used by apollo-optimizerd daemon_freeze_executor.rs and main.rs
/// to access per-cycle metrics state. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
pub struct MetricsState {
    pub metrics: RuntimeMetrics,
    pub throttle_level: String,
    pub thermal_state: String,
    /// Updated by reactor thread on thermal events.
    pub thermal_level_real: String,
    pub fast_tick_until: Option<Instant>,
    pub reactor_event_weight: f64,
    pub reactor_status: ReactorStatus,
}

impl MetricsState {
    /// Synchronize metrics from the lock-free hot path buffer into this Mutex-protected state.
    /// Establish p95/durations based on raw microsecond counters.
    pub fn sync_from_lockfree(&mut self, lf: &crate::engine::lse_counters::MetricsSnapshot) {
        self.metrics.cycles = lf.cycles;
        self.metrics.boosts_applied = lf.actions_applied; // map to boosts for now or split
        self.metrics.freezes_applied = lf.freezes;
        self.metrics.unfreezes_applied = lf.unfreezes;
        self.metrics.throttles_applied = lf.throttles;
        self.metrics.throttle_reverted = lf.throttle_reverted;

        // Latency durations (convert us -> ms)
        self.metrics.p95_cycle_ms = lf.cycle_time_us as f64 / 1000.0;
        self.metrics.refresh_duration_ms = lf.refresh_duration_us as f64 / 1000.0;
        // Sprint 3 Phase B — flush restore_status_* counters from lf to runtime metrics.
        self.metrics.restore_status_missing = lf.restore_status_missing;
        self.metrics.restore_status_restored_n = lf.restore_status_restored_n;
        self.metrics.restore_status_discarded_corrupt = lf.restore_status_discarded_corrupt;
        self.metrics.restore_status_discarded_clock_delta = lf.restore_status_discarded_clock_delta;
        self.metrics.restore_status_discarded_boot_crossed =
            lf.restore_status_discarded_boot_crossed;
        // Sprint 3 Phase A4 — flush identity_cache_* counters from lf to runtime metrics.
        self.metrics.identity_cache_hits = lf.identity_cache_hits;
        self.metrics.identity_cache_misses = lf.identity_cache_misses;
        self.metrics.identity_cache_evictions = lf.identity_cache_evictions;
        self.metrics.identity_cache_ttl_expired = lf.identity_cache_ttl_expired;
        self.metrics.identity_cache_exit_invalidations = lf.identity_cache_exit_invalidations;
        self.metrics.identity_proc_pidpath_calls = lf.identity_proc_pidpath_calls;
        // Sprint 4 Phase 5 — flush actions_pushed_* counters from lf to runtime metrics.
        // Invariant: Σ(typed per-variant) + actions_pushed_raw_total ==
        // total emitted (push_raw does not double-count per-variant).
        self.metrics.actions_pushed_throttle_total = lf.actions_pushed_throttle_total;
        self.metrics.actions_pushed_freeze_total = lf.actions_pushed_freeze_total;
        self.metrics.actions_pushed_unfreeze_total = lf.actions_pushed_unfreeze_total;
        self.metrics.actions_pushed_boost_total = lf.actions_pushed_boost_total;
        self.metrics.actions_pushed_set_memorystatus_total =
            lf.actions_pushed_set_memorystatus_total;
        self.metrics.actions_pushed_set_thread_qos_total = lf.actions_pushed_set_thread_qos_total;
        self.metrics.actions_pushed_set_sysctl_total = lf.actions_pushed_set_sysctl_total;
        self.metrics.actions_pushed_toggle_spotlight_total =
            lf.actions_pushed_toggle_spotlight_total;
        self.metrics.actions_pushed_quarantine_daemon_total =
            lf.actions_pushed_quarantine_daemon_total;
        self.metrics.actions_pushed_raw_total = lf.actions_pushed_raw_total;
        self.metrics.actions_rejected_shape_total = lf.actions_rejected_shape_total;
        self.metrics.memory_budget_duration_ms = lf.memory_budget_duration_us as f64 / 1000.0;
        self.metrics.reactor_duration_ms = lf.reactor_duration_us as f64 / 1000.0;

        // Phase 2 God-Lock Decomposition (Sprint 5)
        self.metrics.profile_floor_hits = lf.profile_floor_hits;
        self.metrics.paging_hints_applied = lf.paging_hints_applied;
        self.metrics.iokit_errors = lf.iokit_errors;
        self.metrics.reactor_pulses = lf.reactor_pulses;
        self.reactor_event_weight = lf.reactor_event_weight;

        // Phase 3.1 — Skill-Aware Prediction observability
        self.metrics.skill_aware_modulations_total = lf.skill_aware_modulations_total;

        // Reactor pulses — 2026-05-12: removed `= lf.signals_sent` overwrite.
        // The lock-free `signals_sent` field is defined in lse_counters.rs:57 but
        // is NEVER incremented anywhere in the codebase. The authoritative writers
        // are `daemon_reactor.rs:164` (per kqueue iter) and `metrics_reporter.rs:389`
        // (per cycle when reactor_weight>0.2). Clobbering with `signals_sent` reset
        // the counter to 0 every cycle and broke the liveness watchdog at
        // main.rs:1553 (would mark reactor "stalled" even when healthy). The
        // in-memory counter is the source of truth — leave it untouched here.
        // Maintenance Purge Gate (2026-05-10) — Sprint 3 telemetry sync chain
        self.metrics.maintenance_purge_total = lf.maintenance_purge_total;
        self.metrics.maintenance_purge_skipped_pressure_total =
            lf.maintenance_purge_skipped_pressure_total;
        self.metrics.maintenance_purge_skipped_swap_floor_total =
            lf.maintenance_purge_skipped_swap_floor_total;
        self.metrics.maintenance_purge_skipped_growing_total =
            lf.maintenance_purge_skipped_growing_total;
        self.metrics.maintenance_purge_skipped_idle_total = lf.maintenance_purge_skipped_idle_total;
        self.metrics.maintenance_purge_skipped_build_mode_total =
            lf.maintenance_purge_skipped_build_mode_total;
        self.metrics.maintenance_purge_skipped_rate_limit_total =
            lf.maintenance_purge_skipped_rate_limit_total;

        // Phase 5.2 — Battery-aware cost penalty (Sprint 8, 2026-05-16).
        // Producers are NOT wired in this commit (OPENS: 1) — the counter
        // remains 0 in prod until decide_actions invokes the penalty
        // function and increments the LSE counter. Plumbing the snapshot
        // surface now keeps this in lockstep with skill_aware_modulations_total
        // (Phase 3.1) and avoids a second touch on daemon_state.rs when wiring.
        self.metrics.battery_aware_penalty_emissions_total =
            lf.battery_aware_penalty_emissions_total;

        // Phase 5.1 — User-presence suppression (Sprint 8, 2026-05-16).
        // Producers (decide_actions cost composition / cognitive tick
        // specialist voting) are NOT wired in this commit (OPENS: 1).
        // Plumb the snapshot surface now so the dashboard counter is
        // ready the moment the modulator is invoked, mirroring 3.1/5.2.
        self.metrics.user_presence_suppressions_total = lf.user_presence_suppressions_total;
    }
}

impl Default for MetricsState {
    fn default() -> Self {
        Self {
            metrics: RuntimeMetrics::default(),
            throttle_level: String::new(),
            thermal_state: String::new(),
            thermal_level_real: String::new(),
            fast_tick_until: None,
            reactor_event_weight: 0.0,
            reactor_status: ReactorStatus::default(),
        }
    }
}

/// Reactor thread counters and status.
///
/// Cross-crate visibility: constructed in apollo-optimizerd main.rs and daemon_memory_budget.rs
/// tests. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
pub struct ReactorStatus {
    pub events_total: u64,
    pub events_mem: u64,
    pub events_thermal: u64,
    pub events_spawn: u64,
    pub events_power: u64,
    pub last_event_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
    /// "normal" | "degraded"
    pub mode: String,
    /// "ok" | "stalled" | "collector-stalled"
    pub health: String,
}

impl Default for ReactorStatus {
    fn default() -> Self {
        Self {
            events_total: 0,
            events_mem: 0,
            events_thermal: 0,
            events_spawn: 0,
            events_power: 0,
            last_event_at: None,
            last_error: None,
            mode: "normal".to_string(),
            health: "ok".to_string(),
        }
    }
}

// ── Policy Domain ───────────────────────────────────────────────────────────

/// Optimization profile, governor, learned policy — the "brain" state.
///
/// Cross-crate visibility: constructed in apollo-optimizerd main.rs and daemon_memory_budget.rs,
/// daemon_dispatch_tick.rs tests; contains `ProfileGovernor`, `AdaptiveGovernor` etc. which
/// are used by daemon ticks. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
pub struct PolicyState {
    pub profile: OptimizationProfile,
    pub governor: ProfileGovernor,
    pub learned_policy: LearnedPolicy,
    pub adaptive_governor: AdaptiveGovernor,
    pub latency_target: LatencyTarget,
    pub timeline: VecDeque<ProfileTransition>,
    /// Resilience: circuit breaker for external calls (LLM, sysctl, etc.).
    pub circuit_breaker: CircuitBreaker,
    /// Resilience: graceful degradation controller for policy quality tiers.
    pub degradation: DegradationController,
}

// ── Process Domain ──────────────────────────────────────────────────────────

/// Blockers + wake state — the "process management" data.
/// Note: frozen_state lives as a flat SharedState field (sentinel coupling; see feedback_lock_migration.md).
///
/// Cross-crate visibility: constructed in apollo-optimizerd main.rs and daemon_memory_budget.rs,
/// daemon_dispatch_tick.rs tests. Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
pub struct ProcessState {
    pub last_blockers: Vec<BlockerScore>,
    pub wake_state: WakeRuntimeState,
}

// ── Hardware Domain ─────────────────────────────────────────────────────────

/// Hardware snapshots, sysctl governor — the "hardware" layer.
/// Note: mach_qos lives as a flat SharedState field (sentinel coupling; see feedback_lock_migration.md).
///
/// Cross-crate visibility: constructed in apollo-optimizerd main.rs and daemon tests.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
pub struct HardwareState {
    pub last_hw_snapshot: Option<HardwareSnapshot>,
    pub sysctl_governor_status: SysctlGovernorStatus,
}

// ── LLM Domain ──────────────────────────────────────────────────────────────

/// LLM configuration, state, and associated file paths.
///
/// Cross-crate visibility: constructed in apollo-optimizerd main.rs and daemon tests;
/// accessed by daemon_skill_tick.rs and llm_daemon.rs. Audited 2026-05-09 during Sprint 5
/// Mes 0 workspace split.
pub struct LlmDomainState {
    pub llm_cfg: LlmConfig,
    pub llm_state: LlmState,
    /// Paths are immutable after initialization.
    pub llm_state_path: PathBuf,
    pub llm_key_path: PathBuf,
    pub learned_policy_path: PathBuf,
    pub feedback_path: PathBuf,
    pub suggestions_path: PathBuf,
}

// ── Usage Domain ────────────────────────────────────────────────────────────

/// Usage model and tracker — the "learning" data.
///
/// Cross-crate visibility: constructed in apollo-optimizerd main.rs and daemon tests.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
pub struct UsageDomainState {
    pub usage_model: UsageModel,
    pub usage_tracker: UsageTrackerState,
    pub usage_model_path: PathBuf,
    pub usage_events_path: PathBuf,
}

/// Usage model lifecycle counters.
///
/// Cross-crate visibility: constructed in apollo-optimizerd main.rs and daemon tests.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
#[derive(Default)]
pub struct UsageTrackerState {
    pub last_persist_at: Option<DateTime<Utc>>,
    pub promotions_day: Option<String>,
    pub promotions_today: u32,
}

// ── Consolidated SharedState ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reactor_status_default_counters_zero() {
        let rs = ReactorStatus::default();
        assert_eq!(rs.events_total, 0);
        assert_eq!(rs.events_mem, 0);
        assert_eq!(rs.events_thermal, 0);
        assert_eq!(rs.events_spawn, 0);
        assert_eq!(rs.events_power, 0);
        assert!(rs.last_event_at.is_none());
        assert!(rs.last_error.is_none());
    }

    #[test]
    fn reactor_status_default_mode_normal() {
        let rs = ReactorStatus::default();
        assert_eq!(rs.mode, "normal");
    }

    #[test]
    fn reactor_status_default_health_ok() {
        let rs = ReactorStatus::default();
        assert_eq!(rs.health, "ok");
    }

    #[test]
    fn usage_tracker_state_default_promotions_zero() {
        let ut = UsageTrackerState::default();
        assert_eq!(ut.promotions_today, 0);
        assert!(ut.last_persist_at.is_none());
        assert!(ut.promotions_day.is_none());
    }

    #[test]
    fn wake_runtime_state_can_be_constructed() {
        let ws = WakeRuntimeState {
            last_cycle_wallclock: chrono::Utc::now(),
            last_wake_at: None,
            post_wake_grace_until: None,
            post_wake_reclaim_until: None,
            post_wake_policy: "normal".to_string(),
        };
        assert_eq!(ws.post_wake_policy, "normal");
        assert!(ws.last_wake_at.is_none());
    }
}

/// The daemon's shared state, grouped into 6 domain-specific Mutex groups.
/// Reduces ~20 individual Mutex fields to 6 coarser-grained locks.
///
/// Cross-crate visibility: the god-node of the daemon. Every daemon tick module
/// (daemon_signal_tick, daemon_chromium_tick, daemon_skill_tick, daemon_ctx_switch_tick,
/// daemon_dispatch_tick, metrics_reporter, socket_handler, etc.) takes `Arc<SharedState>`
/// or `&SharedState`. All fields must remain `pub` for tick modules to access them.
/// Audited 2026-05-09 during Sprint 5 Mes 0 workspace split.
///
/// # Lock ordering (to prevent deadlocks)
/// Never hold two domain locks simultaneously. Acquire one, complete the
/// operation, drop, then acquire the next.
///
/// `frozen_state` and `mach_qos` are intentionally kept as flat Arc<Mutex<>>
/// fields: `frozen_state` is shared with `spawn_resource_sentinel` (16 internal
/// sites in thermal_interrupt.rs use an independent Arc reference), and `mach_qos`
/// is used as a sentinel parameter. Grouping them would cascade to those call sites.
#[derive(Clone)]
pub struct SharedState {
    pub metrics: Arc<Mutex<MetricsState>>,
    pub policy: Arc<Mutex<PolicyState>>,
    pub process: Arc<Mutex<ProcessState>>,
    pub hardware: Arc<Mutex<HardwareState>>,
    pub llm: Arc<Mutex<LlmDomainState>>,
    pub usage: Arc<Mutex<UsageDomainState>>,

    // Sentinel-coupled fields (kept flat — see doc comment above)
    pub frozen_state: Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    pub mach_qos: Arc<Mutex<MachQoSManager>>,

    /// Per-PID post-thaw cooldown set. Prevents gate_e from re-freezing a PID
    /// that was just thawed by the TTL path. See `freeze_cooldown` module.
    pub freeze_cooldown: Arc<Mutex<crate::engine::freeze_cooldown::FreezeCooldown>>,

    // Infrastructure (lock-free or low-frequency)
    pub stop: Arc<AtomicBool>,
    /// Set by socket handler when a `RevertSysctls` RPC is received.
    /// Main loop checks this flag each cycle, executes the revert, then clears it.
    pub revert_sysctls_requested: Arc<AtomicBool>,
    pub cycle_condvar: Arc<(Mutex<bool>, Condvar)>,
    pub resource_interrupt: Arc<ResourceInterruptState>,
    pub subscribers: Arc<Mutex<Vec<UnixStream>>>,

    // Read-only paths (set once at init)
    pub config_path: PathBuf,
    pub user_profile_path: PathBuf,
}
