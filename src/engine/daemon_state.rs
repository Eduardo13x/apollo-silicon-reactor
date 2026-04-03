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
use crate::engine::degradation::DegradationController;
use crate::engine::iokit_sensors::HardwareSnapshot;
use crate::engine::llm::{LearnedPolicy, LlmConfig, LlmState};
use crate::engine::profile_governor::ProfileGovernor;
use crate::engine::sysctl_governor::SysctlGovernorStatus;
use crate::engine::thermal_interrupt::ResourceInterruptState;
use crate::engine::daemon_helpers::WakeRuntimeState;
use crate::engine::mach_qos::MachQoSManager;
use crate::engine::types::{
    BlockerScore, FrozenEntry, LatencyTarget, OptimizationProfile, ProfileTransition, RuntimeMetrics,
};
use crate::engine::usage_model::UsageModel;

// ── Metrics Domain ──────────────────────────────────────────────────────────

/// Runtime metrics, thermal state, reactor counters — the "dashboard" data.
/// Highest contention group (~32 accesses), mitigated by try_lock in socket handler.
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

/// Reactor thread counters and status.
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
pub struct ProcessState {
    pub last_blockers: Vec<BlockerScore>,
    pub wake_state: WakeRuntimeState,
}

// ── Hardware Domain ─────────────────────────────────────────────────────────

/// Hardware snapshots, sysctl governor — the "hardware" layer.
/// Note: mach_qos lives as a flat SharedState field (sentinel coupling; see feedback_lock_migration.md).
pub struct HardwareState {
    pub last_hw_snapshot: Option<HardwareSnapshot>,
    pub sysctl_governor_status: SysctlGovernorStatus,
}

// ── LLM Domain ──────────────────────────────────────────────────────────────

/// LLM configuration, state, and associated file paths.
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
pub struct UsageDomainState {
    pub usage_model: UsageModel,
    pub usage_tracker: UsageTrackerState,
    pub usage_model_path: PathBuf,
    pub usage_events_path: PathBuf,
}

/// Usage model lifecycle counters.
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
            post_wake_policy: "normal".to_string(),
        };
        assert_eq!(ws.post_wake_policy, "normal");
        assert!(ws.last_wake_at.is_none());
    }
}

/// The daemon's shared state, grouped into 6 domain-specific Mutex groups.
/// Reduces ~20 individual Mutex fields to 6 coarser-grained locks.
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
    pub discrepancy_log_path: PathBuf,
    pub user_profile_path: PathBuf,
}
