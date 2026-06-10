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
    /// D5 windowed source for AIS `safety_compliance()`. See
    /// `crate::engine::survival_window`. Written by `daemon_survival_tick`.
    /// Not persisted across process restarts (fresh-on-restart per
    /// design Risk 2 mitigation; future sprint may piggyback LearnedState).
    pub survival_window: crate::engine::survival_window::SurvivalActivationWindow,
}

impl MetricsState {
    /// Synchronize metrics from the lock-free hot path buffer into this Mutex-protected state.
    /// Establish p95/durations based on raw microsecond counters.
    pub fn sync_from_lockfree(&mut self, lf: &crate::engine::lse_counters::MetricsSnapshot) {
        self.metrics.cycles = lf.cycles;
        // Action outcome totals are owned by ExecuteOutcomes/metrics_reporter
        // and direct executor side paths. The legacy LSE counters
        // (`actions_applied`, `freezes`, `unfreezes`, `throttles`,
        // `throttle_reverted`) are not written by the production executor;
        // mapping them here clobbers real action totals with zero during
        // periodic sync.

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

        // Phase 3.2 — Arousal-Modulated NARS Decay observability
        self.metrics.arousal_decay_accelerations_total = lf.arousal_decay_accelerations_total;

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
        // Sprint 12 Convergence #5 (2026-05-17). Producer = should_fire
        // BusSaturated branch in daemon_maintenance_tick. Counter stays
        // at 0 until the caller wires bus_saturated from the existing
        // G12 fallback (entropy_anomaly > 2.0 OR amc_bandwidth_pct > 0.80).
        self.metrics.maintenance_purge_skipped_bus_saturated_total =
            lf.maintenance_purge_skipped_bus_saturated_total;

        // Phase 5.2 — Battery-aware cost penalty (Sprint 8, 2026-05-16).
        // Producers are NOT wired in this commit (OPENS: 1) — the counter
        // remains 0 in prod until decide_actions invokes the penalty
        // function and increments the LSE counter. Plumbing the snapshot
        // surface now keeps this in lockstep with skill_aware_modulations_total
        // (Phase 3.1) and avoids a second touch on daemon_state.rs when wiring.
        self.metrics.battery_aware_penalty_emissions_total =
            lf.battery_aware_penalty_emissions_total;

        // Phase 4.2 — External-event causal attribution (Sprint 7, 2026-05-16).
        // Surface per-kind blame totals so runtime_metrics.json shows how
        // many recent pressure-drop edges had their credit confounded by
        // thermal / disk / network events. The wiring of producers
        // (CausalGraph::record_external_event call sites) is deferred to a
        // follow-up commit; today these counters can only increase via
        // tests, so prod values will be 0 until producers land.
        self.metrics.causal_external_thermal_blames_total = lf.causal_external_thermal_blames_total;
        self.metrics.causal_external_disk_blames_total = lf.causal_external_disk_blames_total;
        self.metrics.causal_external_net_blames_total = lf.causal_external_net_blames_total;

        // Phase 4.3 — Policy Rollback Guard observability (Sprint 7).
        // Per CLAUDE.md observability discipline: surface both counters
        // so dashboards (and the user) can verify the guard is
        // actually running.
        self.metrics.policy_rollback_evaluations_total = lf.policy_rollback_evaluations_total;
        self.metrics.policy_rollback_executions_total = lf.policy_rollback_executions_total;

        // Phase 3.3 — Cross-Group Companion Attention (Sprint 6, 2026-05-16).
        // Producers are NOT wired in this commit (OPENS: 1) — the counter
        // remains 0 in prod until the daemon main-loop invokes
        // CompanionGraph::propagate_attention_across_groups and bumps the
        // LSE counter by the number of inferred triples. Plumbed now to
        // avoid a second touch on daemon_state.rs when wiring lands.
        self.metrics.companion_cross_group_inferences_total =
            lf.companion_cross_group_inferences_total;

        // Phase 4.1 — Adaptive Drift Threshold raises (Sprint 7, 2026-05-16).
        // Producers are NOT wired in this commit (OPENS: 1) — the counter
        // remains 0 in prod until the caller in `learning_tick` invokes
        // `AdaptiveDriftThreshold::recommended_threshold(...)` and calls
        // `add_adaptive_drift_threshold_raises(1)` when the return value
        // exceeds the supplied base. Plumbing the snapshot surface now
        // keeps this in lockstep with skill_aware_modulations_total
        // (Phase 3.1) and the rest of the Sprint 7 batch.
        self.metrics.adaptive_drift_threshold_raises_total =
            lf.adaptive_drift_threshold_raises_total;

        // Phase 5.1 — User-presence suppression (Sprint 8, 2026-05-16).
        // Producers (decide_actions cost composition / cognitive tick
        // specialist voting) are NOT wired in this commit (OPENS: 1).
        // Plumb the snapshot surface now so the dashboard counter is
        // ready the moment the modulator is invoked, mirroring 3.1/5.2.
        self.metrics.user_presence_suppressions_total = lf.user_presence_suppressions_total;

        // Phase 5.3 — Structured-rationale attachments (Sprint 8, 2026-05-16).
        // Producers are NOT wired in this commit (OPENS: 1) — the counter
        // remains 0 in prod until journal write-sites start calling
        // `JournalEntry::with_rationale(..)` and `inc_journal_rationale_attached()`.
        // Plumbing now mirrors 3.1/5.2 and avoids a second touch on
        // daemon_state.rs when wiring lands.
        self.metrics.journal_rationales_attached_total = lf.journal_rationales_attached_total;

        // Phase 4.3.1 — Specialist accuracy purge inhibitions (Sprint 8,
        // 2026-05-16). Mirrors the Phase 2 outcome_tracker / causal_graph
        // post-purge inhibition pattern. Producer is wired in this commit
        // at daemon_cognitive_tick::apply_specialist_voting; the counter
        // increments each cycle where the 30 s purge guard skipped the
        // EMA accuracy update.
        self.metrics.specialist_accuracy_purge_inhibitions_total =
            lf.specialist_accuracy_purge_inhibitions_total;

        // Phase 2 god-lock decomposition (Sprint 8, 2026-05-16): migrate
        // habituation_skips OFF the metrics mutex. Producer is the
        // lock-free `add_habituation_skips` call in
        // `daemon_cognitive_tick::update_habituation_state`. The legacy
        // `RuntimeMetrics.habituation_skips` field stays in place
        // (AIS runtime benchmark reads it via `rm_u("habituation_skips")`),
        // populated FROM the atomic here — single source of truth.
        self.metrics.habituation_skips = lf.habituation_skips_total;

        // Phase C SCORER-OVERRIDE (Sprint 11 finale, 2026-05-16).
        // Asymmetric scorer/gate disagreement counters. Producer is the
        // `apply_scorer_override` call site in `decide_actions` (gate
        // tower verdict → scorer cross-check → conditional override).
        // Both counters stay at 0 until shadow_signals publishes enough
        // signal for the scorer to disagree strongly with the gate, at
        // which point dashboards can verify the partial cutover is
        // actually engaging in prod (the "tautology trap" mitigation
        // CLAUDE.md flags). Mirrors the Phase 3.1 / 5.2 plumbing pattern.
        self.metrics.scorer_override_rejects_total = lf.scorer_override_rejects_total;
        self.metrics.scorer_disagreement_strong_accepts_total =
            lf.scorer_disagreement_strong_accepts_total;

        // Phase D PURGE-INHIBITION (Sprint 12 candidate #1, 2026-05-17).
        // Producer is the `signal_intelligence::step` swap branch when
        // MaintenanceState reports a recent purge. Counter stays at 0
        // until the daemon actually triggers vm_purge — proves the loop
        // closes when the maintenance tick fires.
        self.metrics.purge_inhibition_skips_total = lf.purge_inhibition_skips_total;

        // RAM Phase B (2026-06-03) — mediator chokepoint counters. Stay at
        // 0 until Phase C+ ports actual effectors through mediate(). The
        // wiring here proves the lock-free → runtime metrics path is live
        // and prevents Sprint 9 silent-telemetry-death pattern (4b13a39).
        self.metrics.mediator_blocks_total = lf.mediator_blocks_total;
        self.metrics.mediator_noop_writes_total = lf.mediator_noop_writes_total;
        self.metrics.mediator_postcondition_violation_total =
            lf.mediator_postcondition_violation_total;

        // Sprint follow-up (2026-06-05) — Silent-telemetry-death fix.
        // Mirror the five LSE counters added in this sprint into
        // `RuntimeMetrics` so they reach `runtime_metrics.json`.
        // Without these five lines the counters increment forever but
        // never surface (the Sprint 9 `4b13a39` regression class). See
        // RuntimeMetrics field doc-comments for producer notes.
        self.metrics.ac_cache_evictions_total = lf.ac_cache_evictions_total;
        self.metrics.mediator_thread_policy_total = lf.mediator_thread_policy_total;
        self.metrics.pid_recycle_blocks_total = lf.pid_recycle_blocks_total;
        self.metrics.policy_scorer_uncertainty_saturated_total =
            lf.policy_scorer_uncertainty_saturated_total;
        self.metrics.effect_decay_detected_total = lf.effect_decay_detected_total;
        self.metrics.effect_decay_hp_mach_attempts_total = lf.effect_decay_hp_mach_attempts_total;
        self.metrics.sysctl_governor_realtime_call_inhibit_total =
            lf.sysctl_governor_realtime_call_inhibit_total;
        self.metrics.cooperation_jetsam_hints_total = lf.cooperation_jetsam_hints_total;
        self.metrics.zombie_dead_weight_detected_total = lf.zombie_dead_weight_detected_total;
        self.metrics.zombie_actions_emitted_total = lf.zombie_actions_emitted_total;
        // B.2 replayd gate (2026-06-09). Screen-capture-deciding realtime
        // inhibits — mirrored here to avoid the silent-telemetry-death
        // pattern (Sprint 9 `4b13a39`): without this line the counter
        // increments forever but never reaches `runtime_metrics.json`.
        self.metrics.sysctl_governor_screen_capture_inhibit_total =
            lf.sysctl_governor_screen_capture_inhibit_total;

        // Approach 2 (2026-06-07). OutcomeTracker class-reclassification gate
        // excluded a hard-protected entry from the `low_value_names` signal
        // because `safety::hard_protected_contains(name)` is true. Producer:
        // `PatternWeight::effectiveness_for_classification` in outcome_tracker.
        // Mirroring here closes the silent-telemetry-death pattern (Sprint 9
        // `4b13a39`) — without this line the counter increments forever but
        // never reaches `runtime_metrics.json`.
        self.metrics.hard_protected_reclassify_excluded_total =
            lf.hard_protected_reclassify_excluded_total;

        // Group C (2026-06-06) — Invariant #13 port-hub gate observability.
        // `port_hub_blocks_total` must rise non-zero before this gate can
        // be claimed to do anything in prod; `probe_unavailable_total`
        // distinguishes "no demote candidates exceeded threshold" from
        // "gate observationally dark" (entitlement-denied task_for_pid).
        // The third counter belongs to the parallel Dempster-Shafer
        // aggregator and stays at 0 until `policy_aggregator_mode = "ds"`.
        self.metrics.mediator_port_hub_blocks_total = lf.mediator_port_hub_blocks_total;
        self.metrics.mediator_port_hub_probe_unavailable_total =
            lf.mediator_port_hub_probe_unavailable_total;
        self.metrics.policy_scorer_ds_high_conflict_fallback_total =
            lf.policy_scorer_ds_high_conflict_fallback_total;

        // Brave-Boost feedback loop fix (2026-06-07, APPROACH 1).
        // Producer = `decide_actions` BOOST arm guard. Stays at 0 unless a
        // hard-protected name reaches an unguarded Boost emit site. The
        // explicit copy here is the silent-telemetry-death guard (Sprint 9
        // `4b13a39`): without it the LSE counter increments forever but
        // never surfaces in `runtime_metrics.json`.
        self.metrics.hard_protected_boost_skipped_total = lf.hard_protected_boost_skipped_total;

        // Approach-3 wire (2026-06-07). Producer =
        // `learned_state::poke_rollback_guard_via_decay`. Increments only
        // when ≥5 hard-protected disagreements land in the 5-min effect-
        // decay sliding window AND the rollback guard's cooldown is clear.
        // Same silent-telemetry-death discipline (Sprint 9 `4b13a39`):
        // without this explicit copy the counter would never reach
        // `runtime_metrics.json` even when the rollback fires.
        self.metrics.policy_rollback_triggered_by_decay_total =
            lf.policy_rollback_triggered_by_decay_total;

        // FIX-4-v2 (2026-06-07). Producer = `execute_actions` Boost +
        // SetThreadQoS arms when the pre-syscall capability chain fails
        // (caps.can_taskpolicy = false, qos_mgr = None, or the Mach
        // syscall returned success = false). Silent-telemetry-death
        // discipline (Sprint 9 `4b13a39`) requires this explicit copy
        // — without it the counter increments forever on a degraded
        // capability surface but never surfaces in `runtime_metrics.json`,
        // hiding the very pathology the counter exists to expose.
        self.metrics.effect_decay_phantom_enroll_skipped_total =
            lf.effect_decay_phantom_enroll_skipped_total;

        // Sprint 12 Convergence #4 (2026-05-17). Producer is the daemon
        // main-loop convergence probe (after the cycle's
        // run_signal_tick, when both lse counters are fresh and the
        // causal-graph has had its update). Counter stays at 0 until
        // thermal pressure plus scorer disagreement coincide.
        self.metrics.causal_thermal_scorer_override_alignments_total =
            lf.causal_thermal_scorer_override_alignments_total;

        // Sprint 12 Convergence #1 (2026-05-17). Producer is
        // decide_actions cold-thread loop when `companion_of_fg_pids`
        // hits AND `dram_bandwidth_pct < 0.50`. Stays at 0 until the
        // user is running a multi-process foreground workflow with the
        // bus below the safety floor.
        self.metrics.companion_affinity_alignments_total = lf.companion_affinity_alignments_total;

        // Sprint 13 Pressure-Router Gate (2026-05-30). Producer is the
        // daemon main-loop companion-observation block that skips the
        // observe_cycle + Phase 3.3 propagation under low pressure
        // (pressure < mid_entry, modulo-4 fallback miss). Stays at 0
        // when pressure stays at/above the workload mid_entry — the
        // gate then degenerates into "always observe".
        self.metrics.companion_observe_router_skips_total = lf.companion_observe_router_skips_total;
        // Sprint 12 perf-fix (2026-05-30). Producer is the main loop
        // `companion_of_fg_pids` derivation at
        // `apollo-optimizerd/main.rs:3317`. Bumps every cycle the
        // memoization cache returned a hit instead of rebuilding the
        // HashSet. In steady state (single fg app + stable
        // top_processes) the ratio hits / cycles ≈ 1.0.
        self.metrics.companion_fg_cache_hits_total = lf.companion_fg_cache_hits_total;
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
            survival_window: crate::engine::survival_window::SurvivalActivationWindow::new(),
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

    /// S10 cutover (2026-06-06): Hellerstein settling-time observer.
    /// Producers in `execute_actions.rs` record post-Receipt; consumer
    /// in `daemon_cycle_tail.rs::drain_effect_decay` drains once per
    /// cycle, re-reads the observable, and bumps
    /// `effect_decay_detected_total` on mismatch. Mutex guards a short
    /// VecDeque push/pop — no syscall under guard.
    pub effect_decay: Arc<Mutex<crate::engine::effect_decay::DecayWatchdog>>,

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
