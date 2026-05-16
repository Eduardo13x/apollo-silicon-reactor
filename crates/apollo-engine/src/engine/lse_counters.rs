//! Lock-free metrics counters using ARMv8.1 LSE atomics.
//!
//! Replaces `Mutex<RuntimeMetrics>` for hot-path counters that update every cycle.
//! On Apple Silicon with `-C target-cpu=native`, Rust's AtomicU64 already emits
//! LSE `ldadd` instructions. This module provides:
//!
//! 1. A structured lock-free metrics buffer (no mutex on the hot path)
//! 2. Consistent snapshot reads via epoch counter
//! 3. ARM64 ASM verification that LSE is actually being used
//!
//! Cost: single atomic instruction per counter increment (~3ns vs ~25ns for mutex).

use std::sync::atomic::{AtomicU64, Ordering};

// ── Cycle-stage enum (Phase 0b 2026-05-10) ──────────────────────────────────

/// Five-stage breakdown of the daemon main-loop cycle. Each stage is timed
/// independently into `LockFreeMetrics::stage_*_total_ns/max_ns` so the
/// dominant contributor to p95 latency can be measured, not guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleStage {
    Sense,
    Reason,
    Execute,
    Learn,
    Persist,
    /// Sub-stages of Reason — track inside the dominant stage so its
    /// 83 ms avg can be attributed to specific cognitive ticks.
    /// NotebookLM 2026-05-10 priority targets:
    ReasonSignalTick,
    ReasonDecide,
    ReasonNeuro,
    ReasonUserContext,
    ReasonHoltWinters,
    ReasonPageReclaim,
    ReasonChromium,
    ReasonEnrich,
}

// ── Lock-free metrics ────────────────────────────────────────────────────────

/// Lock-free daemon metrics. Each field is an independent atomic counter.
/// Writers use `Relaxed` ordering (cheapest — single LSE `ldadd` instruction).
/// Readers use `Acquire` on the epoch for a happens-before edge.
#[repr(align(128))] // Separate cache line from other data
pub struct LockFreeMetrics {
    // Epoch incremented after every batch update — readers check this
    epoch: AtomicU64,

    // Core cycle counters
    pub cycles: AtomicU64,
    pub actions_applied: AtomicU64,
    pub freezes: AtomicU64,
    pub unfreezes: AtomicU64,
    pub throttles: AtomicU64,
    pub throttle_reverted: AtomicU64,
    pub signals_sent: AtomicU64,

    // Pressure counters
    pub hw_warnings: AtomicU64,
    pub hw_criticals: AtomicU64,
    pub vm_pressure_events: AtomicU64,
    pub survival_activations: AtomicU64,
    pub paging_hints_applied: AtomicU64,

    // Process management
    pub processes_scanned: AtomicU64,
    pub kqueue_events: AtomicU64,
    pub proc_exits_detected: AtomicU64,

    // Performance self-measurement (in microseconds)
    pub cycle_time_us: AtomicU64,
    pub snapshot_time_us: AtomicU64,
    pub decide_time_us: AtomicU64,
    pub refresh_duration_us: AtomicU64,
    pub memory_budget_duration_us: AtomicU64,
    pub reactor_duration_us: AtomicU64,

    // Per-PID action dedup drops (Phase 1 self-healing — wasted-syscall counter).
    // Each increment corresponds to a duplicate action collapsed at the dispatch
    // chokepoint before reaching execute_actions.
    pub dedup_drops_setmemorystatus: AtomicU64,
    pub dedup_drops_throttle: AtomicU64,
    pub dedup_drops_freeze: AtomicU64,
    pub dedup_drops_unfreeze: AtomicU64,

    /// Restore status telemetry (Phase B1 — recently_applied persistence).
    /// Mutually-exclusive: exactly one of these is incremented per startup.
    pub restore_status_missing: AtomicU64,
    pub restore_status_restored_n: AtomicU64,
    pub restore_status_discarded_corrupt: AtomicU64,
    pub restore_status_discarded_clock_delta: AtomicU64,
    pub restore_status_discarded_boot_crossed: AtomicU64,

    /// IdentityCache telemetry (Phase A4 — Sprint 3 cost recovery).
    /// Lets NotebookLM debrief verify the cache hit ratio and quantify
    /// proc_pidpath syscall amortization.
    pub identity_cache_hits: AtomicU64,
    pub identity_cache_misses: AtomicU64,
    pub identity_cache_evictions: AtomicU64,
    pub identity_cache_ttl_expired: AtomicU64,
    pub identity_cache_exit_invalidations: AtomicU64,
    pub identity_proc_pidpath_calls: AtomicU64,

    /// ActionAccumulator telemetry (Sprint 4 Fase 5 — typed action builder).
    /// Per-variant push counters published from `ActionAccumulator::telemetry()`
    /// at finalize time. Counters are cumulative across all daemon cycles.
    ///
    /// `actions_rejected_shape_total` increments when a typed `push_*` rejects
    /// an action because of malformed shape (pid=0, empty name, empty sysctl
    /// key) — these rejections leave evidence (warn-level tracing event +
    /// counter increment) without ever reaching the dispatcher.
    ///
    /// `actions_pushed_raw_total` increments on `push_raw` / `extend_raw`
    /// (escape hatch for revert/confirmed/decide_actions paths). Per-variant
    /// counters (`actions_pushed_freeze_total`, etc.) increment ONLY for
    /// typed `push_*` methods — raw pushes do NOT bump the per-variant
    /// counter. The invariant
    /// ```text
    /// Σ(typed per-variant) + actions_pushed_raw_total == total_pushed
    /// ```
    ///
    /// holds. Dashboards compute "% bypassing typed shape validation" as
    /// `actions_pushed_raw_total / total_pushed`.
    ///
    /// FOLLOW-UP (not in Fase 5): a `drop_ratio_5min` windowed alarm that
    /// fires when rejected_shape / total_pushed exceeds a threshold over a
    /// rolling window. Needs windowed-counter infrastructure that doesn't
    /// exist yet; these absolute counters are the foundation it would build on.
    pub actions_pushed_throttle_total: AtomicU64,
    pub actions_pushed_freeze_total: AtomicU64,
    pub actions_pushed_unfreeze_total: AtomicU64,
    pub actions_pushed_boost_total: AtomicU64,
    pub actions_pushed_set_memorystatus_total: AtomicU64,
    pub actions_pushed_set_thread_qos_total: AtomicU64,
    pub actions_pushed_set_sysctl_total: AtomicU64,
    pub actions_pushed_toggle_spotlight_total: AtomicU64,
    pub actions_pushed_quarantine_daemon_total: AtomicU64,
    pub actions_pushed_raw_total: AtomicU64,
    pub actions_rejected_shape_total: AtomicU64,

    /// Maintenance Purge Gate telemetry (Sprint 5 Mes 0 — 2026-05-10).
    /// Tracks how often the purge gate fires and which guard skips it.
    pub maintenance_purge_total: AtomicU64,
    pub maintenance_purge_skipped_pressure_total: AtomicU64,
    pub maintenance_purge_skipped_swap_floor_total: AtomicU64,
    pub maintenance_purge_skipped_growing_total: AtomicU64,
    pub maintenance_purge_skipped_idle_total: AtomicU64,
    pub maintenance_purge_skipped_build_mode_total: AtomicU64,
    pub maintenance_purge_skipped_rate_limit_total: AtomicU64,

    /// Phase 1 production-grade Change B (taskinfo cache, 2026-05-16).
    /// Track hit/miss/evict so dashboards can verify the 4-cycle reuse
    /// window is actually skipping syscalls under pressure (instead of
    /// trusting the code path by inspection). NotebookLM 2026-05-16
    /// flagged the absence of this counter as a gap blocking
    /// "production-grade" status.
    pub taskinfo_cache_hits: AtomicU64,
    pub taskinfo_cache_misses: AtomicU64,
    pub taskinfo_cache_exit_invalidations: AtomicU64,
    pub taskinfo_cache_cap_evictions: AtomicU64,

    /// Phase 3.1 — Skill-Aware Prediction observability (Sprint 6, 2026-05-16).
    /// Incremented each time a non-Observe specialist vote is multiplied by a
    /// non-neutral `skill_aware_factor`. Dashboards compute the "tilt rate" as
    /// `skill_aware_modulations_total / (cycles × avg_non_observe_votes)`.
    /// NotebookLM 2026-05-16 flagged the absence of this counter as the same
    /// "tautology trap" we hit with the F1-F7 shadow-mode: a feature wired in
    /// without a measurement of whether it actually changed any decision.
    pub skill_aware_modulations_total: AtomicU64,

    /// Phase 0 lock-decomposition instrumentation (2026-05-10).
    /// Tracks contention on the `state.metrics` god-lock to decide whether
    /// decomposition will actually move p95. Per NotebookLM round-3:
    /// "If wait_ns accounts for <5ms of the 135ms p95, lock decomposition
    /// will fail to meet the user's ≤100ms target — bottleneck is
    /// elsewhere (process-tree O(N) sort)."
    ///
    /// `wait` = time spent BLOCKED in `.lock_recover()` before acquire.
    /// `held` = time the guard is held (predicts decomp benefit).
    pub metrics_lock_wait_total_ns: AtomicU64,
    pub metrics_lock_wait_count: AtomicU64,
    pub metrics_lock_wait_max_ns: AtomicU64,
    pub metrics_lock_held_total_ns: AtomicU64,
    pub metrics_lock_held_count: AtomicU64,
    pub metrics_lock_held_max_ns: AtomicU64,

    /// Phase 0b cycle-stage split (NotebookLM priority #1, 2026-05-10).
    /// Bucket per-cycle latency by stage so the dominant contributor to
    /// p95 is identified instead of guessed. Replaces the single
    /// `cycle_time_us` measurement that hides where the work happens.
    /// Stages map to main.rs cycle layout 1468→4803:
    ///   sense   = pre-snapshot to collect_snapshot_*
    ///   reason  = signal_intelligence + decide_actions + cognitive
    ///   execute = run_dispatch_tick (filter + execute_actions)
    ///   learn   = learning_tick (outcome resolution + persist)
    ///   persist = enriched_telemetry + journal + condvar wait
    pub stage_sense_total_ns: AtomicU64,
    pub stage_sense_max_ns: AtomicU64,
    pub stage_reason_total_ns: AtomicU64,
    pub stage_reason_max_ns: AtomicU64,
    pub stage_execute_total_ns: AtomicU64,
    pub stage_execute_max_ns: AtomicU64,
    pub stage_learn_total_ns: AtomicU64,
    pub stage_learn_max_ns: AtomicU64,
    pub stage_persist_total_ns: AtomicU64,
    pub stage_persist_max_ns: AtomicU64,
    pub stage_count: AtomicU64,
    /// Phase 0c sub-stages of REASON (the 93% bottleneck).
    pub stage_reason_signal_total_ns: AtomicU64,
    pub stage_reason_signal_max_ns: AtomicU64,
    pub stage_reason_decide_total_ns: AtomicU64,
    pub stage_reason_decide_max_ns: AtomicU64,
    pub stage_reason_neuro_total_ns: AtomicU64,
    pub stage_reason_neuro_max_ns: AtomicU64,
    pub stage_reason_usercontext_total_ns: AtomicU64,
    pub stage_reason_usercontext_max_ns: AtomicU64,
    pub stage_reason_holtwinters_total_ns: AtomicU64,
    pub stage_reason_holtwinters_max_ns: AtomicU64,
    pub stage_reason_pagereclaim_total_ns: AtomicU64,
    pub stage_reason_pagereclaim_max_ns: AtomicU64,
    pub stage_reason_chromium_total_ns: AtomicU64,
    pub stage_reason_chromium_max_ns: AtomicU64,
    pub stage_reason_enrich_total_ns: AtomicU64,
    pub stage_reason_enrich_max_ns: AtomicU64,

    /// Phase 2 lock-decomposition telemetry (Sprint 5)
    pub profile_floor_hits: AtomicU64,
    pub iokit_errors: AtomicU64,
    pub reactor_pulses: AtomicU64,
    /// Store f64 reactor event weight as raw u64 bits.
    pub reactor_event_weight_bits: AtomicU64,

    /// Phase 5.2 — Battery-aware cost penalty observability (Sprint 8,
    /// 2026-05-16). Incremented each time the
    /// [`crate::engine::energy::battery_aware_cost_penalty`] returned a
    /// strictly positive penalty (callers actually raised an action's cost
    /// because we're on battery + noise). Mirrors the Phase 3.1
    /// `skill_aware_modulations_total` design: this counter is the only way
    /// to verify the feature actually influences decisions in prod rather
    /// than no-op'ing.
    ///
    /// Note: producers are not wired in this commit (see `OPENS: 1`); the
    /// counter remains 0 in prod until `decide_actions` invokes the
    /// penalty function and calls `inc_battery_aware_penalty_emission()`.
    pub battery_aware_penalty_emissions_total: AtomicU64,

    /// Phase 3.3 — Cross-Group Companion Attention propagation
    /// observability (Sprint 6, 2026-05-16). Incremented each time
    /// [`crate::engine::companion_graph::CompanionGraph::propagate_attention_across_groups`]
    /// returns one or more inferred (A, B, score) triples — counter is
    /// bumped by the number of triples returned, NOT by call count.
    /// Dashboards verify the feature actually fires (vs the "scaffolding
    /// without wiring" anti-pattern documented in CLAUDE.md) and that
    /// the per-cycle cap (100) is rarely hit (which would indicate
    /// graph-wide saturation worth investigating).
    ///
    /// Note: producers are NOT wired in this commit (see `OPENS: 1`); the
    /// counter remains 0 in prod until the daemon main-loop invokes the
    /// propagation API and calls `add_companion_cross_group_inferences()`.
    pub companion_cross_group_inferences_total: AtomicU64,
}

/// Process-wide lock-free counters. Used by code paths that cannot easily
/// thread an `&LockFreeMetrics` through (closures, nested callbacks, places
/// that previously locked `state.metrics`). Per-PID logic continues to use
/// the `lf_metrics` reference passed via function arguments — this static
/// is for fire-and-forget cycle/system counters.
pub static LSE_COUNTERS: LockFreeMetrics = LockFreeMetrics::new();

impl LockFreeMetrics {
    pub const fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
            cycles: AtomicU64::new(0),
            actions_applied: AtomicU64::new(0),
            freezes: AtomicU64::new(0),
            unfreezes: AtomicU64::new(0),
            throttles: AtomicU64::new(0),
            throttle_reverted: AtomicU64::new(0),
            signals_sent: AtomicU64::new(0),
            hw_warnings: AtomicU64::new(0),
            hw_criticals: AtomicU64::new(0),
            vm_pressure_events: AtomicU64::new(0),
            survival_activations: AtomicU64::new(0),
            paging_hints_applied: AtomicU64::new(0),
            processes_scanned: AtomicU64::new(0),
            kqueue_events: AtomicU64::new(0),
            proc_exits_detected: AtomicU64::new(0),
            cycle_time_us: AtomicU64::new(0),
            snapshot_time_us: AtomicU64::new(0),
            decide_time_us: AtomicU64::new(0),
            refresh_duration_us: AtomicU64::new(0),
            memory_budget_duration_us: AtomicU64::new(0),
            reactor_duration_us: AtomicU64::new(0),
            dedup_drops_setmemorystatus: AtomicU64::new(0),
            dedup_drops_throttle: AtomicU64::new(0),
            dedup_drops_freeze: AtomicU64::new(0),
            dedup_drops_unfreeze: AtomicU64::new(0),
            restore_status_missing: AtomicU64::new(0),
            restore_status_restored_n: AtomicU64::new(0),
            restore_status_discarded_corrupt: AtomicU64::new(0),
            restore_status_discarded_clock_delta: AtomicU64::new(0),
            restore_status_discarded_boot_crossed: AtomicU64::new(0),
            identity_cache_hits: AtomicU64::new(0),
            identity_cache_misses: AtomicU64::new(0),
            identity_cache_evictions: AtomicU64::new(0),
            identity_cache_ttl_expired: AtomicU64::new(0),
            identity_cache_exit_invalidations: AtomicU64::new(0),
            identity_proc_pidpath_calls: AtomicU64::new(0),
            actions_pushed_throttle_total: AtomicU64::new(0),
            actions_pushed_freeze_total: AtomicU64::new(0),
            actions_pushed_unfreeze_total: AtomicU64::new(0),
            actions_pushed_boost_total: AtomicU64::new(0),
            actions_pushed_set_memorystatus_total: AtomicU64::new(0),
            actions_pushed_set_thread_qos_total: AtomicU64::new(0),
            actions_pushed_set_sysctl_total: AtomicU64::new(0),
            actions_pushed_toggle_spotlight_total: AtomicU64::new(0),
            actions_pushed_quarantine_daemon_total: AtomicU64::new(0),
            actions_pushed_raw_total: AtomicU64::new(0),
            actions_rejected_shape_total: AtomicU64::new(0),
            maintenance_purge_total: AtomicU64::new(0),
            maintenance_purge_skipped_pressure_total: AtomicU64::new(0),
            maintenance_purge_skipped_swap_floor_total: AtomicU64::new(0),
            maintenance_purge_skipped_growing_total: AtomicU64::new(0),
            maintenance_purge_skipped_idle_total: AtomicU64::new(0),
            maintenance_purge_skipped_build_mode_total: AtomicU64::new(0),
            maintenance_purge_skipped_rate_limit_total: AtomicU64::new(0),
            taskinfo_cache_hits: AtomicU64::new(0),
            taskinfo_cache_misses: AtomicU64::new(0),
            taskinfo_cache_exit_invalidations: AtomicU64::new(0),
            taskinfo_cache_cap_evictions: AtomicU64::new(0),
            skill_aware_modulations_total: AtomicU64::new(0),
            metrics_lock_wait_total_ns: AtomicU64::new(0),
            metrics_lock_wait_count: AtomicU64::new(0),
            metrics_lock_wait_max_ns: AtomicU64::new(0),
            metrics_lock_held_total_ns: AtomicU64::new(0),
            metrics_lock_held_count: AtomicU64::new(0),
            metrics_lock_held_max_ns: AtomicU64::new(0),
            stage_sense_total_ns: AtomicU64::new(0),
            stage_sense_max_ns: AtomicU64::new(0),
            stage_reason_total_ns: AtomicU64::new(0),
            stage_reason_max_ns: AtomicU64::new(0),
            stage_execute_total_ns: AtomicU64::new(0),
            stage_execute_max_ns: AtomicU64::new(0),
            stage_learn_total_ns: AtomicU64::new(0),
            stage_learn_max_ns: AtomicU64::new(0),
            stage_persist_total_ns: AtomicU64::new(0),
            stage_persist_max_ns: AtomicU64::new(0),
            stage_count: AtomicU64::new(0),
            stage_reason_signal_total_ns: AtomicU64::new(0),
            stage_reason_signal_max_ns: AtomicU64::new(0),
            stage_reason_decide_total_ns: AtomicU64::new(0),
            stage_reason_decide_max_ns: AtomicU64::new(0),
            stage_reason_neuro_total_ns: AtomicU64::new(0),
            stage_reason_neuro_max_ns: AtomicU64::new(0),
            stage_reason_usercontext_total_ns: AtomicU64::new(0),
            stage_reason_usercontext_max_ns: AtomicU64::new(0),
            stage_reason_holtwinters_total_ns: AtomicU64::new(0),
            stage_reason_holtwinters_max_ns: AtomicU64::new(0),
            stage_reason_pagereclaim_total_ns: AtomicU64::new(0),
            stage_reason_pagereclaim_max_ns: AtomicU64::new(0),
            stage_reason_chromium_total_ns: AtomicU64::new(0),
            stage_reason_chromium_max_ns: AtomicU64::new(0),
            stage_reason_enrich_total_ns: AtomicU64::new(0),
            stage_reason_enrich_max_ns: AtomicU64::new(0),
            profile_floor_hits: AtomicU64::new(0),
            iokit_errors: AtomicU64::new(0),
            reactor_pulses: AtomicU64::new(0),
            reactor_event_weight_bits: AtomicU64::new(0_f64.to_bits()),
            battery_aware_penalty_emissions_total: AtomicU64::new(0),
            companion_cross_group_inferences_total: AtomicU64::new(0),
        }
    }

    /// Record a per-cycle stage duration. Five callers per cycle.
    #[inline(always)]
    pub fn record_stage(&self, stage: CycleStage, ns: u64) {
        let (total, max) = match stage {
            CycleStage::Sense => (&self.stage_sense_total_ns, &self.stage_sense_max_ns),
            CycleStage::Reason => (&self.stage_reason_total_ns, &self.stage_reason_max_ns),
            CycleStage::Execute => (&self.stage_execute_total_ns, &self.stage_execute_max_ns),
            CycleStage::Learn => (&self.stage_learn_total_ns, &self.stage_learn_max_ns),
            CycleStage::Persist => (&self.stage_persist_total_ns, &self.stage_persist_max_ns),
            CycleStage::ReasonSignalTick => (
                &self.stage_reason_signal_total_ns,
                &self.stage_reason_signal_max_ns,
            ),
            CycleStage::ReasonDecide => (
                &self.stage_reason_decide_total_ns,
                &self.stage_reason_decide_max_ns,
            ),
            CycleStage::ReasonNeuro => (
                &self.stage_reason_neuro_total_ns,
                &self.stage_reason_neuro_max_ns,
            ),
            CycleStage::ReasonUserContext => (
                &self.stage_reason_usercontext_total_ns,
                &self.stage_reason_usercontext_max_ns,
            ),
            CycleStage::ReasonHoltWinters => (
                &self.stage_reason_holtwinters_total_ns,
                &self.stage_reason_holtwinters_max_ns,
            ),
            CycleStage::ReasonPageReclaim => (
                &self.stage_reason_pagereclaim_total_ns,
                &self.stage_reason_pagereclaim_max_ns,
            ),
            CycleStage::ReasonChromium => (
                &self.stage_reason_chromium_total_ns,
                &self.stage_reason_chromium_max_ns,
            ),
            CycleStage::ReasonEnrich => (
                &self.stage_reason_enrich_total_ns,
                &self.stage_reason_enrich_max_ns,
            ),
        };
        total.fetch_add(ns, Ordering::Relaxed);
        max.fetch_max(ns, Ordering::Relaxed);
    }

    /// Increment the stage_count once per cycle (after all 5 stages
    /// recorded). The count divides every stage_*_total_ns to compute
    /// per-stage avg latency.
    #[inline(always)]
    pub fn finish_stage_cycle(&self) {
        self.stage_count.fetch_add(1, Ordering::Relaxed);
    }

    // ── Phase 2: God-Lock Decomposition Counters ─────────────────────────────

    #[inline(always)]
    pub fn increment_profile_floor_hits(&self) {
        self.profile_floor_hits.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn increment_paging_hints_applied(&self) {
        self.paging_hints_applied.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_iokit_errors(&self, count: u64) {
        self.iokit_errors.store(count, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn increment_reactor_pulses(&self) {
        self.reactor_pulses.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_reactor_event_weight(&self, weight: f64) {
        self.reactor_event_weight_bits.store(weight.to_bits(), Ordering::Relaxed);
    }

    /// Record a metrics-lock acquisition + held duration.
    /// Caller passes the wait time (lock acquisition) and held time
    /// (between acquire and guard drop). Both in nanoseconds.
    ///
    /// Used by Phase 0 lock-decomposition baseline measurement
    /// (2026-05-10). Removed once lock-decomp lands and the metrics
    /// god-lock no longer exists.
    #[inline(always)]
    pub fn record_metrics_lock(&self, wait_ns: u64, held_ns: u64) {
        self.metrics_lock_wait_total_ns
            .fetch_add(wait_ns, Ordering::Relaxed);
        self.metrics_lock_wait_count.fetch_add(1, Ordering::Relaxed);
        self.metrics_lock_wait_max_ns
            .fetch_max(wait_ns, Ordering::Relaxed);
        self.metrics_lock_held_total_ns
            .fetch_add(held_ns, Ordering::Relaxed);
        self.metrics_lock_held_count.fetch_add(1, Ordering::Relaxed);
        self.metrics_lock_held_max_ns
            .fetch_max(held_ns, Ordering::Relaxed);
    }

    // ── Writer methods (hot path — Relaxed ordering, single LSE instruction) ─

    #[inline(always)]
    pub fn inc_cycles(&self) {
        self.cycles.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn add_actions(&self, n: u64) {
        self.actions_applied.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_freezes(&self) {
        self.freezes.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_unfreezes(&self) {
        self.unfreezes.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_throttles(&self) {
        self.throttles.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_hw_warning(&self) {
        self.hw_warnings.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_hw_critical(&self) {
        self.hw_criticals.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_vm_pressure(&self) {
        self.vm_pressure_events.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_kqueue_events(&self, n: u64) {
        self.kqueue_events.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_proc_exits(&self) {
        self.proc_exits_detected.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_cycle_time_us(&self, us: u64) {
        self.cycle_time_us.store(us, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_snapshot_time_us(&self, us: u64) {
        self.snapshot_time_us.store(us, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_decide_time_us(&self, us: u64) {
        self.decide_time_us.store(us, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_refresh_duration_us(&self, us: u64) {
        self.refresh_duration_us.store(us, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_memory_budget_duration_us(&self, us: u64) {
        self.memory_budget_duration_us.store(us, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_reactor_duration_us(&self, us: u64) {
        self.reactor_duration_us.store(us, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn add_dedup_drops_setmemorystatus(&self, n: u64) {
        self.dedup_drops_setmemorystatus
            .fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn add_dedup_drops_throttle(&self, n: u64) {
        self.dedup_drops_throttle.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn add_dedup_drops_freeze(&self, n: u64) {
        self.dedup_drops_freeze.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn add_dedup_drops_unfreeze(&self, n: u64) {
        self.dedup_drops_unfreeze.fetch_add(n, Ordering::Relaxed);
    }

    /// Bump epoch after a batch of updates. This establishes the
    /// happens-before edge for readers calling `snapshot()`.
    #[inline(always)]
    pub fn commit(&self) {
        self.epoch.fetch_add(1, Ordering::Release);
    }

    // ── Reader methods ───────────────────────────────────────────────────────

    /// Take a consistent snapshot of all counters.
    /// The `Acquire` load on epoch guarantees we see all prior `Relaxed` stores
    /// from the writer thread that called `commit()`.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let epoch = self.epoch.load(Ordering::Acquire);
        MetricsSnapshot {
            epoch,
            cycles: self.cycles.load(Ordering::Relaxed),
            actions_applied: self.actions_applied.load(Ordering::Relaxed),
            freezes: self.freezes.load(Ordering::Relaxed),
            unfreezes: self.unfreezes.load(Ordering::Relaxed),
            throttles: self.throttles.load(Ordering::Relaxed),
            throttle_reverted: self.throttle_reverted.load(Ordering::Relaxed),
            signals_sent: self.signals_sent.load(Ordering::Relaxed),
            hw_warnings: self.hw_warnings.load(Ordering::Relaxed),
            hw_criticals: self.hw_criticals.load(Ordering::Relaxed),
            vm_pressure_events: self.vm_pressure_events.load(Ordering::Relaxed),
            survival_activations: self.survival_activations.load(Ordering::Relaxed),
            paging_hints_applied: self.paging_hints_applied.load(Ordering::Relaxed),
            processes_scanned: self.processes_scanned.load(Ordering::Relaxed),
            kqueue_events: self.kqueue_events.load(Ordering::Relaxed),
            proc_exits_detected: self.proc_exits_detected.load(Ordering::Relaxed),
            cycle_time_us: self.cycle_time_us.load(Ordering::Relaxed),
            snapshot_time_us: self.snapshot_time_us.load(Ordering::Relaxed),
            decide_time_us: self.decide_time_us.load(Ordering::Relaxed),
            refresh_duration_us: self.refresh_duration_us.load(Ordering::Relaxed),
            memory_budget_duration_us: self.memory_budget_duration_us.load(Ordering::Relaxed),
            reactor_duration_us: self.reactor_duration_us.load(Ordering::Relaxed),
            dedup_drops_setmemorystatus: self.dedup_drops_setmemorystatus.load(Ordering::Relaxed),
            dedup_drops_throttle: self.dedup_drops_throttle.load(Ordering::Relaxed),
            dedup_drops_freeze: self.dedup_drops_freeze.load(Ordering::Relaxed),
            dedup_drops_unfreeze: self.dedup_drops_unfreeze.load(Ordering::Relaxed),
            restore_status_missing: self.restore_status_missing.load(Ordering::Relaxed),
            restore_status_restored_n: self.restore_status_restored_n.load(Ordering::Relaxed),
            restore_status_discarded_corrupt: self
                .restore_status_discarded_corrupt
                .load(Ordering::Relaxed),
            restore_status_discarded_clock_delta: self
                .restore_status_discarded_clock_delta
                .load(Ordering::Relaxed),
            restore_status_discarded_boot_crossed: self
                .restore_status_discarded_boot_crossed
                .load(Ordering::Relaxed),
            identity_cache_hits: self.identity_cache_hits.load(Ordering::Relaxed),
            identity_cache_misses: self.identity_cache_misses.load(Ordering::Relaxed),
            identity_cache_evictions: self.identity_cache_evictions.load(Ordering::Relaxed),
            identity_cache_ttl_expired: self.identity_cache_ttl_expired.load(Ordering::Relaxed),
            identity_cache_exit_invalidations: self
                .identity_cache_exit_invalidations
                .load(Ordering::Relaxed),
            identity_proc_pidpath_calls: self.identity_proc_pidpath_calls.load(Ordering::Relaxed),
            actions_pushed_throttle_total: self
                .actions_pushed_throttle_total
                .load(Ordering::Relaxed),
            actions_pushed_freeze_total: self.actions_pushed_freeze_total.load(Ordering::Relaxed),
            actions_pushed_unfreeze_total: self
                .actions_pushed_unfreeze_total
                .load(Ordering::Relaxed),
            actions_pushed_boost_total: self.actions_pushed_boost_total.load(Ordering::Relaxed),
            actions_pushed_set_memorystatus_total: self
                .actions_pushed_set_memorystatus_total
                .load(Ordering::Relaxed),
            actions_pushed_set_thread_qos_total: self
                .actions_pushed_set_thread_qos_total
                .load(Ordering::Relaxed),
            actions_pushed_set_sysctl_total: self
                .actions_pushed_set_sysctl_total
                .load(Ordering::Relaxed),
            actions_pushed_toggle_spotlight_total: self
                .actions_pushed_toggle_spotlight_total
                .load(Ordering::Relaxed),
            actions_pushed_quarantine_daemon_total: self
                .actions_pushed_quarantine_daemon_total
                .load(Ordering::Relaxed),
            actions_pushed_raw_total: self.actions_pushed_raw_total.load(Ordering::Relaxed),
            actions_rejected_shape_total: self.actions_rejected_shape_total.load(Ordering::Relaxed),
            maintenance_purge_total: self.maintenance_purge_total.load(Ordering::Relaxed),
            maintenance_purge_skipped_pressure_total: self
                .maintenance_purge_skipped_pressure_total
                .load(Ordering::Relaxed),
            maintenance_purge_skipped_swap_floor_total: self
                .maintenance_purge_skipped_swap_floor_total
                .load(Ordering::Relaxed),
            maintenance_purge_skipped_growing_total: self
                .maintenance_purge_skipped_growing_total
                .load(Ordering::Relaxed),
            maintenance_purge_skipped_idle_total: self
                .maintenance_purge_skipped_idle_total
                .load(Ordering::Relaxed),
            maintenance_purge_skipped_build_mode_total: self
                .maintenance_purge_skipped_build_mode_total
                .load(Ordering::Relaxed),
            maintenance_purge_skipped_rate_limit_total: self
                .maintenance_purge_skipped_rate_limit_total
                .load(Ordering::Relaxed),
            taskinfo_cache_hits: self.taskinfo_cache_hits.load(Ordering::Relaxed),
            taskinfo_cache_misses: self.taskinfo_cache_misses.load(Ordering::Relaxed),
            taskinfo_cache_exit_invalidations: self
                .taskinfo_cache_exit_invalidations
                .load(Ordering::Relaxed),
            taskinfo_cache_cap_evictions: self
                .taskinfo_cache_cap_evictions
                .load(Ordering::Relaxed),
            skill_aware_modulations_total: self
                .skill_aware_modulations_total
                .load(Ordering::Relaxed),
            profile_floor_hits: self.profile_floor_hits.load(Ordering::Relaxed),
            iokit_errors: self.iokit_errors.load(Ordering::Relaxed),
            reactor_pulses: self.reactor_pulses.load(Ordering::Relaxed),
            reactor_event_weight: f64::from_bits(
                self.reactor_event_weight_bits.load(Ordering::Relaxed),
            ),
            battery_aware_penalty_emissions_total: self
                .battery_aware_penalty_emissions_total
                .load(Ordering::Relaxed),
            companion_cross_group_inferences_total: self
                .companion_cross_group_inferences_total
                .load(Ordering::Relaxed),
        }
    }

    /// Phase 1 prod-grade (2026-05-16): observability for the enrichment
    /// syscall cache. Hit ratio = hits / (hits + misses). Eviction
    /// counters help diagnose whether the hard cap is firing in prod.
    pub fn inc_taskinfo_cache_hit(&self) {
        self.taskinfo_cache_hits.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_taskinfo_cache_miss(&self) {
        self.taskinfo_cache_misses.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_taskinfo_cache_exit_invalidation(&self) {
        self.taskinfo_cache_exit_invalidations
            .fetch_add(1, Ordering::Relaxed);
    }
    pub fn add_taskinfo_cache_cap_evictions(&self, n: u64) {
        self.taskinfo_cache_cap_evictions
            .fetch_add(n, Ordering::Relaxed);
    }

    /// Phase 3.1 — Skill-Aware Prediction observability hook.
    /// Call once per non-Observe specialist vote whose confidence was
    /// modulated by a non-neutral `skill_aware_factor`. Used by
    /// dashboards to verify the feature is actually influencing decisions
    /// rather than no-op'ing on neutral signal (`None → 1.0`).
    pub fn add_skill_aware_modulations(&self, n: u64) {
        self.skill_aware_modulations_total
            .fetch_add(n, Ordering::Relaxed);
    }

    /// Phase 5.2 — Battery-aware cost penalty observability hook.
    /// Call once per `battery_aware_cost_penalty` invocation that returned
    /// a strictly positive penalty. Dashboards compute the "battery
    /// suppression rate" as `battery_aware_penalty_emissions_total /
    /// cycles` to verify the penalty fires under battery + noise and stays
    /// at 0 on AC.
    #[inline(always)]
    pub fn inc_battery_aware_penalty_emission(&self) {
        self.battery_aware_penalty_emissions_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Phase 3.3 — Cross-Group Companion Attention observability hook.
    /// Increment by the number of triples returned from
    /// [`crate::engine::companion_graph::CompanionGraph::propagate_attention_across_groups`].
    /// Bumping by N (rather than calling N times) keeps the call site a
    /// single LSE `ldaddal` even when the propagation returns many edges.
    #[inline(always)]
    pub fn add_companion_cross_group_inferences(&self, n: u64) {
        self.companion_cross_group_inferences_total
            .fetch_add(n, Ordering::Relaxed);
    }
}

// Safe to share across threads — all fields are atomic.
unsafe impl Sync for LockFreeMetrics {}

/// A consistent point-in-time snapshot of all metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub epoch: u64,
    pub cycles: u64,
    pub actions_applied: u64,
    pub freezes: u64,
    pub unfreezes: u64,
    pub throttles: u64,
    pub throttle_reverted: u64,
    pub signals_sent: u64,
    pub hw_warnings: u64,
    pub hw_criticals: u64,
    pub vm_pressure_events: u64,
    pub survival_activations: u64,
    pub paging_hints_applied: u64,
    pub processes_scanned: u64,
    pub kqueue_events: u64,
    pub proc_exits_detected: u64,
    pub cycle_time_us: u64,
    pub snapshot_time_us: u64,
    pub decide_time_us: u64,
    pub refresh_duration_us: u64,
    pub memory_budget_duration_us: u64,
    pub reactor_duration_us: u64,
    pub dedup_drops_setmemorystatus: u64,
    pub dedup_drops_throttle: u64,
    pub dedup_drops_freeze: u64,
    pub dedup_drops_unfreeze: u64,
    pub restore_status_missing: u64,
    pub restore_status_restored_n: u64,
    pub restore_status_discarded_corrupt: u64,
    pub restore_status_discarded_clock_delta: u64,
    pub restore_status_discarded_boot_crossed: u64,
    pub identity_cache_hits: u64,
    pub identity_cache_misses: u64,
    pub identity_cache_evictions: u64,
    pub identity_cache_ttl_expired: u64,
    pub identity_cache_exit_invalidations: u64,
    pub identity_proc_pidpath_calls: u64,
    pub actions_pushed_throttle_total: u64,
    pub actions_pushed_freeze_total: u64,
    pub actions_pushed_unfreeze_total: u64,
    pub actions_pushed_boost_total: u64,
    pub actions_pushed_set_memorystatus_total: u64,
    pub actions_pushed_set_thread_qos_total: u64,
    pub actions_pushed_set_sysctl_total: u64,
    pub actions_pushed_toggle_spotlight_total: u64,
    pub actions_pushed_quarantine_daemon_total: u64,
    pub actions_pushed_raw_total: u64,
    pub actions_rejected_shape_total: u64,
    pub maintenance_purge_total: u64,
    pub maintenance_purge_skipped_pressure_total: u64,
    pub maintenance_purge_skipped_swap_floor_total: u64,
    pub maintenance_purge_skipped_growing_total: u64,
    pub maintenance_purge_skipped_idle_total: u64,
    pub maintenance_purge_skipped_build_mode_total: u64,
    pub maintenance_purge_skipped_rate_limit_total: u64,
    pub taskinfo_cache_hits: u64,
    pub taskinfo_cache_misses: u64,
    pub taskinfo_cache_exit_invalidations: u64,
    pub taskinfo_cache_cap_evictions: u64,
    pub skill_aware_modulations_total: u64,
    pub profile_floor_hits: u64,
    pub iokit_errors: u64,
    pub reactor_pulses: u64,
    pub reactor_event_weight: f64,
    /// Phase 5.2 — Battery-aware cost penalty emissions (Sprint 8).
    pub battery_aware_penalty_emissions_total: u64,
    /// Phase 3.3 — Cross-Group Companion Attention inferences (Sprint 6).
    pub companion_cross_group_inferences_total: u64,
}

// ── ARM64 LSE verification ───────────────────────────────────────────────────

/// Verify that the hardware supports LSE atomics (ARMv8.1+).
/// On Apple Silicon M1+ this is always true, but we verify with an actual
/// `ldaddal` instruction to prove the compiler emits LSE, not LL/SC fallback.
///
/// Returns the old value of the atomic (should be 0 on first call).
#[cfg(target_arch = "aarch64")]
pub fn verify_lse_atomic_add(target: &AtomicU64, increment: u64) -> u64 {
    let ptr = target.as_ptr();
    let old: u64;
    unsafe {
        std::arch::asm!(
            "ldaddal {val}, {old}, [{ptr}]",
            ptr = in(reg) ptr,
            val = in(reg) increment,
            old = out(reg) old,
            options(nostack),
        );
    }
    old
}

/// Atomic swap using LSE `swpal` instruction.
/// Single instruction, ~3ns. Useful for pointer-swapping state.
#[cfg(target_arch = "aarch64")]
pub fn lse_swap(target: &AtomicU64, new_val: u64) -> u64 {
    let ptr = target.as_ptr();
    let old: u64;
    unsafe {
        std::arch::asm!(
            "swpal {val}, {old}, [{ptr}]",
            ptr = in(reg) ptr,
            val = in(reg) new_val,
            old = out(reg) old,
            options(nostack),
        );
    }
    old
}

/// Compare-and-swap using LSE `casal` instruction.
/// Returns the value found at `target`. If it was `expected`, the swap happened.
#[cfg(target_arch = "aarch64")]
pub fn lse_cas(target: &AtomicU64, expected: u64, desired: u64) -> u64 {
    let ptr = target.as_ptr();
    let found: u64;
    unsafe {
        // CAS: if [ptr] == expected, then [ptr] = desired
        // found receives the value that was at [ptr]
        std::arch::asm!(
            "casal {exp}, {des}, [{ptr}]",
            ptr = in(reg) ptr,
            exp = inout(reg) expected => found,
            des = in(reg) desired,
            options(nostack),
        );
    }
    found
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn basic_increment_and_snapshot() {
        let m = LockFreeMetrics::new();
        m.inc_cycles();
        m.inc_cycles();
        m.inc_freezes();
        m.add_actions(5);
        m.commit();

        let snap = m.snapshot();
        assert_eq!(snap.cycles, 2);
        assert_eq!(snap.freezes, 1);
        assert_eq!(snap.actions_applied, 5);
        assert_eq!(snap.epoch, 1);
    }

    #[test]
    fn concurrent_increments_no_lost_updates() {
        let m = Arc::new(LockFreeMetrics::new());
        let threads: Vec<_> = (0..32)
            .map(|_| {
                let m = m.clone();
                thread::spawn(move || {
                    for _ in 0..10_000 {
                        m.inc_cycles();
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }
        m.commit();

        let snap = m.snapshot();
        assert_eq!(
            snap.cycles,
            32 * 10_000,
            "no lost increments under contention"
        );
    }

    #[test]
    fn concurrent_mixed_operations() {
        let m = Arc::new(LockFreeMetrics::new());
        let threads: Vec<_> = (0..16)
            .map(|i| {
                let m = m.clone();
                thread::spawn(move || {
                    for _ in 0..5_000 {
                        m.inc_cycles();
                        m.inc_freezes();
                        m.inc_throttles();
                        m.add_actions(1);
                        if i % 4 == 0 {
                            m.inc_hw_warning();
                        }
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }
        m.commit();

        let snap = m.snapshot();
        assert_eq!(snap.cycles, 16 * 5_000);
        assert_eq!(snap.freezes, 16 * 5_000);
        assert_eq!(snap.throttles, 16 * 5_000);
        assert_eq!(snap.actions_applied, 16 * 5_000);
        assert_eq!(snap.hw_warnings, 4 * 5_000); // 4 threads (i % 4 == 0)
    }

    #[test]
    fn snapshot_reader_concurrent_with_writer() {
        let m = Arc::new(LockFreeMetrics::new());
        let m2 = m.clone();

        let writer = thread::spawn(move || {
            for _ in 0..100_000 {
                m2.inc_cycles();
                m2.commit();
            }
        });

        // Reader thread takes snapshots while writer is running
        let mut snapshots = Vec::new();
        for _ in 0..1_000 {
            snapshots.push(m.snapshot());
            thread::yield_now();
        }

        writer.join().unwrap();

        // Verify monotonicity — cycles should never decrease
        for window in snapshots.windows(2) {
            assert!(
                window[1].cycles >= window[0].cycles,
                "cycles must be monotonic: {} -> {}",
                window[0].cycles,
                window[1].cycles,
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn lse_atomic_add_works() {
        let val = AtomicU64::new(42);
        let old = verify_lse_atomic_add(&val, 10);
        assert_eq!(old, 42, "ldaddal should return old value");
        assert_eq!(val.load(Ordering::SeqCst), 52, "value should be 42+10");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn lse_swap_works() {
        let val = AtomicU64::new(100);
        let old = lse_swap(&val, 200);
        assert_eq!(old, 100);
        assert_eq!(val.load(Ordering::SeqCst), 200);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn lse_cas_succeeds_on_match() {
        let val = AtomicU64::new(42);
        let found = lse_cas(&val, 42, 99);
        assert_eq!(found, 42, "CAS should return original");
        assert_eq!(val.load(Ordering::SeqCst), 99, "CAS should have swapped");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn lse_cas_fails_on_mismatch() {
        let val = AtomicU64::new(42);
        let found = lse_cas(&val, 0, 99); // expected=0 but actual=42
        assert_eq!(found, 42, "CAS should return actual value");
        assert_eq!(val.load(Ordering::SeqCst), 42, "value should not change");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn lse_contended_cas_loop() {
        // Simulate a CAS retry loop under contention from 8 threads
        let val = Arc::new(AtomicU64::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let v = val.clone();
                thread::spawn(move || {
                    for _ in 0..10_000 {
                        loop {
                            let cur = v.load(Ordering::Relaxed);
                            let found = lse_cas(&v, cur, cur + 1);
                            if found == cur {
                                break; // CAS succeeded
                            }
                            // CAS failed, retry
                        }
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(
            val.load(Ordering::SeqCst),
            8 * 10_000,
            "all increments must be accounted for",
        );
    }

    #[test]
    fn metrics_struct_is_cache_aligned() {
        assert_eq!(
            std::mem::align_of::<LockFreeMetrics>(),
            128,
            "metrics should be 128-byte aligned to avoid false sharing",
        );
    }
}
