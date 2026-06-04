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
    /// counters (`actions_pushed_freeze_total`, etc.) increment on BOTH the
    /// typed `push_*` methods AND `push_raw` — every raw push also bumps the
    /// matching per-variant counter so runtime telemetry reflects the true
    /// emitted-variant volume.
    ///
    /// Invariant (post-ffa0b29):
    /// ```text
    /// Σ(typed per-variant) == total_pushed
    /// ```
    ///
    /// `actions_pushed_raw_total` is an INDEPENDENT diagnostic of escape-hatch
    /// volume — it is a SUBSET of the typed totals, not an addend. Dashboards
    /// compute "% bypassing typed shape validation" as
    /// `actions_pushed_raw_total / total_pushed`.
    ///
    /// DO NOT compute Σ(typed) + raw — this double-counts every escape-hatch
    /// emission and inflates dispatcher volume by the raw fraction.
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
    /// Sprint 12 Convergence #5 (2026-05-17). Maintenance purge skipped
    /// because the unified-memory bus is saturated. On M1 without the
    /// IOReport private entitlement `amc_bandwidth_pct == 0.0` (dead
    /// signal), so callers should compute saturation from the alive
    /// fallback chain — `signal_digest.entropy_anomaly > 2.0` is the
    /// same proxy G12 already uses for DRAM backpressure. Purging while
    /// the bus is busy induces user-visible jank because vm_purge
    /// contends with whatever is driving the bandwidth (typically LLM
    /// inference traffic). [Hennessy & Patterson 2017 §2.2] unified
    /// memory contention.
    pub maintenance_purge_skipped_bus_saturated_total: AtomicU64,

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

    /// Phase 4.2 — External-event causal attribution (Sprint 7, 2026-05-16).
    ///
    /// Each time a `CausalEdge` is tagged with `external_blame` (because an
    /// external event preceded the action inside `EXTERNAL_BLAME_WINDOW`),
    /// the matching kind-specific counter is bumped. The cumulative ratio
    /// `causal_external_*_blames_total / causal_pressure_drop_edges_total`
    /// is the operator-facing "how much of our credit is confounded?"
    /// metric. [Pearl 2009 §4] / [Rubin 1974] — register confounders so
    /// they can be excluded from treatment-effect estimates.
    pub causal_external_thermal_blames_total: AtomicU64,
    pub causal_external_disk_blames_total: AtomicU64,
    pub causal_external_net_blames_total: AtomicU64,

    /// Phase 4.3 — Policy Rollback Guard observability (Sprint 7, 2026-05-16).
    /// Each cycle the daemon will call `PolicyRollbackGuard::evaluate` to
    /// check whether `RestoreQualityMonitor::quality` has fallen below the
    /// safety floor for long enough to revert a recent parameter shift.
    /// `evaluations_total` increments per call (success or not) — dashboards
    /// can compute the fire ratio against `executions_total`. Both are
    /// surfaced through `MetricsSnapshot → RuntimeMetrics → runtime_metrics.json`
    /// so the user can verify the guard is actually running and not
    /// silently dormant.
    pub policy_rollback_evaluations_total: AtomicU64,
    pub policy_rollback_executions_total: AtomicU64,

    /// Phase 3.2 — Arousal-Modulated NARS Decay observability
    /// (Sprint 6, 2026-05-16). Incremented each persist whose
    /// `arousal_modulated_decay_factor(...)` returned a value strictly
    /// less than the base factor (i.e. Stressed or Crisis zone, decay
    /// accelerated). Lets dashboards verify the feature actually engages
    /// in production rather than no-op'ing on a quiescent system.
    /// [McGaugh 2004] arousal → consolidation; [Yerkes & Dodson 1908].
    pub arousal_decay_accelerations_total: AtomicU64,

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

    /// Phase 4.1 — Adaptive Drift Threshold observability (Sprint 7,
    /// 2026-05-16). Incremented each time
    /// [`crate::engine::nars_belief::AdaptiveDriftThreshold::recommended_threshold`]
    /// returns a value strictly greater than the supplied base — i.e. the
    /// adaptive layer raised the bar based on observed noise variance.
    /// Mirrors the Phase 3.1 / 3.2 / 5.2 counter design so dashboards can
    /// verify the feature actually engages in prod rather than silently
    /// no-op'ing.
    /// [Brown 1959] EMA; [Welford 1962] online variance; [Kuncheva 2004]
    /// adaptive drift detection.
    pub adaptive_drift_threshold_raises_total: AtomicU64,

    /// Phase 5.1 — User-presence suppression observability (Sprint 8,
    /// 2026-05-16). Incremented each time
    /// [`crate::engine::user_presence::user_presence_modulator`] returned a
    /// multiplier strictly less than 1.0 (active or semi-active user, no
    /// crisis override). Mirrors the Phase 3.1 / 5.2 design: this counter
    /// is the only way to verify the modulator actually scaled an action
    /// in prod, instead of no-op'ing at the idle tier.
    ///
    /// Note: producers (`decide_actions` cost composition / cognitive tick
    /// specialist voting) are not wired in this commit (see `OPENS: 1` on
    /// the introducing commit). The counter stays at 0 until the caller
    /// invokes the modulator with real HID inputs.
    ///
    /// [Iqbal & Bailey 2008] "Effects of Interruptions on Task Performance".
    pub user_presence_suppressions_total: AtomicU64,

    /// Phase 5.3 — Structured-rationale observability (Sprint 8, 2026-05-16).
    /// Incremented every time a `JournalEntry` is written with a non-`None`
    /// `rationale` field, i.e. the action carried a machine-parseable
    /// explanation. Dashboards compute the "rationale coverage ratio" as
    ///
    /// ```text
    /// journal_rationales_attached_total
    /// ─────────────────────────────────
    ///   total journal entries written
    /// ```
    ///
    /// Without this counter we'd have no way to verify whether the
    /// cross-cutting wiring (deferred to a follow-up commit) ever landed in
    /// prod — the same "tautology trap" NotebookLM flagged for Phase 3.1
    /// (`skill_aware_modulations_total`) and Phase 5.2
    /// (`battery_aware_penalty_emissions_total`).
    ///
    /// Producers are NOT wired in this commit (`OPENS: 1`); the counter
    /// stays at 0 until a journal write-site calls
    /// `inc_journal_rationale_attached()` alongside its
    /// `JournalEntry::with_rationale(..)` invocation.
    ///
    /// References:
    /// - [Doshi-Velez & Kim 2017] interpretable ML observability —
    ///   coverage metric is a precondition for trust.
    /// - [Ribeiro et al. 2016] LIME — every prediction explained.
    pub journal_rationales_attached_total: AtomicU64,

    /// Phase 4.3.1 — Specialist accuracy purge inhibition counter
    /// (Sprint 8, 2026-05-16). Incremented each time the cognitive tick's
    /// specialist voting block SKIPPED the four `specialist_accuracy.update()`
    /// calls because a maintenance purge happened in the previous 30 s.
    ///
    /// A purge causes pressure to drop sharply; without this gate, hazard /
    /// monopoly / kalman specialists that predicted a spike get graded
    /// "wrong" — their EMA weights decay and the next genuine crisis sees a
    /// weaker reaction. Mirrors the inhibition pattern Phase 2 added for
    /// `outcome_tracker` and `causal_graph` post-purge. NotebookLM 2026-05-16
    /// flagged the missing guard as GAP 6.
    ///
    /// [Rubin 1974] "Estimating Causal Effects of Treatments in Randomized
    /// and Nonrandomized Studies" — distinguishing intervention from confounder.
    pub specialist_accuracy_purge_inhibitions_total: AtomicU64,

    /// Phase 2 god-lock decomposition (Sprint 8, 2026-05-16).
    /// Cumulative count of processes skipped by habituation in
    /// `daemon_cognitive_tick::update_habituation_state`. Migrated from
    /// `state.metrics.lock_recover().metrics.habituation_skips += N` to remove
    /// a god-lock contributor on the hot path. The legacy
    /// `RuntimeMetrics.habituation_skips` field is now populated from this
    /// atomic via `sync_from_lockfree` (no duplicate field — single source
    /// of truth, atomic). [Hellerstein 2012] mutex avoidance for hot-path
    /// counters.
    pub habituation_skips_total: AtomicU64,

    /// Phase C SCORER-OVERRIDE (Sprint 11 finale, 2026-05-16).
    /// Asymmetric scorer/gate disagreement gate. Two counters expose how
    /// often the conservative partial cutover actually fires in prod —
    /// both stay at 0 while shadow_signals is silent and only move once
    /// the ActionContext is populated with rich enough signal for the
    /// scorer to disagree confidently with the gate tower.
    ///
    /// * `scorer_override_rejects_total` — gate ACCEPTED a candidate
    ///   action but the scorer's composite was strictly less than
    ///   −0.30 (strong reject). The action was REJECTED (scorer beats
    ///   gate in the safe direction) and a `BlockedActionEvent` was
    ///   emitted to the shadow journal tagged
    ///   `scorer-override-accept-to-reject`.
    /// * `scorer_disagreement_strong_accepts_total` — gate REJECTED but
    ///   the scorer's composite was strictly greater than +0.30 (strong
    ///   accept). Per NotebookLM 2026-05-16 Candidate-C verdict, we
    ///   never let the scorer beat the gate in the *unsafe* direction
    ///   (scorer wants to act, gate said no) — we ONLY journal the
    ///   disagreement for offline analysis. Cutover to symmetric mode
    ///   is a Sprint 12 candidate after the asymmetric mode validates
    ///   with N≥500 events.
    ///
    /// Both counters mirror the Phase 3.1 / 5.2 design: surfaced via
    /// `MetricsSnapshot` → `RuntimeMetrics` → `runtime_metrics.json` so
    /// operators can verify the partial cutover engages at all (the
    /// "tautology trap" mitigation CLAUDE.md flags). Without the
    /// counters there is no way to distinguish "scorer never disagreed
    /// strongly" from "the override code path is dead".
    ///
    /// [Nygard 2018 §8.5] Adaptive capacity limits via shadowing —
    /// observe both the taken decision and the rejected counterfactual
    /// before promoting either side.
    pub scorer_override_rejects_total: AtomicU64,
    pub scorer_disagreement_strong_accepts_total: AtomicU64,

    /// Phase D PURGE-INHIBITION (Sprint 12 candidate #1, 2026-05-17).
    ///
    /// Counts how many times a predictor swap-derived update was inhibited
    /// because [`MaintenanceState::is_in_purge_inhibition_window`] returned
    /// true. Producer = `signal_intelligence::step` swap branch when
    /// `purge_inhibited == true`. Consumer = `runtime_metrics.json` via
    /// `sync_from_lockfree`. The counter stays at 0 unless the daemon
    /// actually purged in the last 5 s — confirms the loop closes when
    /// the maintenance tick fires and proves the predictor sidestepped
    /// the exogenous shock.
    ///
    /// [Hellerstein 2004 §9] disturbance rejection in closed-loop systems.
    pub purge_inhibition_skips_total: AtomicU64,

    /// RAM Phase B (2026-06-03). Mediator chokepoint counters per Saltzer &
    /// Schroeder 1975 complete-mediation principle. Each tracks one failure
    /// mode the mediator interposes on:
    /// - `mediator_blocks_total`: pre-condition violation OR identity
    ///   mismatch OR safety oracle veto refused the effect before syscall.
    /// - `mediator_noop_writes_total`: Effect applied but `before == after`
    ///   in the Receipt — the SetSysctl bug class from Sprint 3 2026-05-07.
    /// - `mediator_postcondition_violation_total`: Effect applied AND
    ///   syscall returned success BUT post-snapshot still failed the
    ///   expected delta check (e.g. SIGSTOP returned 0 but process still
    ///   shown as RUNNING in proc_taskinfo). Catches lying syscalls.
    pub mediator_blocks_total: AtomicU64,
    pub mediator_noop_writes_total: AtomicU64,
    pub mediator_postcondition_violation_total: AtomicU64,

    /// Sprint 12 Convergence #4 (2026-05-17). Counts cycles where
    /// `scorer_override_rejects_total` incremented AND
    /// `CausalGraph::has_recent_external_event(ThermalThrottle, …)`
    /// returned true in the same iteration.
    ///
    /// The conjunction proves the policy scorer disagreed with the gate
    /// in the same window the SoC was thermally throttled — strong
    /// evidence the *learned* policy is misbehaving under thermal stress
    /// (vs simple oscillation). When this counter ramps, the
    /// PolicyRollbackGuard should respond with elevated sensitivity
    /// (lower quality threshold) since the policy has lost authority to
    /// physical reality. Producer = daemon main loop convergence probe.
    ///
    /// [Pearl 2009 §3] confounder adjustment + [Sutton 2018 §11.7]
    /// model-free policy correction.
    pub causal_thermal_scorer_override_alignments_total: AtomicU64,

    /// Sprint 12 Convergence #1 (2026-05-17). Counts cycles where a
    /// cold-thread routing decision flipped from the default E-cluster
    /// (battery friendly) to the P-cluster because the owning process
    /// is a companion of the current foreground app AND DRAM bandwidth
    /// is below the safety floor.
    ///
    /// Converges Companion Graph (logical workflow) with
    /// THREAD_AFFINITY_POLICY (physical cluster topology) — when a
    /// renderer/helper is keeping a foreground window's hot threads on
    /// P-cluster L2, sending the same process's cold threads to
    /// E-cluster forces a cluster-boundary migration the next time the
    /// user clicks the tab. Counter ramping under low pressure proves
    /// the bridge is firing; under high pressure (≥0.50) it falls back
    /// to standard E-routing automatically. Producer = decide_actions
    /// cold-thread loop. [ARM big.LITTLE 2013 §3] cluster-local
    /// scheduling preserves L2 working set across UI interactions.
    pub companion_affinity_alignments_total: AtomicU64,

    /// Sprint 13 Pressure-Router Gate (2026-05-30). Incremented every
    /// cycle the daemon main loop SKIPPED the `companion_graph.observe_cycle`
    /// + Phase 3.3 propagation block because `memory_pressure < mid_entry`
    /// AND `cycle_count % 4 != 0`. Mirrors the existing 4-subsystem
    /// adaptive router gate at signal_intelligence.rs:404-424 (MoR-style
    /// conditional compute) and the Sprint 12 G12 `bus_saturated`
    /// skip-with-counter telemetry shape (commit `5f1c984`).
    ///
    /// The modulo-4 fallback ([Sutton & Barto §2.7] forced exploration)
    /// keeps the Lift denominator updating ~every 20 s @ 5 s/cycle so
    /// statistics don't go stale under sustained low pressure. Ratio
    /// `companion_observe_router_skips_total / cycles` should approach
    /// ~0.75 on an idle laptop (pressure spends most of its time below
    /// the mid-entry threshold) and drop toward 0 under sustained
    /// pressure ≥ mid_entry.
    pub companion_observe_router_skips_total: AtomicU64,
    /// Sprint 12 perf-fix (2026-05-30). Cumulative count of per-cycle
    /// companion-of-foreground-PIDs set rebuilds that hit the
    /// LearningContext-scope memoization cache (no recomputation, no
    /// HashSet allocation). Producer = `apollo-optimizerd` main loop
    /// `companion_of_fg_pids` derivation. See
    /// `RuntimeMetrics::companion_fg_cache_hits_total` for rationale.
    /// [Saltzer & Schroeder 1975] Economy of Mechanism — single
    /// chokepoint memoize keyed on observable mutation witness
    /// (`CompanionGraph::total_cycles + anchor_count`), not wall clock.
    pub companion_fg_cache_hits_total: AtomicU64,
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
            maintenance_purge_skipped_bus_saturated_total: AtomicU64::new(0),
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

            causal_external_thermal_blames_total: AtomicU64::new(0),
            causal_external_disk_blames_total: AtomicU64::new(0),
            causal_external_net_blames_total: AtomicU64::new(0),

            policy_rollback_evaluations_total: AtomicU64::new(0),
            policy_rollback_executions_total: AtomicU64::new(0),

            arousal_decay_accelerations_total: AtomicU64::new(0),

            companion_cross_group_inferences_total: AtomicU64::new(0),

            adaptive_drift_threshold_raises_total: AtomicU64::new(0),

            user_presence_suppressions_total: AtomicU64::new(0),

            journal_rationales_attached_total: AtomicU64::new(0),

            specialist_accuracy_purge_inhibitions_total: AtomicU64::new(0),
            habituation_skips_total: AtomicU64::new(0),

            // Phase C SCORER-OVERRIDE (Sprint 11 finale, 2026-05-16).
            scorer_override_rejects_total: AtomicU64::new(0),
            scorer_disagreement_strong_accepts_total: AtomicU64::new(0),
            purge_inhibition_skips_total: AtomicU64::new(0),
            mediator_blocks_total: AtomicU64::new(0),
            mediator_noop_writes_total: AtomicU64::new(0),
            mediator_postcondition_violation_total: AtomicU64::new(0),
            causal_thermal_scorer_override_alignments_total: AtomicU64::new(0),
            companion_affinity_alignments_total: AtomicU64::new(0),
            companion_observe_router_skips_total: AtomicU64::new(0),
            companion_fg_cache_hits_total: AtomicU64::new(0),
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

    /// Drain the cumulative ns for one stage since the previous drain.
    ///
    /// Windowed counterpart to [`drain_stage_max_ns`]. Both producer
    /// (`record_stage`) and consumer (windowed avg in
    /// `daemon_cycle_tail`) must agree on the same time horizon —
    /// otherwise tail-light stages structurally exhibit avg > max
    /// (lifetime sum / lifetime count vs drained interval max).
    /// Sprint 9 `4b13a39` rule: producer + consumer agree on the
    /// telemetry horizon. [Welford 1962] online statistics windowing.
    #[inline(always)]
    pub fn drain_stage_total_ns(&self, stage: CycleStage) -> u64 {
        let total = match stage {
            CycleStage::Sense => &self.stage_sense_total_ns,
            CycleStage::Reason => &self.stage_reason_total_ns,
            CycleStage::Execute => &self.stage_execute_total_ns,
            CycleStage::Learn => &self.stage_learn_total_ns,
            CycleStage::Persist => &self.stage_persist_total_ns,
            CycleStage::ReasonSignalTick => &self.stage_reason_signal_total_ns,
            CycleStage::ReasonDecide => &self.stage_reason_decide_total_ns,
            CycleStage::ReasonNeuro => &self.stage_reason_neuro_total_ns,
            CycleStage::ReasonUserContext => &self.stage_reason_usercontext_total_ns,
            CycleStage::ReasonHoltWinters => &self.stage_reason_holtwinters_total_ns,
            CycleStage::ReasonPageReclaim => &self.stage_reason_pagereclaim_total_ns,
            CycleStage::ReasonChromium => &self.stage_reason_chromium_total_ns,
            CycleStage::ReasonEnrich => &self.stage_reason_enrich_total_ns,
        };
        total.swap(0, Ordering::Relaxed)
    }

    /// Drain the per-cycle stage count since the previous drain.
    ///
    /// Mirror of [`drain_stage_max_ns`] for the divisor of the windowed
    /// avg. Must be drained exactly once per publish, paired with one
    /// `drain_stage_total_ns` call per stage. [Welford 1962].
    #[inline(always)]
    pub fn drain_stage_count_window(&self) -> u64 {
        self.stage_count.swap(0, Ordering::Relaxed)
    }

    /// Drain the observed max for one stage since the previous drain.
    ///
    /// The total counters remain cumulative for long-run averages, but max
    /// values are interval telemetry. Keeping lifetime maxes made a single
    /// stall poison dashboards for days.
    #[inline(always)]
    pub fn drain_stage_max_ns(&self, stage: CycleStage) -> u64 {
        let max = match stage {
            CycleStage::Sense => &self.stage_sense_max_ns,
            CycleStage::Reason => &self.stage_reason_max_ns,
            CycleStage::Execute => &self.stage_execute_max_ns,
            CycleStage::Learn => &self.stage_learn_max_ns,
            CycleStage::Persist => &self.stage_persist_max_ns,
            CycleStage::ReasonSignalTick => &self.stage_reason_signal_max_ns,
            CycleStage::ReasonDecide => &self.stage_reason_decide_max_ns,
            CycleStage::ReasonNeuro => &self.stage_reason_neuro_max_ns,
            CycleStage::ReasonUserContext => &self.stage_reason_usercontext_max_ns,
            CycleStage::ReasonHoltWinters => &self.stage_reason_holtwinters_max_ns,
            CycleStage::ReasonPageReclaim => &self.stage_reason_pagereclaim_max_ns,
            CycleStage::ReasonChromium => &self.stage_reason_chromium_max_ns,
            CycleStage::ReasonEnrich => &self.stage_reason_enrich_max_ns,
        };
        max.swap(0, Ordering::Relaxed)
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
        self.reactor_event_weight_bits
            .store(weight.to_bits(), Ordering::Relaxed);
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

    /// Drain metrics-lock maxes since the previous publish.
    ///
    /// Average counters remain cumulative, but max values are interval
    /// telemetry like stage maxes. Otherwise one lock stall poisons the
    /// dashboard indefinitely.
    #[inline(always)]
    pub fn drain_metrics_lock_max_ns(&self) -> (u64, u64) {
        (
            self.metrics_lock_wait_max_ns.swap(0, Ordering::Relaxed),
            self.metrics_lock_held_max_ns.swap(0, Ordering::Relaxed),
        )
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
            maintenance_purge_skipped_bus_saturated_total: self
                .maintenance_purge_skipped_bus_saturated_total
                .load(Ordering::Relaxed),
            taskinfo_cache_hits: self.taskinfo_cache_hits.load(Ordering::Relaxed),
            taskinfo_cache_misses: self.taskinfo_cache_misses.load(Ordering::Relaxed),
            taskinfo_cache_exit_invalidations: self
                .taskinfo_cache_exit_invalidations
                .load(Ordering::Relaxed),
            taskinfo_cache_cap_evictions: self.taskinfo_cache_cap_evictions.load(Ordering::Relaxed),
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
            causal_external_thermal_blames_total: self
                .causal_external_thermal_blames_total
                .load(Ordering::Relaxed),
            causal_external_disk_blames_total: self
                .causal_external_disk_blames_total
                .load(Ordering::Relaxed),
            causal_external_net_blames_total: self
                .causal_external_net_blames_total
                .load(Ordering::Relaxed),
            policy_rollback_evaluations_total: self
                .policy_rollback_evaluations_total
                .load(Ordering::Relaxed),
            policy_rollback_executions_total: self
                .policy_rollback_executions_total
                .load(Ordering::Relaxed),
            arousal_decay_accelerations_total: self
                .arousal_decay_accelerations_total
                .load(Ordering::Relaxed),
            companion_cross_group_inferences_total: self
                .companion_cross_group_inferences_total
                .load(Ordering::Relaxed),
            adaptive_drift_threshold_raises_total: self
                .adaptive_drift_threshold_raises_total
                .load(Ordering::Relaxed),
            user_presence_suppressions_total: self
                .user_presence_suppressions_total
                .load(Ordering::Relaxed),
            journal_rationales_attached_total: self
                .journal_rationales_attached_total
                .load(Ordering::Relaxed),

            specialist_accuracy_purge_inhibitions_total: self
                .specialist_accuracy_purge_inhibitions_total
                .load(Ordering::Relaxed),
            habituation_skips_total: self.habituation_skips_total.load(Ordering::Relaxed),
            scorer_override_rejects_total: self
                .scorer_override_rejects_total
                .load(Ordering::Relaxed),
            scorer_disagreement_strong_accepts_total: self
                .scorer_disagreement_strong_accepts_total
                .load(Ordering::Relaxed),
            purge_inhibition_skips_total: self.purge_inhibition_skips_total.load(Ordering::Relaxed),
            mediator_blocks_total: self.mediator_blocks_total.load(Ordering::Relaxed),
            mediator_noop_writes_total: self.mediator_noop_writes_total.load(Ordering::Relaxed),
            mediator_postcondition_violation_total: self
                .mediator_postcondition_violation_total
                .load(Ordering::Relaxed),
            causal_thermal_scorer_override_alignments_total: self
                .causal_thermal_scorer_override_alignments_total
                .load(Ordering::Relaxed),
            companion_affinity_alignments_total: self
                .companion_affinity_alignments_total
                .load(Ordering::Relaxed),
            companion_observe_router_skips_total: self
                .companion_observe_router_skips_total
                .load(Ordering::Relaxed),
            companion_fg_cache_hits_total: self
                .companion_fg_cache_hits_total
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

    // ── Phase 4.2 — External-event causal blame counters ──────────────
    //
    // Called from `CausalGraph::evaluate_with_resources` each time an
    // edge is tagged with `external_blame`. Per-kind so dashboards can
    // distinguish thermal-driven false credits (M1 8GB throttles a lot
    // under sustained load) from disk-driven (Spotlight) and
    // network-driven (Wi-Fi handoff) confounders.

    #[inline(always)]
    pub fn inc_causal_external_thermal_blame(&self) {
        self.causal_external_thermal_blames_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_causal_external_disk_blame(&self) {
        self.causal_external_disk_blames_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_causal_external_net_blame(&self) {
        self.causal_external_net_blames_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Phase 4.3 — Policy Rollback Guard observability hooks (Sprint 7).
    /// Wired from `PolicyRollbackGuard::evaluate` and `mark_executed`
    /// in `learned_state.rs`. Caller code in `daemon` will be wired in
    /// a follow-up commit per the Phase 4.3 OPENS: 1 directive.
    pub fn inc_policy_rollback_evaluation(&self) {
        self.policy_rollback_evaluations_total
            .fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_policy_rollback_execution(&self) {
        self.policy_rollback_executions_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Phase 3.2 — Arousal-Modulated NARS Decay observability hook.
    /// Call once per persist whose
    /// `DriftDetector::arousal_modulated_decay_factor(...)` produced a
    /// factor strictly less than `base_factor` (i.e. Stressed or Crisis
    /// zone, decay accelerated). Mirrors the Phase 3.1 counter design so
    /// dashboards can verify the feature engages in prod.
    /// [McGaugh 2004]; [Yerkes & Dodson 1908].
    pub fn add_arousal_decay_accelerations(&self, n: u64) {
        self.arousal_decay_accelerations_total
            .fetch_add(n, Ordering::Relaxed);
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

    /// Phase 4.1 — Adaptive Drift Threshold observability hook.
    /// Call once per `AdaptiveDriftThreshold::recommended_threshold` that
    /// returned strictly greater than the supplied base (i.e. the noise
    /// variance bumped the threshold above the operator-tuned floor).
    /// Mirrors the Phase 3.1 / 3.2 / 5.2 counter design so dashboards can
    /// detect whether the adaptive layer actually engages in prod.
    /// [Brown 1959]; [Welford 1962]; [Kuncheva 2004].
    pub fn add_adaptive_drift_threshold_raises(&self, n: u64) {
        self.adaptive_drift_threshold_raises_total
            .fetch_add(n, Ordering::Relaxed);
    }

    /// Phase 5.1 — User-presence suppression observability hook.
    /// Call once per `user_presence_modulator` invocation that returned a
    /// multiplier strictly less than 1.0 (active or semi-active tier).
    /// Dashboards compute the "presence suppression rate" as
    /// `user_presence_suppressions_total / cycles` to verify the modulator
    /// fires while the user is at the keyboard and stays at 0 during long
    /// idles. Mirrors `add_skill_aware_modulations` and
    /// `inc_battery_aware_penalty_emission` shape so dashboards can group
    /// the three Sprint-6/8 instrumentation counters uniformly.
    #[inline(always)]
    pub fn add_user_presence_suppressions(&self, n: u64) {
        self.user_presence_suppressions_total
            .fetch_add(n, Ordering::Relaxed);
    }

    /// Phase 5.3 — Structured-rationale observability hook. Call once per
    /// `JournalEntry` written with a non-`None` `rationale` field. Stays at
    /// 0 in prod until the wiring follow-up lands and a journal write-site
    /// emits both `with_rationale(..)` AND `inc_journal_rationale_attached()`.
    ///
    /// [Doshi-Velez & Kim 2017] — observability metric for explanation coverage.
    #[inline(always)]
    pub fn inc_journal_rationale_attached(&self) {
        self.journal_rationales_attached_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Phase 4.3.1 — Specialist accuracy purge inhibition observability hook.
    /// Call once per cycle where the cognitive tick skipped the
    /// `specialist_accuracy.update()` calls because a maintenance purge
    /// happened in the previous 30 s. Dashboards compute the "purge
    /// inhibition rate" as
    /// `specialist_accuracy_purge_inhibitions_total / cycles` to verify the
    /// guard is firing exactly once per qualifying cycle and not spuriously
    /// out-of-window.
    #[inline(always)]
    pub fn inc_specialist_accuracy_purge_inhibitions(&self) {
        self.specialist_accuracy_purge_inhibitions_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Phase 2 god-lock decomposition (2026-05-16): cumulative habituation
    /// skips. Called once per cycle by
    /// `daemon_cognitive_tick::update_habituation_state` with
    /// `habituated_pids.len() as u64`. Replaces the previous
    /// `state.metrics.lock_recover().metrics.habituation_skips += N` god-lock
    /// write — single LSE `ldadd` instead of `mutex.lock + write + drop`.
    #[inline(always)]
    pub fn add_habituation_skips(&self, n: u64) {
        self.habituation_skips_total.fetch_add(n, Ordering::Relaxed);
    }

    /// Phase C SCORER-OVERRIDE (Sprint 11 finale): bump once per action
    /// where the PolicyScorer beat the gate tower in the SAFE direction
    /// (gate accept → composite < −0.30 → final reject). Call from the
    /// `decide_actions` override site only; tests bump directly to
    /// validate the round-trip.
    /// [Nygard 2018 §8.5] — observe the rejected counterfactual.
    #[inline(always)]
    pub fn inc_scorer_override_reject(&self) {
        self.scorer_override_rejects_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Phase C SCORER-OVERRIDE (Sprint 11 finale): bump once per action
    /// where the gate REJECTED but the scorer's composite was strictly
    /// greater than +0.30 (strong accept). The asymmetric design refuses
    /// to let the scorer beat the gate in the unsafe direction —
    /// per NotebookLM 2026-05-16 Candidate-C verdict — so this counter
    /// represents *logged disagreement only*, no action change. Used by
    /// offline analysis (and Sprint 12 cutover gating) to verify the
    /// scorer would have been right.
    #[inline(always)]
    pub fn inc_scorer_disagreement_strong_accept(&self) {
        self.scorer_disagreement_strong_accepts_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Phase D PURGE-INHIBITION: bump once per predictor swap-update gated
    /// during the post-purge inhibition window (see
    /// `MaintenanceState::is_in_purge_inhibition_window`). Producer lives
    /// in `signal_intelligence::step` so a single gate per cycle yields a
    /// single increment, even if multiple predictors share the swap input
    /// downstream of the gate.
    /// [Hellerstein 2004 §9] disturbance rejection — counter proves the
    /// closed-loop system observed and refused the exogenous shock.
    #[inline(always)]
    pub fn inc_purge_inhibition_skip(&self) {
        self.purge_inhibition_skips_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// RAM Phase B — mediator interposed on effect before syscall.
    #[inline(always)]
    pub fn inc_mediator_block(&self) {
        self.mediator_blocks_total.fetch_add(1, Ordering::Relaxed);
    }

    /// RAM Phase B — Receipt showed `before == after` for the effect's target
    /// dimension. Catches SetSysctl no-op class (Sprint 3 2026-05-07 lesson).
    #[inline(always)]
    pub fn inc_mediator_noop_write(&self) {
        self.mediator_noop_writes_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// RAM Phase B — syscall returned success but post-snapshot still failed
    /// the expected delta. Surfaces lying syscalls / kernel rejections that
    /// pretend to succeed.
    #[inline(always)]
    pub fn inc_mediator_postcondition_violation(&self) {
        self.mediator_postcondition_violation_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Sprint 12 Convergence #4: bump once per cycle where the scorer
    /// override count grew AND the causal graph has a recent
    /// `ThermalThrottle` blame inside `EXTERNAL_BLAME_WINDOW`. See
    /// `LockFreeMetrics::causal_thermal_scorer_override_alignments_total`
    /// for the convergence rationale.
    #[inline(always)]
    pub fn inc_causal_thermal_scorer_override_alignment(&self) {
        self.causal_thermal_scorer_override_alignments_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Sprint 12 Convergence #1: bump once per cold-thread routing
    /// decision that flipped to P-cluster because the owning process is
    /// a foreground companion AND DRAM bandwidth is below the safety
    /// floor.
    #[inline(always)]
    pub fn inc_companion_affinity_alignment(&self) {
        self.companion_affinity_alignments_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Sprint 13 Pressure-Router Gate: bump once per cycle the daemon
    /// main loop skipped `companion_graph.observe_cycle` + the Phase 3.3
    /// cross-group propagation because `memory_pressure < mid_entry`
    /// AND the modulo-4 forced-exploration fallback did not fire.
    #[inline(always)]
    pub fn inc_companion_observe_router_skip(&self) {
        self.companion_observe_router_skips_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Sprint 12 perf-fix (2026-05-30): bump once per cycle when the
    /// `companion_of_fg_pids` set was served from the memoization cache
    /// instead of being rebuilt by scanning `top_processes`. Hit ratio
    /// (hits / cycles) should approach 1.0 in steady state — foreground
    /// app rarely flips and top_processes is stable across consecutive
    /// 5-s ticks. Drops only on fg flip OR CompanionGraph mutation.
    /// [Saltzer & Schroeder 1975] Economy of Mechanism.
    #[inline(always)]
    pub fn inc_companion_fg_cache_hit(&self) {
        self.companion_fg_cache_hits_total
            .fetch_add(1, Ordering::Relaxed);
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
    pub maintenance_purge_skipped_bus_saturated_total: u64,
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

    /// Phase 4.2 — External-event causal blame counters (Sprint 7).
    pub causal_external_thermal_blames_total: u64,
    pub causal_external_disk_blames_total: u64,
    pub causal_external_net_blames_total: u64,

    /// Phase 4.3 — Policy Rollback Guard observability (Sprint 7).
    pub policy_rollback_evaluations_total: u64,
    pub policy_rollback_executions_total: u64,

    /// Phase 3.2 — Arousal-Modulated NARS Decay counter.
    pub arousal_decay_accelerations_total: u64,

    /// Phase 3.3 — Cross-Group Companion Attention inferences (Sprint 6).
    pub companion_cross_group_inferences_total: u64,

    /// Phase 4.1 — Adaptive Drift Threshold raises counter (Sprint 7).
    pub adaptive_drift_threshold_raises_total: u64,

    /// Phase 5.1 — User-presence suppressions emitted (Sprint 8).
    /// Count of `user_presence_modulator` calls that returned a
    /// multiplier strictly less than 1.0 (active/semi-active tier).
    pub user_presence_suppressions_total: u64,

    /// Phase 5.3 — JournalEntry rationale attachments (Sprint 8).
    pub journal_rationales_attached_total: u64,

    /// Phase 4.3.1 — Specialist accuracy purge inhibitions (Sprint 8).
    /// Count of cycles where `apply_specialist_voting` skipped the EMA
    /// accuracy updates because a maintenance purge happened ≤30 s ago.
    pub specialist_accuracy_purge_inhibitions_total: u64,
    /// Phase 2 god-lock decomposition (Sprint 8, 2026-05-16).
    /// Cumulative habituation skips, migrated off the `state.metrics`
    /// mutex to a lock-free atomic. Mirrors the legacy
    /// `RuntimeMetrics.habituation_skips` field (which is now populated
    /// FROM this atomic via `sync_from_lockfree`).
    pub habituation_skips_total: u64,

    /// Phase C SCORER-OVERRIDE (Sprint 11 finale, 2026-05-16).
    /// Scorer overrode a gate-ACCEPT into a final REJECT (composite <
    /// −0.30). Surfaced through `RuntimeMetrics → runtime_metrics.json`
    /// so the user can verify the partial cutover engages.
    pub scorer_override_rejects_total: u64,
    /// Phase C SCORER-OVERRIDE (Sprint 11 finale, 2026-05-16).
    /// Gate REJECTED but scorer wanted to ACCEPT strongly (composite >
    /// +0.30). Logged for offline analysis only — per NotebookLM
    /// Candidate-C verdict, the asymmetric mode does NOT let the
    /// scorer beat the gate in the unsafe direction.
    pub scorer_disagreement_strong_accepts_total: u64,
    /// Phase D PURGE-INHIBITION (Sprint 12 candidate #1, 2026-05-17).
    /// Cycles where predictor swap-update was suppressed because a
    /// `vm_purge` fired in the prior 5 s. See
    /// [`crate::engine::lse_counters::LockFreeMetrics::purge_inhibition_skips_total`].
    pub purge_inhibition_skips_total: u64,

    /// RAM Phase B mediator counters. See LSE producer docs.
    pub mediator_blocks_total: u64,
    pub mediator_noop_writes_total: u64,
    pub mediator_postcondition_violation_total: u64,

    /// Sprint 12 Convergence #4 (2026-05-17). Coincidence count of
    /// scorer overrides happening within the thermal external-blame
    /// window. See LSE producer doc — when this ramps, the policy
    /// scorer is consistently disagreeing during thermal pressure,
    /// which is high-signal evidence the learned policy needs a
    /// rollback nudge.
    pub causal_thermal_scorer_override_alignments_total: u64,

    /// Sprint 12 Convergence #1 (2026-05-17). Cumulative count of cold
    /// threads routed to P-cluster (instead of default E) because the
    /// owning process is a foreground companion AND DRAM bandwidth is
    /// below the safety floor. See LSE producer for routing rationale.
    pub companion_affinity_alignments_total: u64,

    /// Sprint 13 Pressure-Router Gate (2026-05-30). Cycles where the
    /// daemon main loop skipped `companion_graph.observe_cycle` + Phase
    /// 3.3 propagation under low pressure. See LSE producer for gate
    /// semantics.
    pub companion_observe_router_skips_total: u64,
    /// Sprint 12 perf-fix (2026-05-30). Cumulative count of per-cycle
    /// hits on the `companion_of_fg_pids` memoization cache (no
    /// rebuild, no HashSet alloc). See the producer doc on
    /// [`LockFreeMetrics::companion_fg_cache_hits_total`] for the
    /// invalidation contract.
    pub companion_fg_cache_hits_total: u64,
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

    /// Phase 2 god-lock fix (2026-05-16): the reactor_event_weight setter must
    /// round-trip via `snapshot()`. The daemon main loop reads the value,
    /// decays it (×0.75), and then writes it back through this setter. If the
    /// setter never lands in the atomic, the snapshot stays pinned at the
    /// last `set_reactor_event_weight(1.0)` call (set by daemon_reactor.rs on
    /// every pulse), the decay is invisible, and `reactor_event_weight` reads
    /// "sticky 1.0" indefinitely — causing the governor to see fake reactive
    /// pressure long after the reactor went quiet. Adversarial review
    /// 2026-05-16.
    #[test]
    fn reactor_event_weight_round_trips_via_set_get() {
        let m = LockFreeMetrics::new();
        m.set_reactor_event_weight(0.42);
        m.commit();
        let snap = m.snapshot();
        assert!(
            (snap.reactor_event_weight - 0.42).abs() < 1e-9,
            "expected 0.42, got {}",
            snap.reactor_event_weight
        );
    }

    #[test]
    fn stage_max_drain_does_not_keep_lifetime_outlier_sticky() {
        let m = LockFreeMetrics::new();
        m.record_stage(CycleStage::Reason, 180_000_000_000);
        assert_eq!(m.drain_stage_max_ns(CycleStage::Reason), 180_000_000_000);

        m.record_stage(CycleStage::Reason, 80_000_000);
        assert_eq!(
            m.drain_stage_max_ns(CycleStage::Reason),
            80_000_000,
            "stage max should reflect recent drained interval, not lifetime max"
        );
    }

    #[test]
    fn metrics_lock_max_drain_does_not_keep_lifetime_outlier_sticky() {
        let m = LockFreeMetrics::new();
        m.record_metrics_lock(40_000_000, 180_000_000);
        assert_eq!(m.drain_metrics_lock_max_ns(), (40_000_000, 180_000_000));

        m.record_metrics_lock(4_000, 8_000);
        assert_eq!(
            m.drain_metrics_lock_max_ns(),
            (4_000, 8_000),
            "metrics lock max should reflect recent drained interval, not lifetime max"
        );
    }
}
