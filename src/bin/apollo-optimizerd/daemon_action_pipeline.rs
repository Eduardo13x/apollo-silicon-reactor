//! # Daemon Action Pipeline — Filter Stage
//!
//! Pre-execution filter pipeline extracted from the daemon main loop as
//! part of the V1.1.0 Strangler Fig Wave 9 (Pass 1) [Fowler 2004].
//!
//! ## Scope (Pass 1)
//!
//! This module handles the **filter pipeline** that sits between
//! Phase 1 (budget-filtered action computation) and Phase 2 (action
//! execution).  It does NOT execute actions; it only reshapes the
//! `final_actions` vector according to:
//!
//! 1. **Circuit-breaker snapshot** (read-only) — samples the current
//!    breaker state so Phase 2 can skip execution when Open.
//! 2. **Degradation tier** — [Nygard 2018] Release It! circuit-breaker
//!    + degradation composition.  Updates the degradation controller
//!    with `new_failures = 0` at cycle start; new failures from
//!    execution are folded in *after* Phase 2 by the caller.
//! 3. **Cognitive gates** — [Lakshminarayanan 2017] + [Sutton 2018 §13]
//!    epistemic uncertainty gates downgrade the operation mode:
//!    - `observe_only` (uncertainty > 0.85) → `Observe`
//!    - `block_aggressive` (uncertainty > 0.70) → `Conservative`
//!    Evaluated in this order (observe_only takes precedence).
//! 4. **Mode filter** — `Emergency`/`Observe`/`Conservative`/`Full`
//!    each keep a specific allowed-action set.
//! 5. **Throttle dedup** — skip `ThrottleProcess` for PIDs already
//!    throttled last cycle (prevents journal I/O saturation).
//! 6. **Causal QoS upgrade** — [Pearl 2009 §3] FreezeProcess →
//!    ThrottleProcess(aggressive) for CPU-dominant processes evidenced
//!    by the causal graph.
//!
//! ## Lock invariant
//!
//! The only mutex this helper touches is `state.policy`, which it
//! acquires briefly three times (breaker read, degradation update,
//! causal QoS is pure).  It does NOT touch `metrics`, `frozen_state`,
//! or `mach_qos` — those belong to the Phase 2 execution path the
//! caller still owns.
//!
//! ## Circuit-breaker attribution
//!
//! The breaker snapshot is read BEFORE the cognitive gates so the
//! breaker's degradation-tier input reflects the real cycle state,
//! but **no record_success / record_failure calls** are made here.
//! Those stay in Phase 2 where the breaker sees actual execution
//! outcomes — preserving attribution semantics [Nygard 2018].

use std::collections::HashSet;

use apollo_engine::collector::SystemSnapshot;
use apollo_engine::engine::circuit_breaker::CircuitState;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::degradation::{DegradationInputs, OperationMode};
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::swap_reclaim::SwapRisk;
use apollo_engine::engine::types::RootAction;

use crate::cognitive_tick::CognitiveDecision;

/// Outcome of the filter pipeline.
///
/// Phase 2 consumes `filtered_actions` and uses `cb_is_open` / `op_mode`
/// to decide whether to skip execution or restrict it.
pub struct FilterOutcome {
    /// Actions surviving all filters, ready for Phase 2 execution.
    pub filtered_actions: Vec<RootAction>,
    /// Circuit-breaker open snapshot — Phase 2 skips execution when true.
    pub cb_is_open: bool,
    /// Effective degradation tier after cognitive-gate downgrade.
    pub op_mode: OperationMode,
    /// Count of FreezeProcess actions rewritten to ThrottleProcess this cycle.
    pub causal_qos_upgrades: u32,
}

/// Previous-cycle throttled PIDs (dedup across cycles).
///
/// Static Mutex preserves the original single-instance semantics when
/// the dedup logic lived inline in `main.rs`.  The daemon only has one
/// hot loop, so there's no contention — the Mutex is just for
/// static-init safety.
static PREV_THROTTLED: std::sync::Mutex<Option<HashSet<u32>>> = std::sync::Mutex::new(None);

/// Run the filter pipeline.
///
/// Must be called after Phase 1 (which produces `final_actions`) and
/// before Phase 2 (which consumes `FilterOutcome.filtered_actions`).
///
/// # Parameters
/// - `final_actions` — actions surviving Phase 1 budget enforcement.
/// - `state` — for the `policy` mutex (circuit breaker + degradation).
/// - `snapshot` — used to read `kernel_task` CPU for degradation input.
/// - `prev_cog_decision` — last-cycle cognitive decision; gates current cycle.
/// - `causal_qos_names` — process names flagged as CPU-dominant by the
///   causal graph; their freezes are upgraded to QoS throttles.
/// - `swap_risk` — ODE physical model risk level; Critical/Overflow overrides
///   `observe_only` gate from Observe to Conservative so swap-pressure actions
///   are never fully suppressed by epistemic uncertainty alone.
pub fn run_filter_pipeline(
    final_actions: Vec<RootAction>,
    state: &SharedState,
    snapshot: &SystemSnapshot,
    prev_cog_decision: Option<&CognitiveDecision>,
    causal_qos_names: &HashSet<String>,
    swap_risk: SwapRisk,
) -> FilterOutcome {
    // ── Circuit breaker snapshot ─────────────────────────────────
    let (cb_is_open, cb_open_duration) = {
        let pg = state.policy.lock_recover();
        let is_open = *pg.circuit_breaker.state() == CircuitState::Open;
        let dur = pg.circuit_breaker.open_duration();
        (is_open, dur)
    };

    // ── Degradation pre-check ────────────────────────────────────
    // new_failures = 0 at cycle start; execution failures get folded
    // in by the caller AFTER Phase 2.
    let op_mode = {
        let kernel_cpu = snapshot
            .top_processes
            .iter()
            .find(|p| p.name == "kernel_task")
            .map(|p| p.cpu_usage as f64)
            .unwrap_or(0.0);
        let mut pg = state.policy.lock_recover();
        let inp = DegradationInputs {
            new_failures: 0,
            kernel_task_cpu_pct: kernel_cpu,
            circuit_open: cb_is_open,
            circuit_open_duration: cb_open_duration,
        };
        pg.degradation.update(&inp).clone()
    };

    // ── Cognitive gates ──────────────────────────────────────────
    // Order matters: observe_only is evaluated before block_aggressive
    // because "no actions" strictly dominates "no aggressive actions".
    // [Lakshminarayanan 2017] predictive-uncertainty action inhibition
    // [Sutton 2018 §13] reduce action scope under high policy uncertainty
    //
    // ODE arbiter: when the physical swap model signals Critical or Overflow,
    // epistemic uncertainty may not suppress all actions — the ODE provides
    // physical certainty that overrides behavioral uncertainty.
    // [Garcia & Fernandez 2015] safe RL — constraint violations bypass uncertainty gates.
    let ode_physical_critical = matches!(swap_risk, SwapRisk::Critical | SwapRisk::Overflow);
    let op_mode =
        if prev_cog_decision.map_or(false, |d| d.observe_only) && op_mode == OperationMode::Full {
            if ode_physical_critical {
                // ODE override: floor at Conservative so freeze/throttle survive.
                tracing::debug!(
                    "cognitive gate: observe_only overridden by ODE physical Critical \
                 → OperationMode::Conservative"
                );
                OperationMode::Conservative
            } else {
                tracing::debug!("cognitive gate: observe_only → OperationMode::Observe");
                OperationMode::Observe
            }
        } else if prev_cog_decision.map_or(false, |d| d.block_aggressive)
            && op_mode == OperationMode::Full
        {
            tracing::debug!("cognitive gate: block_aggressive → OperationMode::Conservative");
            OperationMode::Conservative
        } else {
            op_mode
        };

    // ── Mode filter ──────────────────────────────────────────────
    let filtered_actions: Vec<RootAction> = if op_mode == OperationMode::Emergency {
        // Emergency: only unfreeze, no new actions.
        final_actions
            .into_iter()
            .filter(|a| matches!(a, RootAction::UnfreezeProcess { .. }))
            .collect()
    } else if op_mode == OperationMode::Observe {
        // Observe: no actions at all.
        Vec::new()
    } else if op_mode == OperationMode::Conservative {
        // Conservative: only unfreeze + QoS hints (no SIGSTOP, no throttle).
        final_actions
            .into_iter()
            .filter(|a| {
                matches!(
                    a,
                    RootAction::UnfreezeProcess { .. }
                        | RootAction::SetThreadQoS { .. }
                        | RootAction::BoostProcess { .. }
                )
            })
            .collect()
    } else {
        // Full: all actions pass through.
        final_actions
    };

    // ── ThrottleProcess dedup ────────────────────────────────────
    // Without this, decide_actions re-throttles 30+ PIDs every cycle,
    // each producing a journal write → I/O saturation → system freeze.
    let filtered_actions = {
        let prev = PREV_THROTTLED.lock().unwrap_or_else(|e| e.into_inner());
        let prev_set = prev.clone().unwrap_or_default();
        drop(prev);
        let mut this_cycle = HashSet::new();
        let deduped: Vec<RootAction> = filtered_actions
            .into_iter()
            .filter(|a| {
                if let RootAction::ThrottleProcess { pid, .. } = a {
                    this_cycle.insert(*pid);
                    !prev_set.contains(pid)
                } else {
                    true
                }
            })
            .collect();
        *PREV_THROTTLED.lock().unwrap_or_else(|e| e.into_inner()) = Some(this_cycle);
        deduped
    };

    // ── Causal QoS upgrade ───────────────────────────────────────
    // FreezeProcess → ThrottleProcess for CPU-dominant processes.
    // No-op when `causal_qos_names` is empty (cold start or no
    // CPU-dominant evidence yet — defaults to the safe SIGSTOP path).
    // [Pearl 2009 §3] mediation analysis
    // [Nygard 2018] bulkhead: least-invasive first
    let mut causal_qos_upgrades: u32 = 0;
    let filtered_actions: Vec<RootAction> = if !causal_qos_names.is_empty() {
        filtered_actions
            .into_iter()
            .map(|a| match a {
                RootAction::FreezeProcess {
                    pid,
                    name,
                    reason,
                    start_sec,
                    start_usec,
                    decision_reason,
                } if causal_qos_names.contains(name.as_str()) => {
                    causal_qos_upgrades += 1;
                    RootAction::ThrottleProcess {
                        pid,
                        name,
                        aggressive: true,
                        reason: format!("{} [causal:qos]", reason),
                        start_sec,
                        start_usec,
                        decision_reason,
                    }
                }
                other => other,
            })
            .collect()
    } else {
        filtered_actions
    };

    FilterOutcome {
        filtered_actions,
        cb_is_open,
        op_mode,
        causal_qos_upgrades,
    }
}
