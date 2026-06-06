//! Action accumulator — typed builder replacing the raw `Vec<RootAction>`
//! accumulation in the daemon main loop. Sprint 4 Fase 5.
//!
//! Goals:
//! - Type-safe push: each variant has a typed method (push_throttle, etc.)
//!   that validates shape (required fields, non-empty names, etc.).
//! - Single terminal exit: `finalize()` -> `Vec<RootAction>` consumed by
//!   downstream filter+dispatch. No partial drain.
//! - Read-only peeks: `view()` exposes the partial vec to legacy read sites
//!   (heuristic_pass, stale_apps, paging_hints) without giving them
//!   ownership. Phantom-dedup hazard pre-exists; semantics-preserving.
//! - Telemetry: per-variant push counters + structured tracing per emit.
//! - Escape hatch: `push_raw(action, EmitContext)` for revert_actions and
//!   already-validated paths (freeze_executor confirmed actions, sysctl
//!   governor tick which already returns clamped actions).
//!
//! NOT in scope (Fase 5):
//! - Runtime validation (alive PID, identity cache verify) — that lives
//!   in flush()/filter_pipeline at downstream sites.
//! - Sealing all 8 PID-bearing variants (Fase 9 candidate).
//! - Phase enforcement (telemetry-only via `ActionPhase` enum).
//! - drop_ratio_5min alarm metric — needs windowed counter infrastructure;
//!   tracked as follow-up. The lse_counters per-variant totals are the
//!   foundation that windowed counter would be built on top of.

use crate::engine::audit_types::DecisionReason;
use crate::engine::lse_counters::LockFreeMetrics;
use crate::engine::types::RootAction;
use std::sync::atomic::Ordering;

/// Telemetry phase tag — typed enum, telemetry only, no enforcement.
///
/// Each emit site picks the phase that best describes what subsystem decided
/// the action. Used solely for tracing fields and follow-up debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionPhase {
    Decide,
    LearnedPolicy,
    SkillTick,
    ClusterActions,
    AgentActions,
    PagingHints,
    Heuristic,
    StaleApps,
    Survival,
    FreezeExecutor,
    SysctlGovernor,
    NetworkOptimizer,
    DispatchTick,
    Reactor,
    /// Catch-all for sites not yet categorized. Should converge to zero.
    Other,
}

/// Context carried with every push for tracing / telemetry. Cheap to clone
/// (all fields are `Copy` or `'static` strings).
#[derive(Debug, Clone, Copy)]
pub struct EmitContext {
    pub phase: ActionPhase,
    /// `file:line tag`, e.g. `"main.rs::3461 network_optimizer"`. Free-form
    /// `'static` so call sites just hard-code the literal.
    pub site: &'static str,
    /// Short why, e.g. `"revert"`, `"confirmed"`, `"profile-tcp-tune"`.
    pub reason: &'static str,
}

impl EmitContext {
    pub const fn new(phase: ActionPhase, site: &'static str, reason: &'static str) -> Self {
        Self {
            phase,
            site,
            reason,
        }
    }
}

/// Typed action builder. Replaces direct `Vec<RootAction>` accumulation in
/// the daemon main loop.
///
/// **Control invariant** (Fase 5): "must not change what Apollo decides;
/// must make emitting invalid actions impossible without leaving evidence."
///
/// - Push-time shape validation only — no policy decisions.
/// - Order is preserved exactly: `finalize()` returns the underlying vec
///   in insertion order. Downstream `sort_by` / `retain` operate on it.
#[derive(Default)]
pub struct ActionAccumulator {
    actions: Vec<RootAction>,
    push_count_throttle: u64,
    push_count_freeze: u64,
    push_count_unfreeze: u64,
    push_count_boost: u64,
    push_count_set_memorystatus: u64,
    push_count_set_thread_qos: u64,
    push_count_set_sysctl: u64,
    push_count_toggle_spotlight: u64,
    push_count_quarantine_daemon: u64,
    push_count_raw: u64,
    rejected_shape: u64,
    phase_counts: PhaseCounters,
}

/// Per-`ActionPhase` counters (Fase 5 reviewer fix #5). Records which
/// subsystem decided each push. Used purely for telemetry — increments mirror
/// the variant counters but partitioned by phase. Rejections (`rejected_shape`)
/// do NOT bump these counters, since the action was never accepted.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PhaseCounters {
    pub decide: u64,
    pub learned_policy: u64,
    pub skill_tick: u64,
    pub cluster_actions: u64,
    pub agent_actions: u64,
    pub paging_hints: u64,
    pub heuristic: u64,
    pub stale_apps: u64,
    pub survival: u64,
    pub freeze_executor: u64,
    pub sysctl_governor: u64,
    pub network_optimizer: u64,
    pub dispatch_tick: u64,
    pub reactor: u64,
    pub other: u64,
}

impl ActionAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            actions: Vec::with_capacity(cap),
            ..Self::default()
        }
    }

    /// Read-only peek at currently emitted actions. Used by legacy read
    /// sites (`heuristic_pass`, `stale_apps`, `paging_hints`,
    /// `cluster_actions`, `skill_tick`) that decide based on what's already
    /// been emitted this cycle.
    ///
    /// Pre-existing hazard: actions visible here may be vetoed downstream
    /// (alive check, dedup). Read-site decisions are based on emit-time
    /// state, not flush-time state. Documented; not introduced by Fase 5.
    pub fn view(&self) -> &[RootAction] {
        &self.actions
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Single terminal exit. Caller owns the returned vec; downstream
    /// filter_pipeline / safety / grace / sort / dispatch operate on it.
    /// `#[must_use]` so the migration cannot accidentally drop the result.
    #[must_use]
    pub fn finalize(self) -> Vec<RootAction> {
        self.actions
    }

    /// Internal — increment the per-phase counter for `phase`.
    /// Called by every `push_*` after shape validation succeeds.
    #[inline]
    fn bump_phase_counter(&mut self, phase: ActionPhase) {
        match phase {
            ActionPhase::Decide => self.phase_counts.decide += 1,
            ActionPhase::LearnedPolicy => self.phase_counts.learned_policy += 1,
            ActionPhase::SkillTick => self.phase_counts.skill_tick += 1,
            ActionPhase::ClusterActions => self.phase_counts.cluster_actions += 1,
            ActionPhase::AgentActions => self.phase_counts.agent_actions += 1,
            ActionPhase::PagingHints => self.phase_counts.paging_hints += 1,
            ActionPhase::Heuristic => self.phase_counts.heuristic += 1,
            ActionPhase::StaleApps => self.phase_counts.stale_apps += 1,
            ActionPhase::Survival => self.phase_counts.survival += 1,
            ActionPhase::FreezeExecutor => self.phase_counts.freeze_executor += 1,
            ActionPhase::SysctlGovernor => self.phase_counts.sysctl_governor += 1,
            ActionPhase::NetworkOptimizer => self.phase_counts.network_optimizer += 1,
            ActionPhase::DispatchTick => self.phase_counts.dispatch_tick += 1,
            ActionPhase::Reactor => self.phase_counts.reactor += 1,
            ActionPhase::Other => self.phase_counts.other += 1,
        }
    }

    // ── Typed push methods ───────────────────────────────────────────────

    /// Push a `ThrottleProcess` action with full identity (kernel start
    /// times for A-B-A guard).
    ///
    /// Shape validation: `pid != 0`, `name` non-empty.
    #[allow(clippy::too_many_arguments)]
    pub fn push_throttle(
        &mut self,
        pid: u32,
        name: impl Into<String>,
        aggressive: bool,
        reason: impl Into<String>,
        decision_reason: DecisionReason,
        start_sec: u64,
        start_usec: u64,
        ctx: EmitContext,
        lf_metrics: &LockFreeMetrics,
    ) {
        let name = name.into();
        if pid == 0 || name.is_empty() {
            self.rejected_shape += 1;
            lf_metrics
                .actions_rejected_shape_total
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                target: "apollo.accumulator",
                phase = ?ctx.phase,
                site = ctx.site,
                reason = ctx.reason,
                variant = "throttle",
                pid = pid,
                name_empty = name.is_empty(),
                "rejected shape"
            );
            return;
        }
        let action = RootAction::ThrottleProcess {
            pid,
            name,
            aggressive,
            reason: reason.into(),
            decision_reason,
            start_sec,
            start_usec,
        };
        tracing::debug!(
            target: "apollo.accumulator",
            phase = ?ctx.phase,
            site = ctx.site,
            reason = ctx.reason,
            variant = "throttle",
            pid = pid,
            "action_emitted"
        );
        self.push_count_throttle += 1;
        lf_metrics
            .actions_pushed_throttle_total
            .fetch_add(1, Ordering::Relaxed);
        self.bump_phase_counter(ctx.phase);
        self.actions.push(action);
    }

    /// Push a `FreezeProcess` action with full identity.
    /// Shape validation: `pid != 0`, `name` non-empty.
    #[allow(clippy::too_many_arguments)]
    pub fn push_freeze(
        &mut self,
        pid: u32,
        name: impl Into<String>,
        reason: impl Into<String>,
        decision_reason: DecisionReason,
        start_sec: u64,
        start_usec: u64,
        ctx: EmitContext,
        lf_metrics: &LockFreeMetrics,
    ) {
        let name = name.into();
        if pid == 0 || name.is_empty() {
            self.rejected_shape += 1;
            lf_metrics
                .actions_rejected_shape_total
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                target: "apollo.accumulator",
                phase = ?ctx.phase,
                site = ctx.site,
                reason = ctx.reason,
                variant = "freeze",
                pid = pid,
                name_empty = name.is_empty(),
                "rejected shape"
            );
            return;
        }
        let action = RootAction::FreezeProcess {
            pid,
            name,
            reason: reason.into(),
            decision_reason,
            start_sec,
            start_usec,
        };
        tracing::debug!(
            target: "apollo.accumulator",
            phase = ?ctx.phase,
            site = ctx.site,
            reason = ctx.reason,
            variant = "freeze",
            pid = pid,
            "action_emitted"
        );
        self.push_count_freeze += 1;
        lf_metrics
            .actions_pushed_freeze_total
            .fetch_add(1, Ordering::Relaxed);
        self.bump_phase_counter(ctx.phase);
        self.actions.push(action);
    }

    /// Push a `SetSysctl` action via the Fase 4 sealed factory (clamped).
    pub fn push_set_sysctl_clamped(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
        reason: impl Into<String>,
        decision_reason: DecisionReason,
        ctx: EmitContext,
        lf_metrics: &LockFreeMetrics,
    ) {
        let key = key.into();
        let value = value.into();
        if key.is_empty() {
            self.rejected_shape += 1;
            lf_metrics
                .actions_rejected_shape_total
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                target: "apollo.accumulator",
                phase = ?ctx.phase,
                site = ctx.site,
                reason = ctx.reason,
                variant = "set_sysctl",
                "rejected shape: empty key"
            );
            return;
        }
        let action = RootAction::set_sysctl(key, value, reason, decision_reason);
        tracing::debug!(
            target: "apollo.accumulator",
            phase = ?ctx.phase,
            site = ctx.site,
            reason = ctx.reason,
            variant = "set_sysctl",
            "action_emitted"
        );
        self.push_count_set_sysctl += 1;
        lf_metrics
            .actions_pushed_set_sysctl_total
            .fetch_add(1, Ordering::Relaxed);
        self.bump_phase_counter(ctx.phase);
        self.actions.push(action);
    }

    /// Escape hatch for revert/confirmed-action paths and subsystems whose
    /// `tick()` already returns validated `RootAction`s (e.g. `sysctl_governor`
    /// emits sealed `SetSysctlAction` via the clamping factory). The
    /// `EmitContext` is required and recorded so the audit trail is intact.
    ///
    /// Counter semantics:
    /// - `actions_pushed_raw_total` counts escape-hatch emissions.
    /// - per-variant counters count actual emitted variant volume, including
    ///   actions that came through raw paths.
    ///
    /// This keeps runtime metrics useful: a raw `BoostProcess` should still
    /// move the boost counter, while `raw` separately shows how much traffic
    /// bypassed typed construction.
    pub fn push_raw(&mut self, action: RootAction, ctx: EmitContext, lf_metrics: &LockFreeMetrics) {
        let variant = action_variant_name(&action);
        tracing::debug!(
            target: "apollo.accumulator",
            phase = ?ctx.phase,
            site = ctx.site,
            reason = ctx.reason,
            variant = variant,
            raw = true,
            "action_emitted"
        );
        self.push_count_raw += 1;
        lf_metrics
            .actions_pushed_raw_total
            .fetch_add(1, Ordering::Relaxed);
        match &action {
            RootAction::ThrottleProcess { .. } => {
                self.push_count_throttle += 1;
                lf_metrics
                    .actions_pushed_throttle_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            RootAction::FreezeProcess { .. } => {
                self.push_count_freeze += 1;
                lf_metrics
                    .actions_pushed_freeze_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            RootAction::UnfreezeProcess { .. } => {
                self.push_count_unfreeze += 1;
                lf_metrics
                    .actions_pushed_unfreeze_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            RootAction::BoostProcess { .. } => {
                self.push_count_boost += 1;
                lf_metrics
                    .actions_pushed_boost_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            RootAction::SetMemorystatus { .. } => {
                self.push_count_set_memorystatus += 1;
                lf_metrics
                    .actions_pushed_set_memorystatus_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            RootAction::SetThreadQoS { .. } => {
                self.push_count_set_thread_qos += 1;
                lf_metrics
                    .actions_pushed_set_thread_qos_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            RootAction::SetSysctl(_) => {
                self.push_count_set_sysctl += 1;
                lf_metrics
                    .actions_pushed_set_sysctl_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            RootAction::ToggleSpotlight { .. } => {
                self.push_count_toggle_spotlight += 1;
                lf_metrics
                    .actions_pushed_toggle_spotlight_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            RootAction::QuarantineDaemon { .. } => {
                self.push_count_quarantine_daemon += 1;
                lf_metrics
                    .actions_pushed_quarantine_daemon_total
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        self.bump_phase_counter(ctx.phase);
        self.actions.push(action);
    }

    /// Convenience: push from an iterator of pre-built actions (e.g. from
    /// `decide_actions::run()` or `learned_policy::filter()`). Each becomes
    /// a `push_raw` with shared `EmitContext`. Preserves order.
    pub fn extend_raw(
        &mut self,
        actions: impl IntoIterator<Item = RootAction>,
        ctx: EmitContext,
        lf_metrics: &LockFreeMetrics,
    ) {
        for a in actions {
            self.push_raw(a, ctx, lf_metrics);
        }
    }

    /// Telemetry snapshot. Caller publishes to lse_counters / dashboard.
    pub fn telemetry(&self) -> AccumulatorTelemetry {
        AccumulatorTelemetry {
            throttle: self.push_count_throttle,
            freeze: self.push_count_freeze,
            unfreeze: self.push_count_unfreeze,
            boost: self.push_count_boost,
            set_memorystatus: self.push_count_set_memorystatus,
            set_thread_qos: self.push_count_set_thread_qos,
            set_sysctl: self.push_count_set_sysctl,
            toggle_spotlight: self.push_count_toggle_spotlight,
            quarantine_daemon: self.push_count_quarantine_daemon,
            raw: self.push_count_raw,
            rejected_shape: self.rejected_shape,
            total_pushed: self.actions.len() as u64,
            phase_decide: self.phase_counts.decide,
            phase_learned_policy: self.phase_counts.learned_policy,
            phase_skill_tick: self.phase_counts.skill_tick,
            phase_cluster_actions: self.phase_counts.cluster_actions,
            phase_agent_actions: self.phase_counts.agent_actions,
            phase_paging_hints: self.phase_counts.paging_hints,
            phase_heuristic: self.phase_counts.heuristic,
            phase_stale_apps: self.phase_counts.stale_apps,
            phase_survival: self.phase_counts.survival,
            phase_freeze_executor: self.phase_counts.freeze_executor,
            phase_sysctl_governor: self.phase_counts.sysctl_governor,
            phase_network_optimizer: self.phase_counts.network_optimizer,
            phase_dispatch_tick: self.phase_counts.dispatch_tick,
            phase_reactor: self.phase_counts.reactor,
            phase_other: self.phase_counts.other,
        }
    }
}

/// Per-cycle telemetry of accumulator activity. All counters are absolute
/// (resets when `ActionAccumulator::new()` is called — i.e. once per
/// daemon cycle).
///
/// Invariant (post-ffa0b29): Σ(typed per-variant) == total_pushed.
/// `raw` is an INDEPENDENT diagnostic of escape-hatch volume — it is a SUBSET
/// of the typed totals (every `push_raw` also bumps the matching per-variant
/// counter). Dashboards must NOT add `raw` to the typed sum.
///
/// DO NOT compute Σ(typed) + raw — this double-counts every escape-hatch
/// emission and inflates dispatcher volume by the raw fraction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AccumulatorTelemetry {
    pub throttle: u64,
    pub freeze: u64,
    pub unfreeze: u64,
    pub boost: u64,
    pub set_memorystatus: u64,
    pub set_thread_qos: u64,
    pub set_sysctl: u64,
    pub toggle_spotlight: u64,
    pub quarantine_daemon: u64,
    pub raw: u64,
    pub rejected_shape: u64,
    pub total_pushed: u64,
    // Per-`ActionPhase` totals (Fase 5 reviewer fix #5). Each push
    // increments exactly one of these — sum equals `total_pushed`.
    pub phase_decide: u64,
    pub phase_learned_policy: u64,
    pub phase_skill_tick: u64,
    pub phase_cluster_actions: u64,
    pub phase_agent_actions: u64,
    pub phase_paging_hints: u64,
    pub phase_heuristic: u64,
    pub phase_stale_apps: u64,
    pub phase_survival: u64,
    pub phase_freeze_executor: u64,
    pub phase_sysctl_governor: u64,
    pub phase_network_optimizer: u64,
    pub phase_dispatch_tick: u64,
    pub phase_reactor: u64,
    pub phase_other: u64,
}

fn action_variant_name(a: &RootAction) -> &'static str {
    match a {
        RootAction::ThrottleProcess { .. } => "throttle",
        RootAction::FreezeProcess { .. } => "freeze",
        RootAction::UnfreezeProcess { .. } => "unfreeze",
        RootAction::BoostProcess { .. } => "boost",
        RootAction::SetMemorystatus { .. } => "set_memorystatus",
        RootAction::SetThreadQoS { .. } => "set_thread_qos",
        RootAction::SetSysctl(_) => "set_sysctl",
        RootAction::ToggleSpotlight { .. } => "toggle_spotlight",
        RootAction::QuarantineDaemon { .. } => "quarantine_daemon",
    }
}

/// Outcome-feedback label for a `RootAction`. Used by the daemon's
/// neurocognitive tick to attribute outcome signal back to the action that
/// caused it (Self-Reward, CausalGraph, co-occurrence graph). Returning
/// `Option<String>` keeps the signature future-proof for variants that may
/// not have a meaningful outcome label, but all 9 current variants produce
/// `Some`. The exhaustive match (no `_ =>` arm) plus
/// `#[deny(unreachable_patterns)]` enforces that adding a 10th `RootAction`
/// variant fails the build until this function is updated — closing the
/// silent-feedback gap that previously made Boost / SetSysctl /
/// SetThreadQoS / ToggleSpotlight / QuarantineDaemon / UnfreezeProcess
/// outcomes invisible to learning subsystems.
#[deny(unreachable_patterns)]
pub fn outcome_name(a: &RootAction) -> Option<String> {
    match a {
        RootAction::ThrottleProcess { name, .. } => Some(format!("throttle:{}", name)),
        RootAction::FreezeProcess { name, .. } => Some(format!("freeze:{}", name)),
        RootAction::UnfreezeProcess { name, .. } => Some(format!("unfreeze:{}", name)),
        RootAction::BoostProcess { name, .. } => Some(format!("boost:{}", name)),
        RootAction::SetMemorystatus { pid, .. } => Some(format!("memstatus:pid:{}", pid)),
        RootAction::SetThreadQoS {
            name, tier, ..
        } => Some(format!("thread_qos:{}:tier{}", name, tier)),
        RootAction::SetSysctl(action) => Some(format!("sysctl:{}", action.key())),
        RootAction::ToggleSpotlight { enabled, .. } => {
            Some(format!("spotlight:{}", enabled))
        }
        RootAction::QuarantineDaemon { daemon, .. } => {
            Some(format!("quarantine:{}", daemon))
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> EmitContext {
        EmitContext::new(ActionPhase::Other, "test", "unit")
    }

    fn variant(a: &RootAction) -> &'static str {
        action_variant_name(a)
    }

    #[test]
    fn preserves_emit_order_across_15_pushes() {
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::new();
        // 15 pushes interleaving variants in a fixed order.
        acc.push_throttle(
            101,
            "a",
            false,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_freeze(
            102,
            "b",
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_set_sysctl_clamped(
            "kern.ipc.somaxconn",
            "256",
            "r",
            DecisionReason::PressureContext,
            ctx(),
            &lf,
        );
        acc.push_throttle(
            103,
            "c",
            true,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_freeze(
            104,
            "d",
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_raw(
            RootAction::unfreeze(105, "e", "r", DecisionReason::PressureContext),
            ctx(),
            &lf,
        );
        acc.push_raw(
            RootAction::set_memorystatus(106, -1, "r", DecisionReason::PressureContext),
            ctx(),
            &lf,
        );
        acc.push_raw(
            RootAction::toggle_spotlight(false, "r", DecisionReason::PressureContext),
            ctx(),
            &lf,
        );
        acc.push_throttle(
            107,
            "f",
            false,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_freeze(
            108,
            "g",
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_set_sysctl_clamped(
            "net.inet.tcp.delayed_ack",
            "1",
            "r",
            DecisionReason::PressureContext,
            ctx(),
            &lf,
        );
        acc.push_raw(
            RootAction::BoostProcess {
                pid: 109,
                name: "h".into(),
                reason: "r".into(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: 0,
                start_usec: 0,
            },
            ctx(),
            &lf,
        );
        acc.push_throttle(
            110,
            "i",
            false,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_freeze(
            111,
            "j",
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_set_sysctl_clamped(
            "kern.maxvnodes",
            "200000",
            "r",
            DecisionReason::PressureContext,
            ctx(),
            &lf,
        );

        let t = acc.telemetry();
        assert_eq!(t.total_pushed, 15);
        assert_eq!(t.throttle, 4); // 4 push_throttle
        assert_eq!(t.freeze, 4); // 4 push_freeze
        assert_eq!(t.set_sysctl, 3); // 3 push_set_sysctl_clamped
        assert_eq!(t.unfreeze, 1); // unfreeze went via push_raw
        assert_eq!(t.set_memorystatus, 1); // via push_raw
        assert_eq!(t.toggle_spotlight, 1); // via push_raw
        assert_eq!(t.boost, 1); // via push_raw
                                // raw count = 4 push_raw calls (unfreeze, set_memorystatus,
                                // toggle_spotlight, boost).
        assert_eq!(t.raw, 4);
        // Variant counters now represent total emitted variant volume.
        let typed_sum = t.throttle
            + t.freeze
            + t.unfreeze
            + t.boost
            + t.set_memorystatus
            + t.set_thread_qos
            + t.set_sysctl
            + t.toggle_spotlight
            + t.quarantine_daemon;
        assert_eq!(typed_sum, t.total_pushed);

        let v = acc.finalize();
        let order: Vec<&'static str> = v.iter().map(variant).collect();
        assert_eq!(
            order,
            vec![
                "throttle",
                "freeze",
                "set_sysctl",
                "throttle",
                "freeze",
                "unfreeze",
                "set_memorystatus",
                "toggle_spotlight",
                "throttle",
                "freeze",
                "set_sysctl",
                "boost",
                "throttle",
                "freeze",
                "set_sysctl",
            ]
        );
    }

    #[test]
    fn rejects_throttle_with_empty_name() {
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::new();
        acc.push_throttle(
            42,
            "",
            false,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        let t = acc.telemetry();
        assert_eq!(t.total_pushed, 0);
        assert_eq!(t.rejected_shape, 1);
        assert_eq!(t.throttle, 0);
        assert_eq!(lf.actions_rejected_shape_total.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn rejects_freeze_with_pid_zero() {
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::new();
        acc.push_freeze(
            0,
            "name",
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        let t = acc.telemetry();
        assert_eq!(t.total_pushed, 0);
        assert_eq!(t.rejected_shape, 1);
        assert_eq!(t.freeze, 0);
    }

    #[test]
    fn rejects_set_sysctl_with_empty_key() {
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::new();
        acc.push_set_sysctl_clamped("", "1", "r", DecisionReason::PressureContext, ctx(), &lf);
        let t = acc.telemetry();
        assert_eq!(t.total_pushed, 0);
        assert_eq!(t.rejected_shape, 1);
        assert_eq!(t.set_sysctl, 0);
    }

    #[test]
    fn push_raw_increments_raw_and_variant_counters() {
        // Raw is the escape-hatch path count; per-variant counters still
        // represent actual emitted variant volume for runtime telemetry.
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::new();
        acc.push_raw(
            RootAction::throttle_full(7, "x", false, "r", 0, 0, DecisionReason::PressureContext),
            EmitContext::new(ActionPhase::FreezeExecutor, "test::raw", "confirmed"),
            &lf,
        );
        acc.push_raw(
            RootAction::SetSysctl(crate::engine::types::SetSysctlAction::new_clamped(
                "kern.maxvnodes",
                "200000",
                "r",
                DecisionReason::PressureContext,
            )),
            EmitContext::new(ActionPhase::SysctlGovernor, "test::raw_sysctl", "tick"),
            &lf,
        );
        let t = acc.telemetry();
        assert_eq!(t.raw, 2);
        assert_eq!(t.throttle, 1);
        assert_eq!(t.set_sysctl, 1);
        assert_eq!(t.total_pushed, 2);
        assert_eq!(lf.actions_pushed_raw_total.load(Ordering::Relaxed), 2);
        assert_eq!(lf.actions_pushed_throttle_total.load(Ordering::Relaxed), 1);
        assert_eq!(
            lf.actions_pushed_set_sysctl_total.load(Ordering::Relaxed),
            1
        );
    }

    /// Invariant guard (post-ffa0b29): a single `push_raw` of a Boost bumps
    /// the boost per-variant counter AND the raw diagnostic counter — but
    /// `total_pushed` increments by ONE, not two. Dashboards must not add raw
    /// to the typed sum.
    #[test]
    fn test_push_raw_increments_typed_and_raw() {
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::new();
        acc.push_raw(
            RootAction::BoostProcess {
                pid: 9001,
                name: "x".into(),
                reason: "r".into(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: 0,
                start_usec: 0,
            },
            EmitContext::new(ActionPhase::Other, "test::invariant", "raw_boost"),
            &lf,
        );
        let t = acc.telemetry();
        assert_eq!(t.boost, 1, "boost per-variant counter must bump on raw");
        assert_eq!(t.raw, 1, "raw diagnostic counter must bump");
        assert_eq!(
            t.total_pushed, 1,
            "total_pushed counts dispatcher emissions, not raw+typed"
        );
        // lf_metrics mirrors must match.
        assert_eq!(
            lf.actions_pushed_boost_total.load(Ordering::Relaxed),
            1,
            "lf boost mirror must bump on raw"
        );
        assert_eq!(
            lf.actions_pushed_raw_total.load(Ordering::Relaxed),
            1,
            "lf raw mirror must bump"
        );
    }

    /// Invariant guard (post-ffa0b29): over a mixed batch of typed and raw
    /// pushes, Σ(typed per-variant) == total_pushed. `raw` is an INDEPENDENT
    /// diagnostic — it is a SUBSET of the typed totals, NOT an addend.
    /// Computing Σ(typed) + raw would double-count every escape-hatch push.
    #[test]
    fn test_invariant_typed_sum_equals_total_pushed() {
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::new();
        // 3 typed pushes.
        acc.push_throttle(
            11,
            "t1",
            false,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_freeze(
            12,
            "f1",
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_set_sysctl_clamped(
            "kern.ipc.somaxconn",
            "256",
            "r",
            DecisionReason::PressureContext,
            ctx(),
            &lf,
        );
        // 3 raw pushes (cover 3 different variants).
        acc.push_raw(
            RootAction::unfreeze(13, "u1", "r", DecisionReason::PressureContext),
            ctx(),
            &lf,
        );
        acc.push_raw(
            RootAction::BoostProcess {
                pid: 14,
                name: "b1".into(),
                reason: "r".into(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: 0,
                start_usec: 0,
            },
            ctx(),
            &lf,
        );
        acc.push_raw(
            RootAction::toggle_spotlight(false, "r", DecisionReason::PressureContext),
            ctx(),
            &lf,
        );

        let t = acc.telemetry();
        let typed_sum = t.throttle
            + t.freeze
            + t.unfreeze
            + t.boost
            + t.set_memorystatus
            + t.set_thread_qos
            + t.set_sysctl
            + t.toggle_spotlight
            + t.quarantine_daemon;

        assert_eq!(t.total_pushed, 6, "6 total pushes (3 typed + 3 raw)");
        assert_eq!(
            typed_sum, t.total_pushed,
            "Σ(typed per-variant) MUST equal total_pushed (post-ffa0b29 invariant)"
        );
        assert_eq!(t.raw, 3, "raw diagnostic counts escape-hatch volume");
        // Negative guard: the wrong formula Σ(typed) + raw would inflate.
        assert_ne!(
            typed_sum + t.raw,
            t.total_pushed,
            "Σ(typed) + raw double-counts and MUST NOT match total_pushed"
        );
    }

    #[test]
    fn view_returns_partial_during_build() {
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::new();
        acc.push_throttle(
            1,
            "a",
            false,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_freeze(
            2,
            "b",
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_throttle(
            3,
            "c",
            true,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        assert_eq!(acc.view().len(), 3);
        assert_eq!(acc.len(), 3);
        assert!(!acc.is_empty());
        let v = acc.finalize();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn extend_raw_uses_emit_context_for_all() {
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::new();
        let actions = vec![
            RootAction::throttle_full(11, "x", false, "r", 0, 0, DecisionReason::PressureContext),
            RootAction::freeze_full(12, "y", "r", 0, 0, DecisionReason::PressureContext),
            RootAction::unfreeze(13, "z", "r", DecisionReason::PressureContext),
            RootAction::set_memorystatus(14, -1, "r", DecisionReason::PressureContext),
            RootAction::BoostProcess {
                pid: 15,
                name: "w".into(),
                reason: "r".into(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: 0,
                start_usec: 0,
            },
        ];
        acc.extend_raw(
            actions,
            EmitContext::new(ActionPhase::Decide, "test::extend", "decide"),
            &lf,
        );
        let t = acc.telemetry();
        assert_eq!(t.raw, 5);
        assert_eq!(t.total_pushed, 5);
        assert_eq!(t.throttle, 1);
        assert_eq!(t.freeze, 1);
        assert_eq!(t.unfreeze, 1);
        assert_eq!(t.set_memorystatus, 1);
        assert_eq!(t.boost, 1);
    }

    #[test]
    fn telemetry_pushed_total_matches_actions_len() {
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::new();
        acc.push_throttle(
            1,
            "a",
            false,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        acc.push_freeze(
            2,
            "b",
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        // Rejected push should not move total_pushed.
        acc.push_throttle(
            0,
            "",
            false,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        let t = acc.telemetry();
        assert_eq!(t.total_pushed, 2);
        assert_eq!(t.total_pushed, acc.len() as u64);
    }

    #[test]
    fn finalize_consumes_self_and_returns_owned_vec() {
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::with_capacity(8);
        acc.push_throttle(
            1,
            "a",
            false,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            ctx(),
            &lf,
        );
        let v: Vec<RootAction> = acc.finalize();
        assert_eq!(v.len(), 1);
        // Compile-time guarantee via #[must_use] is checked by clippy lint;
        // we exercise the consumption path here.
    }

    #[test]
    fn extend_raw_empty_iter_no_increments() {
        // Fase 5 reviewer fix #5: edge case — empty iterator must be a no-op.
        let mut acc = ActionAccumulator::new();
        let lf = LockFreeMetrics::new();
        acc.extend_raw(
            std::iter::empty(),
            EmitContext::new(ActionPhase::Decide, "test", "empty"),
            &lf,
        );
        assert_eq!(acc.len(), 0);
        let t = acc.telemetry();
        assert_eq!(t.raw, 0);
        assert_eq!(t.total_pushed, 0);
        assert_eq!(lf.actions_pushed_raw_total.load(Ordering::Relaxed), 0);
        // No phase counter should be touched either.
        assert_eq!(t.phase_decide, 0);
    }

    #[test]
    fn per_phase_counters_track_pushes() {
        // Fase 5 reviewer fix #5: each push must increment exactly one phase
        // counter; sum of phases equals total_pushed.
        let mut acc = ActionAccumulator::new();
        let lf = LockFreeMetrics::new();
        acc.push_throttle(
            123,
            "test",
            false,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            EmitContext::new(ActionPhase::Decide, "test", "x"),
            &lf,
        );
        acc.push_freeze(
            456,
            "test",
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            EmitContext::new(ActionPhase::Heuristic, "test", "x"),
            &lf,
        );
        let t = acc.telemetry();
        assert_eq!(t.phase_decide, 1);
        assert_eq!(t.phase_heuristic, 1);
        assert_eq!(t.phase_skill_tick, 0);
        // Sum of phase counters equals total_pushed.
        let phase_sum = t.phase_decide
            + t.phase_learned_policy
            + t.phase_skill_tick
            + t.phase_cluster_actions
            + t.phase_agent_actions
            + t.phase_paging_hints
            + t.phase_heuristic
            + t.phase_stale_apps
            + t.phase_survival
            + t.phase_freeze_executor
            + t.phase_sysctl_governor
            + t.phase_network_optimizer
            + t.phase_dispatch_tick
            + t.phase_reactor
            + t.phase_other;
        assert_eq!(phase_sum, t.total_pushed);
    }

    #[test]
    fn shape_rejection_does_not_bump_phase_counter() {
        // Phase counter should reflect accepted actions only.
        let mut acc = ActionAccumulator::new();
        let lf = LockFreeMetrics::new();
        acc.push_throttle(
            0, // bad pid -> rejected
            "name",
            false,
            "r",
            DecisionReason::PressureContext,
            0,
            0,
            EmitContext::new(ActionPhase::Decide, "test", "bad"),
            &lf,
        );
        let t = acc.telemetry();
        assert_eq!(t.rejected_shape, 1);
        assert_eq!(t.phase_decide, 0);
    }

    #[test]
    fn push_raw_increments_phase_counter() {
        // push_raw bumps both the phase counter (audit trail) and the
        // per-variant counter (runtime volume telemetry).
        let mut acc = ActionAccumulator::new();
        let lf = LockFreeMetrics::new();
        acc.push_raw(
            RootAction::unfreeze(99, "x", "r", DecisionReason::PressureContext),
            EmitContext::new(ActionPhase::FreezeExecutor, "test", "confirmed"),
            &lf,
        );
        let t = acc.telemetry();
        assert_eq!(t.raw, 1);
        assert_eq!(t.unfreeze, 1);
        assert_eq!(t.phase_freeze_executor, 1); // raw bumps phase
        assert_eq!(t.total_pushed, 1);
    }

    #[test]
    fn test_outcome_name_exhaustive() {
        // Construct one instance of each of the 9 RootAction variants and
        // assert outcome_name() returns Some for every one. Combined with
        // the exhaustive (no `_ =>` arm) match in outcome_name + the
        // `#[deny(unreachable_patterns)]` attribute, this guarantees a
        // build failure if a 10th variant is added without updating the
        // outcome-feedback path — closing the silent-feedback gap that
        // hid Boost / SetSysctl / SetThreadQoS / ToggleSpotlight /
        // QuarantineDaemon / UnfreezeProcess outcomes from learning.
        let variants: [RootAction; 9] = [
            RootAction::throttle(1, "Safari", false, "r", DecisionReason::PressureContext),
            RootAction::freeze(2, "Chrome", "r", DecisionReason::PressureContext),
            RootAction::unfreeze(3, "Slack", "r", DecisionReason::PressureContext),
            RootAction::BoostProcess {
                pid: 4,
                name: "Brave".into(),
                reason: "r".into(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: 0,
                start_usec: 0,
            },
            RootAction::set_memorystatus(5, 10, "r", DecisionReason::PressureContext),
            RootAction::SetThreadQoS {
                pid: 6,
                name: "Xcode".into(),
                thread_index: 0,
                tier: "background".into(),
                reason: "r".into(),
                decision_reason: DecisionReason::PressureContext,
                affinity_tag: Some(2),
                start_sec: 0,
                start_usec: 0,
            },
            RootAction::set_sysctl(
                "kern.maxvnodes",
                "200000",
                "r",
                DecisionReason::PressureContext,
            ),
            RootAction::toggle_spotlight(false, "r", DecisionReason::PressureContext),
            RootAction::QuarantineDaemon {
                daemon: "telemetryd".into(),
                active: true,
                reason: "r".into(),
                decision_reason: DecisionReason::PressureContext,
            },
        ];

        for v in &variants {
            let label = outcome_name(v);
            assert!(
                label.is_some(),
                "outcome_name returned None for variant {:?} — every RootAction must carry an outcome label so learning subsystems (Self-Reward, CausalGraph, co-occurrence graph) receive feedback",
                v
            );
        }

        // Sanity-check the prefix formatting per the brief.
        assert_eq!(outcome_name(&variants[0]).unwrap(), "throttle:Safari");
        assert_eq!(outcome_name(&variants[1]).unwrap(), "freeze:Chrome");
        assert_eq!(outcome_name(&variants[2]).unwrap(), "unfreeze:Slack");
        assert_eq!(outcome_name(&variants[3]).unwrap(), "boost:Brave");
        assert_eq!(outcome_name(&variants[4]).unwrap(), "memstatus:pid:5");
        assert_eq!(
            outcome_name(&variants[5]).unwrap(),
            "thread_qos:Xcode:tierbackground"
        );
        assert_eq!(
            outcome_name(&variants[6]).unwrap(),
            "sysctl:kern.maxvnodes"
        );
        assert_eq!(outcome_name(&variants[7]).unwrap(), "spotlight:false");
        assert_eq!(
            outcome_name(&variants[8]).unwrap(),
            "quarantine:telemetryd"
        );
    }
}
