//! Sprint 4 Fase 5 golden test for `ActionAccumulator`.
//!
//! Goal: prove that pushing actions through the accumulator yields **byte
//! equivalent** output to manually appending them to a raw `Vec<RootAction>`.
//! The migration is supposed to be semantics-preserving — this test pins
//! the equivalence so any future change that re-orders, drops, or reshapes
//! actions is caught at CI time.
//!
//! Building a fully deterministic daemon fixture is impractical: the real
//! main loop depends on `proc_pidpath`, kqueue events, kernel pressure, and
//! many other system-dependent inputs. Instead, we construct a deterministic
//! sequence of action emissions that exercises every emit pattern used by
//! the Fase 5 migration:
//!
//! 1. `extend_raw` — for `decide_actions`, `skill_tick`, `cluster_actions`,
//!    `agent_actions`, `paging_hints`, `heuristic_pass.additional_actions`,
//!    `stale_apps`, `sysctl_governor.tick`.
//! 2. `push_freeze` typed — for the `proc_recovery` site (the only typed
//!    PID-bearing emit in the migration; the rest go through `extend_raw`
//!    because they originate from helper modules).
//! 3. `push_set_sysctl_clamped` typed — for the `network_optimizer` site
//!    (Bug 6 location — Phase 4 sealed, Phase 5 typed-pushed).
//!
//! For each emit pattern, we record the action vec produced by:
//! - the **legacy path**: `let mut v: Vec<RootAction> = vec![]; v.push(a)` /
//!   `v.extend(iter)`,
//! - the **accumulator path**: `acc.push_*` / `acc.extend_raw`.
//!
//! Then we assert variant-sequence equivalence and per-variant counters.

use apollo_engine::engine::action_accumulator::{ActionAccumulator, ActionPhase, EmitContext};
use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::network_optimizer::{NetworkOptimizer, NetworkProfile};
use apollo_engine::engine::sysctl_limits::clamp_to_allowed_range;
use apollo_engine::engine::types::{RootAction, SetSysctlAction};

fn ctx(phase: ActionPhase, site: &'static str, reason: &'static str) -> EmitContext {
    EmitContext::new(phase, site, reason)
}

fn variant_name(a: &RootAction) -> &'static str {
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

/// Build a deterministic mixed-variant sequence resembling a real cycle.
fn fixture_sequence() -> Vec<RootAction> {
    vec![
        // (1) decide_actions chokepoint produces a mixed batch.
        RootAction::throttle_full(
            1001,
            "app-a",
            false,
            "decide:pressure",
            10,
            20,
            DecisionReason::PressureContext,
        ),
        RootAction::freeze_full(
            1002,
            "app-b",
            "decide:idle",
            11,
            22,
            DecisionReason::GraduatedIdle,
        ),
        RootAction::SetMemorystatus {
            pid: 1003,
            priority: -1,
            reason: "decide:hint".into(),
            decision_reason: DecisionReason::MemoryBudget,
        },
        // (2) skill_tick adds 2 boosts.
        RootAction::BoostProcess {
            pid: 2001,
            name: "interactive".into(),
            reason: "skill:focus".into(),
            decision_reason: DecisionReason::InteractiveFocus,
            start_sec: 0,
            start_usec: 0,
        },
        RootAction::BoostProcess {
            pid: 2002,
            name: "interactive2".into(),
            reason: "skill:focus".into(),
            decision_reason: DecisionReason::InteractiveFocus,
            start_sec: 0,
            start_usec: 0,
        },
        // (3) cluster_actions adds 1 freeze + 1 spotlight toggle.
        RootAction::freeze_full(
            3001,
            "cluster-a",
            "cluster:coordinated",
            13,
            24,
            DecisionReason::PressureContext,
        ),
        RootAction::ToggleSpotlight {
            enabled: false,
            reason: "spotlight gate".into(),
            decision_reason: DecisionReason::PressureContext,
        },
        // (4) agent_actions: predictive throttle + memorystatus hint.
        RootAction::throttle_full(
            4001,
            "predictive-a",
            true,
            "agent:pre-throttle",
            14,
            25,
            DecisionReason::CausalInference,
        ),
        RootAction::SetMemorystatus {
            pid: 4002,
            priority: -1,
            reason: "agent:proactive".into(),
            decision_reason: DecisionReason::MemoryBudget,
        },
        // (5) paging_hints: 2 memorystatus hints.
        RootAction::SetMemorystatus {
            pid: 5001,
            priority: -1,
            reason: "paging:hint".into(),
            decision_reason: DecisionReason::MemoryBudget,
        },
        RootAction::SetMemorystatus {
            pid: 5002,
            priority: -1,
            reason: "paging:hint".into(),
            decision_reason: DecisionReason::MemoryBudget,
        },
        // (6) heuristic_pass: 1 throttle.
        RootAction::throttle_full(
            6001,
            "heuristic-a",
            false,
            "heuristic:adaptive",
            15,
            26,
            DecisionReason::SwarmThrottling,
        ),
        // (7) stale_apps: 1 freeze.
        RootAction::freeze_full(
            7001,
            "stale-a",
            "stale:6h_idle",
            16,
            27,
            DecisionReason::GraduatedIdle,
        ),
        // (8) proc_recovery: 1 typed freeze.
        RootAction::freeze_full(
            8001,
            "leak-a",
            "memory-leak recovery: prob=0.95 rss=512MB attempts=3",
            17,
            28,
            DecisionReason::PressureContext,
        ),
        // (9) sysctl_governor: 2 sealed sysctls.
        RootAction::SetSysctl(SetSysctlAction::new_clamped(
            "kern.ipc.somaxconn",
            "256",
            "sysctl_governor:tcp_health",
            DecisionReason::PressureContext,
        )),
        RootAction::SetSysctl(SetSysctlAction::new_clamped(
            "net.inet.tcp.delayed_ack",
            "1",
            "sysctl_governor:tcp_health",
            DecisionReason::PressureContext,
        )),
        // (10) network_optimizer: 1 typed sysctl push (Bug 6 site).
        RootAction::SetSysctl(SetSysctlAction::new_clamped(
            "net.inet.tcp.sendspace",
            "131072",
            "network-optimizer: Balanced profile",
            DecisionReason::PressureContext,
        )),
    ]
}

#[test]
fn legacy_vec_and_accumulator_produce_byte_equivalent_sequences() {
    let lf = LockFreeMetrics::new();

    // Legacy path: build the Vec by hand.
    let mut legacy: Vec<RootAction> = Vec::new();
    let fixture = fixture_sequence();
    legacy.extend(fixture[0..3].iter().cloned()); // decide_actions
    legacy.extend(fixture[3..5].iter().cloned()); // skill_tick
    legacy.extend(fixture[5..7].iter().cloned()); // cluster_actions
    legacy.extend(fixture[7..9].iter().cloned()); // agent_actions
    legacy.extend(fixture[9..11].iter().cloned()); // paging_hints
    legacy.extend(fixture[11..12].iter().cloned()); // heuristic_pass
    legacy.extend(fixture[12..13].iter().cloned()); // stale_apps
    legacy.push(fixture[13].clone()); // proc_recovery typed
    legacy.extend(fixture[14..16].iter().cloned()); // sysctl_governor
                                                    // network_optimizer — typed, but legacy path was actions.push(set_sysctl(...))
    legacy.push(fixture[16].clone());

    // Accumulator path: drive the same emissions through the typed builder.
    let mut acc = ActionAccumulator::with_capacity(16);
    acc.extend_raw(
        fixture[0..3].iter().cloned(),
        ctx(ActionPhase::Decide, "test::decide", "decide_actions"),
        &lf,
    );
    acc.extend_raw(
        fixture[3..5].iter().cloned(),
        ctx(ActionPhase::SkillTick, "test::skill", "skill_tick"),
        &lf,
    );
    acc.extend_raw(
        fixture[5..7].iter().cloned(),
        ctx(
            ActionPhase::ClusterActions,
            "test::cluster",
            "cluster_actions",
        ),
        &lf,
    );
    acc.extend_raw(
        fixture[7..9].iter().cloned(),
        ctx(ActionPhase::AgentActions, "test::agent", "agent_actions"),
        &lf,
    );
    acc.extend_raw(
        fixture[9..11].iter().cloned(),
        ctx(ActionPhase::PagingHints, "test::paging", "paging_hints"),
        &lf,
    );
    acc.extend_raw(
        fixture[11..12].iter().cloned(),
        ctx(ActionPhase::Heuristic, "test::heuristic", "heuristic_pass"),
        &lf,
    );
    acc.extend_raw(
        fixture[12..13].iter().cloned(),
        ctx(ActionPhase::StaleApps, "test::stale", "stale_apps"),
        &lf,
    );
    // proc_recovery typed
    if let RootAction::FreezeProcess {
        pid,
        name,
        reason,
        decision_reason,
        start_sec,
        start_usec,
    } = fixture[13].clone()
    {
        acc.push_freeze(
            pid,
            name,
            reason,
            decision_reason,
            start_sec,
            start_usec,
            ctx(ActionPhase::Survival, "test::recovery", "proc_recovery"),
            &lf,
        );
    } else {
        panic!("fixture[13] expected to be FreezeProcess");
    }
    acc.extend_raw(
        fixture[14..16].iter().cloned(),
        ctx(
            ActionPhase::SysctlGovernor,
            "test::sysctl",
            "sysctl_governor",
        ),
        &lf,
    );
    // network_optimizer typed
    if let RootAction::SetSysctl(s) = &fixture[16] {
        acc.push_set_sysctl_clamped(
            s.key().to_string(),
            s.value().to_string(),
            s.reason().to_string(),
            s.decision_reason().clone(),
            ctx(
                ActionPhase::NetworkOptimizer,
                "test::netopt",
                "profile_tcp_tune",
            ),
            &lf,
        );
    } else {
        panic!("fixture[16] expected to be SetSysctl");
    }

    let acc_telemetry = acc.telemetry();
    let acc_actions = acc.finalize();

    // 1. Length parity.
    assert_eq!(
        legacy.len(),
        acc_actions.len(),
        "legacy and accumulator action counts must match"
    );

    // 2. Variant sequence parity.
    let legacy_variants: Vec<&'static str> = legacy.iter().map(variant_name).collect();
    let acc_variants: Vec<&'static str> = acc_actions.iter().map(variant_name).collect();
    assert_eq!(
        legacy_variants, acc_variants,
        "variant sequence must be byte-equivalent"
    );

    // 3. Per-variant pid extraction parity (PID-bearing variants only).
    let legacy_pids: Vec<Option<u32>> = legacy
        .iter()
        .map(|a| a.identity_fields().map(|(pid, _, _, _)| pid))
        .collect();
    let acc_pids: Vec<Option<u32>> = acc_actions
        .iter()
        .map(|a| a.identity_fields().map(|(pid, _, _, _)| pid))
        .collect();
    assert_eq!(legacy_pids, acc_pids, "pid extraction must match");

    // 4. Per-variant accumulator telemetry — post-ffa0b29 semantics.
    //    The per-variant counters represent ACTUAL emitted variant volume:
    //    a `push_raw` / `extend_raw` bumps BOTH the per-variant counter AND
    //    the `raw` diagnostic counter (escape-hatch path count), while a
    //    typed `push_*` bumps ONLY the per-variant counter. In all cases
    //    `total_pushed` advances by exactly one per action.
    //    Source of truth: action_accumulator.rs unit tests
    //    `push_raw_increments_raw_and_variant_counters` and
    //    `test_push_raw_increments_typed_and_raw` ("Dashboards must not add
    //    raw to the typed sum"). The correct invariant is therefore
    //    Σ(typed per-variant) == total_pushed; `raw` is a SEPARATE diagnostic.
    //
    //    Fixture variant volumes (raw + typed combined):
    //      throttle 3 (idx0/7/11), freeze 4 (idx1/5/12 + idx13 typed),
    //      boost 2 (idx3/4), set_memorystatus 4 (idx2/8/9/10),
    //      toggle_spotlight 1 (idx6), set_sysctl 3 (idx14/15 + idx16 typed).
    assert_eq!(acc_telemetry.throttle, 3);
    assert_eq!(acc_telemetry.freeze, 4);
    assert_eq!(acc_telemetry.unfreeze, 0);
    assert_eq!(acc_telemetry.boost, 2);
    assert_eq!(acc_telemetry.set_memorystatus, 4);
    assert_eq!(acc_telemetry.set_thread_qos, 0);
    assert_eq!(acc_telemetry.set_sysctl, 3);
    assert_eq!(acc_telemetry.toggle_spotlight, 1);
    assert_eq!(acc_telemetry.quarantine_daemon, 0);
    assert_eq!(acc_telemetry.total_pushed, legacy.len() as u64);
    // No shape rejections expected from a well-formed fixture.
    assert_eq!(acc_telemetry.rejected_shape, 0);
    // raw count: every action EXCEPT the two typed pushes (proc_recovery
    // freeze + netopt sysctl) went through extend_raw → raw == total - 2.
    assert_eq!(acc_telemetry.raw, (legacy.len() as u64) - 2);
    // Telemetry invariant (post-ffa0b29): each action lands in exactly one
    // per-variant bucket, so Σ(typed per-variant) == total_pushed. `raw` is
    // a diagnostic overlay and must NOT be added to the typed sum.
    let typed_sum = acc_telemetry.throttle
        + acc_telemetry.freeze
        + acc_telemetry.unfreeze
        + acc_telemetry.boost
        + acc_telemetry.set_memorystatus
        + acc_telemetry.set_thread_qos
        + acc_telemetry.set_sysctl
        + acc_telemetry.toggle_spotlight
        + acc_telemetry.quarantine_daemon;
    assert_eq!(typed_sum, acc_telemetry.total_pushed);
}

/// Fase 5 reviewer fix #4 — strengthened golden: prove that the accumulator's
/// `push_set_sysctl_clamped` produces byte-identical output to the legacy
/// emit pattern at `main.rs:3461-3473` (network_optimizer site, formerly the
/// site of Bug 6). The legacy pattern was:
///
/// ```ignore
/// for (key, value) in net_optimizer.get_sysctl_recommendations(profile) {
///     let clamped = match value.parse::<i64>() {
///         Ok(n) => clamp_to_allowed_range(&key, n).to_string(),
///         Err(_) => value,
///     };
///     actions.push(RootAction::set_sysctl(key, clamped, ..., decision_reason));
/// }
/// ```
///
/// The new path goes through `acc.push_set_sysctl_clamped` which calls
/// `RootAction::set_sysctl` -> `SetSysctlAction::new_clamped` (Fase 4 sealed).
/// This test feeds the same `NetworkOptimizer::get_sysctl_recommendations`
/// output through both paths and asserts every (key, value) pair lines up.
#[test]
// `get_sysctl_recommendations` is deprecated (2026-06-09 single-writer fix:
// the prod write path was deleted) but this golden test still exercises it
// as a fixture generator for accumulator equivalence — not as a write path.
#[allow(deprecated)]
fn golden_network_optimizer_emit_path_equivalent() {
    for profile in [
        NetworkProfile::HighThroughput,
        NetworkProfile::LowLatency,
        NetworkProfile::Balanced,
        NetworkProfile::Battery,
    ] {
        // Legacy path: hand-build the actions the way main.rs:3461-3473 used to.
        let net_optimizer = NetworkOptimizer::new();
        let mut legacy_actions: Vec<RootAction> = Vec::new();
        for (key, value) in net_optimizer.get_sysctl_recommendations(profile) {
            let clamped = match value.parse::<i64>() {
                Ok(n) => clamp_to_allowed_range(&key, n).to_string(),
                Err(_) => value,
            };
            legacy_actions.push(RootAction::set_sysctl(
                key,
                clamped,
                format!("network-optimizer: {:?} profile", profile),
                DecisionReason::PressureContext,
            ));
        }

        // New path: same input, accumulator.
        let lf = LockFreeMetrics::new();
        let mut acc = ActionAccumulator::new();
        let net_optimizer2 = NetworkOptimizer::new();
        for (key, value) in net_optimizer2.get_sysctl_recommendations(profile) {
            acc.push_set_sysctl_clamped(
                key,
                value,
                format!("network-optimizer: {:?} profile", profile),
                DecisionReason::PressureContext,
                EmitContext::new(
                    ActionPhase::NetworkOptimizer,
                    "network_optimizer",
                    "highthroughput",
                ),
                &lf,
            );
        }
        let new_actions = acc.finalize();

        assert_eq!(
            legacy_actions.len(),
            new_actions.len(),
            "len mismatch for profile {:?}",
            profile
        );
        for (a, b) in legacy_actions.iter().zip(new_actions.iter()) {
            // Both should be RootAction::SetSysctl(SetSysctlAction).
            let (lk, lv) = match a {
                RootAction::SetSysctl(s) => (s.key(), s.value()),
                _ => panic!("expected SetSysctl in legacy for profile {:?}", profile),
            };
            let (nk, nv) = match b {
                RootAction::SetSysctl(s) => (s.key(), s.value()),
                _ => panic!("expected SetSysctl in new for profile {:?}", profile),
            };
            assert_eq!(lk, nk, "key mismatch for profile {:?}", profile);
            assert_eq!(
                lv, nv,
                "clamped value mismatch — Bug 6 redux (profile {:?}, key {})",
                profile, lk
            );
        }
    }
}

#[test]
fn shape_rejection_does_not_corrupt_legacy_path_equivalence() {
    // If the typed `push_freeze` rejects (pid=0 / empty name), the legacy
    // path would have pushed garbage and crashed downstream. The
    // accumulator's job is to drop+log+count instead. The Vec produced
    // contains only the remaining valid actions — and downstream pipeline
    // is strictly safer than legacy. Verify the counters reflect this.
    let lf = LockFreeMetrics::new();
    let mut acc = ActionAccumulator::new();
    acc.push_freeze(
        0, // invalid pid
        "name",
        "test",
        DecisionReason::PressureContext,
        0,
        0,
        ctx(ActionPhase::Survival, "test::malformed", "shape_test"),
        &lf,
    );
    acc.push_freeze(
        42, // valid
        "good",
        "test",
        DecisionReason::PressureContext,
        0,
        0,
        ctx(ActionPhase::Survival, "test::valid", "shape_test"),
        &lf,
    );
    let t = acc.telemetry();
    assert_eq!(t.rejected_shape, 1);
    assert_eq!(t.freeze, 1);
    assert_eq!(t.total_pushed, 1);
    let v = acc.finalize();
    assert_eq!(v.len(), 1);
    if let RootAction::FreezeProcess { pid, .. } = &v[0] {
        assert_eq!(*pid, 42);
    } else {
        panic!("expected freeze");
    }
}
