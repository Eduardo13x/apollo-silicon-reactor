//! Invariant #11 — `RootAction::BoostProcess` and `RootAction::SetThreadQoS`
//! must early-return with `BlockReason::PidRecycled` when the live PID's
//! kernel `start_sec` does NOT match the action's recorded `start_sec`.
//!
//! Closes the A-B-A exploit window: prior to Sprint 2026-06-06 these arms
//! verified with hard-coded `0, 0` (legacy fallback) which is a no-op
//! tautology — verify always accepted, the counter was perma-zero across
//! 59 675 cycles in production telemetry.

use std::collections::HashSet;
use std::path::Path;

use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::execute_actions::execute_actions;
use apollo_engine::engine::lse_counters::LSE_COUNTERS;
use apollo_engine::engine::types::{CapabilityReport, RootAction};

fn no_caps() -> CapabilityReport {
    CapabilityReport {
        can_taskpolicy: false,
        can_sysctl: false,
        can_memorystatus: false,
        can_mdutil: false,
        can_tmutil: false,
        is_root: false,
        p_core_count: Some(8),
        e_core_count: Some(4),
        unavailable: vec![],
    }
}

fn null_journal() -> &'static Path {
    Path::new("/dev/null")
}

/// 100 years from now — guaranteed to mismatch the live `launchd` start_sec.
fn impossible_future_start_sec() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + 100 * 365 * 24 * 3600
}

#[test]
fn boost_blocks_on_mismatched_start_sec_and_bumps_counter() {
    let before = LSE_COUNTERS
        .pid_recycle_blocks_total
        .load(std::sync::atomic::Ordering::Relaxed);

    // PID 1 = launchd; very real and reachable. Mismatched start_sec → block.
    let action = RootAction::BoostProcess {
        pid: 1,
        name: "launchd".to_string(),
        reason: "test:inv11".to_string(),
        decision_reason: DecisionReason::PressureContext,
        start_sec: impossible_future_start_sec(),
        start_usec: 0,
    };

    let mut frozen = HashSet::new();
    let outcomes = execute_actions(
        vec![action],
        &no_caps(),
        null_journal(),
        &mut frozen,
        &[],
        &[],
        None,
        true, // dry_run
        0.0,
        0.0,
        None,
        0.0,
    );
    let after = LSE_COUNTERS
        .pid_recycle_blocks_total
        .load(std::sync::atomic::Ordering::Relaxed);

    assert!(after >= before + 1, "pid_recycle_blocks_total must bump");
    assert_eq!(
        outcomes.boosts_applied, 0,
        "boost must NOT apply on identity mismatch"
    );
    assert!(
        outcomes.audit_traces.iter().any(|t| matches!(
            t.block_reason,
            Some(apollo_engine::engine::audit_types::BlockReason::PidRecycled)
        )),
        "audit trace must record BlockReason::PidRecycled"
    );
}

#[test]
fn set_thread_qos_blocks_on_mismatched_start_sec_and_bumps_counter() {
    let before = LSE_COUNTERS
        .pid_recycle_blocks_total
        .load(std::sync::atomic::Ordering::Relaxed);

    // Use a non-protected name so the protected-filter doesn't short-circuit
    // before the identity check. PID 1 (launchd) still satisfies the
    // ProcessIdentity::verify start_sec mismatch (impossible future timestamp).
    // Name field is opaque to verify when start_sec differs.
    let action = RootAction::SetThreadQoS {
        pid: 1,
        name: "synthetic-test-target".to_string(),
        thread_index: 0,
        tier: "background".to_string(),
        reason: "test:inv11".to_string(),
        decision_reason: DecisionReason::PressureContext,
        affinity_tag: None,
        start_sec: impossible_future_start_sec(),
        start_usec: 0,
    };

    let mut frozen = HashSet::new();
    let outcomes = execute_actions(
        vec![action],
        &no_caps(),
        null_journal(),
        &mut frozen,
        &[],
        &[],
        None,
        true,
        0.0,
        0.0,
        None,
        0.0,
    );
    let after = LSE_COUNTERS
        .pid_recycle_blocks_total
        .load(std::sync::atomic::Ordering::Relaxed);

    assert!(after >= before + 1, "pid_recycle_blocks_total must bump");
    assert_eq!(outcomes.thread_qos_applied, 0, "thread QoS must NOT apply");
    assert!(
        outcomes.audit_traces.iter().any(|t| matches!(
            t.block_reason,
            Some(apollo_engine::engine::audit_types::BlockReason::PidRecycled)
        )),
        "audit trace must record BlockReason::PidRecycled (no longer silent skip)"
    );
}

#[test]
fn serde_round_trip_defaults_start_sec_to_zero() {
    // Old persisted JournalEntry/learned_state shapes lack start_sec — must
    // deserialize with the #[serde(default)] zero fallback and remain
    // accept-by-default at verify (legacy semantics preserved).
    let json = r#"{"BoostProcess":{"pid":1,"name":"launchd","reason":"r","decision_reason":"PressureContext"}}"#;
    let parsed: RootAction = serde_json::from_str(json).expect("legacy boost must parse");
    if let RootAction::BoostProcess {
        start_sec,
        start_usec,
        ..
    } = parsed
    {
        assert_eq!(start_sec, 0);
        assert_eq!(start_usec, 0);
    } else {
        panic!("expected BoostProcess");
    }

    let json = r#"{"SetThreadQoS":{"pid":1,"name":"launchd","thread_index":0,"tier":"background","reason":"r","decision_reason":"PressureContext"}}"#;
    let parsed: RootAction = serde_json::from_str(json).expect("legacy thread_qos must parse");
    if let RootAction::SetThreadQoS {
        start_sec,
        start_usec,
        ..
    } = parsed
    {
        assert_eq!(start_sec, 0);
        assert_eq!(start_usec, 0);
    } else {
        panic!("expected SetThreadQoS");
    }
}
