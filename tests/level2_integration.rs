//! Level 2: Integration tests — multi-module interactions with lightweight
//! system calls (no root required).
//!
//! Tests validate: PID validation in execute_actions (BUG 4), sysctl allowlist
//! enforcement, ExecuteOutcomes accumulation, the full enforce_limits pipeline,
//! and safety invariants across protected vs critical-background process sets.

use std::collections::HashSet;
use std::path::Path;

use apollo_optimizer::engine::execute_actions::execute_actions;
use apollo_optimizer::engine::safety::{
    critical_background_processes, enforce_limits_with_budget, protected_processes,
};
use apollo_optimizer::engine::types::{
    ActionBudgetState, CapabilityReport, OptimizationProfile, RootAction, SafetyPolicy,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn no_caps() -> CapabilityReport {
    CapabilityReport {
        can_taskpolicy: false,
        can_sysctl: false,
        can_memorystatus: false,
        can_mdutil: false,
        can_tmutil: false,
        is_root: false,
        unavailable: vec![],
    }
}

/// Returns a PID that is very unlikely to be alive (high value in reserved range).
fn dead_pid() -> u32 {
    // PIDs above 99998 are not used on macOS in practice; MAX/2 is always safe.
    u32::MAX / 2
}

fn null_journal() -> &'static Path {
    Path::new("/dev/null")
}

// ── BUG 4 regression: PID validation before SIGSTOP/SIGCONT ─────────────────

/// Boost action on a dead PID must be silently skipped — no panic, no failure.
#[test]
fn execute_boost_dead_pid_is_skipped() {
    let actions = vec![RootAction::BoostProcess {
        pid: dead_pid(),
        name: "ghost-app".into(),
        reason: "test".into(),
    }];
    let mut frozen = HashSet::new();
    let outcomes = execute_actions(
        actions,
        &no_caps(),
        null_journal(),
        &mut frozen,
        &[],
        &[],
        None,
        false,
        0.0,
    );
    assert_eq!(
        outcomes.failures, 0,
        "dead-PID boost must not count as failure"
    );
    assert_eq!(
        outcomes.boosts_applied, 0,
        "boost must not be counted on dead PID"
    );
}

/// Freeze action on a dead PID must be silently skipped.
#[test]
fn execute_freeze_dead_pid_is_skipped() {
    let actions = vec![RootAction::FreezeProcess {
        pid: dead_pid(),
        name: "ghost-app".into(),
        reason: "test".into(),
        start_sec: 0,
        start_usec: 0,
    }];
    let mut frozen = HashSet::new();
    let outcomes = execute_actions(
        actions,
        &no_caps(),
        null_journal(),
        &mut frozen,
        &[],
        &[],
        None,
        false,
        0.0,
    );
    assert_eq!(
        outcomes.freezes_applied, 0,
        "dead PID must not be counted as frozen"
    );
    assert_eq!(outcomes.failures, 0);
    assert!(
        !frozen.contains(&dead_pid()),
        "dead PID must not appear in frozen set"
    );
}

/// Throttle action on a dead PID must be silently skipped.
#[test]
fn execute_throttle_dead_pid_is_skipped() {
    let actions = vec![RootAction::ThrottleProcess {
        pid: dead_pid(),
        name: "ghost-app".into(),
        aggressive: true,
        reason: "test".into(),
        start_sec: 0,
        start_usec: 0,
    }];
    let mut frozen = HashSet::new();
    let outcomes = execute_actions(
        actions,
        &no_caps(),
        null_journal(),
        &mut frozen,
        &[],
        &[],
        None,
        false,
        0.0,
    );
    assert_eq!(outcomes.throttles_applied, 0);
    assert_eq!(outcomes.failures, 0);
}

/// Unfreeze action on a dead PID must be silently handled — no crash.
#[test]
fn execute_unfreeze_dead_pid_is_safe() {
    let actions = vec![RootAction::UnfreezeProcess {
        pid: dead_pid(),
        name: "ghost-app".into(),
    }];
    let mut frozen = HashSet::new();
    frozen.insert(dead_pid());
    let outcomes = execute_actions(
        actions,
        &no_caps(),
        null_journal(),
        &mut frozen,
        &[],
        &[],
        None,
        false,
        0.0,
    );
    // Unfreeze always increments even on dead PID (SIGCONT to dead PID is a no-op).
    assert_eq!(
        outcomes.failures, 0,
        "unfreeze of dead PID must not be a failure"
    );
    assert!(
        !frozen.contains(&dead_pid()),
        "PID removed from frozen set after unfreeze"
    );
}

// ── Sysctl allowlist enforcement ─────────────────────────────────────────────

/// Non-allowlisted sysctl must not be applied even when cap is granted.
#[test]
fn execute_non_allowlisted_sysctl_is_denied() {
    let actions = vec![RootAction::SetSysctl {
        key: "kern.securelevel".into(), // NOT in the allowlist
        value: "0".into(),
        reason: "test".into(),
    }];
    let mut frozen = HashSet::new();
    let mut caps = no_caps();
    caps.can_sysctl = true; // cap granted, but key is not in allowlist

    let outcomes = execute_actions(
        actions,
        &caps,
        null_journal(),
        &mut frozen,
        &[],
        &[],
        None,
        false,
        0.0,
    );
    assert_eq!(
        outcomes.sysctl_applied, 0,
        "non-allowlisted sysctl must not be applied"
    );
    assert_eq!(
        outcomes.failures, 0,
        "denied allowlist check must not count as failure"
    );
}

/// Allowlisted sysctl without can_sysctl capability must be skipped.
#[test]
fn execute_sysctl_without_cap_is_skipped() {
    let actions = vec![RootAction::SetSysctl {
        key: "vm.compressor_poll_interval".into(), // in allowlist
        value: "20".into(),
        reason: "test".into(),
    }];
    let mut frozen = HashSet::new();
    let caps = no_caps(); // can_sysctl = false

    let outcomes = execute_actions(
        actions,
        &caps,
        null_journal(),
        &mut frozen,
        &[],
        &[],
        None,
        false,
        0.0,
    );
    assert_eq!(
        outcomes.sysctl_applied, 0,
        "sysctl without capability must be skipped"
    );
    assert_eq!(outcomes.failures, 0);
}

// ── ExecuteOutcomes accumulation ─────────────────────────────────────────────

/// Many dead-PID actions produce zero applied-counts and zero failures.
#[test]
fn execute_outcomes_all_zero_for_dead_pids() {
    let dead = dead_pid();
    let actions: Vec<RootAction> = vec![
        RootAction::BoostProcess {
            pid: dead,
            name: "dead1".into(),
            reason: "test".into(),
        },
        RootAction::BoostProcess {
            pid: dead + 1,
            name: "dead2".into(),
            reason: "test".into(),
        },
        RootAction::ThrottleProcess {
            pid: dead + 2,
            name: "dead3".into(),
            aggressive: false,
            reason: "test".into(),
            start_sec: 0,
            start_usec: 0,
        },
        RootAction::FreezeProcess {
            pid: dead + 3,
            name: "dead4".into(),
            reason: "test".into(),
            start_sec: 0,
            start_usec: 0,
        },
        RootAction::SetSysctl {
            key: "kern.securelevel".into(),
            value: "0".into(),
            reason: "bad".into(),
        },
    ];

    let mut frozen = HashSet::new();
    let outcomes = execute_actions(
        actions,
        &no_caps(),
        null_journal(),
        &mut frozen,
        &[],
        &[],
        None,
        false,
        0.0,
    );

    assert_eq!(outcomes.boosts_applied, 0);
    assert_eq!(outcomes.throttles_applied, 0);
    assert_eq!(outcomes.freezes_applied, 0);
    assert_eq!(outcomes.sysctl_applied, 0);
    assert_eq!(outcomes.failures, 0);
    assert!(frozen.is_empty());
}

// ── Full pipeline: enforce_limits + enforce_limits_with_budget ───────────────

/// Feed an oversized action list through the full pipeline; verify all caps hold.
#[test]
fn full_pipeline_respects_all_caps() {
    let profile = OptimizationProfile::BalancedRoot;
    let policy = SafetyPolicy::for_profile(profile);
    let mut budget = ActionBudgetState::default();
    let minute_cap = 8;

    let actions: Vec<RootAction> = (0..60)
        .map(|i| match i % 3 {
            0 => RootAction::BoostProcess {
                pid: (1000 + i) as u32,
                name: format!("app-{}", i),
                reason: "focus".into(),
            },
            1 => RootAction::ThrottleProcess {
                pid: (2000 + i) as u32,
                name: format!("bg-{}", i),
                aggressive: true,
                reason: "noise".into(),
                start_sec: 0,
                start_usec: 0,
            },
            _ => RootAction::FreezeProcess {
                pid: (3000 + i) as u32,
                name: format!("idle-{}", i),
                reason: "pressure".into(),
                start_sec: 0,
                start_usec: 0,
            },
        })
        .collect();

    let final_actions = enforce_limits_with_budget(actions, &policy, &mut budget, minute_cap);

    let boosts = final_actions
        .iter()
        .filter(|a| matches!(a, RootAction::BoostProcess { .. }))
        .count();
    let throttles = final_actions
        .iter()
        .filter(|a| matches!(a, RootAction::ThrottleProcess { .. }))
        .count();
    let freezes = final_actions
        .iter()
        .filter(|a| matches!(a, RootAction::FreezeProcess { .. }))
        .count();

    assert!(
        boosts <= policy.max_boosts_per_cycle,
        "boosts {} > cap {}",
        boosts,
        policy.max_boosts_per_cycle
    );
    assert!(throttles <= policy.max_throttles_per_cycle);
    assert!(freezes <= policy.max_freezes_per_cycle);
    assert!(
        final_actions.len() <= minute_cap,
        "total {} > minute cap {}",
        final_actions.len(),
        minute_cap
    );
    assert_eq!(budget.minute_actions, final_actions.len());
}

// ── protected_processes vs critical_background_processes ─────────────────────

/// Kernel-critical processes must be in protected but NOT in critical_bg.
/// Mixing them would give critical_bg processes kernel-level safety, or leave
/// protected processes accidentally throttleable.
#[test]
fn kernel_processes_not_in_critical_bg() {
    let protected = protected_processes();
    let critical = critical_background_processes();

    for p in &["kernel_task", "launchd", "WindowServer", "securityd"] {
        assert!(
            protected.contains(*p),
            "'{}' must be in protected_processes",
            p
        );
        assert!(
            !critical.contains(*p),
            "'{}' must NOT be in critical_background_processes",
            p
        );
    }
}

/// Dev workloads must be in critical_bg but NOT in protected (they're not
/// system-critical; they should still be protected from throttling, not from
/// all optimization).
#[test]
fn dev_workloads_in_critical_bg_not_protected() {
    let protected = protected_processes();
    let critical = critical_background_processes();

    for p in &["docker", "postgres", "redis-server"] {
        assert!(
            critical.contains(*p),
            "'{}' must be in critical_background_processes",
            p
        );
        assert!(
            !protected.contains(*p),
            "'{}' must NOT be in protected_processes",
            p
        );
    }
}

// ── Budget reset simulation ───────────────────────────────────────────────────

/// Simulates the daemon pattern: cycle counters reset each tick while the
/// minute counter accumulates across ticks.
#[test]
fn budget_cycle_counters_reset_but_minute_counter_persists() {
    let policy = SafetyPolicy::for_profile(OptimizationProfile::BalancedRoot);
    let mut budget = ActionBudgetState::default();

    // Cycle 1: 3 boosts
    let actions1: Vec<RootAction> = (0..3)
        .map(|i| RootAction::BoostProcess {
            pid: (1000 + i) as u32,
            name: format!("app-{}", i),
            reason: "test".into(),
        })
        .collect();
    enforce_limits_with_budget(actions1, &policy, &mut budget, 100);
    assert_eq!(budget.cycle_boosts, 3);
    let minute_after_cycle1 = budget.minute_actions;

    // Simulate daemon cycle reset (only cycle counters are zeroed).
    budget.cycle_boosts = 0;
    budget.cycle_throttles = 0;
    budget.cycle_hints = 0;
    budget.cycle_freezes = 0;

    // Cycle 2: 2 throttles
    let actions2: Vec<RootAction> = (0..2)
        .map(|i| RootAction::ThrottleProcess {
            pid: (2000 + i) as u32,
            name: format!("bg-{}", i),
            aggressive: false,
            reason: "test".into(),
            start_sec: 0,
            start_usec: 0,
        })
        .collect();
    enforce_limits_with_budget(actions2, &policy, &mut budget, 100);

    assert_eq!(
        budget.cycle_boosts, 0,
        "cycle boost counter should be reset"
    );
    assert_eq!(budget.cycle_throttles, 2);
    assert_eq!(
        budget.minute_actions,
        minute_after_cycle1 + 2,
        "minute counter must accumulate across cycles"
    );
}

/// execute_actions with a mix of protected process name and dead PID:
/// protected name check fires before PID check for BoostProcess.
#[test]
fn execute_actions_skips_protected_name_regardless_of_pid() {
    // Use a protected process name; the name check fires first.
    let actions = vec![RootAction::BoostProcess {
        pid: 1, // PID 1 (launchd) — definitely exists, but name check fires before PID check
        name: "kernel_task".into(),
        reason: "test".into(),
    }];
    let mut frozen = HashSet::new();
    let outcomes = execute_actions(
        actions,
        &no_caps(),
        null_journal(),
        &mut frozen,
        &[],
        &[],
        None,
        false,
        0.0,
    );
    // Protected name → skipped. No failure, no boost counted.
    assert_eq!(outcomes.failures, 0);
    assert_eq!(outcomes.boosts_applied, 0);
}
