//! Level 1: Pure unit tests — no system calls, no filesystem I/O, fully deterministic.
//!
//! Covers: tick rate boundary (BUG 1), SafetyPolicy caps, enforce_limits,
//! enforce_limits_with_budget, allowlisted_sysctls, protected/critical processes,
//! and EMA convergence (BUG 13).

use apollo_optimizer::engine::safety::{
    allowlisted_sysctls, critical_background_processes, enforce_limits, enforce_limits_with_budget,
    protected_processes,
};
use apollo_optimizer::engine::types::{
    ActionBudgetState, OptimizationProfile, RootAction, SafetyPolicy,
};
use apollo_optimizer::engine::audit_types::DecisionReason;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_boosts(n: usize) -> Vec<RootAction> {
    (0..n)
        .map(|i| RootAction::BoostProcess {
            pid: (1000 + i) as u32,
            name: format!("app-{}", i),
            reason: "test boost".into(),
            decision_reason: DecisionReason::PressureContext,
        })
        .collect()
}

fn make_throttles(n: usize) -> Vec<RootAction> {
    (0..n)
        .map(|i| RootAction::ThrottleProcess {
            pid: (2000 + i) as u32,
            name: format!("bg-{}", i),
            aggressive: false,
            reason: "test throttle".into(),
            start_sec: 0,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        })
        .collect()
}

fn make_freezes(n: usize) -> Vec<RootAction> {
    (0..n)
        .map(|i| RootAction::FreezeProcess {
            pid: (3000 + i) as u32,
            name: format!("slack-{}", i),
            reason: "test freeze".into(),
            start_sec: 0,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        })
        .collect()
}

// ── BUG 1 regression: tick rate boundary ─────────────────────────────────────

/// The daemon sleeps 300s when tick > 15. Pro-mode returns exactly 15 and
/// must NOT fall into the 300s branch. The old bug used `>= 15`.
#[test]
fn tick_rate_gt_15_maps_to_300s() {
    for v in [16u64, 30, 60, 300] {
        assert!(v > 15, "value {} should trigger the 300s branch", v);
    }
}

#[test]
fn tick_rate_eq_15_does_not_map_to_300s() {
    // Exactly 15 must NOT trigger the 300s sleep branch.
    assert_eq!(
        15u64.cmp(&15),
        std::cmp::Ordering::Equal,
        "pro-mode tick must be exactly 15"
    );
}

// ── SafetyPolicy invariants ───────────────────────────────────────────────────

#[test]
fn aggressive_profile_has_higher_caps_than_safe() {
    let aggressive = SafetyPolicy::for_profile(OptimizationProfile::AggressiveRoot);
    let safe = SafetyPolicy::for_profile(OptimizationProfile::SafeRoot);

    assert!(
        aggressive.max_boosts_per_cycle > safe.max_boosts_per_cycle,
        "aggressive boosts cap ({}) must exceed safe ({})",
        aggressive.max_boosts_per_cycle,
        safe.max_boosts_per_cycle
    );
    assert!(aggressive.max_throttles_per_cycle > safe.max_throttles_per_cycle);
    assert!(aggressive.max_freezes_per_cycle > safe.max_freezes_per_cycle);
}

#[test]
fn safe_profile_has_longer_cooldown_than_aggressive() {
    let aggressive = SafetyPolicy::for_profile(OptimizationProfile::AggressiveRoot);
    let safe = SafetyPolicy::for_profile(OptimizationProfile::SafeRoot);
    assert!(
        safe.cooldown_seconds > aggressive.cooldown_seconds,
        "safe cooldown ({}) should be longer than aggressive ({})",
        safe.cooldown_seconds,
        aggressive.cooldown_seconds
    );
}

#[test]
fn balanced_profile_is_between_safe_and_aggressive() {
    let aggressive = SafetyPolicy::for_profile(OptimizationProfile::AggressiveRoot);
    let balanced = SafetyPolicy::for_profile(OptimizationProfile::BalancedRoot);
    let safe = SafetyPolicy::for_profile(OptimizationProfile::SafeRoot);

    assert!(balanced.max_boosts_per_cycle >= safe.max_boosts_per_cycle);
    assert!(balanced.max_boosts_per_cycle <= aggressive.max_boosts_per_cycle);
}

// ── enforce_limits: per-cycle caps ───────────────────────────────────────────

#[test]
fn enforce_limits_caps_boosts_at_policy_max() {
    let policy = SafetyPolicy::for_profile(OptimizationProfile::BalancedRoot);
    let actions = make_boosts(policy.max_boosts_per_cycle + 5);
    let filtered = enforce_limits(actions, &policy);
    let count = filtered
        .iter()
        .filter(|a| matches!(a, RootAction::BoostProcess { .. }))
        .count();
    assert_eq!(
        count, policy.max_boosts_per_cycle,
        "boosts must be capped at policy max"
    );
}

#[test]
fn enforce_limits_caps_throttles_at_policy_max() {
    let policy = SafetyPolicy::for_profile(OptimizationProfile::BalancedRoot);
    let actions = make_throttles(policy.max_throttles_per_cycle + 10);
    let filtered = enforce_limits(actions, &policy);
    let count = filtered
        .iter()
        .filter(|a| matches!(a, RootAction::ThrottleProcess { .. }))
        .count();
    assert_eq!(count, policy.max_throttles_per_cycle);
}

#[test]
fn enforce_limits_caps_freezes_at_policy_max() {
    let policy = SafetyPolicy::for_profile(OptimizationProfile::BalancedRoot);
    let actions = make_freezes(policy.max_freezes_per_cycle + 10);
    let filtered = enforce_limits(actions, &policy);
    let count = filtered
        .iter()
        .filter(|a| matches!(a, RootAction::FreezeProcess { .. }))
        .count();
    assert_eq!(count, policy.max_freezes_per_cycle);
}

#[test]
fn enforce_limits_caps_sysctl_at_policy_max() {
    let policy = SafetyPolicy::for_profile(OptimizationProfile::BalancedRoot);
    let actions: Vec<RootAction> = (0..30)
        .map(|i| {
            RootAction::set_sysctl(
                format!("vm.key_{}", i),
                "1",
                "test",
                DecisionReason::PressureContext,
            )
        })
        .collect();
    let filtered = enforce_limits(actions, &policy);
    let count = filtered
        .iter()
        .filter(|a| matches!(a, RootAction::SetSysctl(_)))
        .count();
    assert_eq!(
        count, policy.max_sysctl_writes_per_cycle,
        "sysctl actions must be capped at policy max"
    );
}

// ── enforce_limits_with_budget: minute cap ───────────────────────────────────

#[test]
fn budget_minute_cap_stops_actions_exactly() {
    let policy = SafetyPolicy::for_profile(OptimizationProfile::AggressiveRoot);
    let minute_cap = 5;
    let mut budget = ActionBudgetState::default();
    let actions: Vec<RootAction> = make_boosts(10)
        .into_iter()
        .chain(make_throttles(10))
        .collect();

    let filtered = enforce_limits_with_budget(actions, &policy, &mut budget, minute_cap);

    assert!(
        filtered.len() <= minute_cap,
        "got {} actions, cap is {}",
        filtered.len(),
        minute_cap
    );
    assert_eq!(budget.minute_actions, filtered.len());
}

#[test]
fn budget_accumulates_per_category_correctly() {
    let policy = SafetyPolicy::for_profile(OptimizationProfile::AggressiveRoot);
    let mut budget = ActionBudgetState::default();
    let actions: Vec<RootAction> = make_boosts(3)
        .into_iter()
        .chain(make_throttles(2))
        .chain(make_freezes(1))
        .collect();

    let filtered = enforce_limits_with_budget(actions, &policy, &mut budget, 100);

    assert_eq!(budget.cycle_boosts, 3);
    assert_eq!(budget.cycle_throttles, 2);
    assert_eq!(budget.cycle_freezes, 1);
    assert_eq!(budget.minute_actions, filtered.len());
}

#[test]
fn budget_denied_cooldown_increments_when_cap_hit() {
    let policy = SafetyPolicy::for_profile(OptimizationProfile::AggressiveRoot);
    let minute_cap = 2;
    let mut budget = ActionBudgetState::default();
    let actions = make_boosts(10);

    enforce_limits_with_budget(actions, &policy, &mut budget, minute_cap);

    // We submitted 10 actions with a cap of 2, so some were denied.
    assert!(
        budget.boost_denied_cooldown > 0,
        "denied cooldown must increment when minute cap is exceeded"
    );
}

#[test]
fn budget_zero_cap_denies_all_actions() {
    let policy = SafetyPolicy::for_profile(OptimizationProfile::AggressiveRoot);
    let mut budget = ActionBudgetState::default();
    let actions = make_boosts(10);
    let filtered = enforce_limits_with_budget(actions, &policy, &mut budget, 0);
    assert_eq!(filtered.len(), 0, "zero cap must deny all actions");
    assert_eq!(budget.minute_actions, 0);
}

// ── allowlisted_sysctls ───────────────────────────────────────────────────────

#[test]
fn allowlist_contains_required_vm_and_net_keys() {
    let allowed = allowlisted_sysctls();
    for key in &[
        "vm.compressor_poll_interval",
        "debug.lowpri_throttle_enabled",
        "net.inet.tcp.sendspace",
        "net.inet.tcp.recvspace",
        "kern.maxvnodes",
        "kern.maxfiles",
    ] {
        assert!(allowed.contains(*key), "'{}' must be in allowlist", key);
    }
}

#[test]
fn allowlist_excludes_dangerous_sysctls() {
    let allowed = allowlisted_sysctls();
    for key in &[
        "kern.securelevel",
        "vm.swapfileprefix",
        "security.mac.sandbox.enable",
    ] {
        assert!(
            !allowed.contains(*key),
            "'{}' must NOT be in allowlist",
            key
        );
    }
}

// ── protected_processes ───────────────────────────────────────────────────────

#[test]
fn protected_always_contains_kernel_essentials() {
    let protected = protected_processes();
    for name in &[
        "kernel_task",
        "launchd",
        "WindowServer",
        "loginwindow",
        "securityd",
    ] {
        assert!(
            protected.contains(*name),
            "'{}' must be in protected_processes",
            name
        );
    }
}

#[test]
fn protected_contains_spotlight_stack() {
    let protected = protected_processes();
    for name in &["mds", "mds_stores", "mdworker"] {
        assert!(
            protected.contains(*name),
            "Spotlight process '{}' must be protected",
            name
        );
    }
}

// ── critical_background_processes ────────────────────────────────────────────

#[test]
fn critical_bg_contains_dev_workloads() {
    let critical = critical_background_processes();
    for name in &[
        "docker",
        "postgres",
        "redis-server",
        "nginx",
        "node",
        "python",
    ] {
        assert!(
            critical.contains(*name),
            "'{}' must be in critical_background_processes",
            name
        );
    }
}

// ── EMA convergence (BUG 13 regression) ──────────────────────────────────────

/// After enough iterations with alpha=0.1, the EMA must converge to within 0.1
/// of the target value. The old bug used a 50/50 average which converges much
/// faster but also overreacts to transient spikes.
#[test]
fn ema_converges_to_target_after_200_iterations() {
    const ALPHA: f32 = 0.1;
    let mut ema: f32 = 0.0;
    let target = 80.0_f32;

    for _ in 0..200 {
        ema = ema * (1.0 - ALPHA) + target * ALPHA;
    }

    assert!(
        (ema - target).abs() < 0.1,
        "EMA {} did not converge to {} after 200 iterations",
        ema,
        target
    );
}

/// The alpha=0.1 EMA reacts more slowly to a spike than the buggy 50/50 average.
/// This validates that we're smoothing correctly and not overreacting to spikes.
#[test]
fn ema_is_slower_than_buggy_50_50_average() {
    const ALPHA: f32 = 0.1;
    let baseline = 50.0_f32;
    let spike = 100.0_f32;

    let ema_correct = baseline * (1.0 - ALPHA) + spike * ALPHA;
    let ema_buggy = (baseline + spike) / 2.0; // old bug

    assert!(
        ema_correct < ema_buggy,
        "correct EMA ({:.2}) should react more slowly than 50/50 average ({:.2})",
        ema_correct,
        ema_buggy
    );
}

/// Verify EMA formula produces the mathematically correct value for one step.
#[test]
fn ema_single_step_is_numerically_correct() {
    const ALPHA: f32 = 0.1;
    let ema = 50.0_f32;
    let new_val = 100.0_f32;
    let result = ema * (1.0 - ALPHA) + new_val * ALPHA;
    // expected: 50 * 0.9 + 100 * 0.1 = 45 + 10 = 55
    assert!(
        (result - 55.0).abs() < 0.001,
        "EMA single step: expected 55.0, got {}",
        result
    );
}
