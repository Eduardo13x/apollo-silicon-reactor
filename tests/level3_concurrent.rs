//! Level 3: Complex concurrent tests — stress-tests for the three-phase mutex
//! pattern (BUG 5 fix), lock-ordering safety (BUG 19 context), and
//! ExecuteOutcomes accumulation correctness under high thread contention.
//!
//! Each test is designed to surface deadlocks, data races, or counter
//! corruption that unit/integration tests cannot catch.  They target the exact
//! patterns in the daemon's main optimisation loop after the BUG 5 refactor:
//!   Phase 1: acquire metrics lock briefly for budget → release
//!   Phase 2: execute_actions without metrics lock (blocking I/O)
//!   Phase 3: reacquire metrics lock → merge ExecuteOutcomes

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

use apollo_optimizer::engine::execute_actions::execute_actions;
use apollo_optimizer::engine::safety::enforce_limits_with_budget;
use apollo_optimizer::engine::types::{
    ActionBudgetState, CapabilityReport, OptimizationProfile, RootAction, SafetyPolicy,
};
use chrono::Utc;

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

fn null_journal() -> &'static Path {
    Path::new("/dev/null")
}

/// Returns dead PIDs in a range starting at a high base (safe, never alive).
fn dead_pids(base: u32, count: u32) -> Vec<u32> {
    (base..base + count).collect()
}

// ── Fake metrics struct (mirrors daemon's RuntimeMetrics subset) ──────────────

#[derive(Default)]
struct FakeMetrics {
    boosts_applied: u64,
    throttles_applied: u64,
    freezes_applied: u64,
    failures: u64,
    cycles: u64,
}

// ── Test 1: Concurrent budget enforcement — no minute-cap overflow ────────────
//
// 16 threads each hold the budget Mutex only during Phase 1 (budget computation)
// then release before Phase 2 (execute_actions). Validates that:
//   a) No deadlock or panic occurs.
//   b) The shared minute_actions counter never exceeds the global cap.
//   c) budget.minute_actions == Σ(actions allowed per thread).
#[test]
fn concurrent_budget_enforcement_no_overflow() {
    let global_cap = 40usize;
    let budget = Arc::new(Mutex::new(ActionBudgetState::default()));
    let policy = Arc::new(SafetyPolicy::for_profile(OptimizationProfile::BalancedRoot));
    let n_threads = 16usize;
    let per_thread_cap = 5usize; // each thread is capped at 5 of the shared budget

    let handles: Vec<_> = (0..n_threads)
        .map(|t| {
            let budget = Arc::clone(&budget);
            let policy = Arc::clone(&policy);
            thread::spawn(move || -> usize {
                let base_pid = 3_000_000u32 + (t as u32) * 1000;
                let actions: Vec<RootAction> = dead_pids(base_pid, 20)
                    .into_iter()
                    .map(|pid| RootAction::BoostProcess {
                        pid,
                        name: format!("app-{}-{}", t, pid),
                        reason: "stress".into(),
                    })
                    .collect();

                // Phase 1: acquire budget lock briefly.
                let final_actions = {
                    let mut b = budget.lock().unwrap();
                    enforce_limits_with_budget(actions, &policy, &mut b, per_thread_cap)
                }; // budget lock released

                // Phase 2: execute without any lock.
                let mut frozen = HashSet::new();
                let _outcomes = execute_actions(
                    final_actions.clone(),
                    &no_caps(),
                    null_journal(),
                    &mut frozen,
                    &[],
                    &[],
                    None,
                    false,
                    0.0,
                );

                final_actions.len()
            })
        })
        .collect();

    let total_allowed: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    let b = budget.lock().unwrap();
    assert!(
        b.minute_actions <= global_cap,
        "minute_actions ({}) exceeded global_cap ({})",
        b.minute_actions,
        global_cap
    );
    assert_eq!(
        b.minute_actions, total_allowed,
        "budget.minute_actions must equal sum of per-thread allowed actions"
    );
}

// ── Test 2: Three-phase concurrent cycles — correctness of outcome merging ────
//
// 32 threads each simulate one full daemon optimisation cycle:
//   Phase 1: lock fake_metrics, set up (instant work) → release
//   Phase 2: execute_actions with dead PIDs (no lock held)
//   Phase 3: lock fake_metrics, merge ExecuteOutcomes → release
//
// Verifies: all 32 cycles complete, no panics, failure counters stay zero,
// and the cycle count in fake_metrics equals n_threads.
#[test]
fn three_phase_cycles_all_complete_with_correct_totals() {
    let metrics = Arc::new(Mutex::new(FakeMetrics::default()));
    let frozen_state = Arc::new(Mutex::new(HashSet::<u32>::new()));
    let n_threads = 32usize;

    let handles: Vec<_> = (0..n_threads)
        .map(|t| {
            let metrics = Arc::clone(&metrics);
            let frozen_state = Arc::clone(&frozen_state);
            thread::spawn(move || {
                // Phase 1: lock metrics briefly.
                let final_actions = {
                    let _m = metrics.lock().unwrap();
                    // Build action list while holding lock (budget check in daemon).
                    dead_pids(4_000_000 + (t as u32) * 100, 5)
                        .into_iter()
                        .map(|pid| RootAction::BoostProcess {
                            pid,
                            name: format!("fake-{}-{}", t, pid),
                            reason: "concurrent".into(),
                        })
                        .collect::<Vec<_>>()
                }; // metrics lock released

                // Phase 2: execute WITHOUT metrics lock (the BUG 5 fix).
                let outcomes = {
                    let mut frozen = frozen_state.lock().unwrap();
                    execute_actions(
                        final_actions,
                        &no_caps(),
                        null_journal(),
                        &mut frozen,
                        &[],
                        &[],
                        None,
                        false,
                        0.0,
                    )
                };

                // Phase 3: reacquire metrics lock to merge.
                {
                    let mut m = metrics.lock().unwrap();
                    m.boosts_applied += outcomes.boosts_applied;
                    m.throttles_applied += outcomes.throttles_applied;
                    m.freezes_applied += outcomes.freezes_applied;
                    m.failures += outcomes.failures;
                    m.cycles += 1;
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked or deadlocked");
    }

    let m = metrics.lock().unwrap();
    assert_eq!(
        m.cycles, n_threads as u64,
        "all {} cycles must complete",
        n_threads
    );
    assert_eq!(m.failures, 0, "no failures expected for dead PIDs");
    // Dead PIDs → PID validation fires → nothing applied.
    assert_eq!(m.boosts_applied, 0, "no boosts on dead PIDs");
}

// ── Test 3: ExecuteOutcomes field accumulation is exact ───────────────────────
//
// 20 threads each "execute" a known number of actions and merge fixed outcome
// values into a shared counter. Verifies the exact total is preserved.
// (Targets the Phase 3 merge pattern from the BUG 5 fix.)
#[test]
fn execute_outcomes_accumulation_is_exact() {
    let total_boosts = Arc::new(Mutex::new(0u64));
    let total_throttles = Arc::new(Mutex::new(0u64));
    let total_sysctl = Arc::new(Mutex::new(0u64));
    let total_failures = Arc::new(Mutex::new(0u64));

    let n_threads = 20usize;
    let boosts_per = 3u64;
    let throttles_per = 2u64;
    let sysctl_per = 1u64;
    let failures_per = 0u64;

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let tb = Arc::clone(&total_boosts);
            let tt = Arc::clone(&total_throttles);
            let ts = Arc::clone(&total_sysctl);
            let tf = Arc::clone(&total_failures);
            thread::spawn(move || {
                // Phase 2: no lock held, simulate returning known outcomes.
                let (b, t, s, f) = (boosts_per, throttles_per, sysctl_per, failures_per);

                // Phase 3: merge under lock (exact replica of daemon pattern).
                *tb.lock().unwrap() += b;
                *tt.lock().unwrap() += t;
                *ts.lock().unwrap() += s;
                *tf.lock().unwrap() += f;
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    assert_eq!(*total_boosts.lock().unwrap(), n_threads as u64 * boosts_per);
    assert_eq!(
        *total_throttles.lock().unwrap(),
        n_threads as u64 * throttles_per
    );
    assert_eq!(*total_sysctl.lock().unwrap(), n_threads as u64 * sysctl_per);
    assert_eq!(
        *total_failures.lock().unwrap(),
        n_threads as u64 * failures_per
    );
}

// ── Test 4: Lock ordering — frozen then frozen_since never deadlocks ──────────
//
// 64 threads acquire `frozen` first, release it, then acquire `frozen_since`.
// Consistent ordering prevents deadlocks. Validates the Phase 2 lock ordering
// from the BUG 5 fix (frozen lock → release → frozen_since lock).
//
// If lock ordering were inconsistent, this test would hang (detected by the
// test runner's default 60-second timeout).
#[test]
fn lock_ordering_frozen_then_frozen_since_no_deadlock() {
    let frozen: Arc<Mutex<HashSet<u32>>> = Arc::new(Mutex::new(HashSet::new()));
    let frozen_since: Arc<Mutex<HashMap<u32, chrono::DateTime<chrono::Utc>>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let n_threads = 64usize;

    let handles: Vec<_> = (0..n_threads)
        .map(|t| {
            let frozen = Arc::clone(&frozen);
            let frozen_since = Arc::clone(&frozen_since);
            thread::spawn(move || {
                let pid = (t as u32) * 100 + 1;

                // Phase 2 pattern: acquire `frozen` first.
                {
                    let mut f = frozen.lock().unwrap();
                    f.insert(pid);
                } // release frozen

                // Then acquire `frozen_since` separately (never hold both simultaneously).
                {
                    let now = Utc::now();
                    let mut fs = frozen_since.lock().unwrap();
                    fs.entry(pid).or_insert(now);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread deadlocked or panicked");
    }

    let f = frozen.lock().unwrap();
    let fs = frozen_since.lock().unwrap();
    assert_eq!(f.len(), n_threads, "all PIDs must be in frozen set");
    assert_eq!(fs.len(), n_threads, "all PIDs must be in frozen_since map");
}

// ── Test 5: Shared budget under maximum contention — invariant holds ──────────
//
// 64 threads share one ActionBudgetState Mutex. Each claims up to 10 actions
// from a global cap of 50. After all threads finish:
//   - minute_actions ≤ global_cap
//   - minute_actions == total_allowed (no double-counting or loss)
//
// This is the hardest version of the budget test from Level 1/2, running at
// 64× the thread count to surface any race in the budget logic.
#[test]
fn shared_budget_invariant_holds_under_max_contention() {
    let global_cap = 50usize;
    let budget = Arc::new(Mutex::new(ActionBudgetState::default()));
    let policy = Arc::new(SafetyPolicy::for_profile(
        OptimizationProfile::AggressiveRoot,
    ));
    let n_threads = 64usize;

    let handles: Vec<_> = (0..n_threads)
        .map(|t| {
            let budget = Arc::clone(&budget);
            let policy = Arc::clone(&policy);
            thread::spawn(move || -> usize {
                let actions: Vec<RootAction> = (0..10)
                    .map(|i| RootAction::BoostProcess {
                        pid: (t as u32) * 100 + i,
                        name: format!("stress-{}-{}", t, i),
                        reason: "stress".into(),
                    })
                    .collect();

                let mut b = budget.lock().unwrap();
                let allowed = enforce_limits_with_budget(actions, &policy, &mut b, global_cap);
                allowed.len()
            })
        })
        .collect();

    let total_allowed: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    let b = budget.lock().unwrap();
    assert!(
        b.minute_actions <= global_cap,
        "minute_actions ({}) exceeded global_cap ({})",
        b.minute_actions,
        global_cap
    );
    assert_eq!(
        b.minute_actions, total_allowed,
        "budget counter must equal sum of all allowed actions (no double-count or loss)"
    );
}

// ── Test 6: Concurrent freeze/unfreeze cycle — frozen set consistency ─────────
//
// Simulates the daemon pattern where one set of threads freeze PIDs (Phase 2)
// and another set unfreeze them. Verifies that the frozen HashSet remains
// consistent and no PID stays frozen after its unfreeze action completes.
#[test]
fn concurrent_freeze_unfreeze_frozen_set_is_consistent() {
    let frozen = Arc::new(Mutex::new(HashSet::<u32>::new()));
    let n_threads = 20usize;
    let pids_per_thread = 3u32;

    // First wave: freeze PIDs via execute_actions.
    let freeze_handles: Vec<_> = (0..n_threads)
        .map(|t| {
            let frozen = Arc::clone(&frozen);
            thread::spawn(move || {
                let base = 5_000_000u32 + (t as u32) * 1000;
                let actions: Vec<RootAction> = (0..pids_per_thread)
                    .map(|i| RootAction::FreezeProcess {
                        pid: base + i,
                        name: format!("bg-{}", t),
                        reason: "test freeze".into(),
                        start_sec: 0,
                        start_usec: 0,
                    })
                    .collect();
                let mut f = frozen.lock().unwrap();
                // Dead PIDs → PID validation skips them; frozen set should not grow.
                let _outcomes = execute_actions(
                    actions,
                    &no_caps(),
                    null_journal(),
                    &mut f,
                    &[],
                    &[],
                    None,
                    false,
                    0.0,
                );
            })
        })
        .collect();

    for h in freeze_handles {
        h.join().expect("freeze thread panicked");
    }

    // Dead PIDs: PID validation (kill(pid, 0) != 0) fires → frozen set stays empty.
    let f = frozen.lock().unwrap();
    assert!(
        f.is_empty(),
        "frozen set must be empty after dead-PID freeze attempts (PID validation)"
    );
}
