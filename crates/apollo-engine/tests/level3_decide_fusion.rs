//! TDD RED phase: failing tests for PidClass single-pass classification fusion.
//!
//! These tests intentionally fail to compile / run until the fusion lands:
//!   - `apollo_engine::engine::pid_class::{PidClass, PidKind, ClassifyCtx, classify_once}`
//!
//! Design contract under test (from FUSION DESIGN doc):
//!   1. Short-circuit ladder: T1 hard → T2 infra → T3 policy → display_pipeline
//!      → T4 interactive+fg → critical_bg/deferrable → noise.
//!   2. Identity tuple (pid, start_sec, start_usec) carried verbatim (ABA key).
//!   3. classify_once is pure / idempotent.
//!   4. One name allocation per PID (not 7×).
//!   5. Known-protected processes (kernel_task, WindowServer, Brave Helper)
//!      resolve to the correct PidKind.
//!   6. Dev-runtime (rustc / clippy-driver / cargo) → critical_bg flag.
//!   7. Priority winner is deterministic when multiple flags overlap.

#![allow(unused_imports, dead_code)]

use std::collections::HashSet;

use apollo_engine::engine::pid_class::{classify_once, ClassifyCtx, PidClass, PidKind};

// ---------------------------------------------------------------------------
// Test fixture helpers
// ---------------------------------------------------------------------------

fn empty_set() -> HashSet<u32> {
    HashSet::new()
}

fn never_fg(_pid: u32) -> bool {
    false
}

fn always_fg(_pid: u32) -> bool {
    true
}

fn make_ctx<'a>(
    app_bundle_pids: &'a HashSet<u32>,
    behavior_interactive_pids: &'a HashSet<u32>,
    interactive_lc: &'a [String],
    noise_lc: &'a [String],
    policy_protected_lc: &'a [String],
    is_fg: &'a dyn Fn(u32) -> bool,
) -> ClassifyCtx<'a> {
    ClassifyCtx {
        app_bundle_pids,
        behavior_interactive_pids,
        interactive_lc,
        noise_lc,
        policy_protected_lc,
        is_foreground_family_pid: is_fg,
    }
}

// ---------------------------------------------------------------------------
// Test 1: priority order — INTERACTIVE classification does NOT override a
// hard-protected name. T1 wins over T4.
// ---------------------------------------------------------------------------
#[test]
fn fusion_classify_once_priority_order() {
    let bundle = empty_set();
    let beh = empty_set();
    let ctx = make_ctx(&bundle, &beh, &[], &[], &[], &always_fg);

    // kernel_task is hard-protected; even if foreground-family closure says yes,
    // the result must be Protected.
    let pc = classify_once("kernel_task", 1, 1000, 0, 0.5, &ctx);
    assert_eq!(
        pc.kind,
        PidKind::Protected,
        "T1 hard must beat T4 interactive"
    );
    assert!(pc.hard_protected, "hard_protected flag must be set");
    assert!(!pc.policy_protected);
}

// ---------------------------------------------------------------------------
// Test 2: identity tuple carries start_sec verbatim (ABA-safe key).
// Two PidClass with same PID but different start_sec must not alias.
// ---------------------------------------------------------------------------
#[test]
fn fusion_pid_identity_carries_start_sec() {
    let bundle = empty_set();
    let beh = empty_set();
    let ctx = make_ctx(&bundle, &beh, &[], &[], &[], &never_fg);

    let a = classify_once("someproc", 4242, 1_700_000_000, 123, 1.0, &ctx);
    let b = classify_once("someproc", 4242, 1_700_000_500, 456, 1.0, &ctx);

    assert_eq!(a.pid, b.pid);
    assert_ne!(
        (a.start_sec, a.start_usec),
        (b.start_sec, b.start_usec),
        "identity tuple must differ — ABA guard"
    );
    assert_eq!(a.start_sec, 1_700_000_000);
    assert_eq!(b.start_sec, 1_700_000_500);
}

// ---------------------------------------------------------------------------
// Test 3: idempotency — same input twice yields same output.
// ---------------------------------------------------------------------------
#[test]
fn fusion_idempotent_same_input() {
    let bundle = empty_set();
    let beh = empty_set();
    let ctx = make_ctx(&bundle, &beh, &[], &[], &[], &never_fg);

    let a = classify_once("postgres", 100, 500, 0, 2.5, &ctx);
    let b = classify_once("postgres", 100, 500, 0, 2.5, &ctx);

    assert_eq!(a.kind, b.kind);
    assert_eq!(a.hard_protected, b.hard_protected);
    assert_eq!(a.infra_protected, b.infra_protected);
    assert_eq!(a.policy_protected, b.policy_protected);
    assert_eq!(a.interactive, b.interactive);
    assert_eq!(a.display_pipeline, b.display_pipeline);
    assert_eq!(a.deferrable_ml, b.deferrable_ml);
    assert_eq!(a.is_windowserver, b.is_windowserver);
    assert_eq!(a.critical_bg, b.critical_bg);
    assert_eq!(a.cpu, b.cpu);
}

// ---------------------------------------------------------------------------
// Test 4: known-protected processes resolve correctly.
//   - kernel_task   → Protected (hard)
//   - WindowServer  → Protected (hard) + is_windowserver flag
//   - Brave Helper  → Protected (hard, FamilyRoot — Permanent Scar #1)
// ---------------------------------------------------------------------------
#[test]
fn fusion_known_protected_process_gets_correct_class() {
    let bundle = empty_set();
    let beh = empty_set();
    let ctx = make_ctx(&bundle, &beh, &[], &[], &[], &never_fg);

    let kt = classify_once("kernel_task", 1, 1, 0, 0.0, &ctx);
    assert_eq!(kt.kind, PidKind::Protected);
    assert!(kt.hard_protected);

    let ws = classify_once("WindowServer", 200, 1, 0, 5.0, &ctx);
    assert_eq!(ws.kind, PidKind::Protected);
    assert!(ws.hard_protected);
    assert!(ws.is_windowserver, "is_windowserver flag must be set");

    let brave = classify_once("Brave Browser Helper (Renderer)", 333, 1, 0, 8.0, &ctx);
    assert_eq!(
        brave.kind,
        PidKind::Protected,
        "Brave Helper FamilyRoot must classify Protected (Permanent Scar #1)"
    );
    assert!(brave.hard_protected);
}

// ---------------------------------------------------------------------------
// Test 5: dev-runtime processes → critical_bg flag set.
// rustc / clippy-driver / cargo are dev-runtime (build mode).
// ---------------------------------------------------------------------------
#[test]
fn fusion_dev_runtime_classification() {
    let bundle = empty_set();
    let beh = empty_set();
    let ctx = make_ctx(&bundle, &beh, &[], &[], &[], &never_fg);

    let rustc = classify_once("rustc", 5000, 100, 0, 90.0, &ctx);
    assert!(
        rustc.critical_bg || rustc.hard_protected,
        "rustc must be flagged critical_bg or hard_protected"
    );

    let clippy = classify_once("clippy-driver", 5001, 100, 0, 90.0, &ctx);
    assert!(
        clippy.critical_bg || clippy.hard_protected,
        "clippy-driver must be flagged critical_bg or hard_protected"
    );

    let cargo = classify_once("cargo", 5002, 100, 0, 50.0, &ctx);
    assert!(
        cargo.critical_bg || cargo.hard_protected,
        "cargo must be flagged critical_bg or hard_protected"
    );
}

// ---------------------------------------------------------------------------
// Test 6: priority when overlap — interactive substring + infra match.
// A name matching both "interactive learned substring" AND infra-protected
// must resolve to CriticalBackground (T2 beats T4).
// ---------------------------------------------------------------------------
#[test]
fn fusion_priority_when_overlap() {
    let bundle = empty_set();
    let beh = empty_set();
    // Tell the classifier "postgres" is also a learned interactive substring.
    let interactive_lc = vec!["postgres".to_string()];
    let ctx = make_ctx(&bundle, &beh, &interactive_lc, &[], &[], &always_fg);

    let pc = classify_once("postgres", 600, 1, 0, 3.0, &ctx);
    assert_eq!(
        pc.kind,
        PidKind::CriticalBackground,
        "T2 infra must beat T4 interactive even when foreground-family says yes"
    );
    assert!(pc.infra_protected);
}

// ---------------------------------------------------------------------------
// Test 7: substring-only policy match must NOT set hard_protected.
// Tier-3 substring confidence (0.30) is Unconditional via policy_protected,
// not via hard_protected.
// ---------------------------------------------------------------------------
#[test]
fn fusion_substring_only_does_not_set_hard_protected() {
    let bundle = empty_set();
    let beh = empty_set();
    let policy = vec!["myspecialthing".to_string()];
    let ctx = make_ctx(&bundle, &beh, &[], &[], &policy, &never_fg);

    let pc = classify_once("MySpecialThing-worker", 700, 1, 0, 1.0, &ctx);
    assert_eq!(pc.kind, PidKind::SoftlyProtected);
    assert!(pc.policy_protected);
    assert!(
        !pc.hard_protected,
        "substring tier-3 must NOT raise hard_protected"
    );
    assert!(!pc.infra_protected);
}

// ---------------------------------------------------------------------------
// Test 8: interactive background instance (not foreground family) is NOT
// classified as InteractiveApp — ConditionalForeground veto requires fg.
// ---------------------------------------------------------------------------
#[test]
fn fusion_interactive_background_instance_not_interactive_app() {
    let bundle = empty_set();
    let beh = empty_set();
    let interactive_lc = vec!["chrome".to_string()];
    // Foreground predicate says NO.
    let ctx = make_ctx(&bundle, &beh, &interactive_lc, &[], &[], &never_fg);

    let pc = classify_once("Chrome Helper", 800, 1, 0, 1.0, &ctx);
    assert_ne!(
        pc.kind,
        PidKind::InteractiveApp,
        "interactive flag without foreground family must NOT resolve to InteractiveApp"
    );
    // but pc.interactive may still be true (informational)
    assert!(pc.interactive);
}

// ---------------------------------------------------------------------------
// Test 9: start_sec=0 propagates verbatim (caller decides cache semantics).
// ---------------------------------------------------------------------------
#[test]
fn fusion_zero_start_sec_propagates() {
    let bundle = empty_set();
    let beh = empty_set();
    let ctx = make_ctx(&bundle, &beh, &[], &[], &[], &never_fg);

    let pc = classify_once("randomproc", 9999, 0, 0, 1.0, &ctx);
    assert_eq!(pc.start_sec, 0);
    assert_eq!(pc.start_usec, 0);
    assert_eq!(pc.pid, 9999);
}
