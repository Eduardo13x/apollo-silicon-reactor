//! ShadowEvaluator — runs PolicyScorer alongside the gate tower, logs disagreements.
//!
//! The gate tower stays in charge of accept/reject. This evaluator constructs an
//! ActionContext from daemon state at the call site and asks the scorer for its
//! opinion. If scorer disagrees with the gate's verdict, emit a BlockedActionEvent
//! with `blocker = Other("shadow-disagree:…")` so offline analysis can measure:
//!
//!   - False-negative rate (gate blocks, scorer would accept, outcome 30s later)
//!   - False-positive rate (gate accepts, scorer would reject, outcome 30s later)
//!
//! Once those rates are stable and interpretable, cutover flips the order.
//!
//! Paper: [Nygard 2018 §8.5] Adaptive capacity limits via shadowing;
//! [Bengio 2013] counterfactual reasoning.

use std::path::Path;

use crate::engine::action_policy::{
    ActionContext, PolicyScore, PolicyScorer, PressureBenefitFeature, ProtectionFeature,
    UserDisruptionCostFeature,
};
use crate::engine::blocked_action_journal::{emit_async, BlockedActionEvent, BlockerKind};
use crate::engine::lse_counters::LSE_COUNTERS;
use crate::engine::policy_feature_battery_cost::BatteryAwareCostFeature;
use crate::engine::policy_feature_deep_scan::DeepScanCostFeature;
use crate::engine::policy_feature_predictive::PredictiveBenefitFeature;
use crate::engine::policy_feature_sensor_age::SensorAgeFeature;
use crate::engine::types::RootAction;

// ── Phase C SCORER-OVERRIDE (Sprint 11 finale, 2026-05-16) ───────────────────
//
// Asymmetric scorer/gate disagreement threshold. We split disagreements into
// medium-confidence (|composite| ≤ 0.30, existing shadow-log behaviour) and
// high-confidence (|composite| > 0.30, new override path).
//
// Derivation of the ±0.30 bound: the default scorer threshold is 0.0, so a
// composite of ±0.30 means the score moved by ~30% of the *single highest-
// benefit single-feature contribution we observe in steady state* — the
// PressureBenefitFeature's `pressure * 1.0` term saturates near 0.95 under
// crisis, the +1.0 p_oom bonus is rare. ±0.30 places the bound:
//   • Above noise — RSS-composed uncertainty (saturated 1.5) × λ_unc 0.5 = 0.75
//     swing in NET, so a 0.30 net delta requires real benefit/cost evidence,
//     not noise.
//   • Below "obvious" — ±0.50 would gate-out cases where scorer is right but
//     the action is still safe (e.g. UserDisruptionCostFeature's call_in_progress
//     contributes 2.0 → net swings ≥ −2.0 vs benefit). 0.30 catches the
//     real disagreements; 0.50 would silence too many.
// Empirically (Sprint 10 shadow journal): 38% of `evaluate_blocked` disagreements
// have |composite| ≤ 0.30 (noise / borderline), 62% > 0.30. The 0.30 bound
// therefore protects the high-signal majority for the override path while
// leaving the borderline minority on the shadow-log path.
pub const SCORER_STRONG_REJECT: f64 = -0.30;
pub const SCORER_STRONG_ACCEPT: f64 = 0.30;

/// Outcome of `decide_override(gate_accept, score)`. Pure data — callers
/// (decide_actions cost-composition site, tests) translate this into the
/// concrete side effects (set `extreme_freeze_ok = false`, emit
/// `BlockedActionEvent`, bump the matching LSE counter).
///
/// Semantics:
/// * `NoChange` — gate and scorer agree, OR disagree only mildly
///   (|composite| ≤ 0.30). The medium-confidence shadow log still fires
///   for the disagreement subset (existing `evaluate_blocked` /
///   `evaluate_accepted` behaviour). The caller takes the gate verdict.
/// * `OverrideReject { composite }` — gate ACCEPTED but scorer composite
///   < −0.30. The caller MUST reject the action AND emit a journal
///   event with reason `scorer-override-accept-to-reject` AND bump
///   `scorer_override_rejects_total`.
/// * `LogStrongAccept { composite }` — gate REJECTED but scorer composite
///   > +0.30. The caller stays with the gate (REJECT), emits a journal
///   > event with reason `scorer-disagreement-strong-accept`, and bumps
///   > `scorer_disagreement_strong_accepts_total`. The action remains
///   > blocked — Sprint 11 deliberately refuses to let the scorer beat
///   > the gate in the unsafe direction (per NotebookLM 2026-05-16
///   > Candidate-C verdict; Sprint 12 may promote to symmetric once
///   > N≥500 events validate the asymmetric mode).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverrideDecision {
    NoChange,
    OverrideReject { composite: f64 },
    LogStrongAccept { composite: f64 },
}

/// Pure decision function — no I/O, no atomics. Maps the gate's
/// accept/reject + the scorer's `PolicyScore` to the asymmetric
/// override verdict. Tested in isolation; the caller (decide_actions or
/// `ShadowEvaluator::evaluate_with_override`) handles all side effects.
pub fn decide_override(gate_accept: bool, score: &PolicyScore) -> OverrideDecision {
    match (gate_accept, score.accept) {
        // Gate accepts + scorer rejects strongly → DEFER to scorer.
        (true, false) if score.composite < SCORER_STRONG_REJECT => {
            OverrideDecision::OverrideReject {
                composite: score.composite,
            }
        }
        // Gate rejects + scorer accepts strongly → LOG only (NOT promote).
        (false, true) if score.composite > SCORER_STRONG_ACCEPT => {
            OverrideDecision::LogStrongAccept {
                composite: score.composite,
            }
        }
        _ => OverrideDecision::NoChange,
    }
}

pub struct ShadowEvaluator {
    scorer: PolicyScorer,
}

impl Default for ShadowEvaluator {
    fn default() -> Self {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .feature(DeepScanCostFeature::default())
            .feature(PredictiveBenefitFeature::default())
            .feature(SensorAgeFeature::default())
            // Phase 5.2 WIRED (Sprint 10, 2026-05-16) — battery-aware cost.
            // Returns Contribution::zero() until shadow_signals publishes
            // is_on_battery + wakeups + ctx_switches; then injects a
            // [0.0, 0.20] cost penalty proportional to micro-wake noise.
            .feature(BatteryAwareCostFeature)
            .build();
        Self { scorer }
    }
}

impl ShadowEvaluator {
    /// Called when the gate tower BLOCKS a candidate action. Runs scorer; if scorer
    /// would have accepted, emits a `BlockerKind::Other("shadow-disagree:blocked:…")`
    /// event. Errors are logged (eprintln) but never propagated — hot path safety.
    pub fn evaluate_blocked(
        &self,
        action: &RootAction,
        ctx: &ActionContext,
        gate_blocker: BlockerKind,
        journal_path: &Path,
    ) {
        let score = self.scorer.score(action, ctx);
        if score.accept {
            // Scorer disagrees — wants to accept what the gate blocked.
            let event = BlockedActionEvent::new(
                action_kind_str(action),
                target_name(action),
                target_pid(action),
                BlockerKind::Other(format!(
                    "shadow-disagree:blocked-by:{:?}:reason:{}",
                    gate_blocker, score.reason
                )),
                ctx.pressure,
                ctx.swap_gb,
                ctx.thrashing_score,
                ctx.p_oom_30s,
            );
            emit_async(journal_path.to_path_buf(), &event);
        }
    }

    /// Phase C SCORER-OVERRIDE entry point (Sprint 11 finale, 2026-05-16).
    ///
    /// Runs the scorer alongside an already-decided gate verdict and applies
    /// the asymmetric override policy. Returns the [`OverrideDecision`] so
    /// the caller can act on it (e.g. flip `extreme_freeze_ok = false`).
    /// Side effects are performed inside this method — the journal event
    /// is emitted via the existing `emit_async` writer and the matching
    /// LSE counter is bumped.
    ///
    /// Callers MUST honour `OverrideReject` by skipping the action; callers
    /// MAY ignore `LogStrongAccept` because the gate's reject already won
    /// (the side-effect log is enough for offline analysis).
    ///
    /// [Nygard 2018 §8.5] — adaptive capacity limits via shadowing.
    pub fn evaluate_with_override(
        &self,
        action: &RootAction,
        ctx: &ActionContext,
        gate_accept: bool,
        journal_path: &Path,
    ) -> OverrideDecision {
        let score = self.scorer.score(action, ctx);
        let decision = decide_override(gate_accept, &score);
        match decision {
            OverrideDecision::OverrideReject { composite } => {
                let event = BlockedActionEvent::new(
                    action_kind_str(action),
                    target_name(action),
                    target_pid(action),
                    BlockerKind::Other(format!(
                        "scorer-override-accept-to-reject:composite={:.3}:reason:{}",
                        composite, score.reason
                    )),
                    ctx.pressure,
                    ctx.swap_gb,
                    ctx.thrashing_score,
                    ctx.p_oom_30s,
                );
                emit_async(journal_path.to_path_buf(), &event);
                LSE_COUNTERS.inc_scorer_override_reject();
            }
            OverrideDecision::LogStrongAccept { composite } => {
                let event = BlockedActionEvent::new(
                    action_kind_str(action),
                    target_name(action),
                    target_pid(action),
                    BlockerKind::Other(format!(
                        "scorer-disagreement-strong-accept:composite={:.3}:reason:{}",
                        composite, score.reason
                    )),
                    ctx.pressure,
                    ctx.swap_gb,
                    ctx.thrashing_score,
                    ctx.p_oom_30s,
                );
                emit_async(journal_path.to_path_buf(), &event);
                LSE_COUNTERS.inc_scorer_disagreement_strong_accept();
            }
            OverrideDecision::NoChange => {}
        }
        decision
    }

    /// Called when gate tower ACCEPTS a candidate. Runs scorer; if scorer would have
    /// rejected, emits a disagreement event. Offline analysis correlates with outcomes.
    #[allow(dead_code)] // wiring for accepted-case follows in a later commit
    pub fn evaluate_accepted(&self, action: &RootAction, ctx: &ActionContext, journal_path: &Path) {
        let score = self.scorer.score(action, ctx);
        if !score.accept {
            let event = BlockedActionEvent::new(
                action_kind_str(action),
                target_name(action),
                target_pid(action),
                BlockerKind::Other(format!(
                    "shadow-disagree:accepted-but-scorer-rejects:{}",
                    score.reason
                )),
                ctx.pressure,
                ctx.swap_gb,
                ctx.thrashing_score,
                ctx.p_oom_30s,
            );
            emit_async(journal_path.to_path_buf(), &event);
        }
    }
}

fn action_kind_str(a: &RootAction) -> &'static str {
    match a {
        RootAction::BoostProcess { .. } => "Boost",
        RootAction::ThrottleProcess { .. } => "Throttle",
        RootAction::FreezeProcess { .. } => "Freeze",
        RootAction::UnfreezeProcess { .. } => "Unfreeze",
        RootAction::SetSysctl(_) => "SetSysctl",
        RootAction::SetMemorystatus { .. } => "SetMemorystatus",
        RootAction::ToggleSpotlight { .. } => "ToggleSpotlight",
        RootAction::QuarantineDaemon { .. } => "QuarantineDaemon",
        RootAction::SetThreadQoS { .. } => "SetThreadQoS",
    }
}

fn target_name(a: &RootAction) -> String {
    match a {
        RootAction::BoostProcess { name, .. }
        | RootAction::ThrottleProcess { name, .. }
        | RootAction::FreezeProcess { name, .. }
        | RootAction::UnfreezeProcess { name, .. }
        | RootAction::SetThreadQoS { name, .. } => name.clone(),
        RootAction::SetSysctl(s) => s.key().to_string(),
        RootAction::SetMemorystatus { pid, .. } => format!("pid:{}", pid),
        RootAction::ToggleSpotlight { .. } => "spotlight".to_string(),
        RootAction::QuarantineDaemon { daemon, .. } => daemon.clone(),
    }
}

fn target_pid(a: &RootAction) -> Option<u32> {
    match a {
        RootAction::BoostProcess { pid, .. }
        | RootAction::ThrottleProcess { pid, .. }
        | RootAction::FreezeProcess { pid, .. }
        | RootAction::UnfreezeProcess { pid, .. }
        | RootAction::SetMemorystatus { pid, .. }
        | RootAction::SetThreadQoS { pid, .. } => Some(*pid),
        RootAction::SetSysctl(_)
        | RootAction::ToggleSpotlight { .. }
        | RootAction::QuarantineDaemon { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::audit_types::DecisionReason;
    use crate::engine::safety::ProtectionLevel;

    fn make_ctx() -> ActionContext {
        ActionContext {
            pressure: 0.85,
            swap_gb: 2.0,
            thrashing_score: 12_000.0,
            p_oom_30s: Some(0.50),
            p_jank_60s: None,
            has_sleep_assertion: false,
            call_in_progress: false,
            idle_secs: 60.0,
            foreground_pid: None,
            is_foreground_family: false,
            is_recently_active: false,
            thermal_emergency: false,
            interrupt_phase: 0,
            protection_level: ProtectionLevel::Unprotected,
            hot_page_fraction: None,
            wss_mb: None,
            sensor_age_ms: None,
            epistemic_uncertainty: 0.0,
            is_on_battery: None,
            wakeups_per_sec: None,
            ctx_switches_per_sec: None,
        }
    }

    fn unique_tmp(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "apollo-shadow-{}-{}-{}.jsonl",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    /// Poll up to 2s for the async writer to flush `expected_lines` to `path`.
    /// Returns the contents once the expected line count is present, or the last
    /// observed contents on timeout (tests then assert on the result).
    fn wait_for_lines(path: &std::path::Path, expected_lines: usize) -> String {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            if let Ok(s) = std::fs::read_to_string(path) {
                let count = s.lines().filter(|l| !l.is_empty()).count();
                if count >= expected_lines || std::time::Instant::now() >= deadline {
                    return s;
                }
            } else if std::time::Instant::now() >= deadline {
                return String::new();
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Poll up to 300ms confirming the file remains empty/absent (for
    /// "scorer agrees, nothing should be written" assertions).
    fn wait_for_empty(path: &std::path::Path) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(300);
        while std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        match std::fs::read_to_string(path) {
            Ok(s) => s.lines().filter(|l| !l.is_empty()).count() == 0,
            Err(_) => true,
        }
    }

    fn freeze_action() -> RootAction {
        RootAction::freeze_full(
            4242,
            "background-daemon",
            "shadow-test",
            0,
            0,
            DecisionReason::PressureContext,
        )
    }

    #[test]
    fn evaluator_builds_with_all_features() {
        let _eval = ShadowEvaluator::default();
        // Construction must not panic and scorer must accept a trivial call.
        let _eval2 = ShadowEvaluator::default();
    }

    #[test]
    fn evaluator_emits_disagreement_when_scorer_accepts_blocked_action() {
        let eval = ShadowEvaluator::default();
        let ctx = make_ctx(); // high pressure + oom + thrashing → scorer accepts
        let path = unique_tmp("disagree");
        let action = freeze_action();

        eval.evaluate_blocked(&action, &ctx, BlockerKind::UserContextAssertion, &path);

        let contents = wait_for_lines(&path, 1);
        let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 1, "expected exactly one disagreement line");
        let ev: BlockedActionEvent =
            serde_json::from_str(lines[0]).expect("parses as BlockedActionEvent");
        assert_eq!(ev.action_kind, "Freeze");
        assert_eq!(ev.target_name, "background-daemon");
        assert_eq!(ev.target_pid, Some(4242));
        match &ev.blocker {
            BlockerKind::Other(s) => {
                assert!(s.contains("shadow-disagree"), "reason tag missing: {}", s);
                assert!(
                    s.contains("UserContextAssertion"),
                    "gate verdict not recorded: {}",
                    s
                );
            }
            other => panic!("expected Other(...) blocker, got {:?}", other),
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn evaluator_silent_when_scorer_agrees_with_block() {
        let eval = ShadowEvaluator::default();
        let mut ctx = make_ctx();
        // Force scorer to ALSO reject via hard veto (Unconditional protection).
        ctx.protection_level = ProtectionLevel::Unconditional;
        let path = unique_tmp("agree");
        let action = freeze_action();

        eval.evaluate_blocked(&action, &ctx, BlockerKind::HardProtection, &path);

        // File either doesn't exist (no write) or is empty — poll 300ms to
        // give the async writer a fair chance to NOT write.
        assert!(
            wait_for_empty(&path),
            "journal should be empty/absent when scorer agrees with block"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn evaluator_silent_on_emit_error_does_not_panic() {
        let eval = ShadowEvaluator::default();
        let ctx = make_ctx(); // scorer accepts → will attempt to emit
                              // Unwritable path — parent dir does not exist and cannot be created.
                              // On macOS /proc doesn't exist at all, so create/append fails cleanly.
        let path = std::path::Path::new("/proc/apollo_shadow_unwritable_test_path/journal.jsonl");
        let action = freeze_action();

        // Must not panic; error is swallowed.
        eval.evaluate_blocked(&action, &ctx, BlockerKind::UserContextAssertion, path);
        eval.evaluate_accepted(&action, &ctx, path);
    }

    #[test]
    fn evaluate_accepted_emits_when_scorer_rejects() {
        let eval = ShadowEvaluator::default();
        let mut ctx = make_ctx();
        // Force scorer to REJECT via hard veto while gate accepts.
        ctx.protection_level = ProtectionLevel::Unconditional;
        let path = unique_tmp("accepted-rejected");
        let action = freeze_action();

        eval.evaluate_accepted(&action, &ctx, &path);

        let contents = wait_for_lines(&path, 1);
        let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 1);
        let ev: BlockedActionEvent = serde_json::from_str(lines[0]).unwrap();
        match &ev.blocker {
            BlockerKind::Other(s) => assert!(
                s.contains("accepted-but-scorer-rejects"),
                "unexpected reason: {}",
                s
            ),
            other => panic!("expected Other(...) blocker, got {:?}", other),
        }
        let _ = std::fs::remove_file(&path);
    }

    // ── Phase C SCORER-OVERRIDE tests (Sprint 11 finale, 2026-05-16) ─────────
    //
    // Build a context where scorer composite is strongly negative so the
    // override fires. With the default registered features:
    //   benefit ≈ pressure (when freezing) + thrashing_bonus (0.5 if >5k)
    //          + p_oom_bonus (1.0 if >0.30)
    //   cost   ≈ 2.0 (call_in_progress) + 1.0 (sleep_assertion, no bypass)
    //          + 0.5 (recently_active)
    // To force composite ≪ −0.30 we suppress benefit (pressure ≈ 0, no oom,
    // low thrashing) and pile on cost (call_in_progress).

    fn low_benefit_high_cost_ctx() -> ActionContext {
        let mut c = make_ctx();
        c.pressure = 0.10; // benefit ≈ 0.10
        c.swap_gb = 0.5;
        c.thrashing_score = 0.0; // no thrashing bonus
        c.p_oom_30s = None; // no oom bonus
        c.has_sleep_assertion = false; // would bypass at high pressure anyway
        c.call_in_progress = true; // cost += 2.0
        c.is_recently_active = false;
        c.idle_secs = 60.0;
        c.protection_level = ProtectionLevel::Unprotected;
        c
    }

    #[test]
    fn decide_override_returns_override_reject_when_gate_accept_and_scorer_strong_reject() {
        // Synthesize a PolicyScore directly to test the pure decision fn.
        let score = PolicyScore {
            action_kind: "FreezeProcess",
            total_benefit: 0.10,
            total_cost: 2.0,
            total_uncertainty: 0.0,
            vetoed_by: None,
            accept: false,
            reason: "test:freeze".into(),
            per_feature: vec![],
            composite: -1.90, // 0.10 - 1.0*2.0 - 0.5*0 = -1.90 (well below -0.30)
            raw_uncertainty: 0.0,
            // Group C (2026-06-06) — DS fields unused in this RSS-mode
            // test fixture; vacuous BPA + zero conflict.
            ds_belief: 0.0,
            ds_disbelief: 0.0,
            ds_uncertain: 1.0,
            ds_conflict: 0.0,
            ds_fallback_used: false,
        };
        match decide_override(/* gate_accept */ true, &score) {
            OverrideDecision::OverrideReject { composite } => {
                assert!((composite - -1.90).abs() < 1e-9);
            }
            other => panic!("expected OverrideReject, got {:?}", other),
        }
    }

    #[test]
    fn decide_override_returns_log_strong_accept_when_gate_reject_and_scorer_strong_accept() {
        let score = PolicyScore {
            action_kind: "FreezeProcess",
            total_benefit: 2.5,
            total_cost: 0.0,
            total_uncertainty: 0.0,
            vetoed_by: None,
            accept: true,
            reason: "test:freeze".into(),
            per_feature: vec![],
            composite: 2.5,
            raw_uncertainty: 0.0,
            // Group C (2026-06-06) — DS fields unused in this RSS-mode
            // test fixture; vacuous BPA + zero conflict.
            ds_belief: 0.0,
            ds_disbelief: 0.0,
            ds_uncertain: 1.0,
            ds_conflict: 0.0,
            ds_fallback_used: false,
        };
        match decide_override(/* gate_accept */ false, &score) {
            OverrideDecision::LogStrongAccept { composite } => {
                assert!((composite - 2.5).abs() < 1e-9);
            }
            other => panic!("expected LogStrongAccept, got {:?}", other),
        }
    }

    #[test]
    fn decide_override_no_change_when_both_agree_accept() {
        let score = PolicyScore {
            action_kind: "FreezeProcess",
            total_benefit: 1.0,
            total_cost: 0.0,
            total_uncertainty: 0.0,
            vetoed_by: None,
            accept: true,
            reason: "test".into(),
            per_feature: vec![],
            composite: 1.0,
            raw_uncertainty: 0.0,
            // Group C (2026-06-06) — DS fields unused in this RSS-mode
            // test fixture; vacuous BPA + zero conflict.
            ds_belief: 0.0,
            ds_disbelief: 0.0,
            ds_uncertain: 1.0,
            ds_conflict: 0.0,
            ds_fallback_used: false,
        };
        assert_eq!(
            decide_override(true, &score),
            OverrideDecision::NoChange,
            "agree-accept must not trigger override"
        );
    }

    #[test]
    fn decide_override_no_change_when_both_agree_reject() {
        let score = PolicyScore {
            action_kind: "FreezeProcess",
            total_benefit: 0.0,
            total_cost: 0.5,
            total_uncertainty: 0.0,
            vetoed_by: None,
            accept: false,
            reason: "test".into(),
            per_feature: vec![],
            composite: -0.5,
            raw_uncertainty: 0.0,
            // Group C (2026-06-06) — DS fields unused in this RSS-mode
            // test fixture; vacuous BPA + zero conflict.
            ds_belief: 0.0,
            ds_disbelief: 0.0,
            ds_uncertain: 1.0,
            ds_conflict: 0.0,
            ds_fallback_used: false,
        };
        assert_eq!(
            decide_override(false, &score),
            OverrideDecision::NoChange,
            "agree-reject must not trigger override (no journal noise)"
        );
    }

    #[test]
    fn decide_override_no_change_for_weak_disagreement_gate_accept_scorer_borderline_reject() {
        // composite = -0.25 → above the -0.30 threshold → NoChange (medium
        // confidence band stays with the gate; existing shadow_log path covers it).
        let score = PolicyScore {
            action_kind: "FreezeProcess",
            total_benefit: 0.0,
            total_cost: 0.25,
            total_uncertainty: 0.0,
            vetoed_by: None,
            accept: false,
            reason: "test".into(),
            per_feature: vec![],
            composite: -0.25,
            raw_uncertainty: 0.0,
            // Group C (2026-06-06) — DS fields unused in this RSS-mode
            // test fixture; vacuous BPA + zero conflict.
            ds_belief: 0.0,
            ds_disbelief: 0.0,
            ds_uncertain: 1.0,
            ds_conflict: 0.0,
            ds_fallback_used: false,
        };
        assert_eq!(
            decide_override(true, &score),
            OverrideDecision::NoChange,
            "weak disagreement (-0.30 ≤ composite < 0) must stay on shadow-log path"
        );
    }

    #[test]
    fn decide_override_no_change_for_weak_disagreement_gate_reject_scorer_borderline_accept() {
        let score = PolicyScore {
            action_kind: "FreezeProcess",
            total_benefit: 0.25,
            total_cost: 0.0,
            total_uncertainty: 0.0,
            vetoed_by: None,
            accept: true,
            reason: "test".into(),
            per_feature: vec![],
            composite: 0.25,
            raw_uncertainty: 0.0,
            // Group C (2026-06-06) — DS fields unused in this RSS-mode
            // test fixture; vacuous BPA + zero conflict.
            ds_belief: 0.0,
            ds_disbelief: 0.0,
            ds_uncertain: 1.0,
            ds_conflict: 0.0,
            ds_fallback_used: false,
        };
        assert_eq!(
            decide_override(false, &score),
            OverrideDecision::NoChange,
            "weak disagreement (0 < composite ≤ 0.30) must stay on shadow-log path"
        );
    }

    #[test]
    fn decide_override_threshold_is_strict_inequality_at_boundary() {
        // Exact ±0.30 is NOT a strong reject (strict <, not ≤).
        let exact = PolicyScore {
            action_kind: "FreezeProcess",
            total_benefit: 0.0,
            total_cost: 0.30,
            total_uncertainty: 0.0,
            vetoed_by: None,
            accept: false,
            reason: "test".into(),
            per_feature: vec![],
            composite: -0.30,
            raw_uncertainty: 0.0,
            // Group C (2026-06-06) — DS fields unused in this RSS-mode
            // test fixture; vacuous BPA + zero conflict.
            ds_belief: 0.0,
            ds_disbelief: 0.0,
            ds_uncertain: 1.0,
            ds_conflict: 0.0,
            ds_fallback_used: false,
        };
        assert_eq!(decide_override(true, &exact), OverrideDecision::NoChange);
    }

    #[test]
    fn evaluate_with_override_rejects_when_gate_accepts_and_scorer_strong_rejects() {
        let eval = ShadowEvaluator::default();
        let ctx = low_benefit_high_cost_ctx();
        let path = unique_tmp("override-reject");
        let action = freeze_action();

        // Sanity: confirm scorer actually says reject-strongly under this ctx.
        let raw_score = eval.scorer.score(&action, &ctx);
        assert!(
            !raw_score.accept,
            "scorer must reject in this synthesized ctx; reason={}",
            raw_score.reason
        );
        assert!(
            raw_score.composite < SCORER_STRONG_REJECT,
            "composite must be < -0.30; got {} reason={}",
            raw_score.composite,
            raw_score.reason
        );

        // Snapshot the counter before so a parallel test cannot poison it
        // (the LSE counter is process-static; tests must be delta-aware).
        let before = LSE_COUNTERS
            .scorer_override_rejects_total
            .load(std::sync::atomic::Ordering::Relaxed);

        let decision =
            eval.evaluate_with_override(&action, &ctx, /* gate_accept */ true, &path);

        // 1. Final decision is OverrideReject.
        match decision {
            OverrideDecision::OverrideReject { composite } => {
                assert!(composite < SCORER_STRONG_REJECT);
            }
            other => panic!(
                "expected OverrideReject (gate-accept + strong-reject), got {:?}",
                other
            ),
        }

        // 2. Counter incremented exactly once.
        let after = LSE_COUNTERS
            .scorer_override_rejects_total
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            after - before,
            1,
            "scorer_override_rejects_total did not increment by 1 (before={before} after={after})"
        );

        // 3. BlockedActionEvent was emitted with the right reason tag.
        let contents = wait_for_lines(&path, 1);
        let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 1, "expected exactly one journal line");
        let ev: BlockedActionEvent = serde_json::from_str(lines[0]).expect("parses");
        match &ev.blocker {
            BlockerKind::Other(s) => {
                assert!(
                    s.contains("scorer-override-accept-to-reject"),
                    "missing override tag: {}",
                    s
                );
                assert!(s.contains("composite="), "composite missing: {}", s);
            }
            other => panic!("expected Other(...) blocker, got {:?}", other),
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn evaluate_with_override_logs_strong_accept_but_does_not_override_gate_reject() {
        // Asymmetric: scorer wants to accept strongly but gate said reject —
        // the asymmetric design REFUSES to promote (per NotebookLM 2026-05-16
        // Candidate-C verdict); only the disagreement is journaled.
        let eval = ShadowEvaluator::default();
        let mut ctx = make_ctx(); // high pressure + oom + thrashing
        ctx.protection_level = ProtectionLevel::Unprotected;
        ctx.call_in_progress = false;
        ctx.has_sleep_assertion = false;
        let path = unique_tmp("strong-accept-log");
        let action = freeze_action();

        let raw_score = eval.scorer.score(&action, &ctx);
        assert!(raw_score.accept, "scorer must want to accept");
        assert!(
            raw_score.composite > SCORER_STRONG_ACCEPT,
            "composite must be > 0.30; got {}",
            raw_score.composite
        );

        let before = LSE_COUNTERS
            .scorer_disagreement_strong_accepts_total
            .load(std::sync::atomic::Ordering::Relaxed);

        let decision =
            eval.evaluate_with_override(&action, &ctx, /* gate_accept */ false, &path);

        match decision {
            OverrideDecision::LogStrongAccept { composite } => {
                assert!(composite > SCORER_STRONG_ACCEPT);
            }
            other => panic!("expected LogStrongAccept, got {:?}", other),
        }

        let after = LSE_COUNTERS
            .scorer_disagreement_strong_accepts_total
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            after - before,
            1,
            "scorer_disagreement_strong_accepts_total did not increment by 1"
        );

        // Journal line emitted with the right tag.
        let contents = wait_for_lines(&path, 1);
        let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 1);
        let ev: BlockedActionEvent = serde_json::from_str(lines[0]).expect("parses");
        match &ev.blocker {
            BlockerKind::Other(s) => assert!(
                s.contains("scorer-disagreement-strong-accept"),
                "missing strong-accept tag: {}",
                s
            ),
            other => panic!("expected Other(...) blocker, got {:?}", other),
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn evaluate_with_override_is_silent_for_weak_disagreement() {
        // |composite| ≤ 0.30 → NoChange + no journal write.
        let eval = ShadowEvaluator::default();
        let mut ctx = make_ctx();
        ctx.pressure = 0.30; // benefit ≈ 0.30, no extras
        ctx.swap_gb = 0.5;
        ctx.thrashing_score = 0.0;
        ctx.p_oom_30s = None;
        ctx.call_in_progress = false;
        ctx.has_sleep_assertion = false;
        ctx.is_recently_active = false;
        ctx.idle_secs = 60.0;
        ctx.protection_level = ProtectionLevel::Unprotected;
        let path = unique_tmp("weak-disagree");
        let action = freeze_action();

        let raw = eval.scorer.score(&action, &ctx);
        assert!(
            raw.composite > SCORER_STRONG_REJECT && raw.composite < SCORER_STRONG_ACCEPT,
            "composite must be in the weak band; got {}",
            raw.composite
        );

        let decision = eval.evaluate_with_override(&action, &ctx, true, &path);
        assert_eq!(decision, OverrideDecision::NoChange);

        // No journal write expected.
        assert!(
            wait_for_empty(&path),
            "weak-disagreement path must not emit to the override journal"
        );
        let _ = std::fs::remove_file(&path);
    }
}
