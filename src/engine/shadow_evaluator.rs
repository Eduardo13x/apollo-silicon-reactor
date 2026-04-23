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
    ActionContext, PolicyScorer, PressureBenefitFeature, ProtectionFeature,
    UserDisruptionCostFeature,
};
use crate::engine::blocked_action_journal::{emit_async, BlockedActionEvent, BlockerKind};
use crate::engine::policy_feature_deep_scan::DeepScanCostFeature;
use crate::engine::policy_feature_predictive::PredictiveBenefitFeature;
use crate::engine::policy_feature_sensor_age::SensorAgeFeature;
use crate::engine::types::RootAction;

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

    /// Called when gate tower ACCEPTS a candidate. Runs scorer; if scorer would have
    /// rejected, emits a disagreement event. Offline analysis correlates with outcomes.
    #[allow(dead_code)] // wiring for accepted-case follows in a later commit
    pub fn evaluate_accepted(
        &self,
        action: &RootAction,
        ctx: &ActionContext,
        journal_path: &Path,
    ) {
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
        RootAction::SetSysctl { .. } => "SetSysctl",
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
        RootAction::SetSysctl { key, .. } => key.clone(),
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
        RootAction::SetSysctl { .. }
        | RootAction::ToggleSpotlight { .. }
        | RootAction::QuarantineDaemon { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        RootAction::freeze_full(4242, "background-daemon", "shadow-test", 0, 0)
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
}
