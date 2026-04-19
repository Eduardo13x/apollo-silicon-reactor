//! # Daemon Cognitive Tick
//!
//! Post-decision, pre-execution cognitive-layer processing.  Extracted from
//! the daemon main loop as part of the V1.1.0 Strangler Fig pass
//! [Fowler 2004].
//!
//! ## Three blocks
//!
//! 1. **Specialist voting + accuracy feedback** (Super Learner ensemble).
//!    Grades the previous cycle's specialist firing signals against the
//!    observed pressure delta, then assembles the current cycle's
//!    [`SpecialistVote`] bundle (LinUCB + Hazard + Monopoly + Kalman +
//!    Proactive-30s) and tallies the winning [`Intervention`].
//!
//! 2. **Habituation per-process state tracking** [Thompson & Spencer 1966].
//!    Buckets every process by (cpu_bucket, rss_bucket); processes whose
//!    bucket pair stays unchanged for ≥ `HABITUATION_THRESHOLD` cycles are
//!    skipped in `decide_actions`.  Dishabituation on any bucket change.
//!
//! 3. **User context "telepathy"** [Riva & Mantovani 2014].  Merges
//!    `IOHIDSystem` idle time, `pmset` sleep/call/audio assertions (polled
//!    every 3 cycles; carried forward between polls to avoid flicker), and
//!    SMC P-cluster temperature into a single [`UserContext`] value.
//!
//! ## Ordering invariants (preserved from original inline code)
//!
//! The three blocks are **not contiguous** in the main loop — a causal-graph
//! confidence-map block sits between habituation and user_context.  Each
//! block's call site therefore remains a standalone `let` in `main.rs`; this
//! module only provides the per-block logic, not a monolithic orchestrator.
//!
//! The original ordering is **specialist → habituation → user_context** and
//! is preserved verbatim.  Do not reorder without rerunning the
//! NotebookLM peer review: user_context arguably feeds LinUCB's context on
//! the *next* cycle and the current order is an intentional 1-cycle lag.
//!
//! ## Shared-state carry-overs
//!
//! All cross-cycle state is owned by the caller and passed in by `&mut`:
//!
//! - [`SpecialistFeedbackState`] — last cycle's firing signals + previous
//!   pressure for the accuracy tracker.
//! - `habituation_map: HashMap<u32, (u8, u8, u32)>` — per-pid bucket state.
//! - `last_user_assertions: (bool, bool, bool)` — cached pmset tuple.
//! - `last_specialist_votes` — disagreement outcome-feedback record.

use std::collections::{HashMap, HashSet};

use sysinfo::System;

use apollo_optimizer::engine::daemon_helpers::audit_log;
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::iokit_sensors::HardwareSnapshot;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::overflow_guard::OverflowThresholds;
use apollo_optimizer::engine::pipeline::learning_context::LearningContext;
use apollo_optimizer::engine::predictive_agent::{
    specialist, tally_votes, Intervention, SpecialistVote,
};
use apollo_optimizer::engine::lotka_volterra::StabilityRegime;
use apollo_optimizer::engine::signal_intelligence::SignalDigest;
use apollo_optimizer::engine::types::OptimizationProfile;
use apollo_optimizer::engine::user_context::UserContext;

/// Per-process habituation bucket window size: unchanged ≥ this many cycles
/// ⇒ habituated and skipped in `decide_actions`.
pub const HABITUATION_THRESHOLD: u32 = 5;

/// Cross-cycle state needed by the Super Learner accuracy feedback loop.
///
/// At the end of each cycle the daemon updates this with the *actual* firing
/// signals (`p_oom_30s > 0.30` etc.) so the next cycle can grade them
/// against the observed pressure spike.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecialistFeedbackState {
    pub prev_pressure_smooth: f64,
    pub prev_hazard_fired: bool,
    pub prev_monopoly_fired: bool,
    pub prev_kalman_fired: bool,
    pub prev_linucb_intervened: bool,
}

/// Output of [`apply_specialist_voting`] — the winning intervention plus the
/// raw disagreement record for next-cycle outcome feedback.
pub struct SpecialistVotingOutput {
    /// Intervention selected by the weighted ensemble (Observe on low-score
    /// disagreement).
    pub intervention: Intervention,
    /// If specialists disagreed this cycle, carries the `(votes, intervention)`
    /// tuple for the daemon to store in `last_specialist_votes` so Loop 3 can
    /// issue outcome feedback next cycle.  `None` when consensus.
    pub disagreement_record: Option<(Vec<SpecialistVote>, Intervention)>,
}

/// Super Learner specialist voting + accuracy feedback.
///
/// Runs once per cycle, **after** `PredictiveAgent::select_action_with_confidence`
/// has produced `(linucb_choice, linucb_confidence)` and **before**
/// `decision_stage::run`.  Mutates `feedback` in-place so the next cycle can
/// read back the actual firing signals.
///
/// Side effects: on disagreement, emits a `specialist_disagreement` audit
/// line; on `SuggestAggressive`, sets a 5-minute governor override via
/// `state.policy`.
#[allow(clippy::too_many_arguments)]
pub fn apply_specialist_voting(
    state: &SharedState,
    lctx: &mut LearningContext<'_>,
    signal_digest: &SignalDigest,
    feedback: &mut SpecialistFeedbackState,
    overflow_thresholds: &mut OverflowThresholds,
    linucb_choice: Intervention,
    linucb_confidence: f64,
) -> SpecialistVotingOutput {
    // ── Specialist accuracy feedback (Super Learner) ─────────────────
    // Compare prev cycle's ACTUAL specialist signals against observed outcome.
    // Using real firing conditions (not pressure proxies) ensures the tracker
    // measures what the specialist actually predicted, not a heuristic stand-in.
    // A spike is a pressure rise of ≥0.08 over the previous cycle.
    {
        let pressure_spiked =
            signal_digest.pressure_smooth >= feedback.prev_pressure_smooth + 0.08;
        // Hazard: did prev cycle's hazard specialist fire (p_oom_30s > 0.30)?
        let hazard_correct = (feedback.prev_hazard_fired && pressure_spiked)
            || (!feedback.prev_hazard_fired && !pressure_spiked);
        lctx.specialist_accuracy
            .update(specialist::HAZARD, hazard_correct);

        // Monopoly: did prev cycle's monopoly specialist fire (monopoly_risk > 0.5)?
        let monopoly_correct = (feedback.prev_monopoly_fired && pressure_spiked)
            || (!feedback.prev_monopoly_fired && !pressure_spiked);
        lctx.specialist_accuracy
            .update(specialist::MONOPOLY, monopoly_correct);

        // Kalman: did prev cycle's Kalman predict spike (pressure_predicted_5s > 0.85)?
        let kalman_correct = (feedback.prev_kalman_fired && pressure_spiked)
            || (!feedback.prev_kalman_fired && !pressure_spiked);
        lctx.specialist_accuracy
            .update(specialist::KALMAN, kalman_correct);

        // LinUCB: voted for non-Observe intervention. Correct if pressure spiked.
        let linucb_correct = (feedback.prev_linucb_intervened && pressure_spiked)
            || (!feedback.prev_linucb_intervened && !pressure_spiked);
        lctx.specialist_accuracy
            .update(specialist::LINUCB, linucb_correct);
    }
    // Save current cycle's actual specialist firing signals for next cycle's feedback.
    feedback.prev_pressure_smooth = signal_digest.pressure_smooth;
    feedback.prev_hazard_fired = signal_digest.p_oom_30s > 0.30;
    feedback.prev_monopoly_fired = signal_digest.monopoly_risk > 0.5;
    feedback.prev_kalman_fired = signal_digest.pressure_predicted_5s > 0.85;
    feedback.prev_linucb_intervened = linucb_choice != Intervention::Observe;

    // ── Specialist voting: weighted ensemble replaces override chain ──
    // Confidences are modulated by learned accuracy weights (Super Learner).
    // SpecialistAccuracyTracker EMA-tracks per-specialist correctness;
    // a specialist consistently right gets weight→1.0, wrong gets→0.0.
    let mut votes = vec![
        // LinUCB: primary agent — UCB confidence × learned accuracy weight.
        // linucb_confidence is the normalized margin of the winning arm [0.5, 1.0]:
        // dominant winner → near 1.0, all arms tied → 0.5.
        SpecialistVote {
            name: "linucb",
            intervention: linucb_choice,
            confidence: linucb_confidence
                * lctx.specialist_accuracy.weight(specialist::LINUCB),
        },
    ];

    // Hazard specialist: high P(OOM) → use MPC recommendation.
    if signal_digest.p_oom_30s > 0.30 {
        votes.push(SpecialistVote {
            name: "hazard",
            intervention: Intervention::from_index(signal_digest.mpc_recommendation),
            confidence: signal_digest.p_oom_30s.min(1.0)
                * lctx.specialist_accuracy.weight(specialist::HAZARD),
        });
    }

    // Monopoly specialist: one process hogging RAM → throttle noise.
    // Ecological instability (Jacobian eigenvalue sign) amplifies confidence:
    // an Unstable regime means competition dynamics are diverging — act sooner.
    // [Strogatz 2015 §6.4 + Pei Wang 2013 §3.3.1]
    if signal_digest.monopoly_risk > 0.5 {
        let stability_boost = match signal_digest.stability_regime {
            StabilityRegime::Unstable => 1.15_f64,
            StabilityRegime::UnstableSaddle => 1.08_f64,
            _ => 1.0_f64,
        };
        let confidence = (signal_digest.monopoly_risk.min(1.0)
            * lctx.specialist_accuracy.weight(specialist::MONOPOLY)
            * stability_boost)
            .min(1.0);
        // Log NARS maturity horizon for monopoly belief (observations to 0.80 confidence).
        if let Some(rem) = lctx.outcome_tracker.drift_detector.observations_remaining("monopoly_freeze", 0.80) {
            if rem > 0 {
                audit_log(&serde_json::json!({
                    "t": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                    "event": "nars_maturity_horizon",
                    "belief": "monopoly_freeze",
                    "obs_remaining_to_0.80": rem,
                    "stability_regime": format!("{:?}", signal_digest.stability_regime),
                }));
            }
        }
        votes.push(SpecialistVote {
            name: "monopoly",
            intervention: Intervention::PreThrottleNoise,
            confidence,
        });
    }

    // Kalman specialist: predicted pressure spike → tighten.
    if signal_digest.pressure_predicted_5s > 0.85 {
        votes.push(SpecialistVote {
            name: "kalman",
            intervention: Intervention::TightenThresholds,
            confidence: (signal_digest.pressure_predicted_5s - 0.85).min(0.15) / 0.15
                * lctx.specialist_accuracy.weight(specialist::KALMAN),
        });
    }

    // Proactive-30s specialist: Kalman projects overflow in ~30s but we're
    // still below the action threshold — act NOW before RAM fills up.
    // This is the key advantage over purely reactive systems:
    // the OS can only react; Apollo can predict and pre-empt.
    let p30_trigger = overflow_thresholds.bg_pressure as f64 - 0.05;
    let p30_clear = overflow_thresholds.bg_pressure as f64 - 0.08;
    if signal_digest.pressure_predicted_30s > p30_trigger
        && signal_digest.pressure_smooth < p30_clear
    {
        let strength = ((signal_digest.pressure_predicted_30s - p30_trigger) / 0.10)
            .clamp(0.0, 1.0);
        votes.push(SpecialistVote {
            name: "proactive-30s",
            intervention: Intervention::TightenThresholds,
            confidence: strength * lctx.specialist_accuracy.weight(specialist::KALMAN),
        });
    }

    let vote_result = tally_votes(&votes);
    let intervention = vote_result.intervention;

    // Loop 3: store votes for disagreement outcome feedback next cycle.
    let disagreement_record = if vote_result.had_disagreement {
        Some((votes.clone(), intervention))
    } else {
        None
    };

    // Cable: had_disagreement → conservative safety route.
    // When specialists disagree AND the winning score is weak (<0.4),
    // the signal is ambiguous. Fall back to Observe instead of risking
    // a wrong aggressive action. Only override if not in survival mode.
    let intervention = if vote_result.had_disagreement {
        audit_log(&serde_json::json!({
            "t": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            "event": "specialist_disagreement",
            "winner": format!("{:?}", intervention),
            "score": (vote_result.winning_score * 100.0).round() / 100.0,
            "n_votes": votes.len(),
            "pressure": (signal_digest.pressure_smooth * 1000.0).round() / 1000.0,
        }));
        if vote_result.winning_score < 0.4 && signal_digest.pressure_smooth < 0.80 {
            // Low confidence + not critical pressure → play it safe.
            Intervention::Observe
        } else {
            intervention
        }
    } else {
        intervention
    };

    // Apply threshold tightening if selected.
    *overflow_thresholds = lctx.predictive_agent.adjust_thresholds(*overflow_thresholds);

    // SuggestAggressive: set a 5-minute manual override to aggressive profile.
    if intervention == Intervention::SuggestAggressive {
        let mut pg = state.policy.lock_recover();
        if pg.governor.manual_override.is_none() {
            pg.governor.set_manual_override(
                OptimizationProfile::AggressiveRoot,
                5,
                "predictive-agent: proactive pressure mitigation".to_string(),
            );
        }
    }

    SpecialistVotingOutput {
        intervention,
        disagreement_record,
    }
}

/// Habituation per-process state tracking.
///
/// Buckets every process by `(cpu_usage/5%, rss/50MB)` and increments an
/// unchanged-count when both buckets hold steady; resets (dishabituation)
/// on any change.  Processes whose counter reaches
/// [`HABITUATION_THRESHOLD`] land in the returned set and are skipped by
/// `decide_actions`.
///
/// Mutates `habituation_map` in-place; GCs dead PIDs every 100 cycles.
/// Also bumps `metrics.habituation_skips` by the returned set's size so
/// the AIS runtime benchmark can read it.
pub fn update_habituation_state(
    state: &SharedState,
    system: &System,
    habituation_map: &mut HashMap<u32, (u8, u8, u32)>,
    cycle_count: u64,
) -> HashSet<u32> {
    let habituated_pids: HashSet<u32> = {
        let mut hab_set = HashSet::new();
        for (pid, process) in system.processes() {
            let pid_u32 = pid.as_u32();
            let cpu_bucket = (process.cpu_usage() / 5.0) as u8;
            let rss_bucket = (process.memory() / (50 * 1024 * 1024)) as u8;
            match habituation_map.get_mut(&pid_u32) {
                Some(entry) => {
                    if entry.0 == cpu_bucket && entry.1 == rss_bucket {
                        entry.2 += 1; // unchanged
                        if entry.2 >= HABITUATION_THRESHOLD {
                            hab_set.insert(pid_u32);
                        }
                    } else {
                        // Dishabituation: state changed.
                        *entry = (cpu_bucket, rss_bucket, 0);
                    }
                }
                None => {
                    habituation_map.insert(pid_u32, (cpu_bucket, rss_bucket, 0));
                }
            }
        }
        // GC dead PIDs every 100 cycles.
        if cycle_count % 100 == 0 {
            let live: HashSet<u32> = system.processes().keys().map(|p| p.as_u32()).collect();
            habituation_map.retain(|pid, _| live.contains(pid));
        }
        hab_set
    };
    // Emit habituation count so AIS runtime benchmark can read it.
    {
        let mut m = state.metrics.lock_recover();
        m.metrics.habituation_skips += habituated_pids.len() as u64;
    }
    habituated_pids
}

/// User context "telepathy" — infer what the user is doing right now.
///
/// - `idle_secs` from `IOHIDSystem HIDIdleTime` — fast ioreg call, safe every
///   cycle.
/// - Sleep assertions + call + audio from `pmset` — amortised: polled every
///   3 cycles (`cycle_count % 3 == 0`).  On non-poll cycles the last known
///   tuple is carried forward via `last_user_assertions` to prevent
///   `freeze_gate` flicker [Cook et al. 2019].
/// - P-cluster temperature from the latest SMC snapshot: > 75 °C and not
///   long-idle ⇒ clamp `idle_secs ≤ 10.0` so thermal headroom is preserved
///   for the user's workload.
///
/// [Riva & Mantovani 2014] idle time + media state = highest-signal context
/// cues for user presence.
pub fn compute_user_context(
    cycle_count: u64,
    last_user_assertions: &mut (bool, bool, bool),
    cycle_hw_snap: Option<&HardwareSnapshot>,
) -> UserContext {
    // Poll pmset every 3 cycles (~9s) — balances subprocess cost vs
    // responsiveness (call starts → detected within 9s, not 15s).
    let collect_assertions = cycle_count % 3 == 0;
    let mut ctx = UserContext::collect(collect_assertions);
    // Merge: on non-assertion cycles, carry forward last known state.
    // Prevents freeze_gate from flickering between "user-protected" and
    // "delta/committed" every cycle. [Cook et al. 2019]
    if collect_assertions {
        *last_user_assertions = (
            ctx.has_sleep_assertion,
            ctx.call_in_progress,
            ctx.audio_active,
        );
    } else {
        ctx.has_sleep_assertion = last_user_assertions.0;
        ctx.call_in_progress = last_user_assertions.1;
        ctx.audio_active = last_user_assertions.2;
    }
    // Merge cpu_temp from hw_snapshot (already in RuntimeMetrics).
    // If P-cluster temp > 75°C, treat as if more active (raise pressure gate)
    // so Apollo conserves thermal headroom for the user's workload.
    if let Some(hw) = cycle_hw_snap {
        if let Some(p_temp) = hw.temps.p_cluster_celsius {
            if p_temp > 75.0 && !ctx.is_idle_long() {
                // Simulate "recently active" to raise freeze gates and
                // protect thermal headroom — overrides any idle signal.
                ctx.idle_secs = ctx.idle_secs.min(10.0);
            }
        }
    }
    ctx
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn habituation_threshold_is_five() {
        assert_eq!(HABITUATION_THRESHOLD, 5);
    }

    #[test]
    fn specialist_feedback_defaults_are_neutral() {
        let fb = SpecialistFeedbackState::default();
        assert_eq!(fb.prev_pressure_smooth, 0.0);
        assert!(!fb.prev_hazard_fired);
        assert!(!fb.prev_monopoly_fired);
        assert!(!fb.prev_kalman_fired);
        assert!(!fb.prev_linucb_intervened);
    }

    #[test]
    fn user_context_carry_forward_preserves_assertions() {
        // cycle 1 is a non-poll cycle (cycle % 3 != 0); expect last-known carried.
        let mut last = (true, true, false);
        let ctx = compute_user_context(1, &mut last, None);
        assert!(ctx.has_sleep_assertion);
        assert!(ctx.call_in_progress);
        assert!(!ctx.audio_active);
    }
}
