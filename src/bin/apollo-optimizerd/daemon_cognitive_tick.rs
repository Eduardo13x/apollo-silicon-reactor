//! # Daemon Cognitive Tick
//!
//! Post-decision, pre-execution cognitive-layer processing.  Extracted from
//! the daemon main loop as part of the V1.1.0 Strangler Fig pass
//! [Fowler 2004].
//!
//! ## Wave 38 Note: Distinction from `cognitive_tick.rs`
//!
//! Despite the similar naming, this module is **not** a duplicate of `cognitive_tick.rs`.
//! `cognitive_tick.rs` contains the core neurocognitive state pipeline (RewardBus,
//! MetaCognition, EpistemicUncertainty, etc.). This file (`daemon_cognitive_tick.rs`)
//! contains disjoint pre-execution logic (Specialist Voting, Habituation, User Context)
//! that was extracted from `main.rs`. Both are intentional and co-exist safely.
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

use apollo_engine::engine::daemon_helpers::audit_log;
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::iokit_sensors::HardwareSnapshot;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::lotka_volterra::StabilityRegime;
use apollo_engine::engine::maintenance_state::MaintenanceState;
use apollo_engine::engine::nars_belief::TruthValue;
use apollo_engine::engine::overflow_guard::OverflowThresholds;
use apollo_engine::engine::pipeline::learning_context::LearningContext;
use apollo_engine::engine::predictive_agent::{
    specialist, tally_votes, Intervention, SpecialistVote,
};
use apollo_engine::engine::signal_intelligence::SignalDigest;
use apollo_engine::engine::types::OptimizationProfile;
use apollo_engine::engine::workload_classifier::WorkloadMode;

/// NARS belief confidence target for monopoly_freeze maturity gate.
/// At c=0.80 the belief carries enough evidence to act on without
/// second-guessing [Pei Wang 2013 §3.3.1].
const MONOPOLY_BELIEF_CONFIDENCE_TARGET: f32 = 0.80;

/// Floor for maturity factor — even with zero evidence, keep a minimal
/// vote weight so pathological monopoly_risk > 0.5 never gets silenced
/// completely [Gray & Reuter 1992 §11, evidence-gated decisions].
const MONOPOLY_MATURITY_FLOOR: f64 = 0.4;
use apollo_engine::engine::user_context::UserContext;

/// Per-process habituation bucket window size: unchanged ≥ this many cycles
/// ⇒ habituated and skipped in `decide_actions`.
pub const HABITUATION_THRESHOLD: u32 = 5;

/// Phase 5.1 wiring (2026-05-16) — inputs for the user-presence modulator
/// applied inside [`apply_specialist_voting`].
///
/// All four signals are sampled from **last cycle's** [`UserContext`] plus
/// the current cycle's `ArousalState::level`. The one-cycle lag on
/// idle/audio/sleep_assertion is intentional and harmless: the
/// `compute_user_context` block runs strictly *after*
/// `apply_specialist_voting` in the daemon main loop ordering (see the
/// "Ordering invariants" doc-block at the top of this file), so the current
/// cycle's UserContext is not yet known when we need it for voting. A user
/// who was typing 1 cycle ago (~80 ms) is overwhelmingly still typing now;
/// the same logic the Phase 0c idle interpolation already relies on.
///
/// Defaults to "no suppression": `idle_seconds=120.0` (idle tier),
/// `hid_events_per_minute=0.0`, `audio_active=false`, `has_sleep_assertion=false`,
/// `arousal=0.0`. The first cycle of the daemon therefore returns
/// `IDLE_MULTIPLIER` from `user_presence_modulator_narrowed_no_counter` and
/// does not modulate votes. Subsequent cycles use real values.
///
/// `hid_events_per_minute` currently always 0.0 — no daemon-side accessor
/// exists yet (TODO 2026-05-16). The 0.0 value cannot trigger the
/// HID-rate clause; the modulator still operates correctly off `idle_seconds`
/// alone. When a future activity-sensor lands, this field will pick up the
/// real rate automatically.
#[derive(Debug, Default, Clone, Copy)]
pub struct PresenceInputs {
    pub idle_seconds: f64,
    pub hid_events_per_minute: f64,
    pub current_arousal: f64,
    pub audio_active: bool,
    pub has_sleep_assertion: bool,
    /// Phase 5.1.1 production fix (2026-05-16) — raw memory pressure used
    /// by the modulator's critical-pressure bypass. When `>= 0.65` the
    /// modulator returns 1.0 (no suppression) regardless of HID activity,
    /// so memory-survival actions are not zombified during interactive
    /// sessions. See `user_presence::CRITICAL_PRESSURE_BYPASS` for the
    /// empirical motivation (Score 0.85 + 0 actions cascade paralysis).
    pub memory_pressure: f64,
}

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

/// Phase 5.1 wiring — apply `presence_factor` to a vote bundle in place.
///
/// Returns the number of non-Observe votes that were modulated (caller uses
/// this to drive the `user_presence_suppressions_total` counter on a
/// per-modulated-vote basis, per NotebookLM 2026-05-16, Q2). Pure function
/// over the vote slice — no I/O, no global mutation. Factored out of
/// [`apply_specialist_voting`] for unit-testability without constructing the
/// full [`LearningContext`] graph.
///
/// Contract:
///   - `factor` is the value already produced by
///     `user_presence_modulator_narrowed_no_counter`, so this helper does
///     NOT itself decide whether the user is present.
///   - When `factor` ≈ 1.0 the helper is a no-op and returns 0.
///   - Otherwise non-Observe votes have their `confidence` multiplied by
///     `factor` and clamped to `[0.0, 1.0]`; Observe votes are untouched
///     (a "do nothing" vote must not be further suppressed — see the
///     comment block in `apply_specialist_voting` for the rationale).
#[inline]
pub fn apply_presence_factor(votes: &mut [SpecialistVote], factor: f64) -> u64 {
    if (factor - 1.0).abs() <= f64::EPSILON {
        return 0;
    }
    let mut modulated = 0_u64;
    for v in votes.iter_mut() {
        if v.intervention != Intervention::Observe {
            v.confidence = (v.confidence * factor).clamp(0.0, 1.0);
            modulated += 1;
        }
    }
    modulated
}

/// Phase 3.1 — Skill-Aware Prediction confidence multiplier.
///
/// Maps the workload-conditional skill success signal `s ∈ [0, 1]` into a
/// multiplicative factor for non-Observe specialist votes:
///
/// * `None`          → `1.0` (no reliable evidence yet — leave votes neutral)
/// * `Some(0.0)`     → `0.85` (max damp: all matched skills failed)
/// * `Some(0.5)`     → `1.00` (neutral: 50/50 history)
/// * `Some(1.0)`     → `1.15` (max boost: all matched skills succeeded)
///
/// Band intentionally narrow ([0.85, 1.15]) after NotebookLM 2026-05-16 adversarial
/// review: a wider [0.7, 1.3] band stacks multiplicatively with
/// `SpecialistAccuracyTracker::weight()` (≈0.6–1.0) and the per-specialist signal
/// (0.0–1.0) into a "cascade attenuation" — three factors at low end produce
/// ≈0.33, dragging votes below the disagreement-safety floor and forcing Observe
/// on M1 8GB regardless of physical pressure. ±15% keeps the tilt informative
/// without losing dynamic range. [Sutton 2018 §2.5 — pessimism initialisation]
#[inline]
fn skill_aware_factor(signal: Option<f32>) -> f64 {
    match signal {
        Some(s) => 0.85 + 0.30 * (s as f64),
        None => 1.0,
    }
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
    cycle_count: u64,
    // G13 — Epistemic Arbiter: ODE T_sat urgency boosts hazard vote 1.5× when > 0.5.
    // When ODE physics predict imminent saturation, the hazard specialist's confidence
    // is amplified to resolve Lotka-Volterra vs ODE-swap specialist ties decisively.
    // [Kuncheva 2004 §5.2 — ensemble arbitration under conflicting specialist signals]
    ode_t_sat_urgency: f64,
    // Phase 3.1 — Skill-Aware Prediction: current workload feeds the
    // `SkillRegistry::workload_success_signal` lookup, which modulates non-Observe
    // specialist votes by the historical success rate of past throttle-class
    // actions in this workload context.
    workload_mode: WorkloadMode,
    // Phase 5.1 wiring (2026-05-16) — last cycle's UserContext + current
    // arousal, used to scale non-Observe vote confidences by the narrowed
    // user-presence multiplier ∈ [0.7, 1.0]. See [`PresenceInputs`].
    presence_inputs: PresenceInputs,
    // Phase 4.3.1 — Specialist accuracy purge inhibition (Sprint 8, 2026-05-16).
    // SharedState has no `maintenance` field; the daemon main loop owns the
    // `MaintenanceState` value and threads it down (mirrors learning_tick.rs
    // signature). When `is_purge_recent(30)` is true the EMA accuracy update
    // block is skipped — see body for full rationale.
    maintenance_state: &MaintenanceState,
) -> SpecialistVotingOutput {
    // ── Specialist accuracy feedback (Super Learner) ─────────────────
    // Compare prev cycle's ACTUAL specialist signals against observed outcome.
    // Using real firing conditions (not pressure proxies) ensures the tracker
    // measures what the specialist actually predicted, not a heuristic stand-in.
    // A spike is a pressure rise of ≥0.08 over the previous cycle.
    //
    // Phase 4.3.1 — purge-inhibition for specialist accuracy (2026-05-16).
    // A maintenance purge causes pressure to drop. Without this guard,
    // hazard/monopoly/kalman specialists who predicted a spike get marked
    // "wrong" because the pressure dropped (due to purge, not their prediction
    // being incorrect). Their EMA weights would depress; next real crisis they
    // react weaker. Mirrors the inhibition pattern Phase 2 added for
    // outcome_tracker + causal_graph post-purge.
    // [Rubin 1974] intervention vs confounder.
    if !maintenance_state.is_purge_recent(30) {
        let pressure_spiked = signal_digest.pressure_smooth >= feedback.prev_pressure_smooth + 0.08;
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
    } else {
        // Surface the inhibition in runtime_metrics.json so we can verify the
        // guard is firing in prod (mirrors Phase 3.1 `skill_aware_modulations_total`).
        apollo_engine::engine::lse_counters::LSE_COUNTERS
            .inc_specialist_accuracy_purge_inhibitions();
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
            confidence: linucb_confidence * lctx.specialist_accuracy.weight(specialist::LINUCB),
        },
    ];

    // Hazard specialist: high P(OOM) → use MPC recommendation.
    // G13: when ODE urgency > 0.5, boost confidence 1.5× to resolve arbiter ties
    // decisively in favour of the physics-informed signal.
    // [Kuncheva 2004 §5.2 — ensemble arbitration under conflicting specialist signals]
    if signal_digest.p_oom_30s > 0.30 {
        let ode_boost = if ode_t_sat_urgency > 0.5 {
            1.5_f64
        } else {
            1.0_f64
        };
        votes.push(SpecialistVote {
            name: "hazard",
            intervention: Intervention::from_index(signal_digest.mpc_recommendation),
            confidence: (signal_digest.p_oom_30s.min(1.0)
                * lctx.specialist_accuracy.weight(specialist::HAZARD)
                * ode_boost)
                .min(1.0),
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
        // NARS belief maturity gate: scale confidence by accumulated evidence.
        // Young beliefs get floor-weight (0.4); mature beliefs get full weight.
        // [Pei Wang 2013 §3.3.1, Gray & Reuter 1992 §11]
        let obs_remaining = lctx.outcome_tracker.drift_detector.observations_remaining(
            specialist::NAMES[specialist::MONOPOLY],
            MONOPOLY_BELIEF_CONFIDENCE_TARGET,
        );
        let maturity_factor = monopoly_maturity_factor(obs_remaining);
        let confidence = (signal_digest.monopoly_risk.min(1.0)
            * lctx.specialist_accuracy.weight(specialist::MONOPOLY)
            * stability_boost
            * maturity_factor)
            .min(1.0);
        // Log NARS maturity horizon every 30 cycles to avoid journal spam.
        if cycle_count.is_multiple_of(30) {
            if let Some(rem) = obs_remaining {
                if rem > 0 {
                    audit_log(&serde_json::json!({
                        "t": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                        "event": "nars_maturity_horizon",
                        "belief": specialist::NAMES[specialist::MONOPOLY],
                        "obs_remaining_to_0.80": rem,
                        "maturity_factor": (maturity_factor * 1000.0).round() / 1000.0,
                        "stability_regime": format!("{:?}", signal_digest.stability_regime),
                    }));
                }
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
    let p30_trigger = overflow_thresholds.bg_pressure - 0.05;
    let p30_clear = overflow_thresholds.bg_pressure - 0.08;
    if signal_digest.pressure_predicted_30s > p30_trigger
        && signal_digest.pressure_smooth < p30_clear
    {
        let strength =
            ((signal_digest.pressure_predicted_30s - p30_trigger) / 0.10).clamp(0.0, 1.0);
        votes.push(SpecialistVote {
            name: "proactive-30s",
            intervention: Intervention::TightenThresholds,
            confidence: strength * lctx.specialist_accuracy.weight(specialist::KALMAN),
        });
    }

    // Phase 3.1 — Skill-Aware Prediction (Sprint 6).
    //
    // Modulate non-Observe votes by the workload-conditional skill success
    // signal: in workloads where past throttle-class actions have empirically
    // worked, specialist votes amplify; where they have failed, votes damp.
    //
    // Factor map: signal s ∈ [0, 1] → factor ∈ [0.7, 1.3], neutral 1.0 at s=0.5.
    // No reliable skill matching → factor stays 1.0 (no signal, no change).
    //
    // Observe votes are intentionally NOT modulated: a "do nothing" vote should
    // not be penalised because past actions failed — that would create a
    // feedback loop where the system can't recover its own confidence after
    // a bad run. Only positive-action votes are graded against skill history.
    let skill_factor = skill_aware_factor(
        lctx.skill_registry.workload_success_signal(workload_mode.as_str()),
    );
    if (skill_factor - 1.0).abs() > f64::EPSILON {
        let mut modulated = 0_u64;
        for v in votes.iter_mut() {
            if v.intervention != Intervention::Observe {
                v.confidence = (v.confidence * skill_factor).clamp(0.0, 1.0);
                modulated += 1;
            }
        }
        if modulated > 0 {
            apollo_engine::engine::lse_counters::LSE_COUNTERS
                .add_skill_aware_modulations(modulated);
        }
    }

    // Phase 5.1 wiring (2026-05-16) — User-Presence + passive-content
    // suppression. Mirrors the Phase 3.1 skill-aware pattern: a multiplier
    // ∈ [0.7, 1.0] (narrowed band, per NotebookLM 2026-05-16) scales each
    // non-Observe vote's confidence. Observe votes are intentionally NOT
    // modulated, identical reasoning to the skill-aware block: a "do
    // nothing" vote should not be further suppressed by user-presence — that
    // would let an active user permanently silence the system, with no way
    // for the cognitive layer to recover its confidence.
    //
    // Counter `user_presence_suppressions_total` fires once per modulated
    // vote (NotebookLM 2026-05-16, Q2 — per-vote granularity matches the
    // skill_aware_modulations pattern and gives D1 Decision Precision
    // enough signal density to diagnose which specialist paths are being
    // damped).
    //
    // Wiring site chosen over `decide_actions.rs` cost composition per
    // NotebookLM 2026-05-16, Q3 — user presence is a cognitive "telepathy"
    // signal that modulates *confidence* (System 1 / System 2 separation
    // [Kahneman 2011]), not a physical action cost. Folding it into
    // decide_actions would conflate cognitive uncertainty with execution
    // gates and re-introduce the God-Node coupling pattern flagged by NARS
    // belief B001 / B008.
    let presence_factor =
        apollo_engine::engine::user_presence::user_presence_modulator_narrowed_no_counter(
            presence_inputs.idle_seconds,
            presence_inputs.hid_events_per_minute,
            presence_inputs.current_arousal,
            presence_inputs.audio_active,
            presence_inputs.has_sleep_assertion,
            presence_inputs.memory_pressure,
        );
    let presence_modulated = apply_presence_factor(&mut votes, presence_factor);
    if presence_modulated > 0 {
        apollo_engine::engine::lse_counters::LSE_COUNTERS
            .add_user_presence_suppressions(presence_modulated);
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
    *overflow_thresholds = lctx
        .predictive_agent
        .adjust_thresholds(*overflow_thresholds);

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
    // Retained for ABI stability across the daemon's many call sites; the
    // previous body needed `state.metrics.lock_recover()` to bump
    // `habituation_skips`. Phase 2 god-lock decomposition (2026-05-16)
    // moved that write to an LSE atomic, so the parameter is currently
    // unused. Removing it is a separate commit (touches main.rs:3102 and
    // every test caller).
    _state: &SharedState,
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
        if cycle_count.is_multiple_of(100) {
            let live: HashSet<u32> = system.processes().keys().map(|p| p.as_u32()).collect();
            habituation_map.retain(|pid, _| live.contains(pid));
        }
        hab_set
    };
    // Emit habituation count so AIS runtime benchmark can read it.
    // Phase 2 god-lock decomposition (2026-05-16): migrated from
    // `state.metrics.lock_recover().metrics.habituation_skips += N` to a
    // lock-free LSE counter. The legacy `RuntimeMetrics.habituation_skips`
    // field is populated FROM this atomic in
    // `daemon_state::sync_from_lockfree` — single source of truth.
    apollo_engine::engine::lse_counters::LSE_COUNTERS
        .add_habituation_skips(habituated_pids.len() as u64);
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
    last_idle_sample: &mut Option<(f64, std::time::Instant)>,
    cycle_hw_snap: Option<&HardwareSnapshot>,
) -> UserContext {
    // Phase 0c performance fix (Profile evidence 2026-05-10):
    // user_context was 44ms avg = 51% of REASON stage. Root cause: ioreg
    // subprocess (25ms) fired every cycle, pmset (13ms) every 3rd. The
    // user can't perceive a 50s idle-time error, so cadence is throttled
    // and intermediate samples are interpolated.
    //
    //   ioreg  (idle time)       : every 10 cycles → was every cycle
    //   pmset  (sleep assertions): every 15 cycles → was every 3
    //   Between samples:
    //     idle_secs += elapsed since last sample (continuous)
    //     assertions: carry-forward
    //
    // Acceptable error: idle_secs ±5 s, assertions ±15 s detection lag.
    // Both well under the 15 s recently_active / 120 s idle_long thresholds.
    let sample_idle = cycle_count.is_multiple_of(10);
    let collect_assertions = cycle_count.is_multiple_of(15);
    // UserContext::collect always runs ioreg internally; on non-sample
    // cycles we still call it (cheap path elides ioreg via collect_idle_secs
    // fallback default 30.0) — better to use a cached value + interpolate.
    let mut ctx = if sample_idle || collect_assertions {
        UserContext::collect(collect_assertions)
    } else {
        // Skip both subprocesses entirely; fill in below.
        UserContext::default()
    };
    if sample_idle {
        *last_idle_sample = Some((ctx.idle_secs, std::time::Instant::now()));
    } else if let Some((cached, sampled_at)) = last_idle_sample.as_ref() {
        // Interpolate: idle has grown by elapsed wall-clock seconds since
        // we last sampled (assumes no keyboard/mouse since — if user is
        // typing the next sample will reset it).
        ctx.idle_secs = cached + sampled_at.elapsed().as_secs_f64();
    }
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

/// Maturity-weighted confidence factor for the monopoly specialist.
///
/// Scales in `[MONOPOLY_MATURITY_FLOOR, 1.0]` based on how close the
/// `monopoly_freeze` NARS belief is to the confidence target.
///
/// - `None` (belief absent): returns floor — no evidence, cautious.
/// - `Some(0)` (mature): returns 1.0 — full trust.
/// - `Some(rem > 0)`: linear interpolation from floor toward 1.0 as
///   evidence accumulates.
///
/// [Pei Wang 2013 §3.3.1, Gray & Reuter 1992 §11]
pub fn monopoly_maturity_factor(obs_remaining: Option<u32>) -> f64 {
    match obs_remaining {
        None => MONOPOLY_MATURITY_FLOOR,
        Some(0) => 1.0,
        Some(rem) => {
            let needed =
                TruthValue::observations_to_reach(MONOPOLY_BELIEF_CONFIDENCE_TARGET).max(1) as f64;
            let progress = 1.0 - ((rem as f64 / needed).min(1.0));
            MONOPOLY_MATURITY_FLOOR + (1.0 - MONOPOLY_MATURITY_FLOOR) * progress
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn habituation_threshold_is_five() {
        assert_eq!(HABITUATION_THRESHOLD, 5);
    }

    // ── Phase 5.1 wiring — `apply_presence_factor` helper ────────────────────

    fn vote(name: &'static str, intervention: Intervention, confidence: f64) -> SpecialistVote {
        SpecialistVote {
            name,
            intervention,
            confidence,
        }
    }

    /// Active-user input (idle=2s, low HID, sub-crisis arousal, no passive
    /// flags) → narrowed band returns 0.7 → non-Observe vote confidences
    /// scale by 0.7×; Observe vote untouched.
    ///
    /// This is the round-trip test the spec calls out: with
    ///   user_idle=2.0, hid_per_min=60.0, arousal=0.3,
    ///   audio_active=false, has_sleep_assertion=false
    /// the narrowed modulator returns 0.7 and `apply_presence_factor` must
    /// multiply each non-Observe vote's confidence by 0.7.
    #[test]
    fn apply_specialist_voting_presence_factor_modulates_votes() {
        let factor = apollo_engine::engine::user_presence::user_presence_modulator_narrowed_no_counter(
            2.0,    // user_idle
            60.0,   // hid_per_min
            0.3,    // arousal (sub-crisis)
            false,  // audio_active
            false,  // has_sleep_assertion
            0.0,    // memory_pressure (sub-critical — test the original suppression path)
        );
        // Sanity: this input combo MUST land in the active tier.
        assert!(
            (factor - 0.7).abs() < 1e-9,
            "narrowed active tier expected 0.7, got {factor}"
        );

        let mut votes = vec![
            vote("hazard", Intervention::TightenThresholds, 0.80),
            vote("monopoly", Intervention::PreThrottleNoise, 0.60),
            vote("linucb", Intervention::Observe, 0.55),
        ];
        let modulated = apply_presence_factor(&mut votes, factor);

        assert_eq!(modulated, 2, "two non-Observe votes must be modulated");
        // hazard: 0.80 × 0.7 = 0.56
        assert!(
            (votes[0].confidence - 0.56).abs() < 1e-9,
            "hazard confidence: expected 0.56, got {}",
            votes[0].confidence
        );
        // monopoly: 0.60 × 0.7 = 0.42
        assert!(
            (votes[1].confidence - 0.42).abs() < 1e-9,
            "monopoly confidence: expected 0.42, got {}",
            votes[1].confidence
        );
        // linucb (Observe): untouched.
        assert!(
            (votes[2].confidence - 0.55).abs() < 1e-9,
            "Observe vote must NOT be modulated, got {}",
            votes[2].confidence
        );
    }

    /// Factor ≈ 1.0 (no suppression) → no-op, returns 0.
    #[test]
    fn apply_presence_factor_neutral_is_noop() {
        let mut votes = vec![
            vote("hazard", Intervention::TightenThresholds, 0.80),
            vote("linucb", Intervention::Observe, 0.55),
        ];
        let before = votes.clone();
        let modulated = apply_presence_factor(&mut votes, 1.0);
        assert_eq!(modulated, 0);
        for (a, b) in votes.iter().zip(before.iter()) {
            assert!((a.confidence - b.confidence).abs() < f64::EPSILON);
        }
    }

    /// Clamp to [0, 1]: a factor that would push confidence above 1.0 is
    /// clamped. (Defensive — the narrowed band's max is 1.0 so this is a
    /// belt-and-braces test against future caller misuse.)
    #[test]
    fn apply_presence_factor_clamps_into_unit_interval() {
        let mut votes = vec![vote("h", Intervention::TightenThresholds, 0.9)];
        // Pathological factor > 1 — should be clamped.
        apply_presence_factor(&mut votes, 2.0);
        assert!(votes[0].confidence <= 1.0);
        assert!(votes[0].confidence >= 0.0);
    }

    // ── Phase 3.1 — Skill-Aware Prediction factor ────────────────────────────

    #[test]
    fn skill_aware_factor_none_signal_is_neutral() {
        assert_eq!(skill_aware_factor(None), 1.0);
    }

    #[test]
    fn skill_aware_factor_full_failure_damps_to_085() {
        assert!((skill_aware_factor(Some(0.0)) - 0.85).abs() < 1e-9);
    }

    #[test]
    fn skill_aware_factor_full_success_boosts_to_115() {
        assert!((skill_aware_factor(Some(1.0)) - 1.15).abs() < 1e-9);
    }

    #[test]
    fn skill_aware_factor_half_signal_is_neutral_one() {
        assert!((skill_aware_factor(Some(0.5)) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn skill_aware_factor_band_is_30_percent_centered_on_neutral() {
        // Phase 3.1 invariant: width(band) = max_boost - min_damp = 0.30,
        // centered on 1.0. This keeps the multiplier from stacking into a
        // cascade-attenuation with SpecialistAccuracyTracker::weight() — see
        // NotebookLM 2026-05-16 adversarial pass on commit 66e4d16. If a
        // future refactor widens the band, this test will catch it before
        // the cascade re-emerges in prod.
        let max_boost = skill_aware_factor(Some(1.0));
        let min_damp = skill_aware_factor(Some(0.0));
        assert!((max_boost - 1.15).abs() < 1e-9, "max boost = {max_boost}");
        assert!((min_damp - 0.85).abs() < 1e-9, "min damp = {min_damp}");
        assert!(
            ((max_boost - min_damp) - 0.30).abs() < 1e-9,
            "band width = {}",
            max_boost - min_damp
        );
    }

    #[test]
    fn skill_aware_factor_monotone_in_signal() {
        let f_low = skill_aware_factor(Some(0.2));
        let f_mid = skill_aware_factor(Some(0.5));
        let f_high = skill_aware_factor(Some(0.9));
        assert!(f_low < f_mid && f_mid < f_high);
    }

    #[test]
    fn maturity_factor_no_belief_returns_floor() {
        assert_eq!(monopoly_maturity_factor(None), MONOPOLY_MATURITY_FLOOR);
    }

    #[test]
    fn maturity_factor_mature_belief_returns_one() {
        assert_eq!(monopoly_maturity_factor(Some(0)), 1.0);
    }

    #[test]
    fn maturity_factor_half_progress_is_midway() {
        let needed = TruthValue::observations_to_reach(MONOPOLY_BELIEF_CONFIDENCE_TARGET) as f64;
        let half = (needed / 2.0).ceil() as u32;
        let factor = monopoly_maturity_factor(Some(half));
        // ~midway between floor and 1.0 (0.7 ± ε)
        assert!(
            factor > MONOPOLY_MATURITY_FLOOR + 0.2 && factor < 1.0,
            "half-progress factor was {}",
            factor
        );
    }

    #[test]
    fn maturity_factor_monotone_in_evidence() {
        // more evidence (smaller rem) ⇒ higher factor
        let f_young = monopoly_maturity_factor(Some(100));
        let f_mid = monopoly_maturity_factor(Some(4));
        let f_mature = monopoly_maturity_factor(Some(1));
        assert!(f_young <= f_mid, "{} <= {}", f_young, f_mid);
        assert!(f_mid <= f_mature, "{} <= {}", f_mid, f_mature);
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

    // ── Phase 4.3.1 — Specialist accuracy purge inhibition ───────────────────

    /// When a maintenance purge fired in the previous 30 s, the
    /// `apply_specialist_voting` accuracy-update block must NOT run: the
    /// pressure drop is caused by the purge (a confounder), not by the
    /// specialists' predictions being wrong. The test verifies the gate
    /// contract: with `is_purge_recent(30)=true`, calling four
    /// `specialist_accuracy.update(*, false)` (the "all-wrong" cycle a
    /// purge would produce) is the wrong thing to do. Skipping them
    /// leaves weights at the 0.70 init.
    #[test]
    fn purge_recent_inhibits_specialist_accuracy_update() {
        use apollo_engine::engine::maintenance_state::MaintenanceState;
        use apollo_engine::engine::predictive_agent::{specialist, SpecialistAccuracyTracker};

        let mut maint = MaintenanceState::default();
        maint.mark_purged();
        assert!(maint.is_purge_recent(30), "marked purge must register as recent");

        let mut tracker = SpecialistAccuracyTracker::new();
        // Capture initial weights — all four start at the 0.70 init.
        let w_hazard_before = tracker.weight(specialist::HAZARD);
        let w_monopoly_before = tracker.weight(specialist::MONOPOLY);
        let w_kalman_before = tracker.weight(specialist::KALMAN);
        let w_linucb_before = tracker.weight(specialist::LINUCB);
        assert!((w_hazard_before - 0.70).abs() < 1e-9);
        assert!((w_monopoly_before - 0.70).abs() < 1e-9);
        assert!((w_kalman_before - 0.70).abs() < 1e-9);
        assert!((w_linucb_before - 0.70).abs() < 1e-9);

        // Mirror the exact body of `apply_specialist_voting`'s accuracy block:
        // when `is_purge_recent` is true, the four `update()` calls are skipped.
        if !maint.is_purge_recent(30) {
            tracker.update(specialist::HAZARD, false);
            tracker.update(specialist::MONOPOLY, false);
            tracker.update(specialist::KALMAN, false);
            tracker.update(specialist::LINUCB, false);
        }

        // Weights must be unchanged — the gate prevented the post-purge
        // pressure drop from poisoning the EMA accuracy estimates.
        assert!(
            (tracker.weight(specialist::HAZARD) - w_hazard_before).abs() < 1e-9,
            "HAZARD weight drifted under purge inhibition: {} vs {}",
            tracker.weight(specialist::HAZARD),
            w_hazard_before
        );
        assert!(
            (tracker.weight(specialist::MONOPOLY) - w_monopoly_before).abs() < 1e-9,
            "MONOPOLY weight drifted under purge inhibition"
        );
        assert!(
            (tracker.weight(specialist::KALMAN) - w_kalman_before).abs() < 1e-9,
            "KALMAN weight drifted under purge inhibition"
        );
        assert!(
            (tracker.weight(specialist::LINUCB) - w_linucb_before).abs() < 1e-9,
            "LINUCB weight drifted under purge inhibition"
        );
    }

    /// Mirror test: with NO recent purge, the same "all-wrong" updates
    /// SHOULD depress weights — this proves the inhibition above is doing
    /// real work, not just observing trivially-unchanged state.
    #[test]
    fn no_recent_purge_allows_specialist_accuracy_update() {
        use apollo_engine::engine::maintenance_state::MaintenanceState;
        use apollo_engine::engine::predictive_agent::{specialist, SpecialistAccuracyTracker};

        let maint = MaintenanceState::default(); // last_any_purge_at = None
        assert!(!maint.is_purge_recent(30));

        let mut tracker = SpecialistAccuracyTracker::new();
        let before = tracker.weight(specialist::HAZARD);
        if !maint.is_purge_recent(30) {
            tracker.update(specialist::HAZARD, false);
        }
        let after = tracker.weight(specialist::HAZARD);
        assert!(
            after < before,
            "HAZARD weight should depress when accepting 'wrong' update outside purge window: {} -> {}",
            before,
            after
        );
    }

    #[test]
    fn user_context_carry_forward_preserves_assertions() {
        // cycle 1: not a sample cycle (10%) and not assertion cycle (15%).
        // Expect both carry-forwards: assertions from last_user, idle from cache.
        let mut last = (true, true, false);
        let mut last_idle: Option<(f64, std::time::Instant)> = Some((50.0, std::time::Instant::now()));
        let ctx = compute_user_context(1, &mut last, &mut last_idle, None);
        assert!(ctx.has_sleep_assertion);
        assert!(ctx.call_in_progress);
        assert!(!ctx.audio_active);
        // idle_secs = cached 50 + ~0 elapsed ≈ 50.
        assert!(ctx.idle_secs >= 50.0 && ctx.idle_secs < 51.0);
    }
}
