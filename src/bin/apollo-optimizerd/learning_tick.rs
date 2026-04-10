//! # Learning Tick
//!
//! Per-cycle learning pipeline work extracted from the daemon main loop.
//!
//! ## What this module does
//!
//! Every cycle, after actions are executed, the daemon needs to feed outcome
//! information back into the learning subsystems:
//!
//! 1. **Outcome tracking** — record throttles, observe pressure drift, tick resolved outcomes
//! 2. **Causal graph** — record throttle/freeze actions and evaluate pending cause-effect pairs
//! 3. **Bayesian weight sync** — propagate resolved pattern weights to LearnedPolicy
//! 4. **Restore quality monitoring** — track post-restore effectiveness
//! 5. **LearningPipeline fan-out** — push resolved observations to all three learners
//! 6. **Lifelong zone learning** — feed effectiveness signal to SignalIntelligence zones
//! 7. **Cable A/D** — inject RL reward signals (penalty from outcome tracker, power-reduction reward)
//! 8. **Dr. Zero feedback** — read external autoresearch score → RL reward (every 60 cycles)
//! 9. **Predictive agent** — observe pressure outcome, MPC feedback
//! 10. **Periodic persist (% 100)** — flush pipeline, persist signal/state/skills, causal edge learning

use apollo_optimizer::collector::SystemCollector;
use apollo_optimizer::collector::SystemSnapshot;
use apollo_optimizer::engine::daemon_helpers::{hop_groups_path, signal_intelligence_path};
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::effectiveness_tracker::EffectivenessTracker;
use apollo_optimizer::engine::execute_actions::ExecuteOutcomes;
use apollo_optimizer::engine::iokit_sensors::HardwareSnapshot;
use apollo_optimizer::engine::learned_state::{LearnableParams, LearnedState, RestoreQualityMonitor};
use apollo_optimizer::engine::learning_pipeline::{LearningObservation, LearningPipeline};
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::nars_belief::{ArousalState, Salience};
use apollo_optimizer::engine::pipeline::learning_context::LearningContext;
use apollo_optimizer::engine::predictive_agent::{Intervention, SpecialistVote};
use apollo_optimizer::engine::signal_intelligence::SignalDigest;
use apollo_optimizer::engine::system_log_ingester::{SystemEvent, SystemLogIngester};
use apollo_optimizer::engine::types::{FrozenPidEntry, FrozenStatePersisted};
use apollo_optimizer::engine::nested_learner::NestedLearner;
use apollo_optimizer::engine::workload_classifier::WorkloadMode;

/// Run all per-cycle learning pipeline work.
///
/// See module-level documentation for the list of operations performed.
///
/// # Parameters
///
/// - `snapshot` — current system snapshot (pressure, processes, memory stats)
/// - `cycle_hw_snap` — hardware snapshot for this cycle (CPU/package watts)
/// - `exec_outcomes` — execution outcomes from this cycle's action dispatch
/// - `throttle_names_for_outcome` — names of processes throttled this cycle
/// - `signal_digest` — current signal digest (pressure_smooth, mpc_recommendation, …)
/// - `workload_mode` — current workload mode (idle, build, browser, …)
/// - `cycle_count` — monotonically increasing cycle counter
/// - `state` — shared daemon state (policy + frozen_state are accessed)
/// - `collector` — system collector used to resolve PID → process name
/// - `lctx` — all 9 learning subsystems
/// - `learning_pipeline` — fans out resolved observations to three learners
/// - `effectiveness_tracker` — tracks specialist effectiveness
/// - `restore_monitor` — tracks post-restore quality
/// - `last_restore_quality` — updated when restore monitor produces a verdict
/// - `prev_package_watts` — previous cycle's package watts (Cable D, updated by this call)
/// - `pending_trial_skill` — current trial skill (passed to persist, not modified here)
/// - `last_specialist_votes` — if specialists disagreed last cycle: (votes, chosen intervention)
/// - `prev_workload_mode` — previous cycle's workload mode (updated by this call for next cycle)
/// - `arousal_state` — global EMA arousal tracker; updated here, used to adjust recalibration threshold
/// - `ls_path` — filesystem path for unified learned state
/// - `persist_generations` — generation counter passed to `LearnedState::persist_improved`
/// - `skills_path` — filesystem path for optimization skills
/// - `nested_learner` — L0/L1/L2 hierarchy coordinator (context flow gating)
#[allow(clippy::too_many_arguments)]
pub fn run_learning_tick<'a>(
    snapshot: &SystemSnapshot,
    cycle_hw_snap: &Option<HardwareSnapshot>,
    exec_outcomes: &ExecuteOutcomes,
    throttle_names_for_outcome: &[String],
    signal_digest: &SignalDigest,
    workload_mode: WorkloadMode,
    cycle_count: u64,
    state: &SharedState,
    collector: &SystemCollector,
    lctx: &mut LearningContext<'a>,
    learning_pipeline: &mut LearningPipeline,
    effectiveness_tracker: &mut EffectivenessTracker,
    restore_monitor: &mut RestoreQualityMonitor,
    last_restore_quality: &mut Option<f64>,
    prev_package_watts: &mut Option<f64>,
    prev_workload_mode: &mut WorkloadMode,
    arousal_state: &mut ArousalState,
    pending_trial_skill: Option<(String, f64)>,
    last_specialist_votes: Option<(&[SpecialistVote], Intervention)>,
    log_ingester: &mut SystemLogIngester,
    learnable_params: &mut LearnableParams,
    ls_path: &str,
    persist_generations: u32,
    skills_path: &str,
    nested_learner: &mut NestedLearner,
) {
    // ── Arousal EMA: update every cycle from current pressure + swap ─────────
    // p_oom_est ∈ [0,1]: proxy for OOM risk derived from pressure above 0.70.
    // [Yerkes & Dodson 1908] arousal modulates learning rate.
    {
        let mem_pressure = snapshot.pressure.memory_pressure;
        let swap_gb = snapshot.pressure.swap_used_bytes as f64 / 1_073_741_824.0;
        let p_oom_est = ((mem_pressure - 0.70) / 0.30).clamp(0.0, 1.0);
        let cycle_salience = Salience::compute(mem_pressure, 0.0, p_oom_est, swap_gb);
        arousal_state.update(cycle_salience);
    }

    // ── NestedLearner L0: per-cycle signal quality tick ─────────────────────
    // [Google Nested Learning 2025] L0 updates every cycle; gates L1 updates.
    // Signal quality = fluidity_score × (1 - transformer_anomaly × 0.5).
    // High fluidity + low anomaly → signal is trustworthy → L1 gate opens.
    {
        let signal_quality = (signal_digest.fluidity_score as f64)
            * (1.0 - (signal_digest.transformer_anomaly * 0.5).clamp(0.0, 1.0));
        nested_learner.tick_l0(signal_quality);
    }

    // ── Outcome tracking: record throttled processes ─────────────────────────
    if exec_outcomes.throttles_applied > 0 {
        let cpu_watts = cycle_hw_snap
            .as_ref()
            .and_then(|h| h.power.cpu_watts)
            .unwrap_or(0.0) as f64;
        let total_cpu_pct: f64 = snapshot
            .top_processes
            .iter()
            .map(|p| p.cpu_usage as f64)
            .sum::<f64>()
            .max(0.01);
        let mem_pressure_now = snapshot.pressure.memory_pressure;
        let swap_gb_now = snapshot.pressure.swap_used_bytes as f64 / 1_073_741_824.0;
        for name in throttle_names_for_outcome {
            let proc_watts = snapshot
                .top_processes
                .iter()
                .find(|p| &p.name == name)
                .map(|p| (p.cpu_usage as f64 / total_cpu_pct) * cpu_watts)
                .unwrap_or(0.0);
            // Capture swap context for affective salience weighting.
            // High swap at throttle time → high arousal → stronger NARS belief.
            lctx.outcome_tracker.record_throttle_with_swap(
                name,
                mem_pressure_now,
                proc_watts,
                swap_gb_now,
            );
        }
    }

    // ── Causal graph (Pearl 2009 + Granger 1969) ──────────────────────────────
    // Record throttle/freeze actions for causal evaluation with resource snapshots
    // for mechanism attribution. Multi-horizon eval: 3 cycles (fast) + 15 cycles (slow).
    {
        use apollo_optimizer::engine::causal_graph::ResourceSnapshot;
        let pressure_now = snapshot.pressure.memory_pressure as f32;
        let swap_mb_now = snapshot.pressure.swap_used_bytes as f32 / 1_048_576.0;
        for name in throttle_names_for_outcome {
            // Build per-process resource snapshot for mechanism attribution.
            let res = snapshot
                .top_processes
                .iter()
                .find(|p| &p.name == name)
                .map(|p| ResourceSnapshot {
                    rss_mb: p.memory_usage as f32 / 1_048_576.0,
                    cpu_pct: p.cpu_usage,
                    swap_mb: swap_mb_now, // system-level (no per-process swap on macOS)
                })
                .unwrap_or_default();
            lctx.causal_graph.record_action_with_resources(
                &format!("throttle:{}", name),
                pressure_now,
                cycle_count,
                res,
            );
        }
        // Record freeze actions — only PIDs frozen THIS cycle, not all active ones.
        for &pid in &exec_outcomes.newly_frozen_pids {
            if let Some(process) = collector.system().process(sysinfo::Pid::from_u32(pid)) {
                let res = ResourceSnapshot {
                    rss_mb: process.memory() as f32 / 1_048_576.0,
                    cpu_pct: process.cpu_usage(),
                    swap_mb: swap_mb_now,
                };
                lctx.causal_graph.record_action_with_resources(
                    &format!("freeze:{}", process.name()),
                    pressure_now,
                    cycle_count,
                    res,
                );
            }
        }
        // Evaluate pending actions with current resource snapshot.
        // Both fast (3-cycle) and slow (15-cycle) horizons are evaluated.
        let current_res = ResourceSnapshot {
            rss_mb: snapshot
                .top_processes
                .iter()
                .map(|p| p.memory_usage as f32 / 1_048_576.0)
                .sum(),
            cpu_pct: snapshot.top_processes.iter().map(|p| p.cpu_usage).sum(),
            swap_mb: swap_mb_now,
        };
        lctx.causal_graph
            .evaluate_with_resources(pressure_now, cycle_count, &current_res);
    }

    // ── Causal graph: process co-occurrence at high pressure ─────────────────
    if snapshot.pressure.memory_pressure >= 0.60 {
        let active: Vec<String> = snapshot
            .top_processes
            .iter()
            .take(10)
            .map(|p| p.name.clone())
            .collect();
        lctx.outcome_tracker.record_co_occurrence(&active);
    }

    // ── Counterfactual: observe pressure drift ───────────────────────────────
    // If no throttles this cycle, the tracker learns the natural drift rate
    // (what happens without action).
    lctx.outcome_tracker.observe_cycle(
        snapshot.pressure.memory_pressure,
        !throttle_names_for_outcome.is_empty(),
    );

    // ── Outcome tracker tick (urgency-aware, Phase 7) ────────────────────────
    // Urgency flush: when pressure > 0.80, resolve all pending immediately.
    // Normal: use adaptive params from LearnableParams (outcome_wait_secs,
    // outcome_effective_threshold) instead of hardcoded 30s / 0.01.
    {
        let wl_id = match workload_mode {
            WorkloadMode::Build => 1,
            WorkloadMode::LlmInference => 2,
            WorkloadMode::Browsing => 3,
            WorkloadMode::Idle => 0,
        };
        let batch = if snapshot.pressure.memory_pressure > 0.80 {
            lctx.outcome_tracker
                .urgency_flush(snapshot.pressure.memory_pressure)
        } else {
            lctx.outcome_tracker.tick_with_params(
                snapshot.pressure.memory_pressure,
                learnable_params.outcome_wait_secs,
                learnable_params.outcome_effective_threshold,
                wl_id,
            )
        };
        if batch.savings_watts > 0.0 {
            lctx.energy_tracker
                .record_savings(batch.savings_watts, 30.0);
        }
        // Cable 1: causal_effect() → correct PatternWeight using real causal signal.
        // For each effective throttle, check if the drop was truly caused by the
        // action (causal_effect > 0) or just natural drift. Demote weights that
        // only appear effective due to natural pressure fluctuation.
        if !batch.effective_names.is_empty() {
            let drift = lctx.outcome_tracker.natural_drift();
            let short_velocity = lctx.outcome_tracker.pressure_velocity_short();
            // Demote if long-term drift is present OR short-window velocity
            // already explains the drop (faster 3-cycle attribution).
            if drift > 0.01 || short_velocity > 0.01 {
                // Pre-compute causal effects per process before mutating weights.
                let demotions: Vec<String> = batch
                    .effective_names
                    .iter()
                    .filter_map(|name| {
                        let avg_drop = lctx
                            .outcome_tracker
                            .experience
                            .query_similar_with_band(
                                name,
                                snapshot.pressure.memory_pressure,
                                learnable_params.experience_pressure_band,
                            )
                            .map(|(drop, _)| drop)
                            .unwrap_or(0.05);
                        let causal_long = lctx.outcome_tracker.causal_effect(avg_drop);
                        let causal_fast = lctx.outcome_tracker.causal_effect_fast(avg_drop);
                        // Demote if EITHER signal says natural drift explains the drop.
                        if causal_long < 0.005 || causal_fast < 0.005 {
                            Some(name.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                // Roll back effective_count for drift-only "successes".
                for name in &demotions {
                    if let Some(w) = lctx.outcome_tracker.weights.get_mut(name) {
                        if w.effective_count > 0 {
                            w.effective_count -= 1;
                        }
                    }
                }
            }
        }
        // NARS drift recalibration: detect regime changes in per-process effectiveness.
        // When ≥2 beliefs have drifted ≥20pp (or EMA score > arousal-adjusted threshold),
        // the Bayesian weights no longer reflect current system behavior. Apply soft decay:
        // halve accumulated counts toward the Laplace prior (effectiveness→0.5).
        // Threshold is Yerkes-Dodson adaptive: high arousal → 0.06 (hair-trigger),
        // low arousal → 0.10 (conservative). [Murphy 2012] §3.3 "reset toward prior".
        let recalib_threshold = arousal_state.adjusted_drift_threshold(0.08);
        if lctx
            .outcome_tracker
            .nars_needs_recalibration_at(recalib_threshold)
        {
            for w in lctx.outcome_tracker.weights.values_mut() {
                // Soft decay: halve counts. Minimum 1 each to keep Laplace structure.
                w.effective_count = (w.effective_count / 2).max(1);
                w.throttle_count = (w.throttle_count / 2).max(2);
            }
            lctx.outcome_tracker.nars_acknowledge_recalibration();
        }

        // Sync Bayesian weights to the persisted LearnedPolicy.
        if !batch.effective_names.is_empty() || !batch.low_value_names.is_empty() {
            let mut pg = state.policy.lock_recover();
            for (name, weight) in &lctx.outcome_tracker.weights {
                pg.learned_policy
                    .pattern_weights
                    .insert(name.clone(), weight.clone());
            }
        }
        // Phase 4 — Novel pattern logger: when OutcomeTracker sees a high-value
        // pattern crossing the `is_high_value()` threshold for the first time,
        // append it to `novel_patterns.jsonl`.
        //
        // Robust against off-by-one races: condition is `throttle_count ∈ [3, 4]`
        // (catches both 2→3 and 2→4 jumps when multiple outcomes resolve in the
        // same tick). File is rotated every 100 cycles when it exceeds 64 KiB
        // (≈1000 entries) to bound disk usage.
        //
        // [Simon 1955] satisficing — reliable "just confirmed" signal.
        // [Kleppmann 2017] DDIA Ch.3 — log compaction via size-triggered rotation.
        for name in &batch.effective_names {
            if let Some(w) = lctx.outcome_tracker.weights.get(name) {
                if (w.throttle_count == 3 || w.throttle_count == 4) && w.is_high_value() {
                    let pressure = snapshot.pressure.memory_pressure;
                    let workload = format!("{:?}", workload_mode).to_lowercase();
                    let entry = format!(
                        "{{\"process\":{:?},\"effectiveness\":{:.3},\"pressure\":{:.3},\"workload\":{:?},\"cycle\":{}}}\n",
                        name, w.effectiveness(), pressure, workload, cycle_count
                    );
                    let novel_path = apollo_optimizer::engine::daemon_helpers::novel_patterns_path();
                    let _ = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(novel_path)
                        .and_then(|mut f| {
                            use std::io::Write;
                            f.write_all(entry.as_bytes())
                        });
                }
            }
        }
        // Opportunistic rotation: every 100 cycles, check size and rotate if > 64 KiB.
        if cycle_count % 100 == 0 {
            let novel_path = apollo_optimizer::engine::daemon_helpers::novel_patterns_path();
            if let Ok(meta) = std::fs::metadata(novel_path) {
                if meta.len() > 64 * 1024 {
                    let old_path = format!("{}.old", novel_path);
                    let _ = std::fs::rename(novel_path, &old_path);
                }
            }
        }

        // Restore quality monitor: track post-restore effectiveness.
        //
        // The correct "resolved this tick" count is `resolved_outcomes.len()`,
        // NOT `effective_names.len() + low_value_names.len()`. The previous
        // code conflated two unrelated lists:
        //   - effective_names: outcomes that resolved effectively in THIS tick
        //   - low_value_names: ALL patterns historically marked low-value,
        //                      recomputed every tick from self.weights
        // Summing them inflated `resolved` with the entire lifetime low-value
        // list, producing quality ≈ 0 even in a healthy system (observed in
        // production: 0/3000 over 120s). `resolved_outcomes` is the authoritative
        // per-tick resolution list documented as "includes both effective and
        // ineffective resolutions" in OutcomeBatch.
        //
        // The verdict is compared against the long-term steady-state effective
        // rate (not a hardcoded constant) so it only flags real regressions
        // rather than healthy-but-unspectacular behavior.
        if !restore_monitor.is_done() {
            let batch_res = batch.resolved_outcomes.len() as u32;
            let batch_eff = batch.effective_names.len() as u32;
            restore_monitor.observe(batch_eff, batch_res);
            let steady_state = lctx.outcome_tracker.overall_effectiveness();
            if let Some(verdict) = restore_monitor.verdict(steady_state) {
                *last_restore_quality = Some(verdict.quality);
                if verdict.stale {
                    lctx.signal_intel.reset_zones();
                }
            }
        }

        // LearningPipeline: fan out resolved outcomes to all three learners.
        // Each resolved throttle becomes a LearningObservation with the
        // pre/post pressure captured by tick(). Cross-feeds are applied
        // at batch flush (every 8 observations or at persist time).
        for (name, pre_pressure, post_pressure) in batch.resolved_outcomes {
            // ── Loop 1 fix: skills observe their outcomes ───────────────
            // When a resolved outcome matches a known skill target, feed
            // the result back so the skill adapts its min_pressure and
            // success_rate from real data instead of only causal graph edges.
            let effective = post_pressure < pre_pressure - 0.01;
            {
                let skill_name = format!("throttle:{}", name);
                lctx.skill_registry
                    .record_result_with_pressure(&skill_name, effective, pre_pressure as f32);
            }
            // Also match pending_trial_skill: induced skills (group:/batch:)
            // that were trialed get outcome feedback through their skill name.
            if let Some((ref trial_name, _)) = pending_trial_skill {
                // Check if this process is a target of the trialed skill.
                // The trial skill targets may include this process name.
                lctx.skill_registry
                    .record_result_with_pressure(trial_name, effective, pre_pressure as f32);
            }

            let obs = LearningObservation {
                process_name: name,
                skill_name: None, // skill attribution tracked by pending_trial_skill path
                pre_pressure,
                post_pressure,
                workload: workload_mode.as_str().to_string(),
                cycle: cycle_count,
            };
            // NestedLearner L1: tick per resolved outcome.
            // Effectiveness = normalized pressure drop (0.3 drop → 1.0 effective).
            // Context flow: L0 quality weights how much this outcome shifts L1 aggregate.
            {
                let effectiveness =
                    ((pre_pressure - post_pressure) / 0.30).clamp(0.0, 1.0);
                if nested_learner.tick_l1(effectiveness) {
                    let _l2_ctx = nested_learner.flush_l2();
                    // L2 context wired to meta-learning in a follow-up iteration.
                }
            }

            learning_pipeline.push(
                obs,
                lctx.outcome_tracker,
                lctx.causal_graph,
                lctx.skill_registry,
                effectiveness_tracker,
            );
        }
    }

    // ── Lifelong zone learning (workload-aware, Phase 4) ────────────────────
    // Effective actions → lower zone thresholds (engage earlier).
    // Ineffective actions → raise thresholds (be more conservative).
    // Per-workload offsets accumulate so zones auto-adapt to workload type.
    {
        let effectiveness = lctx.outcome_tracker.overall_effectiveness();
        let pressure = signal_digest.pressure_smooth;
        if lctx.outcome_tracker.total_resolved > 10 {
            let wl_id = match workload_mode {
                WorkloadMode::Build => 1,
                WorkloadMode::LlmInference => 2,
                WorkloadMode::Browsing => 3,
                WorkloadMode::Idle => 0,
            };
            lctx.signal_intel.zone_feedback_workload(
                pressure,
                effectiveness > 0.50,
                wl_id,
            );
        }
    }

    // ── RL pressure histogram sample ──────────────────────────────────────────
    // Record every cycle for quantile-based band auto-tuning.
    if let Some(rl) = &mut lctx.overflow_guard.rl_agent {
        rl.record_pressure_sample(
            snapshot.pressure.memory_pressure,
            signal_digest.pressure_smooth, // compressor proxy: smooth pressure
        );
    }

    // ── Auto-tune (Phase 2): Kalman R, RL bands, zone alpha ─────────────────
    // Every 50 cycles: Kalman R auto-tune from innovation variance.
    // Every 200 cycles: RL bands from pressure histogram quantiles.
    // Every 100 cycles: zone alpha from oscillation/stall detection.
    if cycle_count % 50 == 25 {
        if let Some(new_r) = lctx.signal_intel.auto_tune_kalman_r() {
            learnable_params.kalman_pressure_r = new_r; // persist the auto-tuned value
        }
    }
    if cycle_count % 200 == 100 {
        if let Some(rl) = &lctx.overflow_guard.rl_agent {
            if let Some((p_bands, c_bands)) = rl.auto_tune_bands() {
                learnable_params.rl_pressure_bands = p_bands;
                learnable_params.rl_compressor_bands = c_bands;
            }
        }
    }
    // Apply all learned params to live subsystems every 100 cycles.
    // Closes the wiring gap: cusum_k/h, kalman Q, pid_target/decay now consumed.
    if cycle_count % 100 == 50 {
        lctx.signal_intel.apply_learnable_params(learnable_params);
        // Wire nars_drift_threshold → DriftDetector sensitivity.
        lctx.outcome_tracker
            .drift_detector
            .set_drift_threshold(learnable_params.nars_drift_threshold as f32);
    }

    // ── Loop 2 fix: hazard model batch retrain ──────────────────────────────
    // Every 50 cycles, replay buffered OOM events through the hazard model
    // for mini-batch gradient refinement. This makes the hazard model learn
    // from ALL observed overflows (not just the latest one).
    if cycle_count % 50 == 15 {
        let _steps = lctx.signal_intel.retrain_hazard_batch();
    }

    // ── System log ingestion (Phase 5) ──────────────────────────────────────
    // Poll macOS unified logs for Jetsam/OOM kills and process crashes.
    // OOM events → feed hazard model + NARS (arousal=1.0, valence=-1.0).
    // Crash events → NARS belief update (arousal=0.9, valence=-1.0).
    // Protected processes are observed but never targeted.
    {
        let log_events = log_ingester.poll();
        for event in &log_events {
            match event {
                SystemEvent::OomKill { process_name, .. } => {
                    // Feed hazard model: OOM is a real overflow event
                    let mem_pressure = snapshot.pressure.memory_pressure;
                    let swap_ratio = snapshot.pressure.swap_used_bytes as f64
                        / (snapshot.pressure.swap_total_bytes.max(1) as f64);
                    lctx.signal_intel.record_overflow(
                        mem_pressure,
                        swap_ratio,
                        signal_digest.pressure_smooth, // compressor proxy
                        1.0, // ~1 hour since last (conservative)
                    );
                    // NARS: crisis-level salience for OOM kill
                    let salience = Salience {
                        arousal: 1.0,
                        valence: -1.0,
                    };
                    lctx.outcome_tracker.drift_detector.observe_contextual(
                        &format!("oom:{}", process_name),
                        false, // OOM = negative outcome
                        salience,
                        mem_pressure,
                    );
                }
                SystemEvent::Crash { process_name, .. } => {
                    // NARS: high salience for crashes (slightly less than OOM)
                    let salience = Salience {
                        arousal: 0.9,
                        valence: -1.0,
                    };
                    lctx.outcome_tracker.drift_detector.observe_contextual(
                        &format!("crash:{}", process_name),
                        false,
                        salience,
                        snapshot.pressure.memory_pressure,
                    );
                }
            }
        }
    }

    // ── Loop 3 fix: specialist disagreement outcome feedback ──────────────
    // When specialists disagreed last cycle, observe whether the chosen
    // intervention was effective and feed back to boost/penalize specialists.
    if let Some((votes, chosen)) = last_specialist_votes {
        // Simple heuristic: pressure dropped since last cycle → intervention was effective.
        let was_effective = snapshot.pressure.memory_pressure < signal_digest.pressure_smooth;
        lctx.specialist_accuracy
            .record_disagreement_outcome(votes, chosen, was_effective);
    }

    // ── Meta-learning (Phase 6): learning rates that learn ─────────────────
    // Every 500 cycles, adjust learning rates based on whether the system is
    // stuck (velocity low + effectiveness falling → explore more) or converged
    // (velocity low + effectiveness stable → slow down).
    // Safety: only adjusts rates, never safety thresholds. All clamped.
    if cycle_count % 500 == 250 {
        let effectiveness = lctx.outcome_tracker.overall_effectiveness();
        // Compute param delta as proxy for velocity: zone alpha change rate
        let zone_delta = (signal_digest.pressure_smooth
            - lctx.signal_intel.effective_zones(0).0)
            .abs();
        learnable_params.meta_learn(effectiveness, zone_delta);
    }

    // ── Cable A: OutcomeTracker → RL reward signal ───────────────────────────
    // When throttling is wasteful (low-value patterns detected),
    // penalize the RL agent so it learns to adjust thresholds.
    {
        let penalty = lctx.outcome_tracker.rl_penalty();
        if penalty < 0.0 {
            if let Some(rl) = &mut lctx.overflow_guard.rl_agent {
                rl.inject_external_reward(penalty);
            }
        }
    }

    // ── Cable D: Power-reduction reward → RL ────────────────────────────────
    // When package_watts drops cycle-over-cycle, the RL policy did something
    // good — reinforce it. M1 Air idle ~1-3W, active ~5-15W.
    // A 1W+ reduction is meaningful; cap at 5W (→ +0.3).
    {
        let curr_w = cycle_hw_snap
            .as_ref()
            .and_then(|h| h.power.package_watts)
            .map(|w| w as f64);
        if let (Some(prev), Some(curr)) = (*prev_package_watts, curr_w) {
            let delta = (prev - curr).max(0.0);
            if delta > 1.0 {
                let power_reward = (delta / 5.0 * 0.3).clamp(0.0, 0.3);
                if let Some(rl) = &mut lctx.overflow_guard.rl_agent {
                    rl.inject_external_reward(power_reward);
                }
            }
        }
        *prev_package_watts = curr_w;
    }
    *prev_workload_mode = workload_mode;

    // ── Dr. Zero feedback loop (every 60 cycles) ─────────────────────────────
    // File written by watch-deploy.sh after each autoresearch run.
    if cycle_count % 60 == 30 {
        if let Ok(data) = std::fs::read_to_string("/tmp/apollo-dr-zero-feedback.json") {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(score) = v.get("score").and_then(|s| s.as_f64()) {
                    // Normalize: score 90+ is good (reward), <70 is bad (penalty).
                    // Range maps to [-0.3, +0.3] RL reward.
                    let reward = ((score - 80.0) / 33.3).clamp(-0.3, 0.3);
                    if let Some(rl) = &mut lctx.overflow_guard.rl_agent {
                        rl.inject_external_reward(reward);
                    }
                }
            }
        }
    }

    // ── Predictive agent: observe outcome + MPC feedback ────────────────────
    lctx.predictive_agent
        .observe_outcome(snapshot.pressure.memory_pressure);
    lctx.predictive_agent.maybe_persist();
    // MPC feedback: tell MPC what happened after its recommendation.
    lctx.signal_intel.mpc_feedback(
        signal_digest.mpc_recommendation,
        signal_digest.pressure_smooth,
        snapshot.pressure.memory_pressure,
    );

    // ── Periodic persist: every 100 cycles ───────────────────────────────────
    // Flush any buffered observations before persisting state.
    if cycle_count % 100 == 0 {
        learning_pipeline.flush_remaining(
            lctx.outcome_tracker,
            lctx.causal_graph,
            lctx.skill_registry,
            effectiveness_tracker,
        );
        lctx.signal_intel
            .persist(std::path::Path::new(signal_intelligence_path()));
        lctx.outcome_tracker
            .persist_hop_groups(std::path::Path::new(hop_groups_path()));
        // Snapshot frozen state for unified persistence.
        let frozen_snap: FrozenStatePersisted = {
            let fg = state.frozen_state.lock_recover();
            FrozenStatePersisted {
                frozen: fg
                    .iter()
                    .map(|(pid, e)| FrozenPidEntry {
                        pid: *pid,
                        since: e.frozen_at,
                        name: e.process_name.clone(),
                    })
                    .collect(),
            }
        };
        LearnedState::persist_improved(
            lctx.signal_intel,
            lctx.outcome_tracker,
            lctx.specialist_accuracy,
            lctx.skill_registry,
            effectiveness_tracker,
            Some(lctx.overflow_guard.export_history()),
            Some(frozen_snap),
            std::path::Path::new(ls_path),
            persist_generations,
            *last_restore_quality,
            pending_trial_skill,
            Some(arousal_state.clone()),
            Some(lctx.causal_graph),
            None, // process_baselines: persisted at shutdown via main.rs
            Some(learnable_params.clone()),
        );
        // Causal graph observability: log solid/weak links discovered.
        let solid = lctx.causal_graph.solid_count();
        let total = lctx.causal_graph.edge_count();
        if total > 0 {
            println!(
                "lctx.causal_graph: {}/{} edges solid, {} pending",
                solid,
                total,
                lctx.causal_graph.solid_edges().len()
            );
        }
        // Persist optimization skills (Hermes pattern).
        lctx.skill_registry
            .persist(std::path::Path::new(skills_path));
        // Learn skills from causal graph solid edges, ordered by impact.
        // solid_edges_by_impact() sorts by confidence×avg_delta so high-impact
        // actions (large pressure reduction) are learned with higher priority.
        for edge in lctx.causal_graph.solid_edges_by_impact() {
            if edge.cause.starts_with("throttle:") {
                let target = edge.cause.trim_start_matches("throttle:");
                // Scale trigger pressure by impact: high-impact actions activate
                // at lower pressure (proactive), low-impact ones wait for more pressure.
                let trigger_pressure = if edge.avg_delta > 0.08 {
                    0.55 // proactive: high-impact action, use early
                } else {
                    0.65 // conservative: low-impact action, wait
                };
                lctx.skill_registry.learn(
                    &edge.cause,
                    trigger_pressure,
                    "any",
                    vec![target.to_string()],
                );
                lctx.skill_registry
                    .record_result(&edge.cause, edge.confidence > 0.5);
            }
        }
    }
}
