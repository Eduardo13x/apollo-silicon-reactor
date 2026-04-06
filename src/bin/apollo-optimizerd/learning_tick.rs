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
use apollo_optimizer::engine::learned_state::{LearnedState, RestoreQualityMonitor};
use apollo_optimizer::engine::learning_pipeline::{LearningObservation, LearningPipeline};
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::nars_belief::{ArousalState, Salience};
use apollo_optimizer::engine::pipeline::learning_context::LearningContext;
use apollo_optimizer::engine::signal_intelligence::SignalDigest;
use apollo_optimizer::engine::types::{FrozenPidEntry, FrozenStatePersisted};
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
/// - `prev_workload_mode` — previous cycle's workload mode (updated by this call for next cycle)
/// - `arousal_state` — global EMA arousal tracker; updated here, used to adjust recalibration threshold
/// - `ls_path` — filesystem path for unified learned state
/// - `persist_generations` — generation counter passed to `LearnedState::persist_improved`
/// - `skills_path` — filesystem path for optimization skills
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
    ls_path: &str,
    persist_generations: u32,
    skills_path: &str,
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

    // ── Outcome tracker tick ─────────────────────────────────────────────────
    {
        let batch = lctx.outcome_tracker.tick(snapshot.pressure.memory_pressure);
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
                            .query_similar(name, snapshot.pressure.memory_pressure)
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
        // Restore quality monitor: track post-restore effectiveness.
        if !restore_monitor.is_done() {
            let batch_eff = batch.effective_names.len() as u32;
            let batch_res = (batch.effective_names.len() + batch.low_value_names.len()) as u32;
            restore_monitor.observe(batch_eff, batch_res);
            if let Some(verdict) = restore_monitor.verdict() {
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
            let obs = LearningObservation {
                process_name: name,
                skill_name: None, // skill attribution tracked by pending_trial_skill path
                pre_pressure,
                post_pressure,
                workload: workload_mode.as_str().to_string(),
                cycle: cycle_count,
            };
            learning_pipeline.push(
                obs,
                lctx.outcome_tracker,
                lctx.causal_graph,
                lctx.skill_registry,
                effectiveness_tracker,
            );
        }
    }

    // ── Lifelong zone learning ───────────────────────────────────────────────
    // Effective actions → lower zone thresholds (engage earlier).
    // Ineffective actions → raise thresholds (be more conservative).
    {
        let effectiveness = lctx.outcome_tracker.overall_effectiveness();
        let pressure = signal_digest.pressure_smooth;
        if lctx.outcome_tracker.total_resolved > 10 {
            lctx.signal_intel
                .zone_feedback(pressure, effectiveness > 0.50);
        }
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
            None, // learnable_params: wired in Phase 2
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
