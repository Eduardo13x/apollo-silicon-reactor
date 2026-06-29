//! LLM + Usage Learning Subsystem — extracted from daemon monolith.
//!
//! Contains:
//! - `llm_reactive_tick()` — LLM advisor trigger & call logic (~470 lines)
//! - `usage_learning_tick()` — usage model update & pattern promotion
//! - `apply_learned_policy_actions()` — filter/add actions from learned policy
//! - `windowserver_cpu()` — WindowServer CPU helper
//! - `LlmReactiveCounters` — per-cycle trigger counters

use std::collections::{HashMap, HashSet};

use chrono::{Duration as ChronoDuration, Local, Timelike, Utc};

use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::daemon_helpers::pid_start_time;
use apollo_engine::engine::llm::{
    append_jsonl, delete_file_best_effort, load_repo_config, write_json, LearnedPolicy, LlmAdvisor,
    SuggestionOutcome, TeacherContext,
};
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::safety::pattern_conflicts_with_protected;
use apollo_engine::engine::types::{HardPath, LlmRunMode, RootAction};

use super::SharedState;

// ── Types ──────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct LlmReactiveCounters {
    pub ws_high: u32,
    pub mem_high: u32,
    pub swap_high: u32,
    pub prev_trigger_active: bool,
}

// ── Helpers ────────────────────────────────────────────────────────────────

pub fn windowserver_cpu(snapshot: &apollo_engine::collector::SystemSnapshot) -> f32 {
    snapshot
        .top_processes
        .iter()
        .find(|p| p.name.contains("WindowServer"))
        .map(|p| p.cpu_usage)
        .unwrap_or(0.0)
}

/// Swap ceiling above which the local Metal/MLX teacher inference is skipped.
/// 6 GB: MLX runs cleanly through normal 4-5 GB swap on this 8 GB box (measured
/// 2026-06-14); only a >6 GB thrash crisis warrants bailing.
const METAL_OOM_SWAP_THRESHOLD: u64 = 6 * 1024 * 1024 * 1024;

/// True when current swap is high enough that we should skip the GPU teacher
/// call this trigger. Pure predicate so the threshold is unit-testable.
fn metal_oom_would_skip(swap_used_bytes: u64) -> bool {
    swap_used_bytes > METAL_OOM_SWAP_THRESHOLD
}

// ── LLM Reactive Tick ──────────────────────────────────────────────────────

pub fn llm_reactive_tick(
    state: &SharedState,
    advisor: &mut LlmAdvisor,
    snapshot: &apollo_engine::collector::SystemSnapshot,
    counters: &mut LlmReactiveCounters,
    heuristic_struggling: bool,
) {
    let now = Utc::now();
    let (llm_key_path, llm_state_path, suggestions_path, llm_cfg_default) = {
        let llm = state.llm.lock_recover();
        (
            llm.llm_key_path.clone(),
            llm.llm_state_path.clone(),
            llm.suggestions_path.clone(),
            llm.llm_cfg.clone(),
        )
    };
    let has_key = llm_key_path.exists();

    // Load config early — needed for always_on check.
    let llm_cfg = load_repo_config(&state.config_path)
        .llm
        .unwrap_or(llm_cfg_default);
    if !llm_cfg.enabled() {
        return;
    }

    // Auto-enable for always_on (local models like Gemma 4) — no training TTL needed.
    if llm_cfg.always_on() {
        let mut guard = state.llm.lock_recover();
        if !guard.llm_state.enabled {
            guard.llm_state.enabled = true;
            guard.llm_state.training_started_at = Some(now);
            // 10-year TTL — effectively permanent for local models.
            guard.llm_state.training_expires_at = Some(now + ChronoDuration::days(365 * 10));
            write_json(&llm_state_path, &guard.llm_state, Some(0o600));
        }
    } else {
        // TTL housekeeping: if training expired, disable and delete key.
        let mut guard = state.llm.lock_recover();
        if guard.llm_state.enabled
            && guard
                .llm_state
                .training_expires_at
                .map(|t| t <= now)
                .unwrap_or(true)
        {
            guard.llm_state.enabled = false;
            guard.llm_state.training_expires_at = None;
            guard.llm_state.last_suggestion = None;
            guard.llm_state.mode = LlmRunMode::Off;
            guard.llm_state.last_error = Some("training-expired".to_string());
            write_json(&llm_state_path, &guard.llm_state, Some(0o600));
            drop(guard);
            delete_file_best_effort(&llm_key_path);
            return;
        }
    }

    // ── Fase 3: resolver outcome pendiente si ya pasaron ≥30s ─────────────
    {
        // Extract outcome data without holding the lock across policy access.
        let outcome_data = {
            let guard = state.llm.lock_recover();
            match (
                guard.llm_state.pending_outcome_at,
                guard.llm_state.pending_outcome_pressure,
            ) {
                (Some(pending_at), Some(pending_pressure))
                    if now - pending_at >= ChronoDuration::seconds(30) =>
                {
                    Some((
                        pending_at,
                        pending_pressure,
                        guard
                            .llm_state
                            .pending_outcome_rationale
                            .clone()
                            .unwrap_or_default(),
                        guard.llm_state.pending_added_protected.clone(),
                    ))
                }
                _ => None,
            }
        };

        if let Some((pending_at, pending_pressure, rationale, added_protected)) = outcome_data {
            let pressure_after = snapshot.pressure.memory_pressure;
            let delta = pressure_after - pending_pressure;

            // WORSENED revert: if pressure increased significantly, remove the protected
            // patterns that Gemma added — they were shielding processes that cause pressure.
            // Threshold 0.08 (8pp) avoids reverting on noise while catching real regressions.
            if delta > 0.08 && !added_protected.is_empty() {
                let learned_policy_path = state.llm.lock_recover().learned_policy_path.clone();
                let lp_snap = {
                    let mut pg = state.policy.lock_recover();
                    let before = pg.learned_policy.protected_patterns.len();
                    std::sync::Arc::make_mut(&mut pg.learned_policy.protected_patterns)
                        .retain(|p| !added_protected.contains(p));
                    let reverted = before - pg.learned_policy.protected_patterns.len();
                    if reverted > 0 {
                        std::sync::Arc::make_mut(&mut pg.learned_policy.protected_patterns).sort();
                        pg.learned_policy.learned_at = Some(now);
                        let lp_clone = pg.learned_policy.clone();
                        pg.adaptive_governor.update_learned_policy(&lp_clone);
                        tracing::info!(
                            reverted,
                            delta,
                            ?added_protected,
                            "llm: WORSENED outcome — reverted protected patterns"
                        );
                    }
                    pg.learned_policy.clone()
                };
                write_json(&learned_policy_path, &lp_snap, Some(0o600));
            }

            let mut guard = state.llm.lock_recover();
            guard.llm_state.last_suggestion_outcome = Some(SuggestionOutcome {
                applied_at: pending_at,
                pressure_before: pending_pressure,
                pressure_after,
                pressure_delta: delta,
                rationale_snippet: rationale.chars().take(80).collect(),
            });
            guard.llm_state.pending_outcome_at = None;
            guard.llm_state.pending_outcome_pressure = None;
            guard.llm_state.pending_outcome_rationale = None;
            guard.llm_state.pending_added_protected.clear();
            write_json(&llm_state_path, &guard.llm_state, Some(0o600));
        }
    }

    // Keep advisor in sync with config edits.
    advisor.update_cfg(llm_cfg.clone());
    if !has_key {
        return;
    }

    let api_key = match HardPath::read_to_string_limited(&llm_key_path, 4096) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Determine reactive trigger.
    let ws_cpu = windowserver_cpu(snapshot);
    let mem_pressure = snapshot.pressure.memory_pressure;
    let swap_delta_bps = snapshot.pressure.swap_delta_bytes_per_sec;
    let thermal = snapshot.pressure.thermal_level.as_str();

    // Decide desired mode (sensitive vs strict) using cost governor.
    let now_local = Local::now();
    let today = now_local.date_naive().to_string();
    let quiet_hours = {
        let h = now_local.hour();
        (1..8).contains(&h)
    };

    let (mode, daily_budget, min_interval_secs, max_calls_per_hour, pattern_budget_per_day) = {
        let mut guard = state.llm.lock_recover();
        if !llm_cfg.always_on() && !guard.llm_state.training_active() {
            write_json(&llm_state_path, &guard.llm_state, Some(0o600));
            return;
        }

        // Reset daily budget window.
        if guard.llm_state.calls_today_day.as_deref() != Some(&today) {
            guard.llm_state.calls_today_day = Some(today.clone());
            guard.llm_state.calls_today = 0;
        }

        // Keep trigger events only for a short horizon.
        guard
            .llm_state
            .trigger_events
            .retain(|t| now - *t < ChronoDuration::minutes(30));
        let trigger_len = guard.llm_state.trigger_events.len();
        if trigger_len > 100 {
            guard.llm_state.trigger_events.drain(..trigger_len - 100);
        }
        let triggers_recent = guard.llm_state.trigger_events.len() as u32;

        let bootcamp = guard
            .llm_state
            .training_started_at
            .map(|t| now - t < ChronoDuration::days(5))
            .unwrap_or(false);
        let daily_budget = if bootcamp { 24 } else { 8 };

        // If we've been stable for a while, bias to strict.
        let stable_for = guard
            .llm_state
            .no_trigger_since
            .map(|t| now - t)
            .unwrap_or_else(|| ChronoDuration::seconds(0));
        let stable_long = stable_for > ChronoDuration::hours(3);

        let consumed = guard.llm_state.calls_today;
        let consumed_ratio = if daily_budget == 0 {
            1.0
        } else {
            (consumed as f64) / (daily_budget as f64)
        };

        let mut mode = guard.llm_state.mode;
        if quiet_hours {
            mode = LlmRunMode::Strict;
        } else if consumed >= daily_budget {
            mode = LlmRunMode::Off;
        } else if triggers_recent >= 2 {
            mode = LlmRunMode::Sensitive;
        } else if consumed_ratio >= 0.60 {
            // Once we've consumed most of the daily budget, tighten up.
            mode = LlmRunMode::Strict;
        } else if stable_long && !bootcamp {
            // During bootcamp we keep mode sensitive for faster learning.
            mode = LlmRunMode::Strict;
        } else if mode == LlmRunMode::Off {
            // Recover from off when the budget permits.
            mode = LlmRunMode::Strict;
        }
        guard.llm_state.mode = mode;

        let (base_min_interval, base_max_calls, pattern_budget) = match mode {
            LlmRunMode::Sensitive => (600_u64, 4_u32, if bootcamp { 5_u32 } else { 3_u32 }),
            LlmRunMode::Strict => (1800_u64, 2_u32, 2_u32),
            LlmRunMode::Off => (u64::MAX, 0_u32, 0_u32),
        };

        // Respect config as a hard limiter for cadence.
        let effective_min_interval = base_min_interval.max(llm_cfg.min_interval_secs());
        let effective_max_calls = base_max_calls.min(llm_cfg.max_calls_per_hour().max(1));

        write_json(&llm_state_path, &guard.llm_state, Some(0o600));
        (
            mode,
            daily_budget,
            effective_min_interval,
            effective_max_calls,
            pattern_budget,
        )
    };

    if mode == LlmRunMode::Off {
        return;
    }

    // Thresholds by mode.
    // WindowServer >35% es normal durante uso intensivo de UI (especialmente con TDA).
    // Subimos el umbral para no desperdiciar budget del LLM en síntomas, no causas.
    let (ws_thresh, mem_thresh, swap_thresh_bps, cycles_needed) = match mode {
        LlmRunMode::Sensitive => (65.0_f32, 0.78_f64, 20.0 * 1024.0 * 1024.0, 3_u32),
        LlmRunMode::Strict => (75.0_f32, 0.88_f64, 50.0 * 1024.0 * 1024.0, 5_u32),
        LlmRunMode::Off => (f32::MAX, 1.0, f64::MAX, u32::MAX),
    };

    if ws_cpu >= ws_thresh {
        counters.ws_high += 1;
    } else {
        counters.ws_high = 0;
    }
    if mem_pressure >= mem_thresh {
        counters.mem_high += 1;
    } else {
        counters.mem_high = 0;
    }
    if swap_delta_bps >= swap_thresh_bps {
        counters.swap_high += 1;
    } else {
        counters.swap_high = 0;
    }

    let thermal_critical = matches!(thermal, "serious" | "critical");
    let mut trigger_active = thermal_critical
        || counters.ws_high >= cycles_needed
        || counters.mem_high >= cycles_needed
        || counters.swap_high >= cycles_needed;
    let mut rising_edge = trigger_active && !counters.prev_trigger_active;
    counters.prev_trigger_active = trigger_active;

    // One-time baseline call after enabling training so it doesn't look "stuck"
    // when the system is stable and no triggers fire.
    let baseline_call = {
        let guard = state.llm.lock_recover();
        guard.llm_state.last_attempt_at.is_none()
            && guard
                .llm_state
                .training_started_at
                .map(|t| now - t > ChronoDuration::minutes(2))
                .unwrap_or(false)
    };
    if baseline_call {
        trigger_active = true;
        rising_edge = true;
    }

    // Heurístico fallando: el outcome tracker detectó que throttlear ciertos procesos
    // no baja la presión de memoria. El LLM puede sugerir qué patrones proteger/ruido.
    if heuristic_struggling && !trigger_active {
        trigger_active = true;
        rising_edge = !counters.prev_trigger_active;
        // BUG FIX: update prev_trigger_active to reflect the override. Without this,
        // counters.prev_trigger_active stays false (set from the original trigger_active=false
        // at line above), so rising_edge fires again every cycle — trigger storm.
        counters.prev_trigger_active = true;
    }

    if !trigger_active {
        // Bootcamp sampling: even when the system is "fine", take an occasional sample call
        // so the teacher can learn normal workload patterns.
        let sampling_due = {
            let guard = state.llm.lock_recover();
            let since_last = guard
                .llm_state
                .last_attempt_at
                .map(|t| now - t)
                .unwrap_or_else(|| ChronoDuration::hours(24));
            let user_active_proxy = ws_cpu >= 10.0 || snapshot.cpu.global_usage >= 15.0;
            mode == LlmRunMode::Sensitive
                && guard
                    .llm_state
                    .training_started_at
                    .map(|t| now - t < ChronoDuration::days(5))
                    .unwrap_or(false)
                && user_active_proxy
                && since_last > ChronoDuration::minutes(45)
        };

        let mut guard = state.llm.lock_recover();
        if guard.llm_state.no_trigger_since.is_none() {
            guard.llm_state.no_trigger_since = Some(now);
        }

        if sampling_due {
            guard.llm_state.last_trigger_at = Some(now);
            guard.llm_state.last_trigger_reason = Some("sampling".to_string());
            guard.llm_state.trigger_events.push(now);
            guard.llm_state.no_trigger_since = None;
            write_json(&llm_state_path, &guard.llm_state, Some(0o600));
            drop(guard);
            // Turn sampling into a synthetic rising-edge trigger.
            rising_edge = true;
        } else {
            write_json(&llm_state_path, &guard.llm_state, Some(0o600));
            return;
        }
    }

    // Set/refresh trigger state.
    let trigger_reason = if baseline_call {
        "baseline".to_string()
    } else if thermal_critical {
        format!("thermal:{}", thermal)
    } else if counters.ws_high >= cycles_needed {
        format!("ui-lag windowserver cpu {:.1}%", ws_cpu)
    } else if counters.swap_high >= cycles_needed {
        format!("swap-thrash delta {:.0} B/s", swap_delta_bps)
    } else {
        format!("memory-pressure {:.2}", mem_pressure)
    };

    if rising_edge {
        let mut guard = state.llm.lock_recover();
        guard.llm_state.last_trigger_at = Some(now);
        guard.llm_state.last_trigger_reason = Some(trigger_reason.clone());
        // Backstop cooldown: only push to trigger_events once per 60s regardless
        // of rising-edge rate. Prevents pressure oscillation near threshold from
        // inflating triggers_recent (which controls mode=Sensitive) on restarts.
        let last_push = guard
            .llm_state
            .trigger_events
            .last()
            .copied()
            .unwrap_or(chrono::DateTime::<Utc>::MIN_UTC);
        if now - last_push >= ChronoDuration::seconds(60) {
            guard.llm_state.trigger_events.push(now);
        }
        guard.llm_state.no_trigger_since = None;
        write_json(&llm_state_path, &guard.llm_state, Some(0o600));
    }

    // Call gating: only call on rising edge.
    if !rising_edge {
        return;
    }

    // Metal OOM guard: an extreme-swap crisis means a ~2 GB GPU inference would
    // both risk failure and steal memory/GPU from the user mid-thrash. Skip and
    // let Apollo reduce pressure first; retry next trigger.
    //
    // Threshold history: the old 2 GB value was sized for llama.cpp's Metal path
    // (and was effectively permanent on this box, which idles at 4-5 GB swap — so
    // the teacher had not actually run since 2026-06-03). The runtime is now MLX,
    // whose lazy unified-memory eval is far more graceful: measured 2026-06-14, a
    // live inference completed cleanly at 4.6 GB swap and macOS reclaimed swap to
    // 2.1 GB right after. Raise to 6 GB so the teacher runs during normal pressure
    // (its whole purpose) and only bails in a genuine >6 GB/8 GB thrash crisis.
    if metal_oom_would_skip(snapshot.pressure.swap_used_bytes) {
        let mut guard = state.llm.lock_recover();
        guard.llm_state.last_error = Some(format!(
            "metal-oom-risk swap={:.1}GB — skipped",
            snapshot.pressure.swap_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
        write_json(&llm_state_path, &guard.llm_state, Some(0o600));
        return;
    }

    // ── Smart skip: don't call Gemma if she has nothing new to say ────────
    // Guard 1: if last suggestion was <2 hours ago and pressure hasn't changed
    // more than 10%, Gemma would give the same answer. Skip.
    {
        let guard = state.llm.lock_recover();
        if let Some(last_call) = guard.llm_state.last_call_at {
            if now - last_call < ChronoDuration::hours(2) {
                if let Some(ref prev) = guard.llm_state.last_suggestion_outcome {
                    // Compare against pressure_after (measured outcome), not
                    // pressure_before (pre-intervention baseline).  Using _before
                    // causes false skips when pressure returns to pre-suggestion
                    // levels after the suggestion's effect wears off.
                    let pressure_change =
                        (snapshot.pressure.memory_pressure - prev.pressure_after).abs();
                    if pressure_change < 0.10 {
                        return; // same scenario, Gemma would repeat herself
                    }
                }
            }
        }
    }
    // Guard 2: if all top processes already have high_value pattern_weights,
    // Apollo's S1 already knows what to do — no need to consult S2.
    {
        let policy = state.policy.lock_recover();
        let top_names: Vec<&str> = snapshot
            .top_processes
            .iter()
            .take(5)
            .map(|p| p.name.as_str())
            .collect();
        let all_known = top_names.iter().all(|name| {
            policy
                .learned_policy
                .interactive_patterns
                .iter()
                .any(|p| name.contains(p.as_str()))
                || policy
                    .learned_policy
                    .noise_patterns
                    .iter()
                    .any(|p| name.contains(p.as_str()))
                || policy
                    .learned_policy
                    .protected_patterns
                    .iter()
                    .any(|p| name.contains(p.as_str()))
        });
        if all_known && !heuristic_struggling {
            return; // S1 already has all answers, skip S2
        }
    }

    // Budget + cadence.
    {
        let mut guard = state.llm.lock_recover();

        if guard.llm_state.calls_today >= daily_budget {
            guard.llm_state.mode = LlmRunMode::Off;
            guard.llm_state.last_error = Some("daily-budget-exhausted".to_string());
            write_json(&llm_state_path, &guard.llm_state, Some(0o600));
            return;
        }

        if let Some(last) = guard.llm_state.last_attempt_at {
            if now - last < ChronoDuration::seconds(min_interval_secs as i64) {
                return;
            }
        }

        // Per-hour window.
        if guard
            .llm_state
            .hour_window_started_at
            .map(|t| now - t > ChronoDuration::hours(1))
            .unwrap_or(true)
        {
            guard.llm_state.hour_window_started_at = Some(now);
            guard.llm_state.calls_in_window = 0;
        }
        if guard.llm_state.calls_in_window >= max_calls_per_hour {
            return;
        }

        // Record attempt before the network call so status updates immediately.
        guard.llm_state.last_attempt_at = Some(now);
        guard.llm_state.last_http_status = None;
        guard.llm_state.last_error = None;
        guard.llm_state.calls_in_window += 1;
        guard.llm_state.calls_today += 1;
        write_json(&llm_state_path, &guard.llm_state, Some(0o600));
    }

    // Network call (no locks held).
    let current_policy = state.policy.lock_recover().learned_policy.clone();

    // ── Fase 2: construir TeacherContext con datos ricos ───────────────────
    let pattern_scores_owned: Vec<(String, u32, f64)> = current_policy
        .pattern_weights
        .iter()
        .filter(|(_, w)| w.throttle_count >= 3)
        .map(|(name, w)| (name.clone(), w.throttle_count, w.effectiveness()))
        .collect();
    let previous_outcome_owned = state
        .llm
        .lock_recover()
        .llm_state
        .last_suggestion_outcome
        .clone();
    let frozen_count = state.frozen_state.lock_recover().len();
    let teacher = TeacherContext {
        pattern_scores: &pattern_scores_owned,
        previous_outcome: previous_outcome_owned.as_ref(),
        heuristic_struggling,
        frozen_count,
    };

    let suggestion_res =
        advisor.call_raw(snapshot, &api_key, Some(&current_policy), Some(&teacher));

    // Apply suggestion and persist state.
    match suggestion_res {
        Ok(suggestion) => {
            let accepted = suggestion.confidence >= llm_cfg.min_confidence();
            {
                let mut guard = state.llm.lock_recover();
                guard.llm_state.last_http_status = Some(200);
                guard.llm_state.last_call_at = Some(now);
                guard.llm_state.last_suggestion = Some(suggestion.clone());
                guard.llm_state.consecutive_failures = 0;
                if !accepted {
                    guard.llm_state.last_error = Some("below-min-confidence".to_string());
                }
                write_json(&llm_state_path, &guard.llm_state, Some(0o600));
            }

            append_jsonl(
                &suggestions_path,
                &serde_json::json!({
                    "at": now,
                    "trigger": trigger_reason,
                    "mode": format!("{:?}", mode),
                    "accepted": accepted,
                    "suggestion": suggestion,
                }),
            );

            if !accepted {
                return;
            }

            // 1) Profile: apply as a short-lived override.
            if let Some(p) = suggestion.suggested_profile {
                let mut pg = state.policy.lock_recover();
                if pg.governor.manual_override.is_none() {
                    pg.governor
                        .set_manual_override(p, 20, "llm-reactive".to_string());
                }
            }
            // 2) Latency target.
            if let Some(t) = suggestion.suggested_latency_target {
                state.policy.lock_recover().latency_target = t;
            }

            // 3) Learned patterns: merge with daily cap.
            let remaining = {
                let mut guard = state.llm.lock_recover();
                let day = now.date_naive();
                let reset_day = guard
                    .llm_state
                    .policy_updates_day
                    .map(|d| d.date_naive() != day)
                    .unwrap_or(true);
                if reset_day {
                    guard.llm_state.policy_updates_day = Some(now);
                    guard.llm_state.policy_updates_today = 0;
                }
                let remaining =
                    pattern_budget_per_day.saturating_sub(guard.llm_state.policy_updates_today);
                if remaining == 0 {
                    write_json(&llm_state_path, &guard.llm_state, Some(0o600));
                }
                remaining
            };
            if remaining == 0 {
                return;
            }

            let learned_policy_path = state.llm.lock_recover().learned_policy_path.clone();
            let mut added = 0u32;
            let lp_snap = {
                let mut pg = state.policy.lock_recover();
                for p in suggestion
                    .add_interactive_patterns
                    .iter()
                    .take(remaining as usize)
                {
                    if !pg.learned_policy.interactive_patterns.contains(p)
                        && !pattern_conflicts_with_protected(p)
                    {
                        // Remove from noise if promoted to interactive.
                        std::sync::Arc::make_mut(&mut pg.learned_policy.noise_patterns)
                            .retain(|n| n != p);
                        std::sync::Arc::make_mut(&mut pg.learned_policy.interactive_patterns)
                            .push(p.clone());
                        added += 1;
                    }
                }
                for p in suggestion
                    .add_noise_patterns
                    .iter()
                    .take(remaining.saturating_sub(added) as usize)
                {
                    // Skip if already protected/interactive — cannot downgrade.
                    // Truncation-aware interactive conflict (exact .contains
                    // missed the LSP) + safety-protected names. [2026-06-20]
                    if !pg.learned_policy.noise_patterns.contains(p)
                        && !pattern_conflicts_with_protected(p)
                        && !pg.learned_policy.protected_patterns.contains(p)
                        && !noise_pattern_conflicts(p, &pg.learned_policy.interactive_patterns)
                    {
                        std::sync::Arc::make_mut(&mut pg.learned_policy.noise_patterns)
                            .push(p.clone());
                        added += 1;
                    }
                }
                for p in suggestion
                    .add_protected_patterns
                    .iter()
                    .take(remaining.saturating_sub(added) as usize)
                {
                    if !pg.learned_policy.protected_patterns.contains(p)
                        && !pattern_conflicts_with_protected(p)
                    {
                        // Remove from noise when promoted to protected.
                        std::sync::Arc::make_mut(&mut pg.learned_policy.noise_patterns)
                            .retain(|n| n != p);
                        std::sync::Arc::make_mut(&mut pg.learned_policy.protected_patterns)
                            .push(p.clone());
                        added += 1;
                    }
                }
                if added > 0 {
                    std::sync::Arc::make_mut(&mut pg.learned_policy.interactive_patterns).sort();
                    std::sync::Arc::make_mut(&mut pg.learned_policy.noise_patterns).sort();
                    std::sync::Arc::make_mut(&mut pg.learned_policy.protected_patterns).sort();
                    pg.learned_policy.learned_at = Some(now);
                }
                let snap = pg.learned_policy.clone();
                if added > 0 {
                    pg.adaptive_governor.update_learned_policy(&snap);
                }
                snap
            };
            if added > 0 {
                // Persist after releasing the policy lock.
                write_json(&learned_policy_path, &lp_snap, Some(0o600));
                {
                    let mut guard = state.llm.lock_recover();
                    guard.llm_state.policy_updates_today += added;
                    // ── Fase 3: registrar baseline para medir el outcome ──
                    // Solo sobrescribir si no hay outcome pendiente (evitar drift).
                    if guard.llm_state.pending_outcome_at.is_none() {
                        guard.llm_state.pending_outcome_pressure =
                            Some(snapshot.pressure.memory_pressure);
                        guard.llm_state.pending_outcome_at = Some(now);
                        let snippet: String = suggestion.rationale.chars().take(80).collect();
                        guard.llm_state.pending_outcome_rationale = Some(snippet);
                        // Track protected patterns added so we can revert on WORSENED.
                        guard.llm_state.pending_added_protected =
                            suggestion.add_protected_patterns.clone();
                    }
                    write_json(&llm_state_path, &guard.llm_state, Some(0o600));
                }
            }
        }
        Err(err) => {
            let mut guard = state.llm.lock_recover();
            guard.llm_state.consecutive_failures += 1;
            match err {
                apollo_engine::engine::llm::LlmCallError::Cooldown => {
                    guard.llm_state.last_error = Some("cooldown".to_string());
                }
                apollo_engine::engine::llm::LlmCallError::HttpStatus { code, body_excerpt } => {
                    guard.llm_state.last_http_status = Some(code);
                    guard.llm_state.last_error = Some(format!(
                        "http-status {} {}",
                        code,
                        body_excerpt.unwrap_or_default()
                    ));
                }
                apollo_engine::engine::llm::LlmCallError::Transport(e) => {
                    guard.llm_state.last_error = Some(format!("transport {}", e));
                }
                apollo_engine::engine::llm::LlmCallError::Parse(e) => {
                    guard.llm_state.last_error = Some(format!("parse {}", e));
                }
                apollo_engine::engine::llm::LlmCallError::Rejected(e) => {
                    guard.llm_state.last_error = Some(format!("rejected {}", e));
                }
            }

            // Fail-safe: if it's repeatedly failing, go strict to save cost.
            if guard.llm_state.consecutive_failures >= 3 {
                guard.llm_state.mode = LlmRunMode::Strict;
            }
            write_json(&llm_state_path, &guard.llm_state, Some(0o600));
        }
    }
}

// ── Usage Learning Tick ────────────────────────────────────────────────────

pub fn usage_learning_tick(
    state: &SharedState,
    snapshot: &apollo_engine::collector::SystemSnapshot,
    has_foreground: bool,
    cpu_wall_ratios: &HashMap<String, f32>,
) {
    let now = Utc::now();
    let ws_cpu = windowserver_cpu(snapshot);
    // Refine interactive_proxy: require both CPU activity signals AND an actual
    // foreground app (not idle/screensaver). This prevents background CPU spikes
    // from triggering interactive mode when the user isn't at the keyboard.
    let cpu_proxy = ws_cpu >= 10.0 || snapshot.cpu.global_usage >= 15.0;
    let interactive_proxy = cpu_proxy && has_foreground;
    let mem_pressure = snapshot.pressure.memory_pressure;
    let swap_delta = snapshot.pressure.swap_delta_bytes_per_sec;

    let jank_proxy = ws_cpu >= 35.0
        && (mem_pressure >= 0.75 || swap_delta >= 20.0 * 1024.0 * 1024.0)
        || matches!(
            snapshot.pressure.thermal_level.as_str(),
            "serious" | "critical"
        );

    {
        let mut usage = state.usage.lock_recover();
        usage.usage_model.update_from_snapshot(
            snapshot,
            now,
            interactive_proxy,
            jank_proxy,
            10,
            cpu_wall_ratios,
        );
    }

    // Persist usage model periodically (every ~2 minutes).
    {
        let mut usage = state.usage.lock_recover();
        let due = usage
            .usage_tracker
            .last_persist_at
            .map(|t| now - t > ChronoDuration::minutes(2))
            .unwrap_or(true);
        if due {
            let path = usage.usage_model_path.clone();
            usage.usage_model.persist(&path);
            usage.usage_tracker.last_persist_at = Some(now);
        }
    }

    // Daily promotion counters (conservative).
    let today = Local::now().date_naive().to_string();
    let promotions_used = {
        let mut usage = state.usage.lock_recover();
        if usage.usage_tracker.promotions_day.as_deref() != Some(&today) {
            usage.usage_tracker.promotions_day = Some(today.clone());
            usage.usage_tracker.promotions_today = 0;
        }
        usage.usage_tracker.promotions_today
    };
    // Propose promotions without holding locks across scoring.
    let (started_at, existing_interactive, existing_noise, existing_protected) = {
        let model = state.usage.lock_recover();
        let started_at = model.usage_model.top_report(1).model_started_at;
        drop(model);
        let policy = state.policy.lock_recover().learned_policy.clone();
        (
            started_at,
            policy.interactive_patterns,
            policy.noise_patterns,
            policy.protected_patterns,
        )
    };
    let promotions = {
        let model = state.usage.lock_recover();
        model.usage_model.maybe_promote_patterns(
            now,
            &existing_interactive,
            &existing_noise,
            &existing_protected,
            promotions_used,
            started_at,
        )
    };

    if promotions.is_empty() {
        return;
    }

    // Apply promotions to learned policy.
    let mut applied = 0u32;
    let learned_policy_path = state.llm.lock_recover().learned_policy_path.clone();
    let lp_snap = {
        let mut pg = state.policy.lock_recover();
        for (kind, pattern) in &promotions {
            match kind.as_str() {
                "interactive"
                    if !pg.learned_policy.interactive_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                    => {
                        std::sync::Arc::make_mut(&mut pg.learned_policy.interactive_patterns).push(pattern.clone());
                        applied += 1;
                    }
                "noise"
                    if !pg.learned_policy.noise_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                        && !noise_pattern_conflicts(
                            pattern,
                            &pg.learned_policy.interactive_patterns,
                        )
                    => {
                        std::sync::Arc::make_mut(&mut pg.learned_policy.noise_patterns).push(pattern.clone());
                        applied += 1;
                    }
                "protected"
                    // Protected patterns are safety labels — they bypass the daily
                    // cap and only require that the pattern isn't already present.
                    if !pg.learned_policy.protected_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                    => {
                        std::sync::Arc::make_mut(&mut pg.learned_policy.protected_patterns).push(pattern.clone());
                        applied += 1;
                    }
                _ => {}
            }
        }
        if applied > 0 {
            std::sync::Arc::make_mut(&mut pg.learned_policy.interactive_patterns).sort();
            std::sync::Arc::make_mut(&mut pg.learned_policy.noise_patterns).sort();
            std::sync::Arc::make_mut(&mut pg.learned_policy.protected_patterns).sort();
            pg.learned_policy.learned_at = Some(now);
        }
        let snap = pg.learned_policy.clone();
        if applied > 0 {
            pg.adaptive_governor.update_learned_policy(&snap);
        }
        snap
    };
    if applied > 0 {
        // Persist after releasing the policy lock.
        write_json(&learned_policy_path, &lp_snap, Some(0o600));
    }

    if applied > 0 {
        let events_path = {
            let mut usage = state.usage.lock_recover();
            usage.usage_tracker.promotions_today += applied;
            usage.usage_events_path.clone()
        };
        append_jsonl(
            &events_path,
            &serde_json::json!({"at": now, "promotions": promotions}),
        );
    }
}

// ── Apply Learned Policy Actions ───────────────────────────────────────────

/// 2026-05-16: per-PID TTL dedup for learned-policy boost/throttle emissions.
/// Without this, `apply_learned_policy_actions` re-emitted BoostProcess for
/// every interactive PID every cycle (464/500 journal entries = boosts).
/// Each emit ≡ mach_qos.set_tier syscall; mach_qos IS already sticky so
/// the kernel work was redundant — but each emit also walked the safety
/// stack + journal write, consuming 10% Apollo CPU and contributing to
/// the very thrashing Apollo was trying to mitigate. 30s TTL chosen to
/// match typical foreground app dwell time without re-firing on every
/// per-cycle snapshot. Stale entries pruned lazily on next access.
fn boost_dedup_cache() -> &'static std::sync::Mutex<HashMap<u32, std::time::Instant>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<u32, std::time::Instant>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

fn learned_policy_boost_ttl_secs() -> u64 {
    120
}

/// A boost may only target a process the user is actually using: the
/// foreground app or one with a visible window. Blocks headless background
/// processes (e.g. a `node` dev server matching the bare "node" interactive
/// pattern) from being boosted on a name match alone — boosting them steals
/// P-core scheduling from the foreground app / a live call. [2026-06-18]
///
/// 2026-06-21 — yield to frontmost media: when the FRONTMOST app is actively
/// playing media (`frontmost_media_active`), a visible-but-NOT-frontmost
/// candidate is refused. A boost marks the target TASK_FOREGROUND_APPLICATION +
/// QOS_TIER_0 + nice -10, lying to CLPC about P-core contention; doing that to a
/// background-but-visible terminal steals compositing timeshare from the
/// frontmost 4K app → occasional frame drop (node-54x class). The frontmost app
/// itself is ALWAYS boostable (never yields). Survival bypasses this entirely
/// (the caller gates on `!survival` before this). Subtractive — only ever adds a
/// skip; over-yielding merely leaves a non-frontmost app at its correct default
/// QoS.
fn boost_visibility_ok(
    pid: u32,
    fg_pid: Option<u32>,
    visible_pids: &HashSet<u32>,
    frontmost_media_active: bool,
) -> bool {
    if Some(pid) == fg_pid {
        return true; // the frontmost app is always boostable — never yields
    }
    // visible-but-not-frontmost: yield to the frontmost media app
    visible_pids.contains(&pid) && !frontmost_media_active
}

/// A teacher-suggested NOISE pattern conflicts (must be rejected) if it would
/// shadow an interactive/protected process. The teacher does not know Apollo's
/// protected set — 2026-06-20 it moved `language_server` (the LSP) to noise.
/// The prior guard used exact `.contains()`, which missed `language_server` vs
/// the stored interactive `language_server_macos_arm`. Reject if the pattern is
/// a safety-protected name OR matches an existing interactive pattern
/// (truncation-aware, both directions).
fn noise_pattern_conflicts(pattern: &str, interactive: &[String]) -> bool {
    if apollo_engine::engine::safety::is_protected_name(pattern) {
        return true;
    }
    let pl = pattern.to_lowercase();
    let lpm = apollo_engine::engine::decide_actions::learned_pattern_matches;
    interactive.iter().any(|ip| {
        let il = ip.to_lowercase();
        lpm(&pl, &il) || lpm(&il, &pl)
    })
}

fn should_emit_boost(pid: u32, ttl_secs: u64) -> bool {
    let now = std::time::Instant::now();
    let ttl = std::time::Duration::from_secs(ttl_secs);
    let mut cache = boost_dedup_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // Prune stale entries opportunistically.
    cache.retain(|_, t| now.duration_since(*t) < ttl);
    match cache.get(&pid) {
        Some(t) if now.duration_since(*t) < ttl => false,
        _ => {
            cache.insert(pid, now);
            true
        }
    }
}

pub fn apply_learned_policy_actions(
    snapshot: &apollo_engine::collector::SystemSnapshot,
    policy: &LearnedPolicy,
    mut actions: Vec<RootAction>,
    // 2026-06-21: frontmost app NAME, for the boost yield-to-media gate. The
    // frontmost PID arrives via shadow_signals; the name is only known to the
    // caller (main.rs foreground_app). None = unknown → never yields (safe).
    foreground_app: Option<&str>,
) -> Vec<RootAction> {
    // Filter: never act on protected patterns (case-insensitive).
    if !policy.protected_patterns.is_empty() {
        actions.retain(|a| {
            let name = match a {
                RootAction::BoostProcess { name, .. }
                | RootAction::ThrottleProcess { name, .. }
                | RootAction::FreezeProcess { name, .. }
                | RootAction::UnfreezeProcess { name, .. } => name,
                _ => return true,
            };
            let name_lc = name.to_lowercase();
            !policy
                .protected_patterns
                .iter()
                .any(|p| name_lc.contains(&p.to_lowercase()))
        });
    }

    // Add targeted boost/throttle for top processes if policy matches.
    if policy.interactive_patterns.is_empty() && policy.noise_patterns.is_empty() {
        return actions;
    }
    let mut seen: HashSet<(u32, &'static str)> = HashSet::new();
    for a in &actions {
        match a {
            RootAction::BoostProcess { pid, .. } => {
                seen.insert((*pid, "boost"));
            }
            RootAction::ThrottleProcess { pid, .. } => {
                seen.insert((*pid, "throttle"));
            }
            _ => {}
        }
    }

    let survival = apollo_engine::engine::safety::survival_mode_active_total(
        snapshot.pressure.memory_pressure,
        snapshot.pressure.swap_used_bytes,
        snapshot.pressure.swap_total_bytes,
    );

    // ROOT FIX (2026-06-18 node boost-loop): a BOOST raises QoS to make the app
    // the user is actually using snappier. Firing it on a NAME match alone
    // boosts headless background processes — a `node` dev server matching the
    // bare "node" interactive pattern got boosted 54×, stealing P-core
    // scheduling from Meet's video (microstutter). Same class as Brave-0607.
    // Gate on real foreground / visible-window state: only the foreground app
    // or a process with a visible window may be boosted. The visible-pid
    // syscall (~1-3ms) is computed only when an interactive-name candidate
    // actually exists, so most cycles pay nothing.
    let fg_pid = apollo_engine::engine::shadow_signals::get_foreground_pid();
    let any_interactive_candidate = snapshot.top_processes.iter().any(|p| {
        !seen.contains(&(p.pid, "boost"))
            && policy
                .interactive_patterns
                .iter()
                .any(|pat| p.name.contains(pat))
    });
    let visible_pids = if any_interactive_candidate {
        apollo_engine::engine::cg_window::visible_pids()
    } else {
        HashSet::new()
    };
    // 2026-06-21 — yield-to-frontmost-media: true iff the frontmost app is a
    // media host (browser/player, NOT chat/call) AND audio is live now. Only
    // then do visible-but-not-frontmost boosts yield. Computed once/cycle,
    // gated on a candidate existing so quiet cycles pay nothing. is_audio*
    // is ~50µs and fail-false. None foreground name → false (never yields).
    let frontmost_media_active = any_interactive_candidate
        && foreground_app
            .map(apollo_engine::engine::window_sensor::is_media_host)
            .unwrap_or(false)
        && apollo_engine::engine::coreaudio_active::is_audio_running_somewhere();

    for p in &snapshot.top_processes {
        if policy
            .interactive_patterns
            .iter()
            .any(|pat| p.name.contains(pat))
            && !seen.contains(&(p.pid, "boost"))
            && !survival
            && boost_visibility_ok(p.pid, fg_pid, &visible_pids, frontmost_media_active)
            && should_emit_boost(p.pid, learned_policy_boost_ttl_secs())
        {
            // Round-4 hotfix (2026-06-07): the FIX-1 guard at
            // decide_actions.rs:597/644/871 protected the rule-based
            // boost paths but missed this LLM-daemon emit site, which
            // applies the LEARNED policy classification (the very
            // signal corrupted by the Brave loop). Without this guard,
            // 4 Brave boosts/cycle slipped through post-deploy despite
            // 36 hard_protected_boost_skipped_total bumps elsewhere.
            // Apply complete-mediation here too — `is_boost_forbidden`
            // returns true for hard-protected names + Chromium
            // family-roots.
            if apollo_engine::engine::safety::is_boost_forbidden(&p.name) {
                apollo_engine::engine::lse_counters::LSE_COUNTERS
                    .inc_hard_protected_boost_skipped();
                continue;
            }
            let (ss, su) = pid_start_time(p.pid);
            actions.push(RootAction::BoostProcess {
                pid: p.pid,
                name: p.name.clone(),
                reason: "learned-policy interactive".to_string(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: ss,
                start_usec: su,
            });
            seen.insert((p.pid, "boost"));
        }
        if policy.noise_patterns.iter().any(|pat| p.name.contains(pat))
            && !seen.contains(&(p.pid, "throttle"))
            // Complete mediation (2026-06-18 bug-class sweep): a noise pattern
            // can match a protected/Apple/dev process or a Chromium helper. The
            // survival-mode upgrade makes this an AGGRESSIVE throttle (SIGSTOP
            // pulses) — on Chromium that breaks Brave's IPC contract. Never
            // throttle a protected name or the Chromium family on a name match.
            && !apollo_engine::engine::safety::is_protected_name(&p.name)
            && !apollo_engine::engine::safety::is_chromium_family(&p.name)
            // 2026-06-20: the teacher noise-classified `language_server` (the
            // LSP) even though it is interactive — a name-match conflict the
            // exact-contains add-guard missed. NEVER throttle a process that is
            // also interactive-classified, regardless of how it got into noise.
            // Truncation-aware (language_server vs language_server_macos_arm).
            && !policy.interactive_patterns.iter().any(|ip| {
                let nl = p.name.to_lowercase();
                let il = ip.to_lowercase();
                apollo_engine::engine::decide_actions::learned_pattern_matches(&nl, &il)
                    || apollo_engine::engine::decide_actions::learned_pattern_matches(&il, &nl)
            })
        {
            let (ss, su) = pid_start_time(p.pid);
            // Under survival mode, upgrade to aggressive throttle. Non-aggressive
            // (background QoS demotion) is too weak when swap ≥4GB — the process
            // still pages in/out at the same rate. Aggressive adds SIGSTOP pulses.
            // [Nygard 2018 §5] — under load, shed harder on processes already
            // classified as noise.
            actions.push(RootAction::ThrottleProcess {
                pid: p.pid,
                name: p.name.clone(),
                aggressive: survival,
                reason: if survival {
                    "learned-policy noise (survival)".to_string()
                } else {
                    "learned-policy noise".to_string()
                },
                start_sec: ss,
                start_usec: su,
                decision_reason: DecisionReason::PressureContext,
            });
            seen.insert((p.pid, "throttle"));
        }
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::collector::{
        CpuStats, MemoryStats, PressureStats, ProcessStats, SystemSnapshot,
    };

    const GB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn noise_add_rejects_interactive_and_protected() {
        // The teacher noise-classified the LSP. A noise pattern that shadows an
        // interactive entry (truncation-aware) or a protected name must be
        // rejected — exact .contains() missed language_server vs the stored
        // language_server_macos_arm.
        let interactive = vec!["language_server_macos_arm".to_string(), "Brave".to_string()];
        assert!(
            noise_pattern_conflicts("language_server", &interactive),
            "truncated LSP name must conflict with the full interactive pattern"
        );
        assert!(
            noise_pattern_conflicts("WindowServer", &interactive),
            "a safety-protected name must conflict"
        );
        assert!(
            !noise_pattern_conflicts("GoogleUpdater", &interactive),
            "a genuine noise process must NOT conflict"
        );
    }

    #[test]
    fn boost_only_foreground_or_visible() {
        let mut visible = HashSet::new();
        visible.insert(42u32); // a process with a visible window
        let no_media = false; // frontmost is not an active media host
                              // Foreground app → boostable.
        assert!(boost_visibility_ok(7, Some(7), &visible, no_media));
        // Visible window → boostable.
        assert!(boost_visibility_ok(42, Some(7), &visible, no_media));
        // Headless background (e.g. a node dev server): not foreground, no
        // window → MUST NOT be boosted on a name match alone.
        assert!(!boost_visibility_ok(999, Some(7), &visible, no_media));
        assert!(!boost_visibility_ok(999, None, &HashSet::new(), no_media));
    }

    #[test]
    fn boost_yields_to_frontmost_media() {
        // 2026-06-21: a visible-but-NOT-frontmost interactive process (e.g. a
        // terminal at pid 42) must YIELD its boost while the FRONTMOST app is an
        // active media host (Brave playing 4K) — the boost would steal P-core
        // compositing timeshare → occasional frame drop (node-54x class).
        let mut visible = HashSet::new();
        visible.insert(42u32); // the visible-but-background terminal
        let fg = Some(7u32); // frontmost app is pid 7 (the browser)

        // media active in the frontmost app:
        // (a) the FRONTMOST app itself is ALWAYS boostable — never yields.
        assert!(
            boost_visibility_ok(7, fg, &visible, true),
            "frontmost app must always be boostable, even during its own media"
        );
        // (b) the visible-but-not-frontmost terminal YIELDS.
        assert!(
            !boost_visibility_ok(42, fg, &visible, true),
            "visible non-frontmost boost must yield to the frontmost media app"
        );

        // media NOT active (or frontmost is not a media host) → unchanged:
        // (c) the same terminal is boosted as before.
        assert!(
            boost_visibility_ok(42, fg, &visible, false),
            "without frontmost media, a visible process is boosted as before"
        );
    }

    #[test]
    fn is_media_host_matches_players_not_chat() {
        use apollo_engine::engine::window_sensor::is_media_host;
        // Browsers + dedicated players → media hosts (yield).
        assert!(is_media_host("Brave Browser"));
        assert!(is_media_host("Google Chrome"));
        assert!(is_media_host("Safari"));
        assert!(is_media_host("Spotify"));
        assert!(is_media_host("VLC"));
        assert!(is_media_host("IINA"));
        // Chat/call apps → NOT media hosts (a Slack ping must not starve the
        // terminal; calls are handled by is_realtime_call_active).
        assert!(!is_media_host("Slack"));
        assert!(!is_media_host("Discord"));
        assert!(!is_media_host("zoom.us"));
        assert!(!is_media_host("Microsoft Teams"));
        // A terminal/editor → not a media host.
        assert!(!is_media_host("alacritty"));
        assert!(!is_media_host("Code"));
    }

    /// The teacher must actually run during the normal 4-5 GB swap this 8 GB box
    /// idles at (the 2 GB ceiling left it skipped permanently); it should bail
    /// only in a genuine >6 GB thrash crisis.
    #[test]
    fn metal_oom_guard_runs_under_normal_swap_skips_in_crisis() {
        assert!(!metal_oom_would_skip(0));
        assert!(!metal_oom_would_skip(4 * GB + GB / 2)); // 4.5 GB — measured-OK
        assert!(!metal_oom_would_skip(6 * GB)); // exactly at ceiling: still runs
        assert!(metal_oom_would_skip(6 * GB + 1)); // just over: skip
        assert!(metal_oom_would_skip(8 * GB)); // crisis
    }

    fn snapshot_with(processes: Vec<ProcessStats>) -> SystemSnapshot {
        SystemSnapshot {
            timestamp: chrono::Utc::now(),
            cpu: CpuStats {
                global_usage: 0.0,
                core_count: 1,
            },
            memory: MemoryStats {
                total_ram: 0,
                used_ram: 0,
                free_ram: 0,
                total_swap: 0,
                used_swap: 0,
            },
            pressure: PressureStats {
                memory_pressure: 0.0,
                swap_used_bytes: 0,
                swap_total_bytes: 0,
                swap_delta_bytes_per_sec: 0.0,
                thermal_level: "nominal".into(),
                compressor_pressure: 0.0,
                thrashing_score: 0.0,
                memory_pressure_raw: 0.0,
                refault_delta_per_sec: 0.0,
            },
            disks: vec![],
            networks: vec![],
            top_processes: processes,
        }
    }

    fn proc(pid: u32, name: &str, cpu: f32) -> ProcessStats {
        ProcessStats {
            pid,
            name: name.into(),
            cpu_usage: cpu,
            memory_usage: 0,
            cpu_wall_ratio: None,
        }
    }

    fn policy(interactive: &[&str], noise: &[&str], protected: &[&str]) -> LearnedPolicy {
        LearnedPolicy {
            interactive_patterns: std::sync::Arc::new(
                interactive.iter().map(|s| s.to_string()).collect(),
            ),
            noise_patterns: std::sync::Arc::new(noise.iter().map(|s| s.to_string()).collect()),
            protected_patterns: std::sync::Arc::new(
                protected.iter().map(|s| s.to_string()).collect(),
            ),
            learned_at: None,
            pattern_weights: HashMap::new(),
        }
    }

    // ── windowserver_cpu ─────────────────────────────────────────────────────

    #[test]
    fn windowserver_cpu_empty_snapshot_returns_zero() {
        assert_eq!(windowserver_cpu(&snapshot_with(vec![])), 0.0);
    }

    #[test]
    fn windowserver_cpu_finds_exact_name() {
        let snap = snapshot_with(vec![proc(1, "WindowServer", 42.5)]);
        assert_eq!(windowserver_cpu(&snap), 42.5);
    }

    #[test]
    fn windowserver_cpu_matches_substring() {
        let snap = snapshot_with(vec![proc(1, "com.apple.WindowServer", 10.0)]);
        assert_eq!(windowserver_cpu(&snap), 10.0);
    }

    #[test]
    fn windowserver_cpu_case_sensitive_miss() {
        let snap = snapshot_with(vec![proc(1, "windowserver", 99.0)]);
        assert_eq!(windowserver_cpu(&snap), 0.0, "lookup is case-sensitive");
    }

    // ── apply_learned_policy_actions ─────────────────────────────────────────

    #[test]
    fn apply_empty_policy_passthrough() {
        let snap = snapshot_with(vec![]);
        let actions = vec![RootAction::BoostProcess {
            pid: 1,
            name: "app".into(),
            reason: "r".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }];
        let result = apply_learned_policy_actions(&snap, &policy(&[], &[], &[]), actions, None);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn apply_protected_pattern_removes_freeze() {
        let snap = snapshot_with(vec![]);
        let actions = vec![RootAction::FreezeProcess {
            pid: 1,
            name: "claude".into(),
            reason: "r".into(),
            start_sec: 0,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        }];
        let result =
            apply_learned_policy_actions(&snap, &policy(&[], &[], &["claude"]), actions, None);
        assert!(result.is_empty(), "claude must be protected");
    }

    #[test]
    fn apply_protected_pattern_keeps_non_matching() {
        let snap = snapshot_with(vec![]);
        let actions = vec![RootAction::FreezeProcess {
            pid: 2,
            name: "slack".into(),
            reason: "r".into(),
            start_sec: 0,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        }];
        let result =
            apply_learned_policy_actions(&snap, &policy(&[], &[], &["claude"]), actions, None);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn apply_interactive_pattern_adds_boost() {
        apollo_engine::engine::shadow_signals::set_foreground_pid(Some(42));
        let snap = snapshot_with(vec![proc(42, "Xcode", 20.0)]);
        let result =
            apply_learned_policy_actions(&snap, &policy(&["Xcode"], &[], &[]), vec![], None);
        assert_eq!(result.len(), 1);
        match &result[0] {
            RootAction::BoostProcess { pid, name, .. } => {
                assert_eq!(*pid, 42);
                assert_eq!(name, "Xcode");
            }
            _ => panic!("expected BoostProcess"),
        }
        apollo_engine::engine::shadow_signals::set_foreground_pid(None);
    }

    #[test]
    fn apply_no_duplicate_boost_when_already_present() {
        apollo_engine::engine::shadow_signals::set_foreground_pid(Some(42));
        let snap = snapshot_with(vec![proc(42, "Xcode", 20.0)]);
        let existing = vec![RootAction::BoostProcess {
            pid: 42,
            name: "Xcode".into(),
            reason: "existing".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }];
        let result =
            apply_learned_policy_actions(&snap, &policy(&["Xcode"], &[], &[]), existing, None);
        let boosts = result
            .iter()
            .filter(|a| matches!(a, RootAction::BoostProcess { .. }))
            .count();
        assert_eq!(boosts, 1, "must not duplicate existing boost");
        apollo_engine::engine::shadow_signals::set_foreground_pid(None);
    }

    #[test]
    fn learned_policy_boost_ttl_is_longer_than_thirty_seconds() {
        assert!(
            learned_policy_boost_ttl_secs() >= 120,
            "mach QoS boosts are sticky; re-emitting every 30s creates journal noise"
        );
    }
}
