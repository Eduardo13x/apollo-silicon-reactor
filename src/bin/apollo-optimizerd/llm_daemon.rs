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

use apollo_optimizer::engine::daemon_helpers::pid_start_time;
use apollo_optimizer::engine::llm::{
    append_jsonl, delete_file_best_effort, load_repo_config, write_json, LearnedPolicy, LlmAdvisor,
};
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::safety::pattern_conflicts_with_protected;
use apollo_optimizer::engine::types::{HardPath, LlmRunMode, RootAction};

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

pub fn windowserver_cpu(snapshot: &apollo_optimizer::collector::SystemSnapshot) -> f32 {
    snapshot
        .top_processes
        .iter()
        .find(|p| p.name.contains("WindowServer"))
        .map(|p| p.cpu_usage)
        .unwrap_or(0.0)
}

// ── LLM Reactive Tick ──────────────────────────────────────────────────────

pub fn llm_reactive_tick(
    state: &SharedState,
    advisor: &mut LlmAdvisor,
    snapshot: &apollo_optimizer::collector::SystemSnapshot,
    counters: &mut LlmReactiveCounters,
    heuristic_struggling: bool,
) {
    let now = Utc::now();
    let has_key = state.llm_key_path.exists();

    // TTL housekeeping: if training expired, disable and delete key.
    {
        let mut llm_state = state.llm_state.lock_recover();
        if llm_state.enabled
            && llm_state
                .training_expires_at
                .map(|t| t <= now)
                .unwrap_or(true)
        {
            llm_state.enabled = false;
            llm_state.training_expires_at = None;
            llm_state.last_suggestion = None;
            llm_state.mode = LlmRunMode::Off;
            llm_state.last_error = Some("training-expired".to_string());
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            delete_file_best_effort(&state.llm_key_path);
            return;
        }
    }

    let llm_cfg = load_repo_config(&state.config_path)
        .llm
        .unwrap_or_else(|| state.llm_cfg.as_ref().clone());
    if !llm_cfg.enabled() {
        return;
    }

    // Keep advisor in sync with config edits.
    advisor.update_cfg(llm_cfg.clone());
    if !has_key {
        return;
    }

    let api_key = match HardPath::read_to_string_limited(&state.llm_key_path, 4096) {
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
        let mut llm_state = state.llm_state.lock_recover();
        if !llm_state.training_active() {
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            return;
        }

        // Reset daily budget window.
        if llm_state.calls_today_day.as_deref() != Some(&today) {
            llm_state.calls_today_day = Some(today.clone());
            llm_state.calls_today = 0;
        }

        // Keep trigger events only for a short horizon.
        llm_state
            .trigger_events
            .retain(|t| now - *t < ChronoDuration::minutes(30));
        let trigger_len = llm_state.trigger_events.len();
        if trigger_len > 100 {
            llm_state.trigger_events.drain(..trigger_len - 100);
        }
        let triggers_recent = llm_state.trigger_events.len() as u32;

        let bootcamp = llm_state
            .training_started_at
            .map(|t| now - t < ChronoDuration::days(5))
            .unwrap_or(false);
        let daily_budget = if bootcamp { 24 } else { 8 };

        // If we've been stable for a while, bias to strict.
        let stable_for = llm_state
            .no_trigger_since
            .map(|t| now - t)
            .unwrap_or_else(|| ChronoDuration::seconds(0));
        let stable_long = stable_for > ChronoDuration::hours(3);

        let consumed = llm_state.calls_today;
        let consumed_ratio = if daily_budget == 0 {
            1.0
        } else {
            (consumed as f64) / (daily_budget as f64)
        };

        let mut mode = llm_state.mode;
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
        llm_state.mode = mode;

        let (base_min_interval, base_max_calls, pattern_budget) = match mode {
            LlmRunMode::Sensitive => (600_u64, 4_u32, if bootcamp { 5_u32 } else { 3_u32 }),
            LlmRunMode::Strict => (1800_u64, 2_u32, 2_u32),
            LlmRunMode::Off => (u64::MAX, 0_u32, 0_u32),
        };

        // Respect config as a hard limiter for cadence.
        let effective_min_interval = base_min_interval.max(llm_cfg.min_interval_secs());
        let effective_max_calls = base_max_calls.min(llm_cfg.max_calls_per_hour().max(1));

        write_json(&state.llm_state_path, &*llm_state, Some(0o600));
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
        let llm_state = state.llm_state.lock_recover();
        llm_state.last_attempt_at.is_none()
            && llm_state
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
    }

    if !trigger_active {
        // Bootcamp sampling: even when the system is "fine", take an occasional sample call
        // so the teacher can learn normal workload patterns.
        let sampling_due = {
            let llm_state = state.llm_state.lock_recover();
            let since_last = llm_state
                .last_attempt_at
                .map(|t| now - t)
                .unwrap_or_else(|| ChronoDuration::hours(24));
            let user_active_proxy = ws_cpu >= 10.0 || snapshot.cpu.global_usage >= 15.0;
            mode == LlmRunMode::Sensitive
                && llm_state
                    .training_started_at
                    .map(|t| now - t < ChronoDuration::days(5))
                    .unwrap_or(false)
                && user_active_proxy
                && since_last > ChronoDuration::minutes(45)
        };

        let mut llm_state = state.llm_state.lock_recover();
        if llm_state.no_trigger_since.is_none() {
            llm_state.no_trigger_since = Some(now);
        }

        if sampling_due {
            llm_state.last_trigger_at = Some(now);
            llm_state.last_trigger_reason = Some("sampling".to_string());
            llm_state.trigger_events.push(now);
            llm_state.no_trigger_since = None;
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            drop(llm_state);
            // Turn sampling into a synthetic rising-edge trigger.
            rising_edge = true;
        } else {
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
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
        let mut llm_state = state.llm_state.lock_recover();
        llm_state.last_trigger_at = Some(now);
        llm_state.last_trigger_reason = Some(trigger_reason.clone());
        llm_state.trigger_events.push(now);
        llm_state.no_trigger_since = None;
        write_json(&state.llm_state_path, &*llm_state, Some(0o600));
    }

    // Call gating: only call on rising edge.
    if !rising_edge {
        return;
    }

    // Budget + cadence.
    {
        let mut llm_state = state.llm_state.lock_recover();

        if llm_state.calls_today >= daily_budget {
            llm_state.mode = LlmRunMode::Off;
            llm_state.last_error = Some("daily-budget-exhausted".to_string());
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            return;
        }

        if let Some(last) = llm_state.last_attempt_at {
            if now - last < ChronoDuration::seconds(min_interval_secs as i64) {
                return;
            }
        }

        // Per-hour window.
        if llm_state
            .hour_window_started_at
            .map(|t| now - t > ChronoDuration::hours(1))
            .unwrap_or(true)
        {
            llm_state.hour_window_started_at = Some(now);
            llm_state.calls_in_window = 0;
        }
        if llm_state.calls_in_window >= max_calls_per_hour {
            return;
        }

        // Record attempt before the network call so status updates immediately.
        llm_state.last_attempt_at = Some(now);
        llm_state.last_http_status = None;
        llm_state.last_error = None;
        llm_state.calls_in_window += 1;
        llm_state.calls_today += 1;
        write_json(&state.llm_state_path, &*llm_state, Some(0o600));
    }

    // Network call (no locks held).
    let current_policy = state.learned_policy.lock_recover().clone();
    let suggestion_res = advisor.call_raw(snapshot, &api_key, Some(&current_policy));

    // Apply suggestion and persist state.
    match suggestion_res {
        Ok(suggestion) => {
            let accepted = suggestion.confidence >= llm_cfg.min_confidence();
            {
                let mut llm_state = state.llm_state.lock_recover();
                llm_state.last_http_status = Some(200);
                llm_state.last_call_at = Some(now);
                llm_state.last_suggestion = Some(suggestion.clone());
                llm_state.consecutive_failures = 0;
                if !accepted {
                    llm_state.last_error = Some("below-min-confidence".to_string());
                }
                write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            }

            append_jsonl(
                &state.suggestions_path,
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
                let mut gov = state.governor.lock_recover();
                if gov.manual_override.is_none() {
                    gov.set_manual_override(p, 20, "llm-reactive".to_string());
                }
            }
            // 2) Latency target.
            if let Some(t) = suggestion.suggested_latency_target {
                *state.latency_target.lock_recover() = t;
            }

            // 3) Learned patterns: merge with daily cap.
            {
                let mut llm_state = state.llm_state.lock_recover();
                let day = now.date_naive();
                let reset_day = llm_state
                    .policy_updates_day
                    .map(|d| d.date_naive() != day)
                    .unwrap_or(true);
                if reset_day {
                    llm_state.policy_updates_day = Some(now);
                    llm_state.policy_updates_today = 0;
                }
                let remaining =
                    pattern_budget_per_day.saturating_sub(llm_state.policy_updates_today);
                if remaining == 0 {
                    write_json(&state.llm_state_path, &*llm_state, Some(0o600));
                    return;
                }

                let mut policy = state.learned_policy.lock_recover();

                let mut added = 0u32;
                for p in suggestion
                    .add_interactive_patterns
                    .iter()
                    .take(remaining as usize)
                {
                    if !policy.interactive_patterns.contains(p)
                        && !pattern_conflicts_with_protected(p)
                    {
                        // Remove from noise if promoted to interactive.
                        policy.noise_patterns.retain(|n| n != p);
                        policy.interactive_patterns.push(p.clone());
                        added += 1;
                    }
                }
                for p in suggestion
                    .add_noise_patterns
                    .iter()
                    .take(remaining.saturating_sub(added) as usize)
                {
                    // Skip if already protected or interactive — cannot downgrade.
                    if !policy.noise_patterns.contains(p)
                        && !pattern_conflicts_with_protected(p)
                        && !policy.protected_patterns.contains(p)
                        && !policy.interactive_patterns.contains(p)
                    {
                        policy.noise_patterns.push(p.clone());
                        added += 1;
                    }
                }
                for p in suggestion
                    .add_protected_patterns
                    .iter()
                    .take(remaining.saturating_sub(added) as usize)
                {
                    if !policy.protected_patterns.contains(p)
                        && !pattern_conflicts_with_protected(p)
                    {
                        // Remove from noise when promoted to protected.
                        policy.noise_patterns.retain(|n| n != p);
                        policy.protected_patterns.push(p.clone());
                        added += 1;
                    }
                }

                if added > 0 {
                    policy.interactive_patterns.sort();
                    policy.noise_patterns.sort();
                    policy.protected_patterns.sort();
                    policy.learned_at = Some(now);
                    write_json(&state.learned_policy_path, &*policy, Some(0o600));
                    llm_state.policy_updates_today += added;
                    // Propagate updated patterns to the ML Ligero classifier.
                    {
                        let mut gov = state.adaptive_governor.lock_recover();
                        gov.update_learned_policy(&policy);
                    }
                }
                write_json(&state.llm_state_path, &*llm_state, Some(0o600));
            }
        }
        Err(err) => {
            let mut llm_state = state.llm_state.lock_recover();
            llm_state.consecutive_failures += 1;
            match err {
                apollo_optimizer::engine::llm::LlmCallError::Cooldown => {
                    llm_state.last_error = Some("cooldown".to_string());
                }
                apollo_optimizer::engine::llm::LlmCallError::HttpStatus { code, body_excerpt } => {
                    llm_state.last_http_status = Some(code);
                    llm_state.last_error = Some(format!(
                        "http-status {} {}",
                        code,
                        body_excerpt.unwrap_or_default()
                    ));
                }
                apollo_optimizer::engine::llm::LlmCallError::Transport(e) => {
                    llm_state.last_error = Some(format!("transport {}", e));
                }
                apollo_optimizer::engine::llm::LlmCallError::Parse(e) => {
                    llm_state.last_error = Some(format!("parse {}", e));
                }
                apollo_optimizer::engine::llm::LlmCallError::Rejected(e) => {
                    llm_state.last_error = Some(format!("rejected {}", e));
                }
            }

            // Fail-safe: if it's repeatedly failing, go strict to save cost.
            if llm_state.consecutive_failures >= 3 {
                llm_state.mode = LlmRunMode::Strict;
            }
            write_json(&state.llm_state_path, &*llm_state, Some(0o600));
        }
    }
}

// ── Usage Learning Tick ────────────────────────────────────────────────────

pub fn usage_learning_tick(
    state: &SharedState,
    snapshot: &apollo_optimizer::collector::SystemSnapshot,
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
        let due = usage.usage_tracker
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
        let policy = state.learned_policy.lock_recover().clone();
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
    {
        let mut policy = state.learned_policy.lock_recover();
        for (kind, pattern) in &promotions {
            match kind.as_str() {
                "interactive" => {
                    if !policy.interactive_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                    {
                        policy.interactive_patterns.push(pattern.clone());
                        applied += 1;
                    }
                }
                "noise" => {
                    if !policy.noise_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                    {
                        policy.noise_patterns.push(pattern.clone());
                        applied += 1;
                    }
                }
                "protected" => {
                    // Protected patterns are safety labels — they bypass the daily
                    // cap and only require that the pattern isn't already present.
                    if !policy.protected_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                    {
                        policy.protected_patterns.push(pattern.clone());
                        applied += 1;
                    }
                }
                _ => {}
            }
        }
        if applied > 0 {
            policy.interactive_patterns.sort();
            policy.noise_patterns.sort();
            policy.protected_patterns.sort();
            policy.learned_at = Some(now);
            write_json(&state.learned_policy_path, &*policy, Some(0o600));
            // Propagate updated patterns to the ML Ligero classifier.
            {
                let mut gov = state.adaptive_governor.lock_recover();
                gov.update_learned_policy(&policy);
            }
        }
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

pub fn apply_learned_policy_actions(
    snapshot: &apollo_optimizer::collector::SystemSnapshot,
    policy: &LearnedPolicy,
    mut actions: Vec<RootAction>,
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

    for p in &snapshot.top_processes {
        if policy
            .interactive_patterns
            .iter()
            .any(|pat| p.name.contains(pat))
            && !seen.contains(&(p.pid, "boost"))
        {
            actions.push(RootAction::BoostProcess {
                pid: p.pid,
                name: p.name.clone(),
                reason: "learned-policy interactive".to_string(),
            });
            seen.insert((p.pid, "boost"));
        }
        if policy.noise_patterns.iter().any(|pat| p.name.contains(pat))
            && !seen.contains(&(p.pid, "throttle"))
        {
            let (ss, su) = pid_start_time(p.pid);
            actions.push(RootAction::ThrottleProcess {
                pid: p.pid,
                name: p.name.clone(),
                aggressive: false,
                reason: "learned-policy noise".to_string(),
                start_sec: ss,
                start_usec: su,
            });
            seen.insert((p.pid, "throttle"));
        }
    }

    actions
}
