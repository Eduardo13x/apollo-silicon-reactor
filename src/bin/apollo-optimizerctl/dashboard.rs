//! ASCII dashboard renderer for apollo-optimizerctl.
//!
//! Renders a visual summary of daemon status using Unicode box-drawing,
//! ANSI colors, and emoji indicators.

use apollo_engine::engine::types::{BlockerScore, DaemonStatus, OptimizationProfile, SafetyPolicy};

const CW: usize = 66; // content width (visible columns between ║ padding)

// ── ANSI color helpers ──

fn green(s: &str) -> String {
    format!("\x1b[32m{s}\x1b[0m")
}
fn yellow(s: &str) -> String {
    format!("\x1b[33m{s}\x1b[0m")
}
fn red(s: &str) -> String {
    format!("\x1b[31m{s}\x1b[0m")
}
fn bold(s: &str) -> String {
    format!("\x1b[1m{s}\x1b[0m")
}
fn dim(s: &str) -> String {
    format!("\x1b[2m{s}\x1b[0m")
}

// ── Display width (handles ANSI codes + emoji) ──

fn is_wide_char(c: char) -> bool {
    let cp = c as u32;
    matches!(
        cp,
        0x1F300..=0x1F9FF
            | 0x2600..=0x27BF
            | 0x1FA00..=0x1FAFF
            | 0x2300..=0x23FF
            | 0x2B50..=0x2B55
    )
}

fn display_width(s: &str) -> usize {
    let mut w = 0;
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
            continue;
        }
        if in_escape {
            if c == 'm' {
                in_escape = false;
            }
            continue;
        }
        // Zero-width joiners and variation selectors
        if matches!(c as u32, 0x200B..=0x200F | 0xFE0F) {
            continue;
        }
        w += if is_wide_char(c) { 2 } else { 1 };
    }
    w
}

// ── Box drawing ──

fn box_top() -> String {
    format!("╔{}╗", "═".repeat(CW + 2))
}
fn box_bottom() -> String {
    format!("╚{}╝", "═".repeat(CW + 2))
}
fn box_div() -> String {
    format!("╠{}╣", "═".repeat(CW + 2))
}
fn box_empty() -> String {
    format!("║ {} ║", " ".repeat(CW))
}

fn box_line(content: &str) -> String {
    let dw = display_width(content);
    let pad = CW.saturating_sub(dw);
    format!("║ {}{} ║", content, " ".repeat(pad))
}

// ── Formatters ──

fn pad_right(s: &str, width: usize) -> String {
    let dw = display_width(s);
    let pad = width.saturating_sub(dw);
    format!("{}{}", s, " ".repeat(pad))
}

fn render_bar(ratio: f64, width: usize) -> String {
    let ratio = ratio.clamp(0.0, 1.0);
    let filled = (ratio * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    let fill_str = "█".repeat(filled);
    let empty_str = dim(&"░".repeat(empty));
    let colored_fill = if ratio >= 0.85 {
        red(&fill_str)
    } else if ratio >= 0.60 {
        yellow(&fill_str)
    } else {
        green(&fill_str)
    };
    format!("[{}{}]", colored_fill, empty_str)
}

/// Classify swap status for display. Pure function — testable in isolation.
///
/// Priority: rate signal (growing/falling) > absolute amount.
/// A swap growing rapidly is more urgent than static large swap.
#[cfg(test)]
fn swap_status_label(swap_gb: f64, delta_bps: f64) -> &'static str {
    swap_status_label_for_total(swap_gb, 8.0, delta_bps)
}

fn swap_status_label_for_total(swap_gb: f64, swap_total_gb: f64, delta_bps: f64) -> &'static str {
    if delta_bps > 100.0 {
        "📈 Creciendo"
    } else if delta_bps < -100.0 {
        "📉 Bajando"
    } else if swap_total_gb > 0.0 && swap_gb / swap_total_gb >= 0.85 {
        "🔴 Crítico"
    } else if swap_total_gb > 0.0 && swap_gb / swap_total_gb >= 0.50 {
        "🟠 Alto"
    } else {
        "🟢 Estable"
    }
}

fn swap_visual_ratio(swap_used_bytes: u64, swap_total_bytes: u64) -> f64 {
    let fallback_capacity = 8.0 * 1_073_741_824.0;
    let capacity = if swap_total_bytes > 0 {
        swap_total_bytes as f64
    } else {
        fallback_capacity
    };
    (swap_used_bytes as f64 / capacity).clamp(0.0, 1.0)
}

fn score_emoji(score: f64) -> &'static str {
    if score >= 0.7 {
        "🔴"
    } else if score >= 0.4 {
        "🟡"
    } else {
        "🟢"
    }
}

fn score_label(score: f64) -> &'static str {
    if score >= 0.7 {
        "Crítico"
    } else if score >= 0.4 {
        "Moderado"
    } else {
        "Bajo"
    }
}

fn thermal_label(state: &str) -> &'static str {
    match state {
        "critical" => "Crítico",
        "serious" => "Serio",
        "moderate" | "fair" => "Moderado",
        _ => "Nominal",
    }
}

fn profile_emoji(p: OptimizationProfile) -> &'static str {
    match p {
        OptimizationProfile::AggressiveRoot => "⚡",
        OptimizationProfile::SafeRoot => "🛡️",
        OptimizationProfile::BalancedRoot => "🔵",
    }
}

/// Short, current explanation for a profile that differs from its configured
/// base. The governor refreshes `transition_reason` on every evaluation, so it
/// takes precedence over raw signals that may have been deliberately gated.
fn profile_activity_reason(status: &DaemonStatus) -> &'static str {
    if status.override_active {
        "Override manual"
    } else if status.effective_profile == status.base_profile {
        "Base estable"
    } else if status.effective_profile == OptimizationProfile::SafeRoot
        && status.transition_reason == "steady"
    {
        "Auto: ahorro en reposo"
    } else if status.transition_reason == "context-switch-burst-suppressed-calm" {
        "Auto: liberando por calma"
    } else if status.transition_reason == "context-switch-burst"
        || (status.metrics.context_switch_burst && status.metrics.arousal_level >= 0.25)
    {
        "Auto: cambios de app"
    } else if status.metrics.dev_session_active {
        "Auto: desarrollo"
    } else if status.metrics.interactive_heavy {
        "Auto: interacción"
    } else if status.metrics.memory_pressure >= 0.60 || status.metrics.si_p_oom_30s >= 0.30 {
        "Auto: memoria"
    } else if matches!(status.thermal_state.as_str(), "serious" | "critical") {
        "Auto: térmico"
    } else {
        "Auto: transición"
    }
}

fn format_number(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{},{:03}", n / 1000, n % 1000)
    } else {
        n.to_string()
    }
}

/// Fixed-width counter for the 32-column dashboard quadrants.
fn compact_counter(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.0}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.0}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

// ── Box drawing ──

fn render_blockers(blockers: &[BlockerScore]) -> Vec<String> {
    // Hide section entirely when no blockers — quiet UI, no alarming empty red.
    if blockers.is_empty() {
        return Vec::new();
    }
    let mut lines = vec![bold("🔴 TOP BLOQUEADORES")];

    lines.push("┌────┬──────────────────┬────────┬────────┬─────────────────┐".into());
    lines.push("│ #  │ Proceso          │  PID   │ Score  │ Veredicto       │".into());
    lines.push("├────┼──────────────────┼────────┼────────┼─────────────────┤".into());

    for (i, b) in blockers.iter().take(5).enumerate() {
        let idx = format!("{}", i + 1);
        let name: &str = if b.name.len() > 16 {
            &b.name[..16]
        } else {
            &b.name
        };
        let verdict = format!("{} {}", score_emoji(b.score), score_label(b.score));

        lines.push(format!(
            "│ {} │ {} │ {} │ {} │ {} │",
            pad_right(&idx, 2),
            pad_right(name, 16),
            pad_right(&format!("{}", b.pid), 6),
            pad_right(&format!("{:.2}", b.score), 6),
            pad_right(&verdict, 15),
        ));
    }

    lines.push("└────┴──────────────────┴────────┴────────┴─────────────────┘".into());

    lines
}

// ────────────────────────────────────────────────────────────────────────────
// Cognitive Stack Grid Layout (v2) — 2026-05-10
//
// Replaces linear 11-section render with 4-quadrant grid (SENSE / THINK /
// DECIDE / ACT) + full-width bands (GATES, CHROMIUM, CONSUMERS, VERDICT).
// Surfaces all 8 predictive subsystems + maintenance purge + cognition.
// ────────────────────────────────────────────────────────────────────────────

const QW: usize = 32; // width per quadrant content (lines fit in QW columns)

/// Pair two columns line-by-line. Pads each side to QW.
fn render_pair(left: &[String], right: &[String]) -> Vec<String> {
    let max_lines = left.len().max(right.len());
    let mut out = Vec::with_capacity(max_lines);
    for i in 0..max_lines {
        let l = left.get(i).cloned().unwrap_or_default();
        let r = right.get(i).cloned().unwrap_or_default();
        let lp = pad_right(&l, QW);
        let rp = pad_right(&r, QW);
        out.push(format!("{} {}", lp, rp));
    }
    out
}

/// Quadrant header bar like "─ 🔍 SENSE ──────────────────────".
fn q_header(emoji: &str, title: &str) -> String {
    let prefix = format!("─ {} {} ", emoji, title);
    let dw = display_width(&prefix);
    let pad = QW.saturating_sub(dw);
    format!("{}{}", prefix, "─".repeat(pad))
}

// ── 🔍 SENSE quadrant ────────────────────────────────────────────────────────
fn render_sense_q(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold(&q_header("🔍", "SENSE"))];

    let mp = m.memory_pressure;
    lines.push(format!(
        "RAM    {} {:>3.0}%",
        render_bar(mp, 12),
        mp * 100.0
    ));

    // Swap: normalize against the daemon-reported dynamic swap capacity when
    // available, with the historical 8GB visual fallback for old metrics.
    let swap_gb = m.swap_used_bytes as f64 / 1_073_741_824.0;
    let swap_total_gb = m.swap_total_bytes as f64 / 1_073_741_824.0;
    let swap_visual_ratio = swap_visual_ratio(m.swap_used_bytes, m.swap_total_bytes);
    let swap_label = swap_status_label_for_total(swap_gb, swap_total_gb.max(8.0), m.swap_delta_bps);
    lines.push(format!(
        "Swap   {} {:.1}GB",
        render_bar(swap_visual_ratio, 8),
        swap_gb
    ));
    lines.push(format!("       {}", swap_label));

    let thermal_sensor_available = m.smc_cpu_temp_celsius.is_some()
        || m.smc_gpu_temp_celsius.is_some()
        || m.iokit_p_cluster_temp.is_some()
        || m.iokit_e_cluster_temp.is_some();
    lines.push(format!(
        "Therm  k:{} sensor:{}",
        thermal_label(&status.thermal_state),
        if thermal_sensor_available {
            "ok"
        } else {
            "n/a"
        }
    ));

    lines.push(format!(
        "Pres   p={:.0}% c={:.0}%",
        mp * 100.0,
        m.compressed_memory_ratio * 100.0
    ));

    let score = m.last_pressure_score;
    lines.push(format!(
        "Score  {:.2} {} {}",
        score,
        score_emoji(score),
        score_label(score)
    ));

    lines.push(format!("Throttle {}", status.throttle_level));
    lines.push(format!("WS     {}% CPU", m.windowserver_cpu_pct as i32));
    lines.push(format!(
        "UX-p   {} {:.0}%",
        if m.perceptual_latency_category.is_empty() {
            "unmeasured"
        } else {
            m.perceptual_latency_category.as_str()
        },
        m.perceptual_latency_score * 100.0
    ));
    if m.scheduler_jitter_samples > 0 {
        lines.push(format!(
            "Sched  p95 {:.2}ms n{}",
            m.scheduler_jitter_p95_ms,
            format_number(m.scheduler_jitter_samples)
        ));
    }

    if let Some(top) = m.wakeup_vampires.first() {
        // wakeup_vampires entries are pre-formatted "name(rate/s)" strings
        let truncated: String = top.chars().take(24).collect();
        lines.push(format!("Wake   {}", truncated));
    }

    lines
}

// ── 🧠 THINK quadrant ────────────────────────────────────────────────────────
fn render_think_q(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold(&q_header("🧠", "THINK"))];

    // NARS (concept drift, beliefs)
    let drift_emoji = if m.nars_drift_score < 0.10 {
        "✅"
    } else if m.nars_drift_score < 0.30 {
        "⚠"
    } else {
        "🔴"
    };
    // Show drifted/total so "0/3000" reads as "0 of 3000 beliefs in drift",
    // not the old ambiguous "0 bel" (which looked like zero beliefs total).
    lines.push(format!(
        "NARS   {}/{} drift d={:.2} {}",
        m.nars_drifted_beliefs, m.nars_beliefs_total, m.nars_drift_score, drift_emoji
    ));

    let compact_profile = |profile: &str| match profile {
        "adaptive-multicore" => "multi",
        "sequential" => "seq",
        "" => "n/a",
        _ => "other",
    };
    if !m.parallel_expected_profile.is_empty()
        || !m.parallel_compiled_profile.is_empty()
        || !m.parallel_effective_profile.is_empty()
    {
        lines.push(format!(
            "CPU-P  exp {} build {}",
            compact_profile(&m.parallel_expected_profile),
            compact_profile(&m.parallel_compiled_profile),
        ));
        lines.push(format!(
            "       eff {} W{}/{} {} {}",
            compact_profile(&m.parallel_effective_profile),
            m.parallel_worker_threads,
            m.parallel_max_worker_threads,
            if m.parallel_worker_qos_intent.is_empty() {
                "default"
            } else {
                m.parallel_worker_qos_intent.as_str()
            },
            if !m.parallel_worker_qos_status.is_empty() {
                m.parallel_worker_qos_status.clone()
            } else if m.parallel_worker_qos_failures == 0 {
                "unknown".to_string()
            } else {
                format!("fail{}", m.parallel_worker_qos_failures)
            },
        ));
        if !m.parallel_disabled_reason.is_empty() {
            lines.push(format!(
                "       off {}",
                m.parallel_disabled_reason
                    .chars()
                    .take(20)
                    .collect::<String>()
            ));
        }
    }

    if m.reflex_enabled || !m.reflex_phase.is_empty() {
        lines.push(format!(
            "Reflex {} {}/{}",
            if m.reflex_phase.is_empty() {
                "disabled"
            } else {
                m.reflex_phase.as_str()
            },
            format_number(m.reflex_valid_cycles),
            format_number(m.reflex_shadow_cycles),
        ));
        if !m.reflex_blocker.is_empty() && m.reflex_blocker != "ready" {
            lines.push(format!(
                "       block {}",
                m.reflex_blocker.chars().take(18).collect::<String>()
            ));
        }
        lines.push(format!(
            "R-act  P{} A{} real{} sh{}",
            compact_counter(m.reflex_proposed_total),
            compact_counter(m.reflex_admitted_total),
            compact_counter(m.reflex_applied_total),
            compact_counter(m.reflex_shadowed_total),
        ));
        lines.push(format!(
            "       O{} no{} V{} R{} F{}",
            compact_counter(m.reflex_omitted_total),
            compact_counter(m.reflex_noop_total),
            compact_counter(m.reflex_vetoed_total),
            compact_counter(m.reflex_reverted_total),
            compact_counter(m.reflex_failed_total),
        ));
        lines.push(format!(
            "       fast {}us",
            compact_counter(m.reflex_last_decision_latency_us),
        ));
        lines.push(format!(
            "R-deep W{}/{} d{} age{} {}us",
            compact_counter(m.reflex_reasoning_completed_total),
            compact_counter(m.reflex_reasoning_submitted_total),
            compact_counter(m.reflex_reasoning_dropped_total),
            compact_counter(m.reflex_reasoning_last_result_age_cycles),
            compact_counter(m.reflex_reasoning_last_latency_us),
        ));
    }

    if !m.value_scheduler_phase.is_empty() {
        lines.push(format!(
            "Value  {} obs{} J{}/{} {}/{}ms",
            m.value_scheduler_phase,
            compact_counter(m.value_scheduler_valid_cycles),
            m.value_scheduler_selected_jobs,
            m.value_scheduler_registered_jobs,
            m.value_scheduler_predicted_us / 1_000,
            m.value_scheduler_budget_us / 1_000,
        ));
        // `J` is this cycle; `sel` is lifetime. They used to sit on one line
        // with the cumulative counters, reading as if they shared a window.
        lines.push(format!(
            "       now J{}/{} elig{} · sel{}",
            m.value_scheduler_selected_jobs,
            m.value_scheduler_registered_jobs,
            m.value_scheduler_eligible_jobs,
            compact_counter(m.value_scheduler_selected_total),
        ));
        if m.value_scheduler_invalid_samples_total > 0 {
            lines.push(format!(
                "       inv{} seq{} feat{} pub{}",
                compact_counter(m.value_scheduler_invalid_samples_total),
                compact_counter(m.value_scheduler_invalid_sequence_total),
                compact_counter(m.value_scheduler_invalid_features_total),
                compact_counter(m.value_scheduler_invalid_publication_total),
            ));
            let refusals = [
                ("sleep", m.value_scheduler_unhealthy_sleeping_total),
                ("kill", m.value_scheduler_unhealthy_kill_switch_total),
                ("prof", m.value_scheduler_unhealthy_profile_total),
                ("pres", m.value_scheduler_unhealthy_pressure_total),
                ("therm", m.value_scheduler_unhealthy_thermal_total),
                ("p95", m.value_scheduler_unhealthy_p95_total),
                ("lat", m.value_scheduler_unhealthy_latency_total),
            ];
            let named: u64 = refusals.iter().map(|(_, total)| *total).sum();
            let mut shown = refusals
                .iter()
                .filter(|(_, total)| *total > 0)
                .map(|(label, total)| format!("{label}{}", compact_counter(*total)))
                .collect::<Vec<_>>();
            // Pre-breakdown history has no reason. Say so rather than let the
            // named buckets look like they explain every refusal.
            let unattributed = m
                .value_scheduler_invalid_unhealthy_total
                .saturating_sub(named);
            if unattributed > 0 {
                shown.push(format!("pre{}", compact_counter(unattributed)));
            }
            for (index, chunk) in shown.chunks(3).enumerate() {
                lines.push(format!(
                    "       {}{}",
                    if index == 0 { "why " } else { "    " },
                    chunk.join(" ")
                ));
            }
        }
        if !m.value_scheduler_blocker.is_empty() && m.value_scheduler_blocker != "shadow-ready" {
            lines.push(format!(
                "       block {}",
                m.value_scheduler_blocker
                    .chars()
                    .take(20)
                    .collect::<String>()
            ));
        }
    }

    if !m.fabric_phase.is_empty() {
        lines.push(format!(
            "Fabric {} W{} C{}/{} X{}",
            m.fabric_phase,
            compact_counter(m.fabric_workers_active),
            compact_counter(m.fabric_completed_total),
            compact_counter(m.fabric_submitted_total),
            compact_counter(
                m.fabric_cancelled_total
                    .saturating_add(m.fabric_stale_total)
                    .saturating_add(m.fabric_deadline_misses_total)
            ),
        ));
        lines.push(format!(
            "Cost   cpu{:.2}% rss{}M p95{:.0}",
            m.fabric_cpu_percent,
            m.fabric_rss_delta_bytes / (1024 * 1024),
            m.fabric_control_p95_baseline_ms,
        ));
        // `cfg` prefixes the backend so the row cannot be read as "inference
        // ran here". Core ML publishes no dispatch target, so the ANE column
        // reports the evidence status verbatim rather than a glyph that a
        // reader would resolve to "fine" or "broken".
        let ane = match m.coreml_ane_observation.as_str() {
            "measured-active" => "run",
            "measured-idle" => "idle",
            "unavailable" => "n/a",
            // Includes the empty string from a daemon predating this field.
            _ => "unobservable",
        };
        let ml_backend = match m.coreml_configured_backend.as_str() {
            "cpu-and-neural-engine" => "cpu+ane",
            "cpu-only" => "cpu",
            "all" => "all",
            "" => "unavailable",
            other => other,
        };
        lines.push(format!(
            "ML     cfg:{} E{}/{}",
            ml_backend,
            compact_counter(m.fabric_evaluation_total),
            compact_counter(m.fabric_eligible_total),
        ));
        lines.push(format!("       ANE {ane}"));
        if !m.temporal_prediction_backend.is_empty() {
            lines.push(format!(
                "Pred   {} {} p95{:.0}%",
                m.temporal_prediction_backend
                    .chars()
                    .take(12)
                    .collect::<String>(),
                if m.temporal_prediction_authoritative {
                    "active"
                } else {
                    "shadow"
                },
                m.temporal_prediction_p95 * 100.0,
            ));
        }
        if !m.fabric_blocker.is_empty() {
            lines.push(format!(
                "       block {}",
                m.fabric_blocker.chars().take(20).collect::<String>()
            ));
        }
    }

    if !m.microexperiment_phase.is_empty() {
        lines.push(format!(
            "Lab    {} would{} open{} Gold{}",
            m.microexperiment_phase,
            compact_counter(m.microexperiment_shadow_would_open_total),
            compact_counter(m.microexperiment_open_pairs),
            compact_counter(m.microexperiment_pair_gold_total),
        ));
        if m.microexperiment_rollout_required > 0 {
            lines.push(format!(
                "       gate {}/{} boot{}",
                compact_counter(m.microexperiment_rollout_progress),
                compact_counter(m.microexperiment_rollout_required),
                compact_counter(m.microexperiment_restored_progress_at_boot),
            ));
        }
        if m.microexperiment_progress_resets_total > 0 {
            lines.push(format!(
                "       reset{} {}",
                compact_counter(m.microexperiment_progress_resets_total),
                m.microexperiment_last_progress_reset_reason,
            ));
        }
        lines.push(format!(
            "       eff{} harm{} q{}",
            compact_counter(m.microexperiment_effective_total),
            compact_counter(m.microexperiment_harmful_total),
            compact_counter(m.microexperiment_synthetic_quarantined_total),
        ));
        // Endpoint wire. `wire` is the contract state; the rest separate the
        // stages so a stalled circuit names its own cause.
        lines.push(format!(
            "       wire{} arm{} bind{} end{} wait{}",
            if m.microexperiment_endpoint_contract_ready {
                "+"
            } else {
                "-"
            },
            compact_counter(m.microexperiment_arms_registered_total),
            compact_counter(m.microexperiment_decisions_bound_total),
            compact_counter(m.microexperiment_endpoints_emitted_total),
            compact_counter(m.microexperiment_endpoints_pending_utility),
        ));
        let endpoint_rejects = [
            ("key", m.microexperiment_endpoint_action_mismatch_total),
            ("uncat", m.microexperiment_uncatalogued_episodes_total),
            ("arm", m.microexperiment_endpoint_unknown_arm_total),
            ("dup", m.microexperiment_endpoint_duplicate_total),
            ("exp", m.microexperiment_endpoint_expired_total),
            ("epo", m.microexperiment_endpoint_epoch_rejected_total),
            ("aut", m.microexperiment_endpoint_authority_rejected_total),
            ("met", m.microexperiment_endpoint_incomplete_metadata_total),
            ("cap", m.microexperiment_endpoint_capacity_drops_total),
        ];
        let rejects = endpoint_rejects
            .iter()
            .filter(|(_, total)| *total > 0)
            .map(|(label, total)| format!("{label}{}", compact_counter(*total)))
            .collect::<Vec<_>>();
        if !rejects.is_empty() {
            lines.push(format!("       rej {}", rejects.join(" ")));
        }
        if m.microexperiment_control_withholds_total > 0
            || m.microexperiment_endpoint_rollback_failed_total > 0
        {
            lines.push(format!(
                "       hold{} rbk{}/{}",
                compact_counter(m.microexperiment_control_withholds_total),
                compact_counter(m.microexperiment_endpoint_rollback_observed_total),
                compact_counter(m.microexperiment_endpoint_rollback_failed_total),
            ));
        }
        if m.microexperiment_invalidated_total > 0 {
            lines.push(format!(
                "       inv{} dl{} rbf{}",
                compact_counter(m.microexperiment_invalidated_total),
                compact_counter(m.microexperiment_deadline_expired_total),
                compact_counter(m.microexperiment_rollback_failed_total),
            ));
        }
        if !m.microexperiment_blocker.is_empty()
            && m.microexperiment_blocker != "shadow-observed"
            && m.microexperiment_blocker != "no-candidates"
        {
            lines.push(format!(
                "       block {}",
                m.microexperiment_blocker
                    .chars()
                    .take(20)
                    .collect::<String>()
            ));
        }
    }

    // Local Gold outcomes are compiled into fast System 1 reflexes.
    lines.push(format!(
        "Learn  Gold{} F{} S1+{}",
        compact_counter(m.system_deliberation_local_gold),
        m.system_deliberation_local_families,
        compact_counter(m.local_consolidation_system1_updates),
    ));

    if m.ais_score > 0.0 {
        lines.push(format!(
            "AIS    C{:.1} {} O{:.0}%",
            if m.ais_capability > 0.0 {
                m.ais_capability
            } else {
                m.ais_score
            },
            m.ais_grade,
            m.ais_optimization_opportunity * 100.0
        ));
        lines.push(format!("       L {:.0}%", m.ais_learning * 100.0));
    }
    if m.learning_bronze_total > 0 {
        lines.push(format!(
            "Data   G {}/{} q{:.0}%",
            format_number(m.learning_gold_total),
            format_number(m.learning_bronze_total),
            m.learning_data_quality * 100.0
        ));
    }
    if m.world_model_context_bronze_total > 0 {
        lines.push(format!(
            "Ctx-L  G{} S{} R{} q{:.0}%",
            format_number(m.world_model_context_gold_total),
            format_number(m.world_model_context_silver_total),
            format_number(m.world_model_context_rejected_total),
            m.world_model_context_quality * 100.0
        ));
    }
    if m.world_model_actuator_issued_total > 0 {
        lines.push(format!(
            "Act    G {}/{} P{} q{:.0}%",
            format_number(m.world_model_actuator_gold_total),
            format_number(m.world_model_actuator_bronze_total),
            format_number(m.world_model_actuator_pending_total),
            m.world_model_actuator_quality * 100.0
        ));
        if let Some(family) = m
            .world_model_actuator_families
            .iter()
            .filter(|family| family.resolved > 0)
            .max_by_key(|family| family.resolved)
        {
            let short_family: String = family.family.chars().take(8).collect();
            lines.push(format!(
                "Act+   {} G{}/{}",
                short_family,
                format_number(family.gold),
                format_number(family.resolved)
            ));
            lines.push(format!(
                "       eff {}/{} u{:+.0}%",
                format_number(family.effective),
                format_number(family.resolved),
                family.mean_utility * 100.0
            ));
        }
    }
    if m.acceleration_lease_members_applied_total > 0
        || m.acceleration_lease_io_promotions_total > 0
    {
        let family = if m.acceleration_lease_family.is_empty() {
            m.acceleration_lease_last_family.as_str()
        } else {
            m.acceleration_lease_family.as_str()
        };
        let family = if family.is_empty() { "idle" } else { family };
        lines.push(format!(
            "Lease  {} M{} A{}/R{} I{}",
            family,
            m.acceleration_lease_members_active,
            format_number(m.acceleration_lease_members_applied_total),
            format_number(m.acceleration_lease_member_reverts_total),
            format_number(m.acceleration_lease_io_promotions_total),
        ));
        lines.push(format!(
            "       C{} G{} renew{}",
            format_number(m.acceleration_lease_chromium_total),
            format_number(m.acceleration_lease_general_total),
            format_number(m.acceleration_lease_renewals_total),
        ));
        lines.push(format!(
            "       qos{} nice{} skip{}",
            format_number(m.acceleration_lease_task_qos_applied_total),
            format_number(m.acceleration_lease_nice_fallbacks_total),
            format_number(m.acceleration_lease_capability_skips_total),
        ));
        if m.acceleration_lease_task_port_denied_total > 0
            || m.acceleration_lease_qos_write_rejected_total > 0
        {
            lines.push(format!(
                "       skip port{} write{}",
                format_number(m.acceleration_lease_task_port_denied_total),
                format_number(m.acceleration_lease_qos_write_rejected_total),
            ));
            if !m.acceleration_lease_qos_write_error.is_empty() {
                lines.push(format!(
                    "       {}",
                    m.acceleration_lease_qos_write_error
                        .chars()
                        .take(52)
                        .collect::<String>()
                ));
            }
        }
        if !m.interaction_qos_ttl_band.is_empty() {
            lines.push(format!(
                "       ttl {} {}ms {} exp{}",
                m.interaction_qos_ttl_band,
                m.interaction_qos_ttl_ms,
                if m.interaction_qos_ttl_exploratory {
                    "probe"
                } else {
                    "policy"
                },
                format_number(m.interaction_qos_parameter_explorations_total),
            ));
        }
    }
    if !m.webflow_mode.is_empty() || m.webflow_proposed_total > 0 {
        lines.push(format!(
            "Web    {} N{} P{}/A{}",
            if m.webflow_mode.is_empty() {
                "unavailable"
            } else {
                m.webflow_mode.as_str()
            },
            m.webflow_active_navigations,
            format_number(m.webflow_proposed_total),
            format_number(m.webflow_admitted_total),
        ));
        // The sample count is part of the reading: a p95 over a handful of
        // samples is a guess, and reading one as settled produced a bogus
        // 120000ms LCP once.
        let n = m.browser_latency_samples;
        if let (Some(lcp), Some(inp)) = (m.browser_lcp_p95_ms, m.browser_inp_p95_ms) {
            lines.push(format!("       vitals LCP{lcp:.0}ms INP{inp:.0}ms n{n}"));
        } else if let Some(lcp) = m.browser_lcp_p95_ms {
            lines.push(format!("       vitals LCP{lcp:.0}ms INP- n{n}"));
        } else if let Some(inp) = m.browser_inp_p95_ms {
            lines.push(format!("       vitals LCP- INP{inp:.0}ms n{n}"));
        }
        if !m.webflow_phase.is_empty() {
            lines.push(format!(
                "       {} G{} {}",
                m.webflow_phase,
                format_number(m.webflow_valid_health_cycles),
                if m.webflow_blocker.is_empty() {
                    "ready"
                } else {
                    m.webflow_blocker.as_str()
                },
            ));
        }
    }
    if m.network_flow_proposed_total > 0 || m.network_flow_active {
        lines.push(format!(
            "Net+   {} {:.1}MB/s P{} R{} X{}",
            if m.network_flow_active {
                "active"
            } else {
                "idle"
            },
            m.network_flow_traffic_bps as f64 / 1_000_000.0,
            format_number(m.network_flow_proposed_total),
            format_number(m.network_flow_renewed_total),
            format_number(m.network_flow_suppressed_exact_total),
        ));
    }
    let authority_phase = match m.world_model_context_authority_phase.as_str() {
        "calibrating" => "calibrating",
        "trusted" => "trusted",
        "suspended" => "suspended",
        _ => "protected",
    };
    if m.world_model_actuator_known_models == 0 {
        lines.push("WM-U   protected · no evidence".to_string());
    } else {
        lines.push(format!(
            "WM-U   {} {}/{}",
            authority_phase,
            format_number(m.world_model_actuator_ready_models),
            format_number(m.world_model_actuator_known_models)
        ));
    }
    let readiness_reasons = [
        ("no-gold", m.world_model_readiness_no_gold),
        ("immature", m.world_model_readiness_immature),
        ("dormant", m.world_model_readiness_dormant),
        ("quality", m.world_model_readiness_low_quality),
        ("stale", m.world_model_readiness_stale),
        ("origin", m.world_model_readiness_foreign),
        ("hardware", m.world_model_readiness_hardware),
        ("uncertain", m.world_model_readiness_uncertain),
    ]
    .into_iter()
    .filter(|(_, count)| *count > 0)
    .map(|(reason, count)| format!("{reason}{}", compact_counter(count)))
    .collect::<Vec<_>>();
    // Greedy wrap rather than a fixed chunk size: the reason list grows, and a
    // fixed chunk silently overflows the quadrant the first time it does.
    let mut wait_lines: Vec<String> = Vec::new();
    for reason in readiness_reasons {
        let fits = wait_lines
            .last()
            .is_some_and(|line| display_width(line) + 1 + display_width(&reason) <= QW);
        match wait_lines.last_mut() {
            Some(line) if fits => {
                line.push(' ');
                line.push_str(&reason);
            }
            _ => {
                let prefix = if wait_lines.is_empty() {
                    "WM wait "
                } else {
                    "        "
                };
                wait_lines.push(format!("{prefix}{reason}"));
            }
        }
    }
    lines.extend(wait_lines);
    if m.world_model_action_model_capacity > 0 {
        // Saturation is the difference between "still maturing" and "reborn
        // every time a new key shows up", so it gets stated rather than implied.
        lines.push(format!(
            "WM-C   {}/{}{} ev{} b{}",
            format_number(m.world_model_action_model_len),
            format_number(m.world_model_action_model_capacity),
            if m.world_model_action_model_len >= m.world_model_action_model_capacity {
                " full"
            } else {
                ""
            },
            compact_counter(m.world_model_action_model_evictions_total),
            compact_counter(m.world_model_action_model_births_total),
        ));
    }
    // "Waiting" only means learning is in flight if evidence is still arriving.
    lines.push(if m.world_model_last_evidence_cycle == 0 {
        format!(
            "WM-E   ev{} no evidence yet",
            compact_counter(m.world_model_evidence_updates_total)
        )
    } else {
        format!(
            "WM-E   ev{} last c{} idle{}",
            compact_counter(m.world_model_evidence_updates_total),
            format_number(m.world_model_last_evidence_cycle),
            compact_counter(m.cycles.saturating_sub(m.world_model_last_evidence_cycle)),
        )
    });
    if m.world_model_utility_vetoes_total > 0 || m.world_model_utility_promotions_total > 0 {
        lines.push(format!(
            "WM-S   V{} rank{}",
            format_number(m.world_model_utility_vetoes_total),
            format_number(m.world_model_utility_promotions_total)
        ));
    }
    if m.world_model_decision_credit_sources > 0 {
        lines.push(format!(
            "AU {:+.1}% {} {:+.1}% n{}",
            m.world_model_apollo_utility * 100.0,
            m.world_model_decision_credit_leader
                .chars()
                .take(9)
                .collect::<String>(),
            m.world_model_decision_credit_leader_score * 100.0,
            compact_counter(u64::from(m.world_model_decision_credit_leader_observations)),
        ));
    }
    if m.world_model_counterfactual_issued_total > 0
        || m.world_model_counterfactual_rank_uses_total > 0
    {
        lines.push(format!(
            "CF     R{}/{} H{} rank{}",
            format_number(m.world_model_counterfactual_resolved_total),
            format_number(m.world_model_counterfactual_issued_total),
            format_number(m.world_model_counterfactual_would_help_total),
            format_number(m.world_model_counterfactual_rank_uses_total)
        ));
    }
    if m.world_model_episodic_memory_samples > 0 || m.world_model_episodic_rank_uses_total > 0 {
        lines.push(format!(
            "WM-E   M{} F{} rank{}",
            format_number(m.world_model_episodic_memory_samples),
            format_number(m.world_model_episodic_memory_families),
            format_number(m.world_model_episodic_rank_uses_total)
        ));
    }
    if m.world_model_contextual_markov_total > 0
        || m.world_model_contextual_interaction_total > 0
        || m.world_model_contextual_io_total > 0
        || m.world_model_contextual_predictive_total > 0
        || m.world_model_contextual_chromium_total > 0
    {
        lines.push(format!(
            "WM-X   K{} Q{} O{} P{} C{}",
            format_number(m.world_model_contextual_markov_total),
            format_number(m.world_model_contextual_interaction_total),
            format_number(m.world_model_contextual_io_total),
            format_number(m.world_model_contextual_predictive_total),
            format_number(m.world_model_contextual_chromium_total),
        ));
        let family = m.world_model_contextual_last_action.split_once(':').map_or(
            m.world_model_contextual_last_action.as_str(),
            |(family, _)| family,
        );
        let family = match family {
            "markov_prewarm" => "markov",
            "interaction_qos" => "interact",
            "io_shaping" => "io",
            "predictive_threshold" => "pred-thresh",
            "predictive_profile" => "pred-profile",
            "predictive_prethrottle" => "pred-throttle",
            "predictive_purge" => "pred-purge",
            "chromium_ecore" => "chrom-ecore",
            "chromium_purge" => "chrom-purge",
            other => other,
        };
        lines.push(format!(
            "       last {} b{:+.2}",
            family, m.world_model_contextual_last_bias
        ));
    }
    if m.world_model_family_known_models > 0 {
        lines.push(format!(
            "WM-F   {}/{} prior {}",
            format_number(m.world_model_family_ready_models),
            format_number(m.world_model_family_known_models),
            format_number(m.world_model_family_prior_uses_total)
        ));
    }
    if m.world_model_abstentions_total > 0 {
        lines.push(format!(
            "WM-A   {} {}",
            format_number(m.world_model_abstentions_total),
            m.world_model_last_abstention_reason
                .chars()
                .take(18)
                .collect::<String>()
        ));
    }
    if m.action_planner_proposed_total > 0 {
        lines.push(format!(
            "Plan   {}/{} C{} R{}",
            format_number(m.action_planner_admitted_total),
            format_number(m.action_planner_proposed_total),
            format_number(m.action_planner_conflict_drops_total),
            format_number(m.action_planner_reorders_total)
        ));
    }
    if m.world_model_temporal_memory_samples > 0 {
        lines.push(format!(
            "WM-T   M{} R{} G{:+.1}%",
            format_number(m.world_model_temporal_memory_samples),
            format_number(m.world_model_sequence_rollouts_total),
            m.world_model_sequence_expected_gain * 100.0
        ));
        if !m.world_model_sequence_best_first.is_empty() {
            let first = m
                .world_model_sequence_best_first
                .split_once(':')
                .map_or(m.world_model_sequence_best_first.as_str(), |(family, _)| {
                    family
                });
            let second = m.world_model_sequence_best_second.split_once(':').map_or(
                m.world_model_sequence_best_second.as_str(),
                |(family, _)| family,
            );
            lines.push(if second.is_empty() {
                format!("Seq-E  {first}")
            } else {
                format!("Seq-E  {first}>{second}")
            });
        } else if !m.world_model_sequence_abstention_reason.is_empty() {
            lines.push(format!(
                "Seq-E  {}",
                m.world_model_sequence_abstention_reason
                    .chars()
                    .take(20)
                    .collect::<String>()
            ));
        }
        if !m.world_model_sequence_authoritative_best_first.is_empty() {
            let first = m
                .world_model_sequence_authoritative_best_first
                .split_once(':')
                .map_or(
                    m.world_model_sequence_authoritative_best_first.as_str(),
                    |(family, _)| family,
                );
            let second = m
                .world_model_sequence_authoritative_best_second
                .split_once(':')
                .map_or(
                    m.world_model_sequence_authoritative_best_second.as_str(),
                    |(family, _)| family,
                );
            lines.push(if second.is_empty() {
                format!("Seq-A  {first}")
            } else {
                format!("Seq-A  {first}>{second}")
            });
        } else if !m.world_model_sequence_abstention_reason.is_empty() {
            lines.push(format!(
                "Seq-A  {}",
                m.world_model_sequence_abstention_reason
                    .chars()
                    .take(20)
                    .collect::<String>()
            ));
        }
    }
    if m.world_model_dynamics_action_models > 0 || m.world_model_dynamics_no_action_updates > 0 {
        let phase = match m.world_model_dynamics_phase.as_str() {
            "trusted" => "trusted",
            "shadow" => "shadow",
            "calibrating" => "calibrating",
            _ => "protected",
        };
        lines.push(format!(
            "WM-D   {} A{} R{}/{} V{} e{:.1}%",
            phase,
            format_number(m.world_model_dynamics_authoritative_models),
            format_number(m.world_model_dynamics_ranking_models),
            format_number(m.world_model_dynamics_ready_models),
            format_number(m.world_model_dynamics_validation_samples),
            m.world_model_dynamics_validation_mae * 100.0
        ));
        if m.world_model_dynamics_predictions_total > 0
            || m.world_model_dynamics_baseline_uses_total > 0
        {
            lines.push(format!(
                "MPC-D  P{} R{} A{} B{} u{:.1}%",
                format_number(m.world_model_dynamics_predictions_total),
                format_number(m.world_model_dynamics_ranking_predictions_total),
                format_number(m.world_model_dynamics_authoritative_predictions_total),
                format_number(m.world_model_dynamics_baseline_uses_total),
                m.world_model_dynamics_mean_uncertainty * 100.0
            ));
        }
    }
    if !m.gpu_imagination_backend.is_empty() {
        let mean_gpu_ms = if m.gpu_imagination_jobs_completed_total > 0 {
            m.gpu_imagination_gpu_time_ns_total as f64
                / m.gpu_imagination_jobs_completed_total as f64
                / 1_000_000.0
        } else {
            0.0
        };
        let backend_health = if m.gpu_imagination_circuit_state.is_empty() {
            m.gpu_imagination_backend.clone()
        } else {
            let circuit = match m.gpu_imagination_circuit_state.as_str() {
                "closed" => "C",
                "open" => "O",
                "half-open" => "H",
                _ => "?",
            };
            format!("{}/{}", m.gpu_imagination_backend, circuit)
        };
        lines.push(format!(
            "GPU-I  {} J{} S{} t{:.2}ms",
            backend_health,
            format_number(m.gpu_imagination_jobs_completed_total),
            format_number(m.gpu_imagination_samples_total),
            mean_gpu_ms
        ));
        if m.gpu_imagination_jobs_completed_total == 0
            && !m.gpu_imagination_last_submit_outcome.is_empty()
        {
            lines.push(format!("GPU-G  {}", m.gpu_imagination_last_submit_outcome));
        }
        if !m.gpu_imagination_last_best_action.is_empty() {
            let action = m.gpu_imagination_last_best_action.split_once(':').map_or(
                m.gpu_imagination_last_best_action.as_str(),
                |(family, _)| family,
            );
            lines.push(format!(
                "GPU+   {} p{:.0}% d{:+.1}%",
                action.chars().take(13).collect::<String>(),
                m.gpu_imagination_last_positive_probability * 100.0,
                m.gpu_imagination_last_p10_gain * 100.0
            ));
        }
        let gpu_influence = if m.gpu_imagination_last_influence_scope.is_empty() {
            "awaiting-consumer".to_string()
        } else {
            let scope = match m.gpu_imagination_last_influence_scope.as_str() {
                "root-ranking" => "root",
                "markov-prewarm" => "markov",
                "interaction-qos" => "iqos",
                "io-shaping" => "io",
                "chromium" => "chromium",
                "predictive" => "predict",
                _ => "other",
            };
            let action = m
                .gpu_imagination_last_influence_action
                .split_once(':')
                .map_or(
                    m.gpu_imagination_last_influence_action.as_str(),
                    |(family, _)| family,
                );
            format!(
                "{}:{} {:+.2}%",
                scope,
                action.chars().take(5).collect::<String>(),
                m.gpu_imagination_last_influence_support * 100.0
            )
        };
        lines.push(format!(
            "GPU-U  R{} C{} {}",
            compact_counter(m.gpu_imagination_root_rank_uses_total),
            compact_counter(m.gpu_imagination_contextual_uses_total),
            gpu_influence
        ));
        if m.world_model_gpu_bronze_total > 0 {
            lines.push(format!(
                "GPU-M  B{} S{} G{} P{} M{} q{:.0}% e{:.1}%",
                format_number(m.world_model_gpu_bronze_total),
                format_number(m.world_model_gpu_silver_total),
                format_number(m.world_model_gpu_gold_total),
                format_number(m.world_model_gpu_pending_total),
                format_number(m.world_model_gpu_calibrated_models),
                m.world_model_gpu_calibration_quality * 100.0,
                m.world_model_gpu_calibration_mae * 100.0
            ));
            // GPU-M is lifetime-cumulative (persisted in the medallion) while
            // GPU-I above resets every boot, and a Bronze is one of the many
            // ranked candidates a single job emits — not a job. Stacked in one
            // column they invite a jobs-over-Bronze ratio whose two terms share
            // neither span nor unit. Say the span out loud.
            lines.push("       B/S/G lifetime".to_string());
            if m.world_model_gpu_rejected_total > 0 {
                lines.push(format!(
                    "       rej{} ev{} un{} br{}",
                    format_number(m.world_model_gpu_rejected_total),
                    format_number(m.world_model_gpu_evicted_total),
                    format_number(m.world_model_gpu_unused_total),
                    format_number(m.world_model_gpu_bronze_rejected_total),
                ));
                // Rejections older than the breakdown. Without this the three
                // buckets look like they should sum to `rej` and do not.
                if m.world_model_gpu_unclassified_rejections > 0 {
                    lines.push(format!(
                        "       pre-split{}",
                        format_number(m.world_model_gpu_unclassified_rejections),
                    ));
                }
            }
        }
    }
    if !m.system_deliberation_mode.is_empty() {
        let mode = match m.system_deliberation_mode.as_str() {
            "grounded" => "ground",
            "calibrating" => "cal",
            "observing" => "obs",
            _ => "other",
        };
        let system1 = if m.system_deliberation_system1_struggling {
            "S1?"
        } else {
            "S1"
        };
        lines.push(format!(
            "Delib  {} {} c{:.0}% G{} g{} F{}",
            mode,
            system1,
            m.system_deliberation_confidence * 100.0,
            compact_counter(m.system_deliberation_local_gold),
            m.system_deliberation_gpu_forecasts,
            m.system_deliberation_local_families,
        ));
        lines.push(format!(
            "S2>S1  G{} +{}/-{} n{} s{} c{:.0}%",
            compact_counter(m.local_consolidations),
            compact_counter(m.local_consolidation_improvements),
            compact_counter(m.local_consolidation_regressions),
            compact_counter(m.local_consolidation_neutral),
            compact_counter(m.local_consolidation_system1_updates),
            m.system_deliberation_local_confidence * 100.0,
        ));
    }
    lines.push(format!(
        // This is the pressure-causal imagination model, not the universal
        // actuator-learning stream shown above as `Act` / `Act+`.
        "Causal {}/{} ready G{} q{:.0}%",
        format_number(m.world_model_ready_actions),
        format_number(m.world_model_curated_actions),
        format_number(m.world_model_gold_evidence),
        m.world_model_data_quality * 100.0,
    ));
    if m.world_model_causal_actuator_gold_total > 0 {
        lines.push(format!(
            "Caus+  U{} universal Gold",
            format_number(m.world_model_causal_actuator_gold_total)
        ));
    }

    // RL Q-table
    if m.rl_total_ticks > 0 {
        lines.push(format!(
            "RL     {} adj{:+}",
            format_number(m.rl_total_ticks),
            m.rl_adjustment_pp
        ));
    }

    // Kalman (we just refactored it)
    lines.push("Kalman conv ✅".to_string());

    // Causal Graph
    lines.push(format!(
        "Causal {} slow",
        format_number(m.causal_slow_horizon_count as u64)
    ));
    lines.push(format!(
        "       {} mech",
        format_number(m.causal_mechanism_count as u64)
    ));

    // Hazard / MPC / Markov (compact)
    lines.push("Hazard ✅ low risk".to_string());
    lines.push(format!(
        "Workload {}",
        m.current_workload.chars().take(20).collect::<String>()
    ));
    if m.markov_shadow_predictions_total > 0 {
        lines.push(format!(
            "Mk-S   H{}/{} P{}{}",
            format_number(m.markov_shadow_hits),
            format_number(m.markov_shadow_resolved_total),
            format_number(m.markov_shadow_predictions_total),
            if m.markov_shadow_active {
                " active"
            } else {
                ""
            }
        ));
    }
    if !m.markov_prediction_app.is_empty() {
        let app: String = m.markov_prediction_app.chars().take(11).collect();
        let timing = if m.markov_prediction_overdue_secs > 0.0 {
            format!("stale{:.0}s", m.markov_prediction_overdue_secs)
        } else {
            format!("eta{:.0}s", m.markov_prediction_eta_secs)
        };
        lines.push(format!(
            "Mk-P   {} {:.0}% {} {}",
            app,
            m.markov_prediction_confidence * 100.0,
            timing,
            if m.markov_prewarm_blocker.is_empty() {
                m.markov_prewarm_admission.as_str()
            } else {
                m.markov_prewarm_blocker.as_str()
            }
        ));
        lines.push(format!(
            "Mk-H 5s{:.0} 30s{:.0} 2m{:.0} 10m{:.0}",
            m.markov_prediction_5s * 100.0,
            m.markov_prediction_30s * 100.0,
            m.markov_prediction_2m * 100.0,
            m.markov_prediction_10m * 100.0,
        ));
        lines.push(format!(
            "Warm   M{}/{} H{}/{} T{} C{}",
            format_number(m.markov_prewarm_applied),
            format_number(m.markov_prewarm_attempts),
            format_number(m.markov_prewarm_hits),
            format_number(
                m.markov_prewarm_hits
                    .saturating_add(m.markov_prewarm_misses)
            ),
            format_number(m.temporal_prewarm_applied),
            format_number(m.markov_prewarm_conflict_skips_total),
        ));
    }

    lines
}

// ── 🎯 DECIDE quadrant ───────────────────────────────────────────────────────
fn render_decide_q(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold(&q_header("🎯", "DECIDE"))];

    // MetaCognition
    let humble = m.meta_confidence < 0.40;
    let meta_label = if humble {
        "HUMBLE"
    } else if m.meta_confidence > 0.70 {
        "CONFIDENT"
    } else {
        "NORMAL"
    };
    let meta_emoji = if humble { "🤔" } else { "🎯" };
    lines.push(format!(
        "Meta   {} {} {:.0}%",
        meta_emoji,
        meta_label,
        m.meta_confidence * 100.0
    ));

    // Arousal
    let arousal_pct = (m.arousal_level * 100.0) as i32;
    lines.push(format!("Arousal {} {}%", m.arousal_zone, arousal_pct));

    // UCHS
    lines.push(format!(
        "UCHS   {:.0}% {}",
        m.uchs_composite * 100.0,
        m.uchs_grade
    ));

    // Workload + confidence
    lines.push(format!("Conf   {:.0}%", m.ml_confidence * 100.0));

    // FG app
    if let Some(fg) = &m.foreground_app {
        let name: String = fg.name.chars().take(20).collect();
        let active = if m.foreground_idle { "💤" } else { "🟢" };
        lines.push(format!("FG     {} {}", active, name));
    }

    // Profile: distinguish the configured baseline from the profile currently
    // in force, and never show a stale transition reason as a live cause.
    lines.push(format!(
        "Activo  {} {}",
        profile_emoji(status.effective_profile),
        status.effective_profile.as_str()
    ));
    lines.push(format!(
        "Base    {} {}",
        profile_emoji(status.base_profile),
        status.base_profile.as_str()
    ));
    lines.push(format!("Motivo  {}", profile_activity_reason(status)));

    // Sprint Coalition — guard tower visibility (2026-05-10).
    // Guard% = mean over-protection signal across mature blocked patterns.
    // Coalitions = recently-fg apps protected by 5-min envelope.
    // Yellow when guard >= 0.40 (policy showing over-protection signs).
    // Red when guard >= 0.70 (epistemic likely in HIGH mode).
    let guard_pct = (m.guard_overprotection * 100.0) as i32;
    let guard_label = if m.guard_overprotection >= 0.70 {
        red(&format!("{}%", guard_pct))
    } else if m.guard_overprotection >= 0.40 {
        yellow(&format!("{}%", guard_pct))
    } else {
        dim(&format!("{}%", guard_pct))
    };
    lines.push(format!(
        "Guard  {} · Coalitions {}",
        guard_label, m.active_coalitions_count
    ));

    lines
}

// ── 🎬 ACT quadrant ──────────────────────────────────────────────────────────
fn render_act_q(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold(&q_header("🎬", "ACT"))];

    // SafetyPolicy budgets — derive from profile to mirror per-profile caps.
    let pol = SafetyPolicy::for_profile(status.effective_profile);
    let bar_w = 8;
    let b = &m.budgets;
    let boost_ratio = (b.cycle_boosts as f64 / pol.max_boosts_per_cycle.max(1) as f64).min(1.0);
    let throt_ratio =
        (b.cycle_throttles as f64 / pol.max_throttles_per_cycle.max(1) as f64).min(1.0);
    let frz_ratio = (b.cycle_freezes as f64 / pol.max_freezes_per_cycle.max(1) as f64).min(1.0);
    let hint_ratio = (b.cycle_hints as f64 / pol.max_paging_hints_per_cycle.max(1) as f64).min(1.0);

    lines.push(format!(
        "Cycle B {} {}/{}",
        render_bar(boost_ratio, bar_w),
        b.cycle_boosts,
        pol.max_boosts_per_cycle
    ));
    lines.push(format!(
        "Cycle T {} {}/{}",
        render_bar(throt_ratio, bar_w),
        b.cycle_throttles,
        pol.max_throttles_per_cycle
    ));
    lines.push(format!(
        "Cycle F {} {}/{}",
        render_bar(frz_ratio, bar_w),
        b.cycle_freezes,
        pol.max_freezes_per_cycle
    ));
    lines.push(format!(
        "Cycle H {} {}/{}",
        render_bar(hint_ratio, bar_w),
        b.cycle_hints,
        pol.max_paging_hints_per_cycle
    ));
    lines.push(format!(
        "Total B{} T{} F{} U{}",
        format_number(m.boosts_applied),
        format_number(m.throttles_applied),
        format_number(m.freezes_applied),
        format_number(m.unfreezes_applied)
    ));

    let frozen_n = status.frozen_processes.len();
    let frozen_mb = m.frozen_ram_mb;
    if frozen_n > 0 {
        lines.push(format!("Frozen {} ({:.0}MB)", frozen_n, frozen_mb));
    } else {
        lines.push("Frozen 0".to_string());
    }

    let pkg_w = m.energy_package_watts.unwrap_or(0.0);
    lines.push(format!("Energy {:.2}W", pkg_w));
    lines.push(format!(
        "Used   {:.2}Wh {:.2}gCO₂",
        m.energy_package_wh.unwrap_or(0.0),
        m.energy_co2_emitted_g.unwrap_or(0.0)
    ));
    lines.push(format!(
        "Saved  {:.2}Wh {:.2}gCO₂",
        m.energy_savings_wh.unwrap_or(0.0),
        m.energy_co2_avoided_g.unwrap_or(0.0)
    ));
    lines.push(format!(
        "Reactor {} {}",
        status.reactor_mode, status.reactor_health
    ));

    if m.last_episode_id > 0 {
        let family = m
            .last_episode_action
            .split_once(':')
            .map_or(m.last_episode_action.as_str(), |(family, _)| family);
        lines.push(format!(
            "Ep#{} {} q{:.0}%",
            compact_counter(m.last_episode_id),
            m.last_episode_tier,
            m.last_episode_quality * 100.0
        ));
        lines.push(format!(
            "Did    {} -> {}",
            family.chars().take(10).collect::<String>(),
            m.last_episode_target.chars().take(11).collect::<String>()
        ));
        lines.push(format!(
            "Gain   S{:+.1} H{:+.1} AU{:+.1}%",
            m.last_episode_system_gain * 100.0,
            m.last_episode_human_gain * 100.0,
            m.last_episode_apollo_utility * 100.0,
        ));
        if !m.last_episode_proposer.is_empty() {
            lines.push(format!(
                "By     {} p{:+.1}%",
                m.last_episode_proposer.chars().take(15).collect::<String>(),
                m.last_episode_predicted_gain * 100.0,
            ));
        }
    }

    lines
}

// ── 🚪 GATES band (full-width) ───────────────────────────────────────────────
fn render_gates_band(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold("🚪 GATES")];

    // Survival: based on memory pressure
    let surv = if m.memory_pressure >= 0.85 {
        red("🔴")
    } else {
        green("🟢")
    };

    // Auto-purge: paused by media (audio/call/presentation), idle (low pressure),
    // or armed (ready to fire if pressure crosses 0.65). Short reason chip in row.
    let purge = if m.user_call_in_progress {
        yellow("💤 call")
    } else if m.user_audio_active {
        yellow("💤 audio")
    } else if m.user_has_sleep_assertion {
        yellow("💤 video/media")
    } else if m.memory_pressure < 0.65 {
        dim("⬜ idle")
    } else {
        green("🟢 armed")
    };

    let post_wake = if status.post_wake_grace_active {
        yellow(&format!("⚠ {}s", status.post_wake_grace_remaining_secs))
    } else {
        green("🟢")
    };

    // Single compact row: 4 gates side-by-side
    lines.push(format!(
        "survival {} · purge {} · freeze 🟢 · wake {}",
        surv, purge, post_wake
    ));
    if !m.coreaudio_probe_state.is_empty() && m.coreaudio_probe_state != "direct" {
        if m.coreaudio_probe_state == "session-fallback" {
            let guard =
                if m.user_audio_active || m.user_call_in_progress || m.user_has_sleep_assertion {
                    "media"
                } else {
                    "unknown-safe"
                };
            lines.push(dim(&format!(
                "audio session-fallback · HAL n/a · guard {}",
                guard
            )));
        } else {
            lines.push(dim(&format!(
                "audio {} · HAL samples {} cache {} fail {}",
                m.coreaudio_probe_state,
                m.coreaudio_probe_samples_total,
                m.coreaudio_probe_cache_hits_total,
                m.coreaudio_probe_failures_total
            )));
        }
    }

    // Maintenance purge counters
    let total_skipped = m.maintenance_purge_skipped_pressure_total
        + m.maintenance_purge_skipped_swap_floor_total
        + m.maintenance_purge_skipped_growing_total
        + m.maintenance_purge_skipped_idle_total
        + m.maintenance_purge_skipped_build_mode_total
        + m.maintenance_purge_skipped_rate_limit_total;

    let mut skip_breakdown = Vec::new();
    if m.maintenance_purge_skipped_pressure_total > 0 {
        skip_breakdown.push(format!(
            "pres:{}",
            m.maintenance_purge_skipped_pressure_total
        ));
    }
    if m.maintenance_purge_skipped_idle_total > 0 {
        skip_breakdown.push(format!(
            "idle/media:{}",
            m.maintenance_purge_skipped_idle_total
        ));
    }
    if m.maintenance_purge_skipped_swap_floor_total > 0 {
        skip_breakdown.push(format!(
            "floor:{}",
            m.maintenance_purge_skipped_swap_floor_total
        ));
    }
    if m.maintenance_purge_skipped_growing_total > 0 {
        skip_breakdown.push(format!(
            "grow:{}",
            m.maintenance_purge_skipped_growing_total
        ));
    }
    if m.maintenance_purge_skipped_build_mode_total > 0 {
        skip_breakdown.push(format!(
            "build:{}",
            m.maintenance_purge_skipped_build_mode_total
        ));
    }
    if m.maintenance_purge_skipped_rate_limit_total > 0 {
        skip_breakdown.push(format!(
            "rate:{}",
            m.maintenance_purge_skipped_rate_limit_total
        ));
    }

    let breakdown_str = if skip_breakdown.is_empty() {
        String::new()
    } else {
        format!(" ({})", skip_breakdown.join(" "))
    };

    lines.push(dim(&format!(
        "purge totals: {} fired · {} skipped{}",
        m.maintenance_purge_total, total_skipped, breakdown_str
    )));

    lines
}

// ── 🌳 CHROMIUM band (compact) ───────────────────────────────────────────────
fn render_chromium_band(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    if m.chromium_renderers_total == 0 && m.chromium_browsers_managed.is_empty() {
        return Vec::new();
    }
    let mut lines = vec![bold("🌳 CHROMIUM")];
    let freeze = if m.chromium_gate_regime == "disabled" {
        "off".to_string()
    } else {
        m.chromium_renderers_frozen.to_string()
    };
    lines.push(format!(
        "renderers={} freeze={} e-core={} freed={:.0}MB",
        m.chromium_renderers_total, freeze, m.chromium_renderers_ecore, m.chromium_freed_mb
    ));
    if !m.chromium_browsers_managed.is_empty() {
        let apps = m.chromium_browsers_managed.join(", ");
        let truncated: String = apps.chars().take(60).collect();
        lines.push(dim(&format!("apps: {}", truncated)));
    }
    lines
}

// ── ⚡ TOP CONSUMERS band ────────────────────────────────────────────────────
fn render_consumers_band(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    if m.energy_top_consumers.is_empty() {
        return Vec::new();
    }
    let mut lines = vec![bold("⚡ TOP CONSUMERS")];
    let total_w: f64 = m.energy_top_consumers.iter().map(|c| c.current_watts).sum();
    for (i, c) in m.energy_top_consumers.iter().take(5).enumerate() {
        let pct = if total_w > 0.0 {
            (c.current_watts / total_w * 100.0) as i32
        } else {
            0
        };
        let name: String = c.name.chars().take(22).collect();
        let bar = render_bar(c.current_watts / total_w.max(0.01), 12);
        lines.push(format!(
            "{}. {:<22} {:.2}W {} {:>3}%",
            i + 1,
            name,
            c.current_watts,
            bar,
            pct
        ));
    }
    lines
}

fn learning_percent(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{:.0}%", value.clamp(0.0, 1.0) * 100.0))
        .unwrap_or_else(|| "--".to_string())
}

fn utility_percent(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{:.0}%", value.clamp(-1.0, 1.0) * 100.0))
        .unwrap_or_else(|| "--".to_string())
}

fn bounded_learning_line(line: String) -> String {
    let mut bounded = String::new();
    let mut width = 0;
    for character in line.chars() {
        let character_width = if is_wide_char(character) { 2 } else { 1 };
        if width + character_width > CW {
            break;
        }
        bounded.push(character);
        width += character_width;
    }
    bounded
}

fn render_learning_band(status: &DaemonStatus) -> Vec<String> {
    let metrics = &status.metrics;
    let trust = &metrics.trust_inventory;
    let gold = trust.local_gold_decisions;
    let calibration = metrics.unified_learning_ais.calibrated_accuracy;
    let causal = metrics.unified_learning_ais.causal_resolution;
    let primary = if metrics.unified_learning_schema_version == 0 {
        "Learn  legacy metrics; unified evidence unavailable".to_string()
    } else if trust.degraded > 0 {
        if trust.recovery_target_gold > 0 {
            format!(
                "Learn  relearning {} models; best Gold {}/{} + quality",
                trust.degraded, trust.recovery_best_gold, trust.recovery_target_gold
            )
        } else {
            format!("Learn  relearning {} models; Gold {}", trust.degraded, gold)
        }
    } else if trust.trusted == 0 && metrics.world_model_actuator_ready_models > 0 {
        format!(
            "Learn  WM ready {}; unified Gold {}",
            metrics.world_model_actuator_ready_models, gold
        )
    } else if gold == 0 {
        "Learn  no local Gold evidence yet".to_string()
    } else if gold < 10 {
        format!("Learn  collecting {gold}/10 to candidate")
    } else if gold < 20 {
        format!("Learn  candidate {gold}/20 to validate")
    } else if gold < 50 && trust.trusted == 0 {
        format!("Learn  validated {gold}/50 to trust")
    } else if trust.trusted == 0 {
        "Learn  mature evidence; no trusted predictor yet".to_string()
    } else {
        format!(
            "Learn  trusted {} active {} closure {} cal {} causal {}",
            trust.trusted,
            trust.active_trusted,
            learning_percent(metrics.ledger_closure.closure_coverage),
            learning_percent(calibration),
            learning_percent(causal),
        )
    };
    let worst = if trust.worst_producer.is_empty() || trust.worst_action.is_empty() {
        "Worst  none with local Gold calibration".to_string()
    } else {
        format!(
            "Worst  {}/{}@{} MAE {} cov {}",
            trust.worst_producer,
            trust.worst_action,
            trust.worst_horizon,
            learning_percent(trust.worst_normalized_mae),
            learning_percent(trust.worst_coverage),
        )
    };
    let latest = &metrics.latest_resolved_episode;
    let latest = if !latest.present {
        "Latest none yet".to_string()
    } else if latest.measured_utility.is_none() {
        format!("Latest {} evaluated; no measurement", latest.action)
    } else {
        format!(
            "Latest {} expected {} measured {} {}/{}",
            latest.action,
            utility_percent(latest.expected_utility),
            utility_percent(latest.measured_utility),
            latest.tier,
            latest.scope,
        )
    };
    let ledger_orphans = if metrics.unified_learning_schema_version < 2 {
        "Ledger huérfanos n/d".to_string()
    } else {
        format!(
            "Ledger huérfanos {}",
            metrics.decision_ledger_unattributed_applied_total
        )
    };
    vec![primary, worst, latest, ledger_orphans]
        .into_iter()
        .map(bounded_learning_line)
        .collect()
}

// ── 📋 VERDICT band (cognitive) ──────────────────────────────────────────────
fn render_verdict_band(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold("📋 VEREDICTO")];

    let pressure = m.memory_pressure;
    let media_active = m.user_audio_active || m.user_call_in_progress || m.user_has_sleep_assertion;
    let humble = m.meta_confidence < 0.40;

    let main = if pressure >= 0.85 {
        red("🔴 Crisis: survival mode + emergency purge eligible")
    } else if pressure >= 0.65 && !media_active {
        yellow("🟡 Pressure elevada · maintenance-purge gate evalúa")
    } else if pressure >= 0.65 && media_active {
        yellow("🟡 Pressure elevada · purge BLOCKED (proteger media)")
    } else if pressure >= 0.40 {
        yellow("🟡 Pressure moderada · cognición estable")
    } else {
        green("🟢 Sistema optimizado · sin pressure")
    };
    lines.push(main);

    if humble {
        lines.push(dim("Meta: HUMBLE · 2× exploration activa"));
    }

    if !status.frozen_processes.is_empty() {
        lines.push(dim(&format!(
            "Frozen: {} procesos · {:.0}MB recuperados",
            status.frozen_processes.len(),
            m.frozen_ram_mb
        )));
    }

    lines
}

// ── New header (compact one-line) ────────────────────────────────────────────
fn render_header_v2(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let state = if status.kill_switch {
        yellow("⏸ Pausado")
    } else if status.running {
        green("Activo")
    } else {
        red("Detenido")
    };
    let profile = format!(
        "activo {} {}",
        profile_emoji(status.effective_profile),
        status.effective_profile.as_str()
    );
    let mut lines = vec![bold(&format!(
        "🚀 APOLLO {} │ {} │ c:{} │ c-p95 {:.0}ms",
        state,
        profile,
        format_number(m.cycles),
        m.p95_cycle_ms
    ))];
    if status.kill_switch {
        lines.push(yellow(
            "⚠ Optimización pausada — apollo-optimizerctl resume",
        ));
    }
    lines
}

/// Cognitive-stack grid renderer (replaces linear v1).
pub fn render_dashboard_v2(status: &DaemonStatus) -> String {
    let mut out = String::with_capacity(4096);

    out.push_str(&box_top());
    out.push('\n');
    for line in render_header_v2(status) {
        out.push_str(&box_line(&line));
        out.push('\n');
    }
    out.push_str(&box_div());
    out.push('\n');

    // ── Grid row 1: SENSE | THINK ────────────────────────────────────────────
    out.push_str(&box_empty());
    out.push('\n');
    let sense = render_sense_q(status);
    let think = render_think_q(status);
    for line in render_pair(&sense, &think) {
        out.push_str(&box_line(&line));
        out.push('\n');
    }

    // ── Grid row 2: DECIDE | ACT ─────────────────────────────────────────────
    out.push_str(&box_empty());
    out.push('\n');
    let decide = render_decide_q(status);
    let act = render_act_q(status);
    for line in render_pair(&decide, &act) {
        out.push_str(&box_line(&line));
        out.push('\n');
    }

    // ── Full-width bands ─────────────────────────────────────────────────────
    let bands: Vec<Vec<String>> = vec![
        render_gates_band(status),
        render_chromium_band(status),
        render_consumers_band(status),
        render_learning_band(status),
        render_blockers(&status.last_blockers),
        render_verdict_band(status),
    ];
    for band in bands.iter().filter(|b| !b.is_empty()) {
        out.push_str(&box_empty());
        out.push('\n');
        for line in band {
            out.push_str(&box_line(line));
            out.push('\n');
        }
    }

    out.push_str(&box_empty());
    out.push('\n');
    out.push_str(&box_bottom());
    out.push('\n');

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::types::{LatencyTarget, RuntimeMetrics};

    fn dashboard_status() -> DaemonStatus {
        DaemonStatus {
            running: true,
            profile: OptimizationProfile::BalancedRoot,
            latency_target: LatencyTarget::Normal,
            effective_profile: OptimizationProfile::BalancedRoot,
            kill_switch: false,
            throttle_level: "low".to_string(),
            thermal_state: "nominal".to_string(),
            last_blockers: Vec::new(),
            auto_profile_enabled: true,
            base_profile: OptimizationProfile::BalancedRoot,
            override_active: false,
            override_expires_at: None,
            transition_reason: String::new(),
            post_wake_grace_active: false,
            post_wake_grace_remaining_secs: 0,
            last_wake_at: None,
            post_wake_policy: String::new(),
            reactor_mode: "normal".to_string(),
            reactor_health: "ok".to_string(),
            metrics: RuntimeMetrics::default(),
            frozen_processes: Vec::new(),
        }
    }

    #[test]
    fn unified_learning_band_uses_legacy_degraded_and_collecting_precedence() {
        let mut status = dashboard_status();
        assert_eq!(
            render_learning_band(&status)[0],
            "Learn  legacy metrics; unified evidence unavailable"
        );

        status.metrics.unified_learning_schema_version = 1;
        assert_eq!(
            render_learning_band(&status)[0],
            "Learn  no local Gold evidence yet"
        );
        status.metrics.trust_inventory.local_gold_decisions = 9;
        assert_eq!(
            render_learning_band(&status)[0],
            "Learn  collecting 9/10 to candidate"
        );
        status.metrics.trust_inventory.degraded = 2;
        status.metrics.trust_inventory.trusted = 4;
        assert_eq!(
            render_learning_band(&status)[0],
            "Learn  relearning 2 models; Gold 9"
        );
        status.metrics.trust_inventory.recovery_best_gold = 3;
        status.metrics.trust_inventory.recovery_target_gold = 10;
        assert_eq!(
            render_learning_band(&status)[0],
            "Learn  relearning 2 models; best Gold 3/10 + quality"
        );
    }

    #[test]
    fn unified_learning_band_reports_worst_and_latest_without_zero_as_evidence() {
        let mut status = dashboard_status();
        status.metrics.unified_learning_schema_version = 1;
        let absent = render_learning_band(&status);
        assert_eq!(absent[1], "Worst  none with local Gold calibration");
        assert_eq!(absent[2], "Latest none yet");

        status.metrics.trust_inventory.local_gold_decisions = 50;
        status.metrics.trust_inventory.trusted = 1;
        status.metrics.trust_inventory.active_trusted = 1;
        status.metrics.trust_inventory.worst_producer = "world-model".into();
        status.metrics.trust_inventory.worst_action = "boost:action".into();
        status.metrics.trust_inventory.worst_horizon = "5s".into();
        status.metrics.trust_inventory.worst_normalized_mae = Some(0.12);
        status.metrics.trust_inventory.worst_coverage = Some(0.90);
        status.metrics.unified_learning_ais.calibrated_accuracy = Some(0.88);
        status.metrics.unified_learning_ais.causal_resolution = Some(0.77);
        status.metrics.latest_resolved_episode.present = true;
        status.metrics.latest_resolved_episode.action = "boost:action".into();
        status.metrics.latest_resolved_episode.expected_utility = Some(0.08);
        status.metrics.latest_resolved_episode.measured_utility = Some(-0.02);
        status.metrics.latest_resolved_episode.tier = "gold".into();
        status.metrics.latest_resolved_episode.scope = "treatment".into();
        let lines = render_learning_band(&status);
        assert_eq!(
            lines[1],
            "Worst  world-model/boost:action@5s MAE 12% cov 90%"
        );
        assert_eq!(
            lines[2],
            "Latest boost:action expected 8% measured -2% gold/treatment"
        );
        assert_eq!(
            lines[0],
            "Learn  trusted 1 active 1 closure -- cal 88% causal 77%"
        );
        assert!(lines.iter().all(|line| display_width(line) <= CW));
    }

    #[test]
    fn unified_learning_band_keeps_ready_world_model_visible_while_unified_calibrates() {
        let mut status = dashboard_status();
        status.metrics.unified_learning_schema_version = 2;
        status.metrics.world_model_actuator_ready_models = 18;

        let lines = render_learning_band(&status);

        assert_eq!(lines[0], "Learn  WM ready 18; unified Gold 0");
        assert!(lines.iter().all(|line| display_width(line) <= CW));
    }

    #[test]
    fn unified_learning_band_labels_unmeasured_ledger_event_as_evaluation() {
        let mut status = dashboard_status();
        status.metrics.unified_learning_schema_version = 2;
        status.metrics.latest_resolved_episode.present = true;
        status.metrics.latest_resolved_episode.action = "predictive_purge:maintenance".into();
        status.metrics.latest_resolved_episode.tier = "ledger".into();
        status.metrics.latest_resolved_episode.scope = "local".into();

        let lines = render_learning_band(&status);

        assert_eq!(
            lines[2],
            "Latest predictive_purge:maintenance evaluated; no measurement"
        );
        assert!(lines.iter().all(|line| display_width(line) <= CW));
    }

    #[test]
    fn unified_learning_band_marks_ledger_orphans_unavailable_before_schema_two() {
        let mut status = dashboard_status();
        status.metrics.unified_learning_schema_version = 1;
        status.metrics.decision_ledger_unattributed_applied_total = 3;

        let lines = render_learning_band(&status);

        assert_eq!(lines[3], "Ledger huérfanos n/d");
        assert!(lines.iter().all(|line| display_width(line) <= CW));
    }

    #[test]
    fn unified_learning_band_reports_ledger_orphans_from_schema_two() {
        let mut status = dashboard_status();
        status.metrics.unified_learning_schema_version = 2;
        status.metrics.decision_ledger_unattributed_applied_total = 3;

        let lines = render_learning_band(&status);

        assert_eq!(lines[3], "Ledger huérfanos 3");
        assert!(lines.iter().all(|line| display_width(line) <= CW));

        status.metrics.decision_ledger_unattributed_applied_total = u64::MAX;
        let max_lines = render_learning_band(&status);
        assert_eq!(max_lines[3], "Ledger huérfanos 18446744073709551615");
        assert!(max_lines.iter().all(|line| display_width(line) <= CW));
    }

    #[test]
    fn dashboard_explains_an_automatic_active_profile_with_live_signals() {
        let mut status = dashboard_status();
        status.effective_profile = OptimizationProfile::AggressiveRoot;
        status.metrics.context_switch_burst = true;
        status.metrics.arousal_level = 0.60;
        status.transition_reason = "manual-override-cleared".to_string();

        assert_eq!(profile_activity_reason(&status), "Auto: cambios de app");

        let decide = render_decide_q(&status);
        assert!(decide
            .iter()
            .any(|line| line == "Activo  ⚡ aggressive-root"));
        assert!(decide.iter().any(|line| line == "Base    🔵 balanced-root"));
        assert!(decide
            .iter()
            .any(|line| line == "Motivo  Auto: cambios de app"));
        assert!(decide.iter().all(|line| display_width(line) <= QW));

        let header = render_header_v2(&status);
        assert!(header[0].contains("activo ⚡ aggressive-root"));
        assert!(header.iter().all(|line| display_width(line) <= CW));
    }

    #[test]
    fn dashboard_separates_loop_latency_and_attributed_user_gain() {
        let mut status = dashboard_status();
        status.metrics.p95_cycle_ms = 37.0;
        status.metrics.perceptual_latency_score = 0.18;
        status.metrics.perceptual_latency_category = "responsive".to_string();
        status.metrics.scheduler_jitter_p95_ms = 0.12;
        status.metrics.scheduler_jitter_samples = 42;
        status.metrics.last_episode_id = 7;
        status.metrics.last_episode_action = "boost:Editor".to_string();
        status.metrics.last_episode_target = "Editor".to_string();
        status.metrics.last_episode_tier = "gold".to_string();
        status.metrics.last_episode_quality = 0.94;
        status.metrics.last_episode_utility = 0.03;
        status.metrics.last_episode_latency_improvement = 0.08;
        status.metrics.last_episode_system_gain = 0.02;
        status.metrics.last_episode_human_gain = 0.08;
        status.metrics.last_episode_apollo_utility = 0.05;
        status.metrics.last_episode_proposer = "interaction-specialist".to_string();
        status.metrics.last_episode_predicted_gain = 0.04;

        assert!(render_header_v2(&status)[0].contains("c-p95 37ms"));
        let sense = render_sense_q(&status);
        assert!(sense.iter().any(|line| line == "UX-p   responsive 18%"));
        assert!(sense.iter().any(|line| line == "Sched  p95 0.12ms n42"));
        let act = render_act_q(&status);
        assert!(act.iter().any(|line| line == "Ep#7 gold q94%"));
        assert!(act.iter().any(|line| line == "Gain   S+2.0 H+8.0 AU+5.0%"));
        assert!(act
            .iter()
            .any(|line| line == "By     interaction-spe p+4.0%"));

        status.metrics.last_episode_id = u64::MAX;
        status.metrics.last_episode_action = "predictive-prearm:Browser".to_string();
        status.metrics.last_episode_target = "Chromium Helper Renderer".to_string();
        assert!(render_act_q(&status)
            .iter()
            .all(|line| display_width(line) <= QW));
    }

    #[test]
    fn markov_prediction_labels_expired_timing_as_stale() {
        let mut status = dashboard_status();
        status.metrics.markov_prediction_app = "Alacritty".to_string();
        status.metrics.markov_prediction_confidence = 0.53;
        status.metrics.markov_prediction_eta_secs = 0.0;
        status.metrics.markov_prediction_overdue_secs = 333.0;
        status.metrics.markov_prewarm_blocker = "confidence".to_string();

        assert!(render_think_q(&status)
            .iter()
            .any(|line| line == "Mk-P   Alacritty 53% stale333s confidence"));
    }

    #[test]
    fn dashboard_explains_world_model_waits_and_markov_horizons() {
        let mut status = dashboard_status();
        status.metrics.world_model_actuator_known_models = 7;
        status.metrics.world_model_actuator_ready_models = 1;
        status.metrics.world_model_context_authority_phase = "trusted".to_string();
        status.metrics.world_model_readiness_immature = 4;
        status.metrics.world_model_readiness_uncertain = 2;
        status.metrics.markov_prediction_app = "Terminal".to_string();
        status.metrics.markov_prediction_5s = 0.05;
        status.metrics.markov_prediction_30s = 0.25;
        status.metrics.markov_prediction_2m = 0.70;
        status.metrics.markov_prediction_10m = 0.80;

        let think = render_think_q(&status);
        assert!(think.iter().any(|line| line == "WM-U   trusted 1/7"));
        assert!(think
            .iter()
            .any(|line| line == "WM wait immature4 uncertain2"));
        assert!(think.iter().any(|line| line == "Mk-H 5s5 30s25 2m70 10m80"));
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn value_refusals_name_the_gate_and_flag_pre_breakdown_history() {
        let mut status = dashboard_status();
        status.metrics.value_scheduler_phase = "active".to_string();
        status.metrics.value_scheduler_valid_cycles = 17_149;
        status.metrics.value_scheduler_invalid_samples_total = 3_880;
        status.metrics.value_scheduler_invalid_unhealthy_total = 3_880;
        status.metrics.value_scheduler_unhealthy_sleeping_total = 12;
        status.metrics.value_scheduler_unhealthy_thermal_total = 4;

        let think = render_think_q(&status);
        assert!(
            think
                .iter()
                .any(|line| line == "       why sleep12 therm4 pre4k"),
            "refusals must name their gate and admit the unattributed remainder: {think:?}"
        );
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn value_separates_this_cycle_from_lifetime_selection() {
        let mut status = dashboard_status();
        status.metrics.value_scheduler_phase = "active".to_string();
        status.metrics.value_scheduler_selected_jobs = 0;
        status.metrics.value_scheduler_registered_jobs = 10;
        status.metrics.value_scheduler_eligible_jobs = 0;
        status.metrics.value_scheduler_selected_total = 1_752;

        let think = render_think_q(&status);
        assert!(
            think
                .iter()
                .any(|line| line == "       now J0/10 elig0 · sel2k"),
            "an idle cycle must not read as a scheduler that never selects: {think:?}"
        );
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn dashboard_separates_models_still_maturing_from_models_gone_quiet() {
        let mut status = dashboard_status();
        status.metrics.world_model_actuator_known_models = 247;
        status.metrics.world_model_actuator_ready_models = 21;
        status.metrics.world_model_context_authority_phase = "trusted".to_string();
        status.metrics.world_model_readiness_immature = 20;
        status.metrics.world_model_readiness_dormant = 180;
        status.metrics.world_model_readiness_uncertain = 26;

        let think = render_think_q(&status);
        assert!(think
            .iter()
            .any(|line| line == "WM wait immature20 dormant180"));
        assert!(think.iter().any(|line| line == "        uncertain26"));
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn dashboard_shows_capacity_pressure_and_how_long_evidence_has_been_absent() {
        let mut status = dashboard_status();
        status.metrics.cycles = 11_065;
        status.metrics.world_model_action_model_len = 256;
        status.metrics.world_model_action_model_capacity = 256;
        status.metrics.world_model_action_model_evictions_total = 12;
        status.metrics.world_model_action_model_births_total = 17;
        status.metrics.world_model_evidence_updates_total = 20;
        status.metrics.world_model_last_evidence_cycle = 10_870;

        let think = render_think_q(&status);
        assert!(
            think
                .iter()
                .any(|line| line == "WM-C   256/256 full ev12 b17"),
            "a saturated map must say so: {think:?}"
        );
        assert!(
            think
                .iter()
                .any(|line| line == "WM-E   ev20 last c10,870 idle195"),
            "the age of the newest evidence must be visible: {think:?}"
        );
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn dashboard_says_plainly_when_no_evidence_has_ever_arrived() {
        let mut status = dashboard_status();
        status.metrics.world_model_last_evidence_cycle = 0;
        status.metrics.world_model_evidence_updates_total = 0;

        let think = render_think_q(&status);
        assert!(think
            .iter()
            .any(|line| line == "WM-E   ev0 no evidence yet"));
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn dashboard_reports_fabric_work_separately_from_verified_ane_use() {
        let mut status = dashboard_status();
        status.metrics.fabric_phase = "shadow".to_string();
        status.metrics.fabric_workers_active = 3;
        status.metrics.fabric_completed_total = 12;
        status.metrics.fabric_submitted_total = 14;
        status.metrics.fabric_cancelled_total = 1;
        status.metrics.coreml_configured_backend = "cpu-and-neural-engine".to_string();
        status.metrics.fabric_evaluation_total = 8;
        status.metrics.fabric_eligible_total = 10;
        status.metrics.coreml_ane_observation = "unsupported".to_string();
        status.metrics.temporal_prediction_backend = "cpu-utility".to_string();
        status.metrics.temporal_prediction_p95 = 0.42;

        let think = render_think_q(&status);
        assert!(think
            .iter()
            .any(|line| line == "Fabric shadow W3 C12/14 X1"));
        // `cfg:` marks the backend as configuration, and the ANE column says
        // the evidence is missing rather than showing a glyph the reader would
        // resolve to "fine" or "broken".
        assert!(think.iter().any(|line| line == "ML     cfg:cpu+ane E8/10"));
        assert!(think.iter().any(|line| line == "       ANE unobservable"));
        assert!(think
            .iter()
            .any(|line| line == "Pred   cpu-utility shadow p9542%"));
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn dashboard_separates_reflex_actions_model_support_and_parallel_profiles() {
        let mut status = dashboard_status();
        status.metrics.reflex_enabled = true;
        status.metrics.reflex_phase = "shadow".to_string();
        status.metrics.reflex_blocker = "warming-up".to_string();
        status.metrics.reflex_valid_cycles = 123;
        status.metrics.reflex_shadow_cycles = 500;
        status.metrics.reflex_proposed_total = 9;
        status.metrics.reflex_admitted_total = 3;
        status.metrics.reflex_applied_total = 2;
        status.metrics.reflex_shadowed_total = 6;
        status.metrics.reflex_omitted_total = 4;
        status.metrics.reflex_noop_total = 2;
        status.metrics.reflex_vetoed_total = 1;
        status.metrics.reflex_reverted_total = 1;
        status.metrics.reflex_failed_total = 0;
        status.metrics.reflex_last_decision_latency_us = 7;
        status.metrics.reflex_reasoning_submitted_total = 5;
        status.metrics.reflex_reasoning_completed_total = 4;
        status.metrics.reflex_reasoning_dropped_total = 1;
        status.metrics.reflex_reasoning_last_result_age_cycles = 2;
        status.metrics.reflex_reasoning_last_latency_us = 39;
        status.metrics.parallel_expected_profile = "adaptive-multicore".to_string();
        status.metrics.parallel_compiled_profile = "adaptive-multicore".to_string();
        status.metrics.parallel_effective_profile = "adaptive-multicore".to_string();
        status.metrics.parallel_worker_threads = 4;
        status.metrics.parallel_max_worker_threads = 4;
        status.metrics.parallel_worker_qos_intent = "utility".to_string();
        status.metrics.parallel_worker_qos_status = "ok".to_string();
        status.metrics.world_model_utility_promotions_total = 3;

        let think = render_think_q(&status);
        assert!(think.iter().any(|line| line == "Reflex shadow 123/500"));
        assert!(think.iter().any(|line| line == "R-act  P9 A3 real2 sh6"));
        assert!(think.iter().any(|line| line == "       O4 no2 V1 R1 F0"));
        assert!(think.iter().any(|line| line == "       fast 7us"));
        assert!(think.iter().any(|line| line == "R-deep W4/5 d1 age2 39us"));
        assert!(think
            .iter()
            .any(|line| line == "CPU-P  exp multi build multi"));
        assert!(think
            .iter()
            .any(|line| line == "       eff multi W4/4 utility ok"));
        assert!(think.iter().any(|line| line == "WM-S   V0 rank3"));
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn dashboard_explains_root_audio_fallback_and_cumulative_purges() {
        let mut status = dashboard_status();
        status.metrics.coreaudio_probe_state = "session-fallback".to_string();
        status.metrics.user_has_sleep_assertion = true;
        status.metrics.maintenance_purge_total = 2;

        let gates = render_gates_band(&status);
        assert!(gates
            .iter()
            .any(|line| line.contains("HAL n/a · guard media")));
        assert!(gates
            .iter()
            .any(|line| line.contains("purge totals: 2 fired")));

        status.metrics.user_has_sleep_assertion = false;
        assert!(render_gates_band(&status)
            .iter()
            .any(|line| line.contains("HAL n/a · guard unknown-safe")));
    }

    #[test]
    fn dashboard_reports_calm_release_instead_of_raw_context_switch_burst() {
        let mut status = dashboard_status();
        status.effective_profile = OptimizationProfile::AggressiveRoot;
        status.metrics.context_switch_burst = true;
        status.metrics.arousal_level = 0.15;
        status.transition_reason = "context-switch-burst-suppressed-calm".to_string();

        assert_eq!(
            profile_activity_reason(&status),
            "Auto: liberando por calma"
        );
        assert!(render_decide_q(&status)
            .iter()
            .any(|line| line == "Motivo  Auto: liberando por calma"));
    }

    #[test]
    fn dashboard_explains_safe_root_at_rest() {
        let mut status = dashboard_status();
        status.effective_profile = OptimizationProfile::SafeRoot;
        status.transition_reason = "steady".to_string();

        assert_eq!(profile_activity_reason(&status), "Auto: ahorro en reposo");
    }

    #[test]
    fn dashboard_marks_manual_overrides_explicitly() {
        let mut status = dashboard_status();
        status.effective_profile = OptimizationProfile::AggressiveRoot;
        status.override_active = true;
        status.metrics.context_switch_burst = true;

        assert_eq!(profile_activity_reason(&status), "Override manual");
    }

    #[test]
    fn think_quadrant_surfaces_medallion_quality_and_ais_learning() {
        let mut status = dashboard_status();
        status.metrics.ais_score = 92.4;
        status.metrics.ais_capability = 92.4;
        status.metrics.ais_optimization_opportunity = 0.18;
        status.metrics.ais_grade = "S".to_string();
        status.metrics.ais_learning = 0.88;
        status.metrics.learning_bronze_total = 200;
        status.metrics.learning_gold_total = 198;
        status.metrics.learning_data_quality = 0.99;
        status.metrics.world_model_curated_actions = 3;
        status.metrics.world_model_ready_actions = 2;
        status.metrics.world_model_utility_promotions_total = 3;
        status.metrics.world_model_gold_evidence = 127;
        status.metrics.world_model_contextual_actions = 1;
        status.metrics.world_model_data_quality = 1.0;
        status.metrics.world_model_context_bronze_total = 500;
        status.metrics.world_model_context_silver_total = 7;
        status.metrics.world_model_context_gold_total = 490;
        status.metrics.world_model_context_rejected_total = 3;
        status.metrics.world_model_context_stale_total = 1;
        status.metrics.world_model_context_quality = 0.98;
        status.metrics.world_model_context_authority_phase = "calibrating".to_string();
        status.metrics.world_model_actuator_issued_total = 12;
        status.metrics.world_model_actuator_pending_total = 2;
        status.metrics.world_model_actuator_bronze_total = 10;
        status.metrics.world_model_actuator_gold_total = 8;
        status.metrics.world_model_actuator_quality = 0.94;
        status.metrics.world_model_actuator_known_models = 7;
        status.metrics.world_model_actuator_ready_models = 2;
        status.metrics.world_model_counterfactual_issued_total = 4;
        status.metrics.world_model_counterfactual_resolved_total = 3;
        status.metrics.world_model_counterfactual_would_help_total = 2;
        status.metrics.world_model_counterfactual_rank_uses_total = 7;
        status.metrics.world_model_episodic_memory_samples = 23;
        status.metrics.world_model_episodic_memory_families = 6;
        status.metrics.world_model_episodic_rank_uses_total = 9;
        status.metrics.world_model_contextual_markov_total = 12;
        status.metrics.world_model_contextual_interaction_total = 8;
        status.metrics.world_model_contextual_io_total = 5;
        status.metrics.world_model_contextual_predictive_total = 42;
        status.metrics.world_model_contextual_chromium_total = 4;
        status.metrics.world_model_contextual_last_action =
            "predictive_threshold:tighten".to_string();
        status.metrics.world_model_contextual_last_bias = 0.35;
        status.metrics.markov_shadow_predictions_total = 5;
        status.metrics.markov_shadow_resolved_total = 4;
        status.metrics.markov_shadow_hits = 3;
        status.metrics.world_model_causal_actuator_gold_total = 42;
        status.metrics.world_model_temporal_memory_samples = 32;
        status.metrics.world_model_sequence_rollouts_total = 123;
        status.metrics.world_model_sequence_expected_gain = 0.021;
        status.metrics.world_model_sequence_best_first = "boost:Editor".to_string();
        status.metrics.world_model_sequence_best_second =
            "markov_prewarm:predicted_app".to_string();
        status.metrics.world_model_dynamics_phase = "shadow".to_string();
        status.metrics.world_model_dynamics_action_models = 9;
        status.metrics.world_model_dynamics_ready_models = 6;
        status.metrics.world_model_dynamics_ranking_models = 4;
        status.metrics.world_model_dynamics_validation_samples = 42;
        status.metrics.world_model_dynamics_validation_mae = 0.032;
        status.metrics.world_model_dynamics_predictions_total = 17;
        status
            .metrics
            .world_model_dynamics_ranking_predictions_total = 9;
        status
            .metrics
            .world_model_dynamics_authoritative_predictions_total = 3;
        status.metrics.world_model_dynamics_baseline_uses_total = 8;
        status.metrics.world_model_dynamics_mean_uncertainty = 0.071;
        status.metrics.gpu_imagination_backend = "metal".to_string();
        status.metrics.gpu_imagination_jobs_completed_total = 3;
        status.metrics.gpu_imagination_samples_total = 196_608;
        status.metrics.gpu_imagination_gpu_time_ns_total = 1_260_000;
        status.metrics.gpu_imagination_last_best_action = "boost:Editor".to_string();
        status.metrics.gpu_imagination_last_positive_probability = 0.82;
        status.metrics.gpu_imagination_last_p10_gain = 0.012;
        status.metrics.gpu_imagination_root_rank_uses_total = 4;
        status.metrics.gpu_imagination_contextual_uses_total = 9;
        status.metrics.gpu_imagination_last_influence_scope = "markov-prewarm".to_string();
        status.metrics.gpu_imagination_last_influence_action =
            "markov_prewarm:predicted_app".to_string();
        status.metrics.gpu_imagination_last_influence_support = 0.012;
        status.metrics.system_deliberation_mode = "calibrating".to_string();
        status.metrics.system_deliberation_confidence = 0.78;
        status.metrics.system_deliberation_local_gold = 24;
        status.metrics.system_deliberation_gpu_forecasts = 3;
        status.metrics.system_deliberation_local_confidence = 0.66;
        status.metrics.system_deliberation_local_families = 1;
        status.metrics.local_consolidations = 12;
        status.metrics.local_consolidation_improvements = 8;
        status.metrics.local_consolidation_regressions = 2;
        status.metrics.local_consolidation_neutral = 2;
        status.metrics.local_consolidation_system1_updates = 20;
        status.metrics.world_model_actuator_families =
            vec![apollo_engine::engine::types::ActuatorEvidenceStatus {
                family: "boost".to_string(),
                issued: 7,
                resolved: 6,
                gold: 5,
                effective: 4,
                rejected: 0,
                expired: 0,
                mean_quality: 0.95,
                mean_utility: 0.03,
            }];

        let think = render_think_q(&status);

        assert!(think.iter().any(|line| line == "AIS    C92.4 S O18%"));
        assert!(think.iter().any(|line| line == "       L 88%"));
        assert!(think.iter().any(|line| line == "Data   G 198/200 q99%"));
        assert!(think
            .iter()
            .any(|line| line == "Causal 2/3 ready G127 q100%"));
        assert!(think.iter().any(|line| line == "Caus+  U42 universal Gold"));
        assert!(think.iter().any(|line| line == "Ctx-L  G490 S7 R3 q98%"));
        assert!(think.iter().any(|line| line == "Act    G 8/10 P2 q94%"));
        assert!(think.iter().any(|line| line == "Act+   boost G5/6"));
        assert!(think.iter().any(|line| line == "       eff 4/6 u+3%"));
        assert!(think.iter().any(|line| line == "WM-U   calibrating 2/7"));
        assert!(think.iter().any(|line| line == "WM-S   V0 rank3"));
        assert!(think.iter().any(|line| line == "CF     R3/4 H2 rank7"));
        assert!(think.iter().any(|line| line == "WM-E   M23 F6 rank9"));
        assert!(think.iter().any(|line| line == "WM-X   K12 Q8 O5 P42 C4"));
        assert!(think
            .iter()
            .any(|line| line == "       last pred-thresh b+0.35"));
        assert!(think.iter().any(|line| line == "Mk-S   H3/4 P5"));
        assert!(think.iter().any(|line| line == "WM-T   M32 R123 G+2.1%"));
        assert!(think
            .iter()
            .any(|line| line == "Seq-E  boost>markov_prewarm"));
        assert!(think
            .iter()
            .any(|line| line == "WM-D   shadow A0 R4/6 V42 e3.2%"));
        assert!(think.iter().any(|line| line == "MPC-D  P17 R9 A3 B8 u7.1%"));
        assert!(think
            .iter()
            .any(|line| line == "GPU-I  metal J3 S196,608 t0.42ms"));
        assert!(think.iter().any(|line| line == "GPU+   boost p82% d+1.2%"));
        assert!(think
            .iter()
            .any(|line| line == "GPU-U  R4 C9 markov:marko +1.20%"));
        assert!(think
            .iter()
            .any(|line| line == "Delib  cal S1 c78% G24 g3 F1"));
        assert!(think
            .iter()
            .any(|line| line == "S2>S1  G12 +8/-2 n2 s20 c66%"));
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn think_quadrant_explains_protected_world_model_without_models() {
        let mut status = dashboard_status();
        status.metrics.world_model_context_authority_phase = "protected".to_string();

        let think = render_think_q(&status);

        assert!(think
            .iter()
            .any(|line| line == "WM-U   protected · no evidence"));
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn think_quadrant_explains_gpu_imagination_without_completed_jobs() {
        let mut status = dashboard_status();
        status.metrics.gpu_imagination_backend = "metal".to_string();
        status.metrics.gpu_imagination_last_submit_outcome = "no-candidates".to_string();

        let think = render_think_q(&status);

        assert!(think
            .iter()
            .any(|line| line == "GPU-I  metal J0 S0 t0.00ms"));
        assert!(think.iter().any(|line| line == "GPU-G  no-candidates"));
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn think_quadrant_separates_exact_web_from_universal_network_flow() {
        let mut status = dashboard_status();
        status.metrics.webflow_mode = "lifecycle".to_string();
        status.metrics.webflow_active_navigations = 1;
        status.metrics.webflow_proposed_total = 12;
        status.metrics.webflow_admitted_total = 4;
        status.metrics.network_flow_active = true;
        status.metrics.network_flow_traffic_bps = 2_400_000;
        status.metrics.network_flow_proposed_total = 8;
        status.metrics.network_flow_renewed_total = 6;
        status.metrics.network_flow_suppressed_exact_total = 3;

        let think = render_think_q(&status);
        assert!(think
            .iter()
            .any(|line| line == "Web    lifecycle N1 P12/A4"));
        assert!(think
            .iter()
            .any(|line| line == "Net+   active 2.4MB/s P8 R6 X3"));
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn think_quadrant_exposes_the_metal_circuit_breaker_state() {
        let mut status = dashboard_status();
        status.metrics.gpu_imagination_backend = "metal".to_string();
        status.metrics.gpu_imagination_circuit_state = "half-open".to_string();

        let think = render_think_q(&status);

        assert!(think
            .iter()
            .any(|line| line == "GPU-I  metal/H J0 S0 t0.00ms"));
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn think_quadrant_distinguishes_idle_temporal_model_from_cold_start() {
        let mut status = dashboard_status();
        status.metrics.world_model_temporal_memory_samples = 32;
        status.metrics.world_model_sequence_abstention_reason = "idle_no_accelerator".to_string();

        let think = render_think_q(&status);

        assert!(think
            .iter()
            .any(|line| line == "Seq-E  idle_no_accelerator"));
        assert!(think
            .iter()
            .any(|line| line == "Seq-A  idle_no_accelerator"));
        assert!(think.iter().all(|line| display_width(line) <= QW));
    }

    #[test]
    fn swap_label_stable_when_low_and_no_delta() {
        assert_eq!(swap_status_label(0.5, 0.0), "🟢 Estable");
        assert_eq!(swap_status_label(3.9, 0.0), "🟢 Estable");
    }

    #[test]
    fn swap_label_alto_when_between_4_and_8gb() {
        assert_eq!(swap_status_label(4.0, 0.0), "🟠 Alto");
        assert_eq!(swap_status_label(6.7, 0.0), "🟠 Alto");
    }

    #[test]
    fn swap_label_critico_when_8gb_or_more() {
        assert_eq!(swap_status_label(8.0, 0.0), "🔴 Crítico");
        assert_eq!(swap_status_label(13.5, 0.0), "🔴 Crítico");
    }

    #[test]
    fn swap_label_growing_overrides_amount() {
        // Even at 12 GB, if actively growing, show the dynamic rate label
        assert_eq!(swap_status_label(12.0, 500.0), "📈 Creciendo");
    }

    #[test]
    fn swap_label_falling_overrides_amount() {
        assert_eq!(swap_status_label(12.0, -500.0), "📉 Bajando");
    }

    #[test]
    fn swap_label_stable_bug_is_fixed() {
        // This is the exact bug that was reported: 12.7 GB showing 🟢 Estable
        // because only delta_bps was checked, not the absolute amount.
        let label = swap_status_label(12.7, 0.0); // delta=0 (not growing)
        assert_ne!(label, "🟢 Estable", "12.7 GB swap must NOT show Estable");
        assert_eq!(label, "🔴 Crítico");
    }

    #[test]
    fn swap_visual_ratio_uses_reported_total_when_available() {
        let gib = 1_073_741_824u64;
        assert!((swap_visual_ratio(8 * gib, 16 * gib) - 0.5).abs() < 0.001);
    }

    #[test]
    fn swap_visual_ratio_falls_back_to_8gb_when_total_missing() {
        let gib = 1_073_741_824u64;
        assert!((swap_visual_ratio(4 * gib, 0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn swap_label_uses_relative_capacity_on_16gb_swap() {
        assert_eq!(swap_status_label_for_total(8.0, 16.0, 0.0), "🟠 Alto");
        assert_eq!(swap_status_label_for_total(14.0, 16.0, 0.0), "🔴 Crítico");
    }

    #[test]
    fn act_quadrant_uses_cycle_hint_budget_not_cumulative_hints() {
        let mut metrics = RuntimeMetrics {
            paging_hints_applied: 13,
            boosts_applied: 505,
            freezes_applied: 4,
            unfreezes_applied: 1,
            ..RuntimeMetrics::default()
        };
        metrics.budgets.cycle_hints = 0;

        let status = DaemonStatus {
            running: true,
            profile: OptimizationProfile::AggressiveRoot,
            latency_target: LatencyTarget::Normal,
            effective_profile: OptimizationProfile::AggressiveRoot,
            kill_switch: false,
            throttle_level: "medium".to_string(),
            thermal_state: "nominal".to_string(),
            last_blockers: Vec::new(),
            auto_profile_enabled: true,
            base_profile: OptimizationProfile::AggressiveRoot,
            override_active: false,
            override_expires_at: None,
            transition_reason: String::new(),
            post_wake_grace_active: false,
            post_wake_grace_remaining_secs: 0,
            last_wake_at: None,
            post_wake_policy: String::new(),
            reactor_mode: "normal".to_string(),
            reactor_health: "ok".to_string(),
            metrics,
            frozen_processes: Vec::new(),
        };

        let lines = render_act_q(&status);
        let hint_line = lines
            .iter()
            .find(|line| line.starts_with("Cycle H"))
            .expect("ACT quadrant should render hint budget line");
        let total_line = lines
            .iter()
            .find(|line| line.starts_with("Total "))
            .expect("ACT quadrant should render cumulative action totals");

        assert!(
            hint_line.contains("0/20"),
            "hint budget must show current-cycle count: {hint_line}"
        );
        assert!(
            !hint_line.contains("13/20"),
            "cumulative hints must not be compared to per-cycle cap: {hint_line}"
        );
        assert!(
            total_line.contains("B505") && total_line.contains("F4") && total_line.contains("U1"),
            "cumulative action totals must stay visible: {total_line}"
        );
    }

    #[test]
    fn act_quadrant_separates_consumed_and_avoided_co2() {
        let mut status = dashboard_status();
        status.metrics.energy_package_wh = Some(8.46);
        status.metrics.energy_co2_emitted_g = Some(3.30);
        status.metrics.energy_savings_wh = Some(0.0);
        status.metrics.energy_co2_avoided_g = Some(0.0);

        let lines = render_act_q(&status);

        assert!(lines.iter().any(|line| line == "Used   8.46Wh 3.30gCO₂"));
        assert!(lines.iter().any(|line| line == "Saved  0.00Wh 0.00gCO₂"));
    }
}
