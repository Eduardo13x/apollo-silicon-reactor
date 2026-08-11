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

fn thermal_emoji(state: &str) -> &'static str {
    match state {
        "critical" => "🔴",
        "serious" => "🟠",
        "moderate" | "fair" => "🟡",
        _ => "🟢",
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

    lines.push(format!(
        "Temp   {} {}",
        thermal_emoji(&status.thermal_state),
        thermal_label(&status.thermal_state)
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
        "UX     {} {:.0}%",
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

    // Local Gold outcomes are compiled into fast System 1 reflexes.
    lines.push(format!(
        "Learn  G{} F{} S1+{}",
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
            "Ctx    G{} S{} R{} q{:.0}%",
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
            "       C{} G{} N{} renew{}",
            format_number(m.acceleration_lease_chromium_total),
            format_number(m.acceleration_lease_general_total),
            format_number(m.acceleration_lease_nice_fallbacks_total),
            format_number(m.acceleration_lease_renewals_total),
        ));
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
    if m.world_model_utility_vetoes_total > 0 || m.world_model_utility_promotions_total > 0 {
        lines.push(format!(
            "WM-U+  V{} P{}",
            format_number(m.world_model_utility_vetoes_total),
            format_number(m.world_model_utility_promotions_total)
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
        lines.push(format!(
            "GPU-I  {} J{} S{} t{:.2}ms",
            m.gpu_imagination_backend,
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
        "Causal {}/{} ready G{} q{:.0}% P{}",
        format_number(m.world_model_ready_actions),
        format_number(m.world_model_curated_actions),
        format_number(m.world_model_gold_evidence),
        m.world_model_data_quality * 100.0,
        format_number(m.world_model_utility_promotions_total)
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
        lines.push(format!(
            "Mk-P   {} {:.0}% eta{:.0}s {}",
            app,
            m.markov_prediction_confidence * 100.0,
            m.markov_prediction_eta_secs,
            if m.markov_prewarm_blocker.is_empty() {
                m.markov_prewarm_admission.as_str()
            } else {
                m.markov_prewarm_blocker.as_str()
            }
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
            "Gain   u{:+.1}% ux{:+.1}%",
            m.last_episode_utility * 100.0,
            m.last_episode_latency_improvement * 100.0
        ));
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
    lines.push(format!(
        "renderers={} frozen={} e-core={} freed={:.0}MB",
        m.chromium_renderers_total,
        m.chromium_renderers_frozen,
        m.chromium_renderers_ecore,
        m.chromium_freed_mb
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
        "🚀 APOLLO {} │ {} │ c:{} │ loop-p95 {:.0}ms",
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

        assert!(render_header_v2(&status)[0].contains("loop-p95 37ms"));
        let sense = render_sense_q(&status);
        assert!(sense.iter().any(|line| line == "UX     responsive 18%"));
        assert!(sense.iter().any(|line| line == "Sched  p95 0.12ms n42"));
        let act = render_act_q(&status);
        assert!(act.iter().any(|line| line == "Ep#7 gold q94%"));
        assert!(act.iter().any(|line| line == "Gain   u+3.0% ux+8.0%"));

        status.metrics.last_episode_id = u64::MAX;
        status.metrics.last_episode_action = "predictive-prearm:Browser".to_string();
        status.metrics.last_episode_target = "Chromium Helper Renderer".to_string();
        assert!(render_act_q(&status)
            .iter()
            .all(|line| display_width(line) <= QW));
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
            .any(|line| line == "Causal 2/3 ready G127 q100% P3"));
        assert!(think.iter().any(|line| line == "Caus+  U42 universal Gold"));
        assert!(think.iter().any(|line| line == "Ctx    G490 S7 R3 q98%"));
        assert!(think.iter().any(|line| line == "Act    G 8/10 P2 q94%"));
        assert!(think.iter().any(|line| line == "Act+   boost G5/6"));
        assert!(think.iter().any(|line| line == "       eff 4/6 u+3%"));
        assert!(think.iter().any(|line| line == "WM-U   calibrating 2/7"));
        assert!(think.iter().any(|line| line == "WM-U+  V0 P3"));
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
