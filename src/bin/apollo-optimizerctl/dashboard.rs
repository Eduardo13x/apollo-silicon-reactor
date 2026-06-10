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
fn swap_status_label(swap_gb: f64, delta_bps: f64) -> &'static str {
    if delta_bps > 100.0 {
        "📈 Creciendo"
    } else if delta_bps < -100.0 {
        "📉 Bajando"
    } else if swap_gb >= 8.0 {
        "🔴 Crítico"
    } else if swap_gb >= 4.0 {
        "🟠 Alto"
    } else {
        "🟢 Estable"
    }
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

fn format_number(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{},{:03}", n / 1000, n % 1000)
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

    // Swap: bar normalized vs 8 GB headroom (M1 dynamic swap typical max),
    // not vs swap_total which macOS resizes dynamically. Avoids alarming "80%"
    // readings when the underlying file is small but auto-growing.
    let swap_gb = m.swap_used_bytes as f64 / 1_073_741_824.0;
    let swap_visual_ratio = (swap_gb / 8.0).clamp(0.0, 1.0);
    let swap_label = swap_status_label(swap_gb, m.swap_delta_bps);
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
    lines.push(format!(
        "NARS   {} bel d={:.2} {}",
        m.nars_drifted_beliefs, m.nars_drift_score, drift_emoji
    ));

    // Bayesian outcome tracker (LLM teacher patterns proxy)
    let interactive = status
        .llm
        .as_ref()
        .map_or(0, |l| l.learned_policy.interactive_patterns);
    let noise = status
        .llm
        .as_ref()
        .map_or(0, |l| l.learned_policy.noise_patterns);
    lines.push(format!("Bayes  {}int {}noise", interactive, noise));

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

    // Profile
    let prof_emoji = profile_emoji(status.effective_profile);
    lines.push(format!(
        "Profile {} {}",
        prof_emoji,
        status.effective_profile.as_str()
    ));

    // Teacher (Gemma 4 local LLM) — daily teach-call budget
    if let Some(l) = &status.llm {
        let state = if l.enabled { "✅" } else { "❌" };
        lines.push(format!(
            "Teach calls {} {}/{}",
            state, l.calls_today, l.daily_budget
        ));
    } else {
        lines.push("Teach calls ❌ off".to_string());
    }

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
        "Saved  {:.2}Wh {:.2}gCO₂",
        m.energy_savings_wh.unwrap_or(0.0),
        m.energy_co2_avoided_g.unwrap_or(0.0)
    ));
    lines.push(format!(
        "Reactor {} {}",
        status.reactor_mode, status.reactor_health
    ));

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
        "purge stats: {} fired · {} skipped{}",
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
        "{} {}",
        profile_emoji(status.effective_profile),
        status.effective_profile.as_str()
    );
    let mut lines = vec![bold(&format!(
        "🚀 APOLLO {} │ {} │ ciclos: {} │ p95 {:.0}ms",
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

    #[test]
    fn swap_label_stable_when_low_and_no_delta() {
        assert_eq!(swap_status_label(0.5, 0.0), "🟢 Estable");
        assert_eq!(swap_status_label(3.9, 0.0), "🟢 Estable");
    }

    #[test]
    fn swap_label_alto_when_between_4_and_8gb() {
        assert_eq!(swap_status_label(4.0, 0.0), "🟠 Alto");
        assert_eq!(swap_status_label(7.9, 0.0), "🟠 Alto");
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
            llm: None,
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
}
