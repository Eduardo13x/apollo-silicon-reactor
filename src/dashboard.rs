//! ASCII dashboard renderer for apollo-optimizerctl.
//!
//! Renders a visual summary of daemon status using Unicode box-drawing,
//! ANSI colors, and emoji indicators.

use crate::engine::types::{
    BlockerScore, DaemonStatus, EnergyConsumerInfo, FreezeSource, OptimizationProfile, SafetyPolicy,
};
use chrono::Utc;

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

fn pressure_emoji(p: f64) -> &'static str {
    if p >= 0.85 {
        "🔴"
    } else if p >= 0.60 {
        "🟠"
    } else if p >= 0.40 {
        "🟡"
    } else {
        "🟢"
    }
}

fn pressure_label(p: f64) -> &'static str {
    if p >= 0.85 {
        "Crítica"
    } else if p >= 0.60 {
        "Presión"
    } else if p >= 0.40 {
        "Moderado"
    } else {
        "Normal"
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

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.0} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{:.0} KB", bytes as f64 / 1024.0)
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

// ── Section renderers ──
// Each returns Vec<String> of raw content lines (not box-wrapped).

fn render_header(status: &DaemonStatus) -> Vec<String> {
    let state = if status.kill_switch {
        yellow("⏸️  Pausado")
    } else if status.running {
        green("🟢 Activo")
    } else {
        red("🔴 Detenido")
    };
    let profile = format!(
        "{} {}",
        profile_emoji(status.effective_profile),
        status.effective_profile.as_str()
    );
    let mut lines = vec![
        bold("🚀 APOLLO OPTIMIZER — Dashboard"),
        format!(
            "Estado: {} | Perfil: {} | Ciclos: {}",
            state,
            profile,
            format_number(status.metrics.cycles)
        ),
    ];
    if status.kill_switch {
        lines.push(yellow(
            "⚠ Optimización pausada — ejecuta: apollo-optimizerctl resume",
        ));
    }
    lines
}

fn render_system(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold("📊 SISTEMA")];

    // Memory pressure
    let mp = m.memory_pressure;
    let mp_bar = render_bar(mp, 20);
    lines.push(format!(
        "RAM    {}  {:>3.0}%   {} {}",
        mp_bar,
        mp * 100.0,
        pressure_emoji(mp),
        pressure_label(mp)
    ));

    // Swap
    let swap_str = format_bytes(m.swap_used_bytes);
    let swap_ratio = (m.swap_used_bytes as f64 / (8.0 * 1_073_741_824.0)).min(1.0);
    let swap_bar = render_bar(swap_ratio, 20);
    let swap_gb = m.swap_used_bytes as f64 / 1_073_741_824.0;
    let swap_label = if m.swap_delta_bps > 100.0 {
        "📈 Creciendo"
    } else if m.swap_delta_bps < -100.0 {
        "📉 Bajando"
    } else if swap_gb >= 8.0 {
        "🔴 Crítico"
    } else if swap_gb >= 4.0 {
        "🟠 Alto"
    } else {
        "🟢 Estable"
    };
    lines.push(format!(
        "Swap   {}  {:>7}  {}",
        swap_bar, swap_str, swap_label
    ));

    // Temperature
    if let Some(temp) = m.iokit_p_cluster_temp {
        let temp_ratio = (temp as f64 / 110.0).min(1.0);
        let temp_bar = render_bar(temp_ratio, 20);
        lines.push(format!(
            "Temp   {}  {:>4.0}°C  {} {}",
            temp_bar,
            temp,
            thermal_emoji(&m.thermal_state),
            thermal_label(&m.thermal_state)
        ));
    } else {
        lines.push(format!(
            "Temp   {}     {} {}",
            dim("[sin sensor IOKit]"),
            thermal_emoji(&m.thermal_state),
            thermal_label(&m.thermal_state)
        ));
    }

    // Pressure score
    let ps = m.last_pressure_score;
    let ps_bar = render_bar(ps, 20);
    lines.push(format!(
        "Score  {}  {:>4.2}   {} {}",
        ps_bar,
        ps,
        pressure_emoji(ps),
        pressure_label(ps)
    ));

    lines
}

fn render_intelligence(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold("🧠 INTELIGENCIA")];

    let workload = if m.current_workload.is_empty() {
        "desconocida"
    } else {
        &m.current_workload
    };
    let confidence = format!("{:.0}%", m.ml_confidence * 100.0);
    lines.push(format!(
        "Carga actual: 💻 {} (confianza: {})",
        workload, confidence
    ));

    let dev = if m.dev_session_active {
        green("✅ Activa")
    } else {
        dim("⬜ Inactiva")
    };
    let inter = if m.interactive_heavy {
        green("✅ Alto")
    } else {
        dim("⬜ Normal")
    };
    lines.push(format!("Sesión dev: {} | Interactivo: {}", dev, inter));

    if !m.ml_sources.is_empty() {
        let sources = m
            .ml_sources
            .iter()
            .take(4)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!("Fuentes: {}", dim(&sources)));
    }

    lines
}

fn render_foreground(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold("🎯 FOREGROUND")];

    match &m.foreground_app {
        Some(app) => {
            lines.push(format!("App activa: {} (PID: {})", app.name, app.pid));
            let state = green("🟢 Activo");
            let idle = if m.foreground_idle {
                yellow("Sí")
            } else {
                green("❌")
            };
            lines.push(format!("Estado: {} | Idle: {}", state, idle));
        }
        None => {
            if m.foreground_idle {
                lines.push(format!(
                    "App activa: {} | Estado: {}",
                    dim("ninguna"),
                    yellow("💤 Idle")
                ));
            } else {
                lines.push(format!(
                    "App activa: {} | Estado: {}",
                    dim("N/A"),
                    dim("sin datos")
                ));
            }
        }
    }

    lines
}

fn render_energy(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold("⚡ ENERGÍA")];

    // Power readings: prefer energy tracker fields, fall back to iokit raw values
    let pkg_w = m
        .energy_package_watts
        .unwrap_or_else(|| m.iokit_package_watts.map(|w| w as f64).unwrap_or(0.0));
    let cpu_w = m.energy_cpu_watts.unwrap_or(0.0);
    let gpu_w = m.energy_gpu_watts.unwrap_or(0.0);

    // Only show power line if we have any data
    if pkg_w > 0.0 || cpu_w > 0.0 || gpu_w > 0.0 {
        lines.push(format!(
            "Paquete: {:.1}W | CPU: {:.1}W | GPU: {:.1}W",
            pkg_w, cpu_w, gpu_w
        ));
    } else {
        lines.push(format!("Potencia: {}", dim("sin datos IOKit")));
    }

    // Session energy and CO2
    let session_wh = m.energy_session_wh.unwrap_or(0.0);
    let co2_g = m.energy_co2_avoided_g.unwrap_or(0.0);
    if session_wh > 0.0 || co2_g > 0.0 {
        lines.push(format!(
            "Sesión: {:.2} Wh | CO₂ evitado: {:.2}g",
            session_wh, co2_g
        ));
    }

    // Top energy consumers
    if !m.energy_top_consumers.is_empty() {
        lines.push("Top consumidores:".into());
        for (i, consumer) in m.energy_top_consumers.iter().take(5).enumerate() {
            lines.push(format_energy_consumer(i + 1, consumer));
        }
    }

    lines
}

/// Format a single energy consumer line with a proportional bar.
fn format_energy_consumer(rank: usize, consumer: &EnergyConsumerInfo) -> String {
    let name = if consumer.name.len() > 18 {
        &consumer.name[..18]
    } else {
        &consumer.name
    };
    let bar = render_bar(consumer.percentage / 100.0, 16);
    format!(
        "  {}. {} {:>5.1}W  {}  {:>3.0}%",
        rank,
        pad_right(name, 18),
        consumer.current_watts,
        bar,
        consumer.percentage
    )
}

fn render_actions(status: &DaemonStatus) -> Vec<String> {
    let b = &status.metrics.budgets;
    let policy = SafetyPolicy::for_profile(status.effective_profile);
    let mut lines = vec![bold("🎯 ACCIONES (este ciclo)")];

    let boost_bar = render_bar(
        b.cycle_boosts as f64 / policy.max_boosts_per_cycle.max(1) as f64,
        6,
    );
    let throttle_bar = render_bar(
        b.cycle_throttles as f64 / policy.max_throttles_per_cycle.max(1) as f64,
        6,
    );
    lines.push(format!(
        "Boosts   {}  {}/{}    Throttles  {}  {}/{}",
        boost_bar,
        b.cycle_boosts,
        policy.max_boosts_per_cycle,
        throttle_bar,
        b.cycle_throttles,
        policy.max_throttles_per_cycle
    ));

    let freeze_bar = render_bar(
        b.cycle_freezes as f64 / policy.max_freezes_per_cycle.max(1) as f64,
        6,
    );
    let hint_bar = render_bar(
        b.cycle_hints as f64 / policy.max_paging_hints_per_cycle.max(1) as f64,
        6,
    );
    lines.push(format!(
        "Freezes  {}  {}/{}    Hints      {}  {}/{}",
        freeze_bar,
        b.cycle_freezes,
        policy.max_freezes_per_cycle,
        hint_bar,
        b.cycle_hints,
        policy.max_paging_hints_per_cycle
    ));

    lines
}

fn render_blockers(blockers: &[BlockerScore]) -> Vec<String> {
    let mut lines = vec![bold("🔴 TOP BLOQUEADORES")];

    if blockers.is_empty() {
        lines.push(green("Sin bloqueadores activos"));
        return lines;
    }

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

fn render_frozen(status: &DaemonStatus) -> Vec<String> {
    let mut lines = vec![bold("🧊 PROCESOS CONGELADOS")];
    if status.frozen_processes.is_empty() {
        lines.push(green("Ninguno congelado actualmente"));
        return lines;
    }
    lines.push(format!(
        "{:<6} {:<22} {:>8} {:>6} {:<14}",
        "PID", "NOMBRE", "TIEMPO", "P@FRZ", "FUENTE"
    ));
    lines.push("─".repeat(CW.min(60)));
    let mut sorted = status.frozen_processes.clone();
    sorted.sort_by_key(|p| p.frozen_seconds);
    sorted.reverse();
    for p in sorted.iter().take(10) {
        let source = match p.source {
            FreezeSource::MainLoop => "MainLoop",
            FreezeSource::Sentinel => "Sentinel",
            FreezeSource::Manual => "Manual",
            FreezeSource::ThermalPreThrottle => "ThermalPre",
        };
        let time = if p.frozen_seconds >= 3600 {
            format!("{}h {:02}m", p.frozen_seconds / 3600, (p.frozen_seconds % 3600) / 60)
        } else if p.frozen_seconds >= 60 {
            format!("{}m {:02}s", p.frozen_seconds / 60, p.frozen_seconds % 60)
        } else {
            format!("{}s", p.frozen_seconds)
        };
        let name = if p.name.len() > 22 { &p.name[..22] } else { &p.name };
        lines.push(format!(
            "{:<6} {:<22} {:>8} {:>5.2} {:<14}",
            p.pid, name, time, p.pressure_at_freeze, source
        ));
    }
    if status.frozen_processes.len() > 10 {
        lines.push(format!("  ... y {} más", status.frozen_processes.len() - 10));
    }
    lines
}

fn render_reactor(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold("⚡ REACTOR")];

    let mode_emoji = match m.reactor_mode.as_str() {
        "aggressive" => "🔴",
        "elevated" => "🟡",
        _ => "🔵",
    };
    let health_emoji = match m.reactor_health.as_str() {
        "degraded" | "critical" => "💔",
        _ => "💚",
    };

    lines.push(format!(
        "Modo: {} {} | Salud: {} {} | Pulsos: {}",
        mode_emoji,
        m.reactor_mode,
        health_emoji,
        m.reactor_health,
        format_number(m.reactor_pulses)
    ));

    lines.push(format!(
        "Eventos: {} total (Mem:{} Therm:{} Spawn:{} Power:{})",
        format_number(m.reactor_events_total),
        m.reactor_events_mem,
        m.reactor_events_thermal,
        m.reactor_events_spawn,
        m.reactor_events_power
    ));

    // Resource sentinel (interrupt handler) status.
    let phase_label = match m.resource_interrupt_last_phase {
        1 => "MODERATE",
        2 => "EMERGENCY",
        3 => "SUPER-EMERGENCY",
        _ => "idle",
    };
    let sentinel_emoji = if m.resource_interrupt_active {
        "🚨"
    } else {
        "✅"
    };
    lines.push(format!(
        "Sentinel: {} {} | Fires: {} | Lat: {}μs",
        sentinel_emoji,
        phase_label,
        format_number(m.resource_interrupts_total),
        format_number(m.resource_interrupt_latency_us)
    ));
    if m.resource_interrupt_processes_frozen > 0
        || m.resource_interrupt_processes_migrated > 0
        || m.resource_interrupt_recovery_count > 0
    {
        lines.push(format!(
            "  Frozen: {} | Migrated: {} | Recovered: {}",
            format_number(m.resource_interrupt_processes_frozen),
            format_number(m.resource_interrupt_processes_migrated),
            format_number(m.resource_interrupt_recovery_count)
        ));
    }

    lines
}

fn render_session(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold("📈 SESIÓN")];

    lines.push(format!(
        "Ciclos: {} | Cambios perfil: {} | Zombies: {} | Kills: {}",
        format_number(m.cycles),
        m.profile_switches,
        m.zombies_detected,
        m.kills_applied
    ));

    lines.push(format!(
        "Boosts: {} | Throttles: {} | Freezes: {} | Unfreezes: {}",
        format_number(m.boosts_applied),
        format_number(m.throttles_applied),
        format_number(m.freezes_applied),
        format_number(m.unfreezes_applied)
    ));

    // Use real energy savings from EnergyTracker if available, else rough estimate
    let fallback_wh = m.throttles_applied as f64 * 0.003
        + m.freezes_applied as f64 * 0.005
        + m.kills_applied as f64 * 0.01;
    let wh_saved = m.energy_savings_wh.unwrap_or(fallback_wh);
    let co2_g = m.energy_co2_avoided_g.unwrap_or(wh_saved * 0.075);
    lines.push(format!(
        "⚡ Energía ahorrada: {:.2} Wh | 🌱 CO₂ evitado: {:.2}g",
        wh_saved, co2_g
    ));

    lines
}

fn render_llm(status: &DaemonStatus) -> Vec<String> {
    let mut lines = vec![bold("🤖 LLM TEACHER")];

    match &status.llm {
        Some(llm) => {
            let state = if llm.enabled && llm.training_active {
                green("🟢 Activo")
            } else if llm.enabled {
                yellow("🟡 Standby")
            } else {
                dim("⬜ Desactivado")
            };

            let patterns = format!(
                "{} interactivos, {} ruido",
                llm.learned_policy.interactive_patterns, llm.learned_policy.noise_patterns
            );
            lines.push(format!("Estado: {} | Patrones: {}", state, patterns));

            let last_call = match llm.last_call_at {
                Some(t) => {
                    let secs = Utc::now().signed_duration_since(t).num_seconds().max(0);
                    if secs >= 3600 {
                        format!("hace {}h", secs / 3600)
                    } else if secs >= 60 {
                        format!("hace {}m", secs / 60)
                    } else {
                        format!("hace {}s", secs)
                    }
                }
                None => "nunca".into(),
            };

            let confidence = llm
                .last_suggestion_confidence
                .map(|c| format!("{:.2}", c))
                .unwrap_or_else(|| "-".into());

            lines.push(format!(
                "Última llamada: {} | Confianza: {} | Budget: {}/{}",
                last_call, confidence, llm.daily_budget_remaining, llm.daily_budget
            ));
        }
        None => {
            lines.push(dim("No configurado").to_string());
        }
    }

    lines
}

fn render_verdict(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold("📋 VEREDICTO GENERAL")];

    let (emoji, title, detail) = if m.thermal_state == "critical" {
        (
            "🔴",
            "Emergencia térmica",
            "Throttling activo — reduciendo carga.",
        )
    } else if m.memory_pressure > 0.85 {
        (
            "🔴",
            "Presión de memoria crítica",
            "Swap elevado — liberando recursos.",
        )
    } else if m.survival_mode_activations > 0 {
        (
            "🟠",
            "Modo supervivencia activado",
            "Recursos extremos — kills preventivos aplicados.",
        )
    } else if m.memory_pressure > 0.60 {
        (
            "🟡",
            "Presión de memoria moderada",
            "Monitoreando swap y memoria activamente.",
        )
    } else if m.last_blockers.len() > 5 {
        (
            "🟡",
            "Múltiples bloqueadores detectados",
            "Throttling selectivo en progreso.",
        )
    } else if m.kills_applied > 0 {
        (
            "🟡",
            "Procesos zombie eliminados",
            "Sistema estable post-limpieza.",
        )
    } else {
        (
            "🟢",
            "Sistema optimizado. Rendimiento estable.",
            "Sin problemas detectados.",
        )
    };

    // Inner verdict box (fills content width)
    let inner_w = CW - 2; // 64 dashes for border
    let text_w = inner_w - 2; // 62 cols for text inside │ ... │
    let verdict_line1 = format!("{} {}", emoji, title);
    let verdict_line2 = format!("   {}", detail);

    lines.push(format!("┌{}┐", "─".repeat(inner_w)));
    lines.push(format!("│ {} │", pad_right(&verdict_line1, text_w)));
    lines.push(format!("│ {} │", pad_right(&verdict_line2, text_w)));
    lines.push(format!("└{}┘", "─".repeat(inner_w)));

    lines
}

// ── Main entry point ──

pub fn render_dashboard(status: &DaemonStatus) -> String {
    let mut out = String::with_capacity(4096);

    let sections: Vec<Vec<String>> = vec![
        render_system(status),
        render_intelligence(status),
        render_foreground(status),
        render_actions(status),
        render_frozen(status),
        render_blockers(&status.last_blockers),
        render_reactor(status),
        render_energy(status),
        render_session(status),
        render_llm(status),
        render_verdict(status),
    ];

    // Header
    out.push_str(&box_top());
    out.push('\n');
    for line in render_header(status) {
        out.push_str(&box_line(&line));
        out.push('\n');
    }
    out.push_str(&box_div());
    out.push('\n');

    // Content sections separated by empty lines
    for (i, section) in sections.iter().enumerate() {
        out.push_str(&box_empty());
        out.push('\n');
        for line in section {
            out.push_str(&box_line(line));
            out.push('\n');
        }
        if i == sections.len() - 1 {
            out.push_str(&box_empty());
            out.push('\n');
        }
    }

    // Footer
    out.push_str(&box_bottom());
    out.push('\n');

    out
}
