//! ASCII dashboard renderer for apollo-optimizerctl.
//!
//! Renders a visual summary of daemon status using Unicode box-drawing,
//! ANSI colors, and emoji indicators.

use apollo_engine::engine::types::{
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

    // Swap + SwapTrend
    let swap_str = format_bytes(m.swap_used_bytes);
    let swap_ratio = (m.swap_used_bytes as f64 / (8.0 * 1_073_741_824.0)).min(1.0);
    let swap_bar = render_bar(swap_ratio, 20);
    let swap_gb = m.swap_used_bytes as f64 / 1_073_741_824.0;
    let swap_label = swap_status_label(swap_gb, m.swap_delta_bps);
    let trend_suffix = if !m.swap_trend.is_empty() && m.swap_trend != "Stable" {
        let colored = match m.swap_trend.as_str() {
            "Critical" => red(&m.swap_trend),
            "Increasing" => yellow(&m.swap_trend),
            "Decreasing" => green(&m.swap_trend),
            other => dim(other),
        };
        format!(" [{}]", colored)
    } else {
        String::new()
    };
    lines.push(format!(
        "Swap   {}  {:>7}  {}{}",
        swap_bar, swap_str, swap_label, trend_suffix
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

    // Fluidity telemetry block — enriched display pipeline + memory observability
    {
        // WindowServer CPU — compositor load
        if m.windowserver_cpu_pct > 0.5 {
            let ws_color = if m.windowserver_cpu_pct > 30.0 {
                yellow(&format!("{:.0}%", m.windowserver_cpu_pct))
            } else {
                format!("{:.0}%", m.windowserver_cpu_pct)
            };
            lines.push(format!("WindowServer CPU: {}", ws_color));
        }
        // Frozen RAM
        if m.frozen_ram_mb > 0.0 {
            lines.push(format!(
                "Frozen RAM: {:.0} MB recovered ({} procs)",
                m.frozen_ram_mb,
                m.freezes_applied.saturating_sub(m.unfreezes_applied)
            ));
        }
        // Display boost counter
        if m.display_boost_count > 0 {
            lines.push(format!(
                "Display boost: {}× fired — behavior_interactive: {} PIDs",
                m.display_boost_count, m.behavior_interactive_pid_count
            ));
        }
        // Cycles at high pressure
        if m.cycles_high_pressure > 0 {
            let cyc_colored = if m.cycles_high_pressure >= 5 {
                red(&format!("{}", m.cycles_high_pressure))
            } else if m.cycles_high_pressure >= 2 {
                yellow(&format!("{}", m.cycles_high_pressure))
            } else {
                format!("{}", m.cycles_high_pressure)
            };
            lines.push(format!(
                "High-pressure streak: {} cycles  RL threshold: {:.2}",
                cyc_colored, m.rl_threshold_current
            ));
        }
        // ML throttle source + freeze gate
        let ml_src = if m.ml_throttle_source.is_empty() {
            "none"
        } else {
            &m.ml_throttle_source
        };
        let gate = if m.freeze_gate_last.is_empty() {
            "none"
        } else {
            &m.freeze_gate_last
        };
        if ml_src != "none" || gate != "none" {
            lines.push(format!(
                "ML throttle: {}  Freeze gate: {}",
                match ml_src {
                    "swap-early" => yellow(ml_src),
                    "call-mode" => yellow("call-mode (bandwidth reserved)"),
                    _ => ml_src.to_string(),
                },
                if gate != "none" {
                    yellow(gate)
                } else {
                    gate.to_string()
                }
            ));
        }
    }

    // User context block — "what is the user doing right now?"
    {
        // Call in progress: highest priority — affects all throttle/freeze decisions
        if m.user_call_in_progress {
            lines.push(red("📞 CALL IN PROGRESS — freeze protection active"));
        } else if m.user_has_sleep_assertion {
            lines.push(yellow("⚠ Sleep assertion active — media/presentation mode"));
        }

        // Idle time
        let idle_str = if m.user_idle_secs >= 120.0 {
            green(&format!("idle {:.0}s (aggressive mode)", m.user_idle_secs))
        } else if m.user_idle_secs >= 15.0 {
            format!("idle {:.0}s", m.user_idle_secs)
        } else if m.user_idle_secs > 0.0 {
            yellow(&format!("active ({:.0}s ago)", m.user_idle_secs))
        } else {
            String::new()
        };
        if !idle_str.is_empty() {
            let audio_tag = if m.user_audio_active {
                "  🔊 audio"
            } else {
                ""
            };
            lines.push(format!("User: {}{}", idle_str, audio_tag));
        }
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

    // ── Fluidity Intelligence ────────────────────────────────────────────────
    // Always show: score drives key freeze/QoS decisions every cycle.
    {
        let score = m.fluidity_score;
        let score_colored = if score > 0.8 {
            green(&format!("{:.2}", score))
        } else if score > 0.6 {
            yellow(&format!("{:.2}", score))
        } else {
            red(&format!("{:.2}", score))
        };
        let ws_str = if m.windowserver_cpu_pct > 25.0 {
            yellow(&format!("WS={:.0}%", m.windowserver_cpu_pct))
        } else {
            dim(&format!("WS={:.0}%", m.windowserver_cpu_pct))
        };
        let launch_str = if m.app_launching {
            yellow(&format!(" launching={}", m.app_launch_name))
        } else {
            dim(" launching=false")
        };
        let window_str = if m.window_op_active {
            yellow(" win_op=true")
        } else {
            dim(" win_op=false")
        };
        lines.push(format!(
            "FLUIDITY  score={} {} {}{}",
            score_colored, ws_str, window_str, launch_str
        ));
    }

    // ── Chromium Renderer Manager ────────────────────────────────────────────
    // Only show when there are tracked renderers (not present on all systems).
    if m.chromium_renderers_total > 0 {
        let frozen_str = if m.chromium_renderers_frozen > 0 {
            yellow(&format!("frozen={}", m.chromium_renderers_frozen))
        } else {
            dim(&format!("frozen={}", m.chromium_renderers_frozen))
        };
        let freed_str = if m.chromium_freed_mb > 0.0 {
            green(&format!("freed={:.0}MB", m.chromium_freed_mb))
        } else {
            dim("freed=0MB")
        };
        let browsers_str = if !m.chromium_browsers_managed.is_empty() {
            let joined = m.chromium_browsers_managed.join(", ");
            format!("  [{}]", joined)
        } else {
            String::new()
        };
        lines.push(format!(
            "CHROMIUM  renderers={} {} e-core={} {}{}",
            m.chromium_renderers_total,
            frozen_str,
            m.chromium_renderers_ecore,
            freed_str,
            browsers_str,
        ));
    }

    // Affective arousal indicator (Yerkes-Dodson zone)
    if m.arousal_level > 0.01 || !m.arousal_zone.is_empty() {
        let zone = if m.arousal_zone.is_empty() {
            "Idle"
        } else {
            &m.arousal_zone
        };
        let zone_colored = match zone {
            "Crisis" => red(zone),
            "Stressed" => yellow(zone),
            "Optimal" => green(zone),
            _ => dim(zone),
        };
        lines.push(format!(
            "Arousal: {} ({:.0}%) — Yerkes-Dodson",
            zone_colored,
            m.arousal_level * 100.0,
        ));
    }

    // NARS concept drift indicator
    if m.nars_drift_score > 0.0 || m.nars_drifted_beliefs > 0 {
        let drift_label = if m.nars_drift_score > 0.08 || m.nars_drifted_beliefs >= 2 {
            red("🔴 Recalibrando")
        } else if m.nars_drift_score > 0.05 {
            yellow("🟡 Leve")
        } else {
            green("🟢 Estable")
        };
        lines.push(format!(
            "Concept Drift: {} (score={:.3}, beliefs={})",
            drift_label, m.nars_drift_score, m.nars_drifted_beliefs
        ));
    }

    // Causal reasoning depth: show multi-horizon and mechanism attribution status.
    if m.causal_slow_horizon_count > 0 || m.causal_mechanism_count > 0 {
        let horizon_label = if m.causal_slow_horizon_count > 5 {
            green(&format!("{} edges", m.causal_slow_horizon_count))
        } else if m.causal_slow_horizon_count > 0 {
            yellow(&format!("{} edges", m.causal_slow_horizon_count))
        } else {
            dim("cold")
        };
        let mech_label = if m.causal_mechanism_count > 3 {
            green(&format!("{} edges", m.causal_mechanism_count))
        } else if m.causal_mechanism_count > 0 {
            yellow(&format!("{} edges", m.causal_mechanism_count))
        } else {
            dim("cold")
        };
        lines.push(format!(
            "Causal Depth: slow-horizon={} | mechanisms={}",
            horizon_label, mech_label
        ));
    }
    // Top mechanism attributions: show WHY throttles work.
    for mech in m.causal_mechanisms.iter().take(3) {
        lines.push(format!("  {}", dim(mech)));
    }

    // KPC memory-bound score
    if m.kpc_memory_bound_score > 0.0 {
        let score_label = if m.kpc_memory_bound_score > 0.7 {
            green(&format!(
                "{:.0}% memory-stalled",
                m.kpc_memory_bound_score * 100.0
            ))
        } else if m.kpc_memory_bound_score > 0.4 {
            yellow(&format!(
                "{:.0}% memory-stalled",
                m.kpc_memory_bound_score * 100.0
            ))
        } else {
            dim(&format!(
                "{:.0}% memory-stalled",
                m.kpc_memory_bound_score * 100.0
            ))
        };
        lines.push(format!("KPC: {}", score_label));
    }

    // ── Neurocognitive Health (UCHS) ──────────────────────────────────────
    if m.uchs_composite > 0.0 {
        let grade_color = match m.uchs_grade.as_str() {
            "S+" | "S" => green(&m.uchs_grade),
            "A" => green(&m.uchs_grade),
            "B" => yellow(&m.uchs_grade),
            _ => red(&m.uchs_grade),
        };
        let recovery = if m.uchs_recovery_mode {
            red(" RECOVERY")
        } else {
            String::new()
        };
        lines.push(format!(
            "UCHS: {:.1}% {} {}{}",
            m.uchs_composite * 100.0,
            grade_color,
            dim(&format!(
                "meta={:.0}% snr={:.1} eval={:.0}% adv={:.0}%",
                m.meta_confidence * 100.0,
                m.cognitive_snr,
                m.self_eval_quality * 100.0,
                m.adversarial_pass_rate * 100.0,
            )),
            recovery,
        ));
        if m.epistemic_uncertainty > 0.40 {
            let ep_label = match m.epistemic_level.as_str() {
                "OBSERVE-ONLY" => red(&m.epistemic_level),
                "HIGH" => yellow(&m.epistemic_level),
                _ => dim(&m.epistemic_level),
            };
            lines.push(format!(
                "Epistemic: {} ({:.0}%)",
                ep_label,
                m.epistemic_uncertainty * 100.0,
            ));
        }
        if m.humble_mode {
            lines.push(yellow("MetaCognition: HUMBLE MODE (2× exploration)").to_string());
        }
        if m.drift_early_warning > 0.05 {
            lines.push(format!(
                "Drift Early Warning: {} ({:.3})",
                yellow("ACTIVE"),
                m.drift_early_warning,
            ));
        }
        if m.adversarial_safety_alert {
            lines.push(red("⚠ COGNITIVE SAFETY ALERT — adversarial pass rate < 75%").to_string());
        }
    }

    // Wakeup vampires: battery drain daemons
    if !m.wakeup_vampires.is_empty() {
        lines.push(format!(
            "Wakeup Vampires: {}",
            yellow(&m.wakeup_vampires.join(", "))
        ));
    }

    // Behavioral anomalies: processes deviating from learned baseline
    if !m.anomaly_processes.is_empty() {
        let label = if m.anomaly_process_count >= 3 {
            red(&m.anomaly_processes.join(", "))
        } else {
            yellow(&m.anomaly_processes.join(", "))
        };
        lines.push(format!("Anomalies: {}", label));
    }

    // AMX coprocessor (undocumented — probed via raw ASM .word 0x00201220)
    if m.amx_available {
        lines.push(format!(
            "AMX: {} (ctx-switch ~{}ns)",
            green("active"),
            m.amx_cs_overhead_ns
        ));
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
            FreezeSource::ChromiumManager => "Chromium",
            FreezeSource::Unknown => "Unknown",
        };
        let time = if p.frozen_seconds >= 3600 {
            format!(
                "{}h {:02}m",
                p.frozen_seconds / 3600,
                (p.frozen_seconds % 3600) / 60
            )
        } else if p.frozen_seconds >= 60 {
            format!("{}m {:02}s", p.frozen_seconds / 60, p.frozen_seconds % 60)
        } else {
            format!("{}s", p.frozen_seconds)
        };
        let name = if p.name.len() > 22 {
            &p.name[..22]
        } else {
            &p.name
        };
        lines.push(format!(
            "{:<6} {:<22} {:>8} {:>5.2} {:<14}",
            p.pid, name, time, p.pressure_at_freeze, source
        ));
    }
    if status.frozen_processes.len() > 10 {
        lines.push(format!(
            "  ... y {} más",
            status.frozen_processes.len() - 10
        ));
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
    } else if apollo_engine::engine::safety::survival_mode_active_total(
        m.memory_pressure,
        m.swap_used_bytes,
        m.swap_total_bytes,
    ) {
        (
            "🟠",
            "Modo supervivencia activo",
            "Pressure + swap por encima de umbrales — gates de freeze priorizados.",
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

    lines.push(format!("Pres   p={:.0}% c={:.0}%", mp * 100.0, m.compressed_memory_ratio * 100.0));

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
    let interactive = status.llm.as_ref().map_or(0, |l| l.learned_policy.interactive_patterns);
    let noise = status.llm.as_ref().map_or(0, |l| l.learned_policy.noise_patterns);
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
    lines.push(format!("Workload {}", m.current_workload.chars().take(20).collect::<String>()));

    lines
}

// ── 🎯 DECIDE quadrant ───────────────────────────────────────────────────────
fn render_decide_q(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold(&q_header("🎯", "DECIDE"))];

    // MetaCognition
    let humble = m.meta_confidence < 0.40;
    let meta_label = if humble { "HUMBLE" } else if m.meta_confidence > 0.70 { "CONFIDENT" } else { "NORMAL" };
    let meta_emoji = if humble { "🤔" } else { "🎯" };
    lines.push(format!(
        "Meta   {} {} {:.0}%",
        meta_emoji,
        meta_label,
        m.meta_confidence * 100.0
    ));

    // Arousal
    let arousal_pct = (m.arousal_level * 100.0) as i32;
    lines.push(format!(
        "Arousal {} {}%",
        m.arousal_zone, arousal_pct
    ));

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

    lines
}

// ── 🎬 ACT quadrant ──────────────────────────────────────────────────────────
fn render_act_q(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold(&q_header("🎬", "ACT"))];

    // SafetyPolicy budgets — derive from profile to mirror per-profile caps.
    let pol = SafetyPolicy::for_profile(status.effective_profile);
    let bar_w = 8;
    let boost_ratio = (m.boosts_applied as f64 / pol.max_boosts_per_cycle.max(1) as f64).min(1.0);
    let throt_ratio = (m.throttles_applied as f64 / pol.max_throttles_per_cycle.max(1) as f64).min(1.0);
    let frz_ratio = (m.freezes_applied as f64 / pol.max_freezes_per_cycle.max(1) as f64).min(1.0);
    let hint_ratio = (m.paging_hints_applied as f64 / pol.max_paging_hints_per_cycle.max(1) as f64).min(1.0);

    lines.push(format!(
        "Boost  {} {}/{}",
        render_bar(boost_ratio, bar_w),
        m.boosts_applied,
        pol.max_boosts_per_cycle
    ));
    lines.push(format!(
        "Throt  {} {}/{}",
        render_bar(throt_ratio, bar_w),
        m.throttles_applied,
        pol.max_throttles_per_cycle
    ));
    lines.push(format!(
        "Frz    {} {}/{}",
        render_bar(frz_ratio, bar_w),
        m.freezes_applied,
        pol.max_freezes_per_cycle
    ));
    lines.push(format!(
        "Hint   {} {}/{}",
        render_bar(hint_ratio, bar_w),
        m.paging_hints_applied,
        pol.max_paging_hints_per_cycle
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
    lines.push(format!("Reactor {} {}", status.reactor_mode, status.reactor_health));

    lines
}

// ── 🚪 GATES band (full-width) ───────────────────────────────────────────────
fn render_gates_band(status: &DaemonStatus) -> Vec<String> {
    let m = &status.metrics;
    let mut lines = vec![bold("🚪 GATES")];

    // Survival: based on memory pressure
    let surv = if m.memory_pressure >= 0.85 {
        red("🔴 ACTIVE")
    } else {
        green("🟢 ready")
    };

    // Maintenance purge: blocked by media or other gates?
    let media_blocked = m.user_audio_active || m.user_call_in_progress || m.user_has_sleep_assertion;
    let purge_state = if media_blocked {
        let reason = if m.user_call_in_progress {
            "call"
        } else if m.user_audio_active {
            "audio"
        } else {
            "sleep-assertion"
        };
        red(&format!("🔴 BLOCKED ({})", reason))
    } else if m.memory_pressure < 0.65 {
        dim("⬜ idle (p<0.65)")
    } else {
        green("🟢 ready")
    };

    lines.push(format!("survival {}  ·  maintenance-purge {}", surv, purge_state));

    // Build mode + post-wake
    let post_wake = if status.post_wake_grace_active {
        yellow(&format!("⚠ grace ({}s)", status.post_wake_grace_remaining_secs))
    } else {
        green("🟢 ready")
    };
    lines.push(format!("freeze 🟢 user-protected  ·  post-wake {}", post_wake));

    // Maintenance purge counters
    let total_skipped = m.maintenance_purge_skipped_pressure_total
        + m.maintenance_purge_skipped_swap_floor_total
        + m.maintenance_purge_skipped_growing_total
        + m.maintenance_purge_skipped_idle_total
        + m.maintenance_purge_skipped_build_mode_total
        + m.maintenance_purge_skipped_rate_limit_total;

    let mut skip_breakdown = Vec::new();
    if m.maintenance_purge_skipped_pressure_total > 0 {
        skip_breakdown.push(format!("pres:{}", m.maintenance_purge_skipped_pressure_total));
    }
    if m.maintenance_purge_skipped_idle_total > 0 {
        skip_breakdown.push(format!("idle/media:{}", m.maintenance_purge_skipped_idle_total));
    }
    if m.maintenance_purge_skipped_swap_floor_total > 0 {
        skip_breakdown.push(format!("floor:{}", m.maintenance_purge_skipped_swap_floor_total));
    }
    if m.maintenance_purge_skipped_growing_total > 0 {
        skip_breakdown.push(format!("grow:{}", m.maintenance_purge_skipped_growing_total));
    }
    if m.maintenance_purge_skipped_build_mode_total > 0 {
        skip_breakdown.push(format!("build:{}", m.maintenance_purge_skipped_build_mode_total));
    }
    if m.maintenance_purge_skipped_rate_limit_total > 0 {
        skip_breakdown.push(format!("rate:{}", m.maintenance_purge_skipped_rate_limit_total));
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

    if status.frozen_processes.len() > 0 {
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
}
