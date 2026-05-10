//! Overflow Guard — aprendizaje adaptativo para prevenir OOM/memory overflows.
//!
//! ## Problema
//! En un MacBook Air M1 de 8 GB, compilar Rust + tener Brave + Claude + Antigravity
//! abiertos puede llevar al sistema al límite en minutos. Apollo actuaba demasiado
//! tarde (threshold 90%) porque no sabía que el sistema ya había colapsado antes.
//!
//! ## Solución
//! 1. **Detección**: cuando ocurre un overflow (presión crítica / kqueue Critical),
//!    registrar qué apps corrían y cuánta presión había.
//! 2. **Thresholds adaptativos**: tras un overflow, Apollo baja sus umbrales de
//!    intervención (actúa al 75% en vez del 85%) y los sube lentamente si no hay
//!    nuevos eventos (1 pp por hora de calma).
//! 3. **Build mode**: si hay procesos de compilación corriendo, Apollo anticipa
//!    el pico de RAM y actúa desde el 65% de presión.
//! 4. **Persistencia**: el historial sobrevive reboots en `/var/lib/apollo/overflow_history.json`.
//! 5. **RL tuning (Phase 4)**: Q-learning agent adjusts thresholds online based on
//!    observed stability vs. overflow events.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use crate::engine::rl_threshold::{RlState, RlThresholdAgent, RL_ABSOLUTE_FLOOR};
use crate::engine::workload_classifier::WorkloadMode;

// ── Thresholds ────────────────────────────────────────────────────────────────

/// Umbrales de presión de memoria que usa el sistema de decisión.
#[derive(Debug, Clone, Copy)]
pub struct OverflowThresholds {
    /// Por encima de este valor → BackgroundPressure (throttle moderado).
    pub bg_pressure: f64,
    /// Por encima de este valor → ThermalConstrained + extreme freeze eligible.
    pub critical_pressure: f64,
    /// Por encima de este valor → extreme freeze activo (la vieja 0.90 fija).
    pub extreme_pressure: f64,
    /// Workload mode: determines threshold bonus (Phase 3).
    pub workload_mode: WorkloadMode,
}

impl Default for OverflowThresholds {
    fn default() -> Self {
        Self {
            bg_pressure: 0.78,
            critical_pressure: 0.88,
            extreme_pressure: 0.90,
            workload_mode: WorkloadMode::Idle,
        }
    }
}

// ── Historial persistido ──────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct OverflowEvent {
    /// Unix timestamp (segundos) del evento.
    pub timestamp_secs: u64,
    /// Presión de memoria en el momento del overflow.
    pub memory_pressure: f64,
    /// Swap delta en bytes/s.
    pub swap_delta_bps: f64,
    /// Top N nombres de proceso más pesados en ese momento.
    pub heavy_apps: Vec<String>,
    /// Cómo fue detectado el overflow.
    pub cause: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct OverflowHistory {
    /// Últimos eventos (máximo 20).
    pub events: Vec<OverflowEvent>,
    /// Ajuste acumulado al threshold (negativo = más conservador).
    /// Rango: [-0.20, 0.0]. Cada overflow resta 0.05; cada hora sin overflow suma 0.01.
    pub threshold_offset: f64,
    /// Cuántos overflows en total (contador de vida).
    pub total_overflows: u64,
}

// ── Guard ─────────────────────────────────────────────────────────────────────

/// Detector y aprendiz de overflows. Cargado al arrancar el daemon, persistido
/// tras cada evento relevante.
pub struct OverflowGuard {
    pub history: OverflowHistory,
    path: PathBuf,
    last_decay_check: Instant,
    /// Evitar registrar múltiples eventos del mismo overflow en rápida sucesión.
    last_event_at: Option<Instant>,
    /// Q-learning RL agent for adaptive threshold tuning (Phase 4).
    pub rl_agent: Option<RlThresholdAgent>,
    /// Device-level offset: shifts base thresholds down on memory-constrained devices.
    /// -0.05 for ≤8 GB, 0.0 for ≤16 GB, +0.05 for >16 GB.
    device_offset: f64,
}

/// Herramientas de compilación que causan picos de RAM.
/// "stable" es el wrapper del Rust toolchain que rustup antepone a rustc/cargo —
/// aparece como proceso padre durante compilación y consume RAM proporcional.
pub const BUILD_TOOLS: &[&str] = &[
    "rustc", "cargo", "swift", "clang", "make", "gcc", "ld", "link", "stable",
];

/// Match a process name against build tool patterns.
///
/// Short patterns (≤4 chars) use exact matching to prevent false positives:
/// - `"ld"` should match the linker (`ld`) but NOT `triald`, `linkd`,
///   `IOUserBluetoothSerialDriver`, etc.
/// - `"link"` should match the Windows linker stub but NOT `linkd`,
///   `contentlinkingd`, etc.
///
/// Longer patterns use `contains()` as before (e.g. `"rustc"` in `"rustc-1.77"`).
pub fn is_build_tool_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    BUILD_TOOLS.iter().any(|&t| {
        if t.len() <= 4 {
            lower == t
        } else {
            lower.contains(t)
        }
    })
}

/// Query total physical RAM in GB using `sysctl -n hw.memsize`.
/// Falls back to 8.0 GB if the query fails (conservative default for M1 Air).
fn query_ram_gb() -> f64 {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|b| b as f64 / 1e9)
        .unwrap_or(8.0)
}

/// Compute a device-level threshold offset based on total RAM.
/// Memory-constrained devices (≤8 GB) get -0.03 (act slightly sooner).
/// Previous -0.05 was too aggressive: combined with other offsets it
/// pushed thresholds low enough to trigger false positives on M1 8GB.
/// Mid-range (≤16 GB) get 0.0 (no change).
/// Large-memory devices (>16 GB) get +0.05 (more headroom).
pub fn device_threshold_offset(ram_gb: f64) -> f64 {
    if ram_gb <= 8.0 {
        -0.03
    } else if ram_gb <= 16.0 {
        0.0
    } else {
        0.05
    }
}

impl OverflowGuard {
    /// Carga el historial desde disco, o crea uno vacío si no existe.
    /// If `rl_path` is provided, also loads the RL threshold agent from that path.
    pub fn load_or_default(path: &Path, rl_path: Option<&Path>) -> Self {
        let history = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let rl_agent = rl_path.map(RlThresholdAgent::load_or_default);
        let device_offset = device_threshold_offset(query_ram_gb());
        Self {
            history,
            path: path.to_path_buf(),
            last_decay_check: Instant::now(),
            last_event_at: None,
            rl_agent,
            device_offset,
        }
    }

    /// Registra un evento de overflow. Llama esto cuando:
    /// - kqueue dispara VmPressureLevel::Critical o SuddenTerminate
    /// - survival_mode se activa
    /// - presión RAM > 0.92 sostenida
    ///
    /// `compressor_pressure` is the raw compressor ratio (0.0-1.0) for RL state.
    pub fn record_event(
        &mut self,
        memory_pressure: f64,
        swap_delta_bps: f64,
        heavy_apps: &[String],
        cause: &str,
        compressor_pressure: f64,
        pressure_bands: &[f64; 3],
        compressor_bands: &[f64; 2],
    ) {
        // B3 fix (round-3): reduce dedup window 60s → 8s so bursts of overflow
        // events (e.g. rapid compressor thrashing during a memory storm) each
        // get registered. Previously, 60s dedup collapsed a burst to a single
        // event, leaving the RL agent + dynamic offset with no signal.
        if let Some(last) = self.last_event_at {
            if last.elapsed() < Duration::from_secs(8) {
                return;
            }
        }
        self.last_event_at = Some(Instant::now());

        let timestamp_secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let event = OverflowEvent {
            timestamp_secs,
            memory_pressure,
            swap_delta_bps,
            heavy_apps: heavy_apps.iter().take(10).cloned().collect(),
            cause: cause.to_string(),
        };

        self.history.events.push(event);
        if self.history.events.len() > 20 {
            self.history.events.remove(0);
        }
        self.history.total_overflows += 1;

        // Bajar threshold: cada overflow resta 5%, piso en -20%.
        self.history.threshold_offset = (self.history.threshold_offset - 0.05).max(-0.20);

        eprintln!(
            "overflow-guard: evento #{} — presión={:.0}% swap={:.0}KB/s offset={:+.0}%pp apps=[{}]",
            self.history.total_overflows,
            memory_pressure * 100.0,
            swap_delta_bps / 1024.0,
            self.history.threshold_offset * 100.0,
            heavy_apps
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );

        // RL agent: tick with overflow=true using adaptive bands.
        let overflows_1h = self.recent_overflow_count_hours(1);
        // BUG-D fix: compute dynamic offset BEFORE the mutable borrow of rl_agent
        // to satisfy the borrow checker (compute_dynamic_offset takes &self).
        let dynamic_off_for_rl = self.compute_dynamic_offset();
        let device_offset_snap = self.device_offset;
        if let Some(rl) = &mut self.rl_agent {
            let rl_state = RlState::from_metrics_with_bands(
                memory_pressure,
                compressor_pressure,
                overflows_1h,
                pressure_bands,
                compressor_bands,
            );
            // Compound = dynamic_offset + rl_adjustment + device_offset.
            // When at the hard floor (-0.15), Lower5pp has no real effect.
            let compound = dynamic_off_for_rl + rl.current_adjustment + device_offset_snap;
            rl.compound_at_floor = compound <= -0.15 + 0.005;
            rl.tick(rl_state, true);
            rl.persist();
        }

        self.persist();
    }

    /// Cómputo dinámico del offset de threshold, ponderado exponencialmente
    /// por la antigüedad de cada evento (half-life = 8 horas).
    ///
    /// Cada evento contribuye `-0.05 * 2^(-age_h / 8)` al offset.
    /// Efectos: un evento de hace 8h aporta -0.025; de hace 24h, -0.006;
    /// de hace 48h, < -0.001. Así el offset se recupera naturalmente
    /// sin necesitar un tick de decay — la calma ya es su propia recompensa.
    pub fn compute_dynamic_offset(&self) -> f64 {
        let now_secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let raw: f64 = self
            .history
            .events
            .iter()
            .map(|e| {
                let age_h = now_secs.saturating_sub(e.timestamp_secs) as f64 / 3600.0;
                -0.05 * (2.0_f64).powf(-age_h / 8.0)
            })
            .sum();

        raw.max(-0.20)
    }

    /// Mantiene `history.threshold_offset` sincronizado con el offset dinámico.
    /// Llamar una vez por ciclo para que las métricas sean precisas.
    ///
    /// `memory_pressure` and `compressor_pressure` feed the RL agent state
    /// so it can learn from stable (no-overflow) ticks.
    pub fn tick_decay(
        &mut self,
        memory_pressure: f64,
        compressor_pressure: f64,
        pressure_bands: &[f64; 3],
        compressor_bands: &[f64; 2],
    ) {
        // RL agent: tick every cycle (stable tick, no overflow).
        // Use adaptive bands from LearnableParams so Q-table state
        // discretization stays aligned with auto-tuned boundaries.
        let overflows_1h = self.recent_overflow_count_hours(1);
        // BUG-D fix: compute dynamic offset BEFORE the mutable borrow of rl_agent.
        let dynamic_off_for_rl = self.compute_dynamic_offset();
        let device_offset_snap = self.device_offset;
        if let Some(rl) = &mut self.rl_agent {
            let rl_state = RlState::from_metrics_with_bands(
                memory_pressure,
                compressor_pressure,
                overflows_1h,
                pressure_bands,
                compressor_bands,
            );
            // Inform the RL agent whether the compound offset is already at the
            // hard floor (-0.15) so it can suppress Lower5pp rewards that have
            // no real effect (overflow_guard clamps them away anyway).
            let compound = dynamic_off_for_rl + rl.current_adjustment + device_offset_snap;
            rl.compound_at_floor = compound <= -0.15 + 0.005;
            rl.tick(rl_state, false);
            if rl.total_ticks() % 50 == 0 {
                rl.persist();
            }
        }

        if self.last_decay_check.elapsed() < Duration::from_secs(600) {
            return;
        }
        self.last_decay_check = Instant::now();

        let dynamic = self.compute_dynamic_offset();
        let prev = self.history.threshold_offset;

        if (dynamic - prev).abs() > 0.005 {
            self.history.threshold_offset = dynamic;
            eprintln!(
                "overflow-guard: offset dinámico {:.0}%pp → {:.0}%pp ({} eventos)",
                prev * 100.0,
                dynamic * 100.0,
                self.history.events.len(),
            );
            self.persist();
        }
    }

    /// Calcula los thresholds adaptativos para este ciclo.
    ///
    /// `workload_mode` comes from the nearest-centroid classifier (Phase 3).
    /// Each mode applies a different threshold bonus via `WorkloadMode::threshold_bonus()`.
    pub fn thresholds(&self, workload_mode: WorkloadMode) -> OverflowThresholds {
        let total_offset = self.applied_offset(workload_mode);

        OverflowThresholds {
            bg_pressure: (0.78 + total_offset).max(RL_ABSOLUTE_FLOOR),
            critical_pressure: (0.88 + total_offset).max(0.73),
            extreme_pressure: (0.90 + total_offset).max(0.78),
            workload_mode,
        }
    }

    /// Same as [`thresholds`] but adds a PID derivative term so thresholds
    /// tighten when pressure is rising fast, not only when levels are high.
    ///
    /// `d_offset = -(velocity * kd).clamp(0, 0.05)`; kd=0.30 so 0.17/tick
    /// velocity produces the maximum 0.05 offset (≈5 pp tighter threshold).
    ///
    /// [Hellerstein 2004] §9 — PID operating-regime control.
    pub fn thresholds_with_d_term(
        &self,
        workload_mode: WorkloadMode,
        pressure_velocity: f64,
    ) -> OverflowThresholds {
        const KD: f64 = 0.30;
        const MAX_D_OFFSET: f64 = 0.05;
        let d_offset = -(pressure_velocity.max(0.0) * KD).min(MAX_D_OFFSET);
        let base = self.thresholds(workload_mode);
        OverflowThresholds {
            bg_pressure: (base.bg_pressure + d_offset).max(RL_ABSOLUTE_FLOOR),
            critical_pressure: (base.critical_pressure + d_offset).max(0.68),
            extreme_pressure: (base.extreme_pressure + d_offset).max(0.73),
            workload_mode: base.workload_mode,
        }
    }

    /// The compound offset actually applied to thresholds this cycle.
    ///
    /// B6 fix (round-3): single source of truth for reporting. Metric
    /// `overflow_threshold_offset_pp` previously read only the dynamic
    /// component (`compute_dynamic_offset`) and so could show "recovered"
    /// while the live threshold was still pinned at the floor due to the
    /// RL + device + workload contributions.  Callers that report to the
    /// operator must use this value.
    pub fn applied_offset(&self, workload_mode: WorkloadMode) -> f64 {
        let off = self.compute_dynamic_offset();
        let workload_bonus = workload_mode.threshold_bonus();
        let rl_adj = self
            .rl_agent
            .as_ref()
            .map(|r| r.current_adjustment)
            .unwrap_or(0.0);
        (off + workload_bonus + rl_adj + self.device_offset).max(-0.15)
    }

    /// ¿Hay herramientas de compilación corriendo activamente?
    pub fn detect_build_mode(proc_names: &[&str]) -> bool {
        let count = proc_names.iter().filter(|n| is_build_tool_name(n)).count();
        count >= 2
    }

    /// Número de overflows en los últimos N días.
    pub fn recent_overflow_count(&self, days: u64) -> usize {
        let cutoff = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(days * 86400);
        self.history
            .events
            .iter()
            .filter(|e| e.timestamp_secs >= cutoff)
            .count()
    }

    /// Número de overflows en las últimas N horas.
    pub fn recent_overflow_count_hours(&self, hours: u64) -> usize {
        let cutoff = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(hours * 3600);
        self.history
            .events
            .iter()
            .filter(|e| e.timestamp_secs >= cutoff)
            .count()
    }

    /// ¿Se parece la carga actual a una que causó overflow antes?
    pub fn resembles_past_overflow(&self, proc_names: &[&str]) -> bool {
        for event in &self.history.events {
            let matches = event
                .heavy_apps
                .iter()
                .filter(|app| proc_names.iter().any(|n| n.contains(app.as_str())))
                .count();
            if matches >= 3 && !event.heavy_apps.is_empty() && matches * 2 >= event.heavy_apps.len()
            {
                return true;
            }
        }
        false
    }

    fn persist(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.history) {
            let _ = std::fs::write(&self.path, json);
        }
    }

    /// Export the overflow history for unified persistence (LearnedState).
    /// The returned value is a clone of the internal history struct.
    pub fn export_history(&self) -> OverflowHistory {
        self.history.clone()
    }

    /// Import overflow history from unified persistence (LearnedState).
    /// Replaces the in-memory history; the path-based fallback file is not touched.
    /// Called at startup before the guard begins observing new events.
    pub fn import_history(&mut self, history: OverflowHistory) {
        self.history = history;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_guard_with_events(event_ages_hours: &[f64]) -> OverflowGuard {
        let now_secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut history = OverflowHistory::default();
        for &age_h in event_ages_hours {
            let ts = now_secs.saturating_sub((age_h * 3600.0) as u64);
            history.events.push(OverflowEvent {
                timestamp_secs: ts,
                memory_pressure: 0.90,
                swap_delta_bps: 0.0,
                heavy_apps: vec![],
                cause: "test".to_string(),
            });
        }
        history.threshold_offset = -0.20;
        OverflowGuard {
            history,
            path: PathBuf::from("/tmp/test_overflow.json"),
            last_decay_check: Instant::now(),
            last_event_at: None,
            rl_agent: None,
            device_offset: 0.0,
        }
    }

    fn make_guard_with_device_offset(offset: f64) -> OverflowGuard {
        OverflowGuard {
            history: OverflowHistory::default(),
            path: PathBuf::from("/tmp/test_overflow_device.json"),
            last_decay_check: Instant::now(),
            last_event_at: None,
            rl_agent: None,
            device_offset: offset,
        }
    }

    #[test]
    fn dynamic_offset_zero_with_no_events() {
        let guard = make_guard_with_events(&[]);
        assert_eq!(guard.compute_dynamic_offset(), 0.0);
    }

    #[test]
    fn dynamic_offset_decays_with_age() {
        let guard_fresh = make_guard_with_events(&[0.0]);
        let guard_old = make_guard_with_events(&[8.0]);

        let offset_fresh = guard_fresh.compute_dynamic_offset();
        let offset_old = guard_old.compute_dynamic_offset();

        assert!(
            offset_fresh < -0.04,
            "evento reciente debe tener offset alto: {}",
            offset_fresh
        );
        assert!(
            offset_old > -0.03,
            "evento de 8h debe tener offset reducido: {}",
            offset_old
        );
        assert!(
            offset_old > offset_fresh,
            "evento más viejo debe tener menor impacto"
        );
    }

    #[test]
    fn dynamic_offset_recovers_after_24h_calm() {
        let ages: Vec<f64> = (24..=168).step_by(8).map(|h| h as f64).collect();
        let guard = make_guard_with_events(&ages);
        let offset = guard.compute_dynamic_offset();
        assert!(
            offset > -0.10,
            "20 eventos de 24h+ no deben mantener offset al piso: {}",
            offset
        );
    }

    #[test]
    fn dynamic_offset_capped_at_floor() {
        let guard = make_guard_with_events(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        let offset = guard.compute_dynamic_offset();
        assert!(
            offset >= -0.20,
            "offset nunca debe bajar del piso -20pp: {}",
            offset
        );
    }

    #[test]
    fn thresholds_recover_when_events_age() {
        let ages: Vec<f64> = (24..=168).step_by(8).map(|h| h as f64).collect();
        let guard = make_guard_with_events(&ages);
        let t = guard.thresholds(WorkloadMode::Idle);
        assert!(
            t.bg_pressure > 0.70,
            "bg_pressure debe haberse recuperado: {}",
            t.bg_pressure
        );
        // Idle mode adds +0.03 bonus, so ceiling is 0.78 + 0.03 = 0.81.
        assert!(
            t.bg_pressure <= 0.81,
            "no puede superar el default + idle bonus: {}",
            t.bg_pressure
        );
    }

    #[test]
    fn device_offset_8gb_lowers_thresholds() {
        let guard_8gb = make_guard_with_device_offset(-0.03);
        let guard_base = make_guard_with_device_offset(0.0);
        let t_8gb = guard_8gb.thresholds(WorkloadMode::Idle);
        let t_base = guard_base.thresholds(WorkloadMode::Idle);
        assert!(
            t_8gb.bg_pressure < t_base.bg_pressure,
            "8 GB device should have lower bg_pressure threshold: {} vs {}",
            t_8gb.bg_pressure,
            t_base.bg_pressure
        );
        assert!(
            (t_base.bg_pressure - t_8gb.bg_pressure - 0.03).abs() < 1e-9,
            "offset should be exactly -0.03pp: diff={}",
            t_base.bg_pressure - t_8gb.bg_pressure
        );
    }

    #[test]
    fn device_offset_32gb_raises_thresholds() {
        let guard_32gb = make_guard_with_device_offset(0.05);
        let guard_base = make_guard_with_device_offset(0.0);
        let t_32gb = guard_32gb.thresholds(WorkloadMode::Idle);
        let t_base = guard_base.thresholds(WorkloadMode::Idle);
        assert!(
            t_32gb.bg_pressure > t_base.bg_pressure,
            "32 GB device should have higher bg_pressure threshold: {} vs {}",
            t_32gb.bg_pressure,
            t_base.bg_pressure
        );
        assert!(
            (t_32gb.bg_pressure - t_base.bg_pressure - 0.05).abs() < 1e-9,
            "offset should be exactly +0.05pp"
        );
    }

    #[test]
    fn device_threshold_offset_buckets() {
        assert!(
            (device_threshold_offset(8.0) - (-0.03)).abs() < 1e-9,
            "8 GB exactly -> -0.03"
        );
        assert!(
            (device_threshold_offset(7.9) - (-0.03)).abs() < 1e-9,
            "< 8 GB -> -0.03"
        );
        assert!(
            (device_threshold_offset(16.0) - 0.0).abs() < 1e-9,
            "16 GB exactly -> 0.0"
        );
        assert!(
            (device_threshold_offset(12.0) - 0.0).abs() < 1e-9,
            "12 GB -> 0.0"
        );
        assert!(
            (device_threshold_offset(32.0) - 0.05).abs() < 1e-9,
            "32 GB -> +0.05"
        );
        assert!(
            (device_threshold_offset(16.1) - 0.05).abs() < 1e-9,
            "> 16 GB -> +0.05"
        );
    }

    #[test]
    fn d_term_zero_velocity_matches_base_thresholds() {
        let guard = make_guard_with_device_offset(0.0);
        let base = guard.thresholds(WorkloadMode::Idle);
        let d = guard.thresholds_with_d_term(WorkloadMode::Idle, 0.0);
        assert!(
            (base.bg_pressure - d.bg_pressure).abs() < 1e-9,
            "zero velocity: D-term should add no offset"
        );
    }

    #[test]
    fn d_term_positive_velocity_tightens_thresholds() {
        let guard = make_guard_with_device_offset(0.0);
        let base = guard.thresholds(WorkloadMode::Idle);
        let d = guard.thresholds_with_d_term(WorkloadMode::Idle, 0.1);
        assert!(
            d.bg_pressure < base.bg_pressure,
            "rising pressure: D-term should lower bg threshold {}<{}",
            d.bg_pressure,
            base.bg_pressure
        );
        assert!(
            d.critical_pressure < base.critical_pressure,
            "rising pressure: D-term should lower critical threshold"
        );
    }

    #[test]
    fn d_term_capped_at_max_offset() {
        let guard = make_guard_with_device_offset(0.0);
        let base = guard.thresholds(WorkloadMode::Idle);
        // velocity=10.0 >> kd threshold — should be capped at MAX_D_OFFSET=0.05
        let d = guard.thresholds_with_d_term(WorkloadMode::Idle, 10.0);
        assert!(
            (base.bg_pressure - d.bg_pressure - 0.05).abs() < 1e-9,
            "D-term capped: offset should be exactly 0.05, got {}",
            base.bg_pressure - d.bg_pressure
        );
    }
}
