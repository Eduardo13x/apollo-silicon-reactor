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

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

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
    /// Modo build: compilación detectada → actuar más temprano.
    pub build_mode: bool,
}

impl Default for OverflowThresholds {
    fn default() -> Self {
        Self {
            bg_pressure: 0.78,
            critical_pressure: 0.88,
            extreme_pressure: 0.90,
            build_mode: false,
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
}

/// Herramientas de compilación que causan picos de RAM.
/// "stable" es el wrapper del Rust toolchain que rustup antepone a rustc/cargo —
/// aparece como proceso padre durante compilación y consume RAM proporcional.
const BUILD_TOOLS: &[&str] = &[
    "rustc", "cargo", "swift", "clang", "make", "gcc", "ld", "link", "stable",
];

impl OverflowGuard {
    /// Carga el historial desde disco, o crea uno vacío si no existe.
    pub fn load_or_default(path: &Path) -> Self {
        let history = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self {
            history,
            path: path.to_path_buf(),
            last_decay_check: Instant::now(),
            last_event_at: None,
        }
    }

    /// Registra un evento de overflow. Llama esto cuando:
    /// - kqueue dispara VmPressureLevel::Critical o SuddenTerminate
    /// - survival_mode se activa
    /// - presión RAM > 0.92 sostenida
    pub fn record_event(
        &mut self,
        memory_pressure: f64,
        swap_delta_bps: f64,
        heavy_apps: &[String],
        cause: &str,
    ) {
        // Deduplicar: no registrar dos eventos del mismo overflow (ventana 60s).
        if let Some(last) = self.last_event_at {
            if last.elapsed() < Duration::from_secs(60) {
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
                // Contribución decae a la mitad cada 8 horas.
                -0.05 * (2.0_f64).powf(-age_h / 8.0)
            })
            .sum();

        raw.max(-0.20) // piso: nunca más de 20pp de ajuste
    }

    /// Mantiene `history.threshold_offset` sincronizado con el offset dinámico.
    /// Llamar una vez por ciclo para que las métricas sean precisas.
    pub fn tick_decay(&mut self) {
        if self.last_decay_check.elapsed() < Duration::from_secs(600) {
            return; // Revisar cada 10 minutos máximo.
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
    /// # Parámetros
    /// - `proc_names`: nombres de todos los procesos corriendo (para detectar build mode).
    pub fn thresholds(&self, proc_names: &[&str]) -> OverflowThresholds {
        // Usar offset dinámico para que los thresholds reflejen la edad real
        // de los eventos, no un acumulador estático que queda atrapado en el piso.
        let off = self.compute_dynamic_offset();
        let build_mode = Self::detect_build_mode(proc_names);

        // En build mode, actuar aún más temprano (compile pica RAM rápido).
        let build_bonus = if build_mode { -0.08 } else { 0.0 };
        let total_offset = off + build_bonus;

        OverflowThresholds {
            // BackgroundPressure threshold: default 0.78, piso 0.55
            bg_pressure: (0.78 + total_offset).max(0.55),
            // ThermalConstrained threshold: default 0.88, piso 0.65
            critical_pressure: (0.88 + total_offset).max(0.65),
            // Extreme freeze threshold: default 0.90, piso 0.70
            extreme_pressure: (0.90 + total_offset).max(0.70),
            build_mode,
        }
    }

    /// ¿Hay herramientas de compilación corriendo activamente?
    /// Requiere al menos 2 procesos de build para evitar falsos positivos
    /// (p.ej. un `cargo` idle en background).
    pub fn detect_build_mode(proc_names: &[&str]) -> bool {
        let count = proc_names
            .iter()
            .filter(|n| BUILD_TOOLS.iter().any(|t| n.to_lowercase().contains(t)))
            .count();
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

    /// ¿Se parece la carga actual a una que causó overflow antes?
    /// Útil para mostrar advertencias al usuario y subir el reactor_weight.
    pub fn resembles_past_overflow(&self, proc_names: &[&str]) -> bool {
        for event in &self.history.events {
            let matches = event
                .heavy_apps
                .iter()
                .filter(|app| proc_names.iter().any(|n| n.contains(app.as_str())))
                .count();
            // Coinciden ≥3 apps pesadas Y es al menos la mitad del escenario pasado.
            if matches >= 3 && event.heavy_apps.len() > 0 && matches * 2 >= event.heavy_apps.len() {
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
        history.threshold_offset = -0.20; // simula estado anterior al fix
        OverflowGuard {
            history,
            path: PathBuf::from("/tmp/test_overflow.json"),
            last_decay_check: Instant::now(),
            last_event_at: None,
        }
    }

    #[test]
    fn dynamic_offset_zero_with_no_events() {
        let guard = make_guard_with_events(&[]);
        assert_eq!(guard.compute_dynamic_offset(), 0.0);
    }

    #[test]
    fn dynamic_offset_decays_with_age() {
        // Evento de hace 8 horas contribuye -0.05 * 0.5 = -0.025
        let guard_fresh = make_guard_with_events(&[0.0]); // recién ocurrido
        let guard_old = make_guard_with_events(&[8.0]);   // hace 8 horas

        let offset_fresh = guard_fresh.compute_dynamic_offset();
        let offset_old = guard_old.compute_dynamic_offset();

        // Evento reciente: ≈ -0.05; hace 8h: ≈ -0.025
        assert!(offset_fresh < -0.04, "evento reciente debe tener offset alto: {}", offset_fresh);
        assert!(offset_old > -0.03, "evento de 8h debe tener offset reducido: {}", offset_old);
        assert!(offset_old > offset_fresh, "evento más viejo debe tener menor impacto");
    }

    #[test]
    fn dynamic_offset_recovers_after_24h_calm() {
        // 20 eventos pero todos de hace 24+ horas — simula la situación del usuario
        let ages: Vec<f64> = (24..=168).step_by(8).map(|h| h as f64).collect();
        let guard = make_guard_with_events(&ages);

        let offset = guard.compute_dynamic_offset();
        // Con todos los eventos a 24h+, el offset debe ser mucho menor que -20pp
        assert!(
            offset > -0.10,
            "20 eventos de 24h+ no deben mantener offset al piso: {}",
            offset
        );
    }

    #[test]
    fn dynamic_offset_capped_at_floor() {
        // 5 eventos muy recientes — impacto máximo pero limitado al piso -20pp
        let guard = make_guard_with_events(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        let offset = guard.compute_dynamic_offset();
        assert!(offset >= -0.20, "offset nunca debe bajar del piso -20pp: {}", offset);
    }

    #[test]
    fn thresholds_recover_when_events_age() {
        // Mismo escenario: 20 eventos de hace 24h
        let ages: Vec<f64> = (24..=168).step_by(8).map(|h| h as f64).collect();
        let guard = make_guard_with_events(&ages);

        let t = guard.thresholds(&[]);
        // bg_pressure debe estar por encima de 0.70 (el piso cuando offset=-0.08 aprox)
        assert!(
            t.bg_pressure > 0.70,
            "bg_pressure debe haberse recuperado: {}",
            t.bg_pressure
        );
        assert!(t.bg_pressure <= 0.78, "no puede superar el default: {}", t.bg_pressure);
    }
}
