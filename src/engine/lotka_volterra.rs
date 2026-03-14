//! Lotka-Volterra — dinámica competitiva de procesos por RAM.
//!
//! ## Modelo (competencia interespecífica, Volterra 1926)
//!
//! Para dos "especies" (proceso dominante vs resto del sistema):
//!
//!   dx/dt = r₁·x · (1 - (x + α₁₂·y) / K)
//!   dy/dt = r₂·y · (1 - (y + α₂₁·x) / K)
//!
//! Donde:
//!   - x = fracción de RAM del proceso dominante
//!   - y = fracción de RAM del resto
//!   - K = capacidad total (= 1.0, normalizado)
//!   - rᵢ = tasa de crecimiento (aprendida de la velocidad de cambio de RSS)
//!   - αᵢⱼ = coeficiente de competencia (cuánto afecta la especie j a la i)
//!
//! ## Aplicación
//! Si la simulación hacia adelante muestra que x → K (el dominante acapara todo),
//! Apollo debe intervenir ANTES de que ocurra. El "tiempo al colapso" ecológico
//! es una señal predictiva.
//!
//! ## Simplificación práctica
//! No modelamos N procesos (sería O(N²)). Agrupamos en 2-3 clases:
//! 1. Proceso más pesado (potencial acaparador)
//! 2. Resto del sistema (competidor pasivo)

use serde::{Deserialize, Serialize};

/// Estado de competencia entre el proceso dominante y el resto.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitionState {
    /// Fracción de RAM del proceso dominante (0–1).
    x: f64,
    /// Fracción de RAM del resto del sistema (0–1).
    y: f64,
    /// Tasa de crecimiento del dominante (aprendida online).
    growth_dominant: f64,
    /// Tasa de crecimiento del resto.
    growth_rest: f64,
    /// Coeficiente de competencia: cuánto el dominante desplaza al resto.
    alpha_dom_rest: f64,
    /// Coeficiente de competencia: cuánto el resto limita al dominante.
    alpha_rest_dom: f64,
    /// Nombre del proceso dominante actual.
    dominant_name: String,
    /// RSS del dominante en el ciclo anterior (para calcular growth rate).
    prev_dominant_bytes: u64,
    /// RSS total del sistema en el ciclo anterior.
    prev_total_bytes: u64,
    /// Ciclos observados.
    ticks: u64,
}

impl Default for CompetitionState {
    fn default() -> Self {
        Self::new()
    }
}

impl CompetitionState {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 1.0,
            growth_dominant: 0.0,
            growth_rest: 0.0,
            alpha_dom_rest: 1.0,
            alpha_rest_dom: 0.5,
            dominant_name: String::new(),
            prev_dominant_bytes: 0,
            prev_total_bytes: 0,
            ticks: 0,
        }
    }

    /// Actualiza el modelo con las observaciones del ciclo actual.
    ///
    /// - `dominant_name`: nombre del proceso con más RSS.
    /// - `dominant_bytes`: RSS del proceso dominante.
    /// - `total_used_bytes`: RSS total de todos los procesos.
    /// - `total_available_bytes`: RAM total del sistema.
    /// - `dt_secs`: tiempo desde el último ciclo.
    pub fn update(
        &mut self,
        dominant_name: &str,
        dominant_bytes: u64,
        total_used_bytes: u64,
        total_available_bytes: u64,
        dt_secs: f64,
    ) {
        let dt = dt_secs.max(0.01);
        let total_avail = total_available_bytes.max(1) as f64;

        // Fracciones actuales.
        let x_new = (dominant_bytes as f64 / total_avail).clamp(0.0, 1.0);
        let rest_bytes = total_used_bytes.saturating_sub(dominant_bytes);
        let y_new = (rest_bytes as f64 / total_avail).clamp(0.0, 1.0);

        // Si cambió el proceso dominante, resetear growth tracking.
        if dominant_name != self.dominant_name {
            self.dominant_name = dominant_name.to_string();
            self.growth_dominant = 0.0;
            self.prev_dominant_bytes = dominant_bytes;
            self.prev_total_bytes = total_used_bytes;
        } else if self.ticks > 0 && self.prev_dominant_bytes > 0 {
            // Aprender growth rate del dominante: dr/dt normalizado.
            let dx = x_new - self.x;
            let dy = y_new - self.y;

            // EWMA de growth rates (α = 0.2 para suavizar).
            let raw_growth_dom = dx / dt;
            let raw_growth_rest = dy / dt;
            self.growth_dominant = 0.8 * self.growth_dominant + 0.2 * raw_growth_dom;
            self.growth_rest = 0.8 * self.growth_rest + 0.2 * raw_growth_rest;

            // Actualizar coeficientes de competencia.
            // Si el dominante crece mientras el resto decrece → alta competencia.
            if dx > 0.0 && dy < 0.0 && x_new > 0.01 {
                let competition = (-dy / dx).clamp(0.0, 3.0);
                self.alpha_dom_rest = 0.9 * self.alpha_dom_rest + 0.1 * competition;
            }
        }

        self.x = x_new;
        self.y = y_new;
        self.prev_dominant_bytes = dominant_bytes;
        self.prev_total_bytes = total_used_bytes;
        self.ticks += 1;
    }

    /// Simula hacia adelante `horizon_secs` segundos.
    /// Retorna la fracción de RAM del dominante al final.
    ///
    /// Usa Euler explícito con paso fijo de 1s.
    pub fn simulate_forward(&self, horizon_secs: f64) -> f64 {
        let steps = (horizon_secs as usize).min(120);
        let k = 1.0; // capacidad normalizada
        let mut x = self.x;
        let mut y = self.y;
        let r1 = self.growth_dominant;
        let r2 = self.growth_rest;
        let a12 = self.alpha_dom_rest;
        let a21 = self.alpha_rest_dom;

        for _ in 0..steps {
            let dx = r1 * x * (1.0 - (x + a12 * y) / k);
            let dy = r2 * y * (1.0 - (y + a21 * x) / k);
            x = (x + dx).clamp(0.0, k);
            y = (y + dy).clamp(0.0, k);
        }
        x
    }

    /// ¿El proceso dominante tiende a acaparar toda la RAM?
    ///
    /// Retorna un score 0–1:
    /// - 0.0: sin tendencia de acaparamiento
    /// - 0.5: crecimiento moderado, podría ser problema
    /// - 1.0: acaparamiento inminente
    pub fn monopoly_risk(&self) -> f64 {
        if self.ticks < 5 || self.growth_dominant <= 0.0 {
            return 0.0;
        }

        // Factor 1: fracción actual del dominante (ya alto = más riesgo).
        let share_risk = self.x;

        // Factor 2: velocidad de crecimiento (positivo y rápido = más riesgo).
        // Normalizar: 0.01/s = crecimiento moderado, 0.05/s = rápido.
        let growth_risk = (self.growth_dominant / 0.05).clamp(0.0, 1.0);

        // Factor 3: coeficiente de competencia (alto = desplaza al resto rápido).
        let competition_risk = (self.alpha_dom_rest / 2.0).clamp(0.0, 1.0);

        // Combinar: media geométrica (todos deben ser altos para alarma).
        let product = share_risk * growth_risk * competition_risk;
        if product <= 0.0 {
            return 0.0;
        }
        product.powf(1.0 / 3.0).clamp(0.0, 1.0)
    }

    /// Nombre del proceso dominante.
    pub fn dominant_process(&self) -> &str {
        &self.dominant_name
    }

    /// Fracción de RAM del proceso dominante.
    pub fn dominant_share(&self) -> f64 {
        self.x
    }

    /// Tasa de crecimiento del dominante (fracción/segundo).
    pub fn dominant_growth_rate(&self) -> f64 {
        self.growth_dominant
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let cs = CompetitionState::new();
        assert_eq!(cs.monopoly_risk(), 0.0);
        assert_eq!(cs.dominant_process(), "");
    }

    #[test]
    fn test_stable_system_low_risk() {
        let mut cs = CompetitionState::new();
        let total_ram = 8 * 1024 * 1024 * 1024u64; // 8 GB
                                                   // Proceso estable usando 1 GB de 8 GB.
        for _ in 0..20 {
            cs.update("stable_app", 1_000_000_000, 4_000_000_000, total_ram, 0.5);
        }
        assert!(
            cs.monopoly_risk() < 0.3,
            "stable system risk={}",
            cs.monopoly_risk()
        );
    }

    #[test]
    fn test_growing_process_increasing_risk() {
        let mut cs = CompetitionState::new();
        let total_ram = 8 * 1024 * 1024 * 1024u64;
        // Proceso creciendo 200 MB por ciclo.
        for i in 0..30 {
            let dom_bytes = 1_000_000_000u64 + i * 200_000_000;
            let total_used = 4_000_000_000u64 + i * 150_000_000;
            cs.update("growing_app", dom_bytes, total_used, total_ram, 0.5);
        }
        assert!(
            cs.dominant_growth_rate() > 0.0,
            "growth rate should be positive"
        );
        // After significant growth, risk should be non-trivial.
        assert!(
            cs.monopoly_risk() > 0.0,
            "growing process should have some risk"
        );
    }

    #[test]
    fn test_simulate_forward_bounded() {
        let mut cs = CompetitionState::new();
        let total_ram = 8 * 1024 * 1024 * 1024u64;
        for i in 0..20 {
            let dom_bytes = 1_000_000_000u64 + i * 200_000_000;
            cs.update("hog", dom_bytes, 5_000_000_000, total_ram, 0.5);
        }
        let predicted_share = cs.simulate_forward(30.0);
        assert!(
            predicted_share >= 0.0 && predicted_share <= 1.0,
            "predicted share must be in [0,1], got {}",
            predicted_share
        );
    }

    #[test]
    fn test_dominant_change_resets_growth() {
        let mut cs = CompetitionState::new();
        let total_ram = 8 * 1024 * 1024 * 1024u64;
        for _ in 0..10 {
            cs.update("app_a", 2_000_000_000, 4_000_000_000, total_ram, 0.5);
        }
        // Change dominant.
        cs.update("app_b", 3_000_000_000, 5_000_000_000, total_ram, 0.5);
        assert_eq!(cs.dominant_process(), "app_b");
        assert_eq!(cs.dominant_growth_rate(), 0.0); // reset on dominant change
    }
}
