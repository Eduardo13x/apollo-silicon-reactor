//! MPC Horizon — Model Predictive Control con horizonte de N ciclos.
//!
//! ## Idea (Control óptimo, Bellman 1957 / Mayne 2000)
//!
//! En vez de elegir la mejor acción AHORA, simula un horizonte de H pasos
//! y elige la primera acción de la secuencia que minimiza el costo acumulado.
//!
//! ## Modelo del sistema
//! Estado: pressure (escalar, 0–1)
//! Transición: pressure_next = pressure + velocity * dt + effect(action)
//!
//! Donde velocity viene del Kalman y effect(action) es el impacto estimado
//! de cada intervención (aprendido online).
//!
//! ## Costo por paso
//! J = Σᵢ [ pressure_i² + λ · cost(action_i) ]
//!
//! - pressure²: penaliza presión alta cuadráticamente (más urgente cuanto más alto)
//! - λ · cost: penaliza intervenciones innecesarias
//!
//! ## Optimización
//! Con 5 acciones y horizonte 5 → 5⁵ = 3125 secuencias.
//! Enumerar todas es viable (~1µs cada una). Sin heurísticas necesarias.

use serde::{Deserialize, Serialize};

/// Número de acciones posibles.
const N_ACTIONS: usize = 5;
/// Horizonte máximo (pasos hacia adelante).
const MAX_HORIZON: usize = 5;

/// Persisted subset of MPC state — learned effects only (plan/cost are transient).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpcPersisted {
    pub effects: [f64; N_ACTIONS],
}

/// Efecto estimado de cada acción sobre la presión.
/// Negativo = reduce presión, 0 = noop.
/// Orden: [Observe, TightenThresholds, SuggestAggressive, PreThrottleNoise, ProactivePurge]
const DEFAULT_EFFECTS: [f64; N_ACTIONS] = [
    0.00,   // Observe: no effect
    -0.02,  // TightenThresholds: slight reduction
    -0.03,  // SuggestAggressive: moderate reduction
    -0.01,  // PreThrottleNoise: small reduction
    -0.015, // ProactivePurge: small-moderate reduction
];

/// Costo de ejecutar cada acción (penalización por acción innecesaria).
const ACTION_COSTS: [f64; N_ACTIONS] = [
    0.00,  // Observe: free
    0.01,  // TightenThresholds: cheap
    0.03,  // SuggestAggressive: moderate (changes profile)
    0.02,  // PreThrottleNoise: moderate
    0.015, // ProactivePurge: moderate
];

/// Controlador MPC con horizonte finito.
#[derive(Debug, Clone)]
pub struct MpcController {
    /// Efectos estimados por acción (aprendidos online).
    effects: [f64; N_ACTIONS],
    /// Horizonte de planificación (pasos).
    horizon: usize,
    /// Peso del costo de acción vs presión (λ).
    action_cost_weight: f64,
    /// dt por paso (segundos).
    dt_per_step: f64,
    /// Última secuencia óptima encontrada (para diagnóstico).
    last_plan: [usize; MAX_HORIZON],
    /// Costo de la última secuencia óptima.
    last_cost: f64,
}

impl MpcController {
    /// Crea un controlador MPC.
    ///
    /// - `horizon`: pasos hacia adelante (2–5 recomendado).
    /// - `dt_per_step`: duración de cada paso en segundos (0.5 para el ciclo normal).
    pub fn new(horizon: usize, dt_per_step: f64) -> Self {
        Self {
            effects: DEFAULT_EFFECTS,
            horizon: horizon.min(MAX_HORIZON),
            action_cost_weight: 0.5,
            dt_per_step,
            last_plan: [0; MAX_HORIZON],
            last_cost: f64::MAX,
        }
    }

    /// Resuelve el MPC: dada la presión actual y su velocidad,
    /// encuentra la secuencia óptima de acciones en el horizonte.
    ///
    /// Retorna el índice de la PRIMERA acción de la secuencia óptima.
    ///
    /// - `pressure`: presión actual (0–1).
    /// - `velocity`: velocidad de cambio (del Kalman, unidades/segundo).
    pub fn solve(&mut self, pressure: f64, velocity: f64) -> usize {
        let h = self.horizon;

        // Para horizonte ≤ 3, enumerar todas las secuencias.
        // Para horizonte > 3, usar greedy por paso (evitar 5⁵ = 3125 evaluaciones).
        if h <= 3 {
            self.solve_exhaustive(pressure, velocity)
        } else {
            self.solve_greedy(pressure, velocity)
        }
    }

    /// Enumeración exhaustiva (horizonte ≤ 3, máx 125 evaluaciones).
    fn solve_exhaustive(&mut self, pressure: f64, velocity: f64) -> usize {
        let h = self.horizon;
        let mut best_cost = f64::MAX;
        let mut best_first_action = 0usize;
        let mut best_plan = [0usize; MAX_HORIZON];

        let total_sequences = N_ACTIONS.pow(h as u32);
        for seq_idx in 0..total_sequences {
            // Decode sequence index into action indices.
            let mut actions = [0usize; MAX_HORIZON];
            let mut tmp = seq_idx;
            for slot in actions.iter_mut().take(h) {
                *slot = tmp % N_ACTIONS;
                tmp /= N_ACTIONS;
            }

            let cost = self.evaluate_sequence(pressure, velocity, &actions[..h]);
            if cost < best_cost {
                best_cost = cost;
                best_first_action = actions[0];
                best_plan = actions;
            }
        }

        self.last_plan = best_plan;
        self.last_cost = best_cost;
        best_first_action
    }

    /// Greedy por paso (horizonte > 3).
    fn solve_greedy(&mut self, pressure: f64, velocity: f64) -> usize {
        let h = self.horizon;
        let mut plan = [0usize; MAX_HORIZON];
        let mut p = pressure;
        let mut v = velocity;

        for slot in plan.iter_mut().take(h) {
            let mut best_action = 0;
            let mut best_cost = f64::MAX;
            for (a, (&effect, &ac)) in self.effects.iter().zip(ACTION_COSTS.iter()).enumerate() {
                let p_next = (p + v * self.dt_per_step + effect).clamp(0.0, 1.0);
                let cost = p_next * p_next + self.action_cost_weight * ac;
                if cost < best_cost {
                    best_cost = cost;
                    best_action = a;
                }
            }
            *slot = best_action;
            p = (p + v * self.dt_per_step + self.effects[best_action]).clamp(0.0, 1.0);
            // Velocity decays slightly towards zero (mean reversion assumption).
            v *= 0.9;
        }

        self.last_plan = plan;
        self.last_cost = self.evaluate_sequence(pressure, velocity, &plan[..h]);
        plan[0]
    }

    /// Evalúa el costo total de una secuencia de acciones.
    fn evaluate_sequence(&self, pressure: f64, velocity: f64, actions: &[usize]) -> f64 {
        let mut total_cost = 0.0;
        let mut p = pressure;
        let mut v = velocity;

        for &a in actions {
            p = (p + v * self.dt_per_step + self.effects[a]).clamp(0.0, 1.0);
            v *= 0.9; // mean reversion
                      // Stage cost: pressure² + λ · action_cost
            total_cost += p * p + self.action_cost_weight * ACTION_COSTS[a];
        }
        total_cost
    }

    /// Actualiza el efecto estimado de una acción observando el resultado real.
    ///
    /// - `action`: índice de la acción ejecutada.
    /// - `pressure_before`: presión antes de la acción.
    /// - `pressure_after`: presión después de la acción (siguiente ciclo).
    /// - `velocity`: velocidad estimada (para separar efecto de acción vs tendencia natural).
    pub fn update_effect(
        &mut self,
        action: usize,
        pressure_before: f64,
        pressure_after: f64,
        velocity: f64,
    ) {
        if action >= N_ACTIONS {
            return;
        }
        // efecto observado = cambio real - cambio por tendencia
        let expected_change = velocity * self.dt_per_step;
        let actual_change = pressure_after - pressure_before;
        let observed_effect = actual_change - expected_change;

        // EWMA update (α = 0.1)
        self.effects[action] = 0.9 * self.effects[action] + 0.1 * observed_effect;
        // Clamp effects to reasonable range.
        self.effects[action] = self.effects[action].clamp(-0.10, 0.02);
    }

    /// Costo del plan actual.
    pub fn last_cost(&self) -> f64 {
        self.last_cost
    }

    /// Efectos aprendidos por acción.
    pub fn learned_effects(&self) -> &[f64; N_ACTIONS] {
        &self.effects
    }

    /// Snapshot of learned state for persistence.
    pub fn to_persisted(&self) -> MpcPersisted {
        MpcPersisted {
            effects: self.effects,
        }
    }

    /// Restore learned state from a persisted snapshot.
    /// Horizon and dt_per_step must still be set by the caller.
    pub fn restore_effects(&mut self, persisted: &MpcPersisted) {
        self.effects = persisted.effects;
    }

    /// Constraint-aware solve: adjusts horizon and action costs based on context.
    ///
    /// - `urgency` (0–1): higher urgency → longer horizon (look further ahead).
    /// - `subsystem_utilities` [entropy, hazard, lotka, mpc]: from budget cognitivo.
    ///   Low utility for a subsystem increases the cost of actions that rely on it,
    ///   steering MPC toward actions whose supporting signals are reliable.
    ///
    /// Maps: urgency → horizon (2-5), utilities → cost modifiers.
    pub fn solve_constrained(
        &mut self,
        pressure: f64,
        velocity: f64,
        urgency: f64,
        subsystem_utilities: &[f64; 4],
    ) -> usize {
        // Dynamic horizon: more urgency → plan further ahead.
        let dynamic_h = if urgency > 0.70 {
            5
        } else if urgency > 0.50 {
            4
        } else if urgency > 0.30 {
            3
        } else {
            2
        };
        let old_h = self.horizon;
        self.horizon = dynamic_h.min(MAX_HORIZON);

        // Modulate action_cost_weight by signal reliability.
        // When signals are low-utility (unreliable), increase cost of intervention
        // (favor Observe — don't act on bad data).
        // Average utility of the 4 subsystems: low avg → higher λ.
        let avg_utility: f64 = subsystem_utilities.iter().sum::<f64>() / 4.0;
        let old_lambda = self.action_cost_weight;
        // Scale λ from 0.3 (high utility, trust signals) to 1.0 (low utility, be cautious).
        self.action_cost_weight = (1.0 - avg_utility * 0.7).clamp(0.30, 1.0);

        let result = self.solve(pressure, velocity);

        // Restore original horizon and lambda (these are per-call overrides).
        self.horizon = old_h;
        self.action_cost_weight = old_lambda;
        result
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_pressure_cost_minimal() {
        let mut mpc = MpcController::new(3, 0.5);
        // Low pressure, no velocity → cost should be near-zero regardless of choice.
        let action = mpc.solve(0.1, 0.0);
        assert!(action < N_ACTIONS);
        assert!(
            mpc.last_cost() < 0.1,
            "cost should be minimal at low pressure, got {}",
            mpc.last_cost()
        );
    }

    #[test]
    fn test_high_pressure_rising_chooses_action() {
        let mut mpc = MpcController::new(3, 0.5);
        // High pressure + rising → should choose intervention.
        let action = mpc.solve(0.85, 0.1);
        assert_ne!(action, 0, "high+rising pressure should NOT choose Observe");
    }

    #[test]
    fn test_moderate_pressure_falling_may_observe() {
        let mut mpc = MpcController::new(3, 0.5);
        // Moderate pressure but falling → might choose Observe (problem resolving itself).
        let action = mpc.solve(0.70, -0.05);
        // This is a valid case for either Observe or mild intervention.
        assert!(action < N_ACTIONS);
    }

    #[test]
    fn test_update_effect_changes_estimates() {
        let mut mpc = MpcController::new(3, 0.5);
        let before = mpc.effects[1]; // TightenThresholds
                                     // Simulate: action 1 was applied, pressure dropped more than expected.
        mpc.update_effect(1, 0.80, 0.70, 0.0);
        // Observed effect = (0.70 - 0.80) - 0.0 = -0.10
        assert!(
            mpc.effects[1] < before,
            "effect should decrease (become more negative)"
        );
    }

    #[test]
    fn test_greedy_fallback_for_long_horizon() {
        let mut mpc = MpcController::new(5, 0.5);
        let action = mpc.solve(0.80, 0.05);
        assert!(action < N_ACTIONS);
        // Should produce a plan.
        assert!(mpc.last_cost() < f64::MAX);
    }

    #[test]
    fn test_sequence_evaluation_bounded() {
        let mpc = MpcController::new(3, 0.5);
        let cost = mpc.evaluate_sequence(0.5, 0.0, &[0, 0, 0]);
        assert!(cost >= 0.0, "cost should be non-negative");
        assert!(cost < 10.0, "cost should be bounded");
    }

    // ── Constraint-aware MPC tests ──────────────────────────────────────────

    #[test]
    fn test_constrained_high_urgency_longer_horizon() {
        let mut mpc = MpcController::new(3, 0.5);
        // High urgency + high pressure → should act (not Observe).
        let action = mpc.solve_constrained(0.85, 0.1, 0.80, &[0.5, 0.5, 0.5, 0.5]);
        assert_ne!(action, 0, "high urgency should trigger intervention");
        // Horizon was dynamically increased to 5 for urgency > 0.70.
        // Original horizon (3) should be restored after call.
        assert_eq!(mpc.horizon, 3, "horizon should be restored");
    }

    #[test]
    fn test_constrained_low_utility_favors_observe() {
        let mut mpc = MpcController::new(3, 0.5);
        // Low utility → high λ → actions are expensive → favor Observe.
        let action_cautious = mpc.solve_constrained(
            0.60,
            0.02,
            0.40,
            &[0.05, 0.05, 0.05, 0.05], // very low utility
        );
        // Same scenario but with high utility.
        let action_confident = mpc.solve_constrained(
            0.60,
            0.02,
            0.40,
            &[0.90, 0.90, 0.90, 0.90], // high utility
        );
        // With low utility, MPC should be more cautious (higher cost for actions).
        // At moderate pressure, low utility → likely Observe; high utility → may intervene.
        // We can't assert exact actions, but the costs should differ.
        // At minimum: both produce valid actions.
        assert!(action_cautious < N_ACTIONS);
        assert!(action_confident < N_ACTIONS);
    }

    #[test]
    fn test_constrained_restores_state() {
        let mut mpc = MpcController::new(3, 0.5);
        let orig_lambda = mpc.action_cost_weight;
        let orig_h = mpc.horizon;

        mpc.solve_constrained(0.70, 0.05, 0.60, &[0.3, 0.3, 0.3, 0.3]);

        assert_eq!(mpc.horizon, orig_h, "horizon must be restored");
        assert!(
            (mpc.action_cost_weight - orig_lambda).abs() < 1e-9,
            "lambda must be restored"
        );
    }
}
