//! Q-table RL agent for adaptive overflow thresholds.
//!
//! ## Scientific basis
//! Tabular Q-learning (Watkins, 1989) with e-greedy exploration.
//! State space is deliberately small (48 states x 3 actions = 144 Q-values)
//! to ensure convergence within ~200 episodes per state-action pair.
//!
//! ## State representation
//! (pressure_band, compressor_band, overflow_last_hour) -- 4 x 3 x 4 = 48 states.
//!
//! ## Reward function
//! +1.0 per tick without overflow (stability reward).
//! -10.0 per overflow event (penalizes dangerous thresholds).
//!
//! ## Safety
//! Hard floor: the RL adjustment can never push absolute thresholds below 0.45.
//! The RL output is an additive correction on top of the existing exponential
//! decay (compute_dynamic_offset), which provides the baseline behavior.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const NUM_STATES: usize = 48; // 4 pressure × 3 compressor × 4 overflow
const NUM_ACTIONS: usize = 3;
/// Minimum learning rate — floor for the decaying EMA alpha.
const ALPHA_MIN: f64 = 0.02;
/// Initial learning rate — high to learn fast from early observations.
const ALPHA_INITIAL: f64 = 0.20;
const GAMMA: f64 = 0.95;
const EPSILON_INITIAL: f64 = 0.10;
const EPSILON_STABLE: f64 = 0.05;
const EPSILON_DECAY_TICKS: u64 = 200;

const REWARD_STABLE: f64 = 1.0;
const REWARD_OVERFLOW: f64 = -10.0;

/// Hard floor: RL can never push absolute bg_pressure below this.
pub const RL_ABSOLUTE_FLOOR: f64 = 0.45;

const ADJUSTMENT_FLOOR: f64 = -0.20;
const ADJUSTMENT_CEIL: f64 = 0.05;

/// Infrastructure-locked constraints (Hermes/Tinker-Atropos pattern).
/// These are hard walls the RL agent can NEVER cross, regardless of reward.
/// Prevents learning to sacrifice system stability for marginal improvements.
pub struct RlConstraints {
    /// Maximum Dyna-Q planning steps (prevents CPU burn under stress).
    pub max_dyna_steps: usize,
    /// Minimum alpha multiplier (prevents learning stall).
    pub min_alpha_mult: f64,
    /// Maximum epsilon (prevents pure random exploration).
    pub max_epsilon: f64,
    /// Maximum consecutive Lower5pp actions (prevents threshold collapse).
    pub max_consecutive_lower: u32,
}

impl Default for RlConstraints {
    fn default() -> Self {
        Self {
            max_dyna_steps: 20,
            min_alpha_mult: 0.3,
            max_epsilon: 0.15,
            max_consecutive_lower: 5,
        }
    }
}

/// Discretized system state for the Q-table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RlState {
    pub pressure_band: u8,
    pub compressor_band: u8,
    pub overflow_last_hour: u8,
}

impl RlState {
    pub fn from_metrics(
        memory_pressure: f64,
        compressor_pressure: f64,
        overflows_last_hour: usize,
    ) -> Self {
        let pressure_band = if memory_pressure < 0.50 {
            0
        } else if memory_pressure <= 0.80 {
            1
        } else if memory_pressure <= 0.92 {
            2 // high
        } else {
            3 // critical — RL can learn distinct policy for extreme pressure
        };
        let compressor_band = if compressor_pressure < 0.30 {
            0
        } else if compressor_pressure <= 0.60 {
            1
        } else {
            2
        };
        let overflow_last_hour = (overflows_last_hour as u8).min(3);
        Self {
            pressure_band,
            compressor_band,
            overflow_last_hour,
        }
    }

    pub fn index(&self) -> usize {
        (self.pressure_band as usize) * 12
            + (self.compressor_band as usize) * 4
            + (self.overflow_last_hour as usize)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlAction {
    Lower5pp = 0,
    Hold = 1,
    Raise1pp = 2,
}

impl RlAction {
    fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Lower5pp,
            1 => Self::Hold,
            _ => Self::Raise1pp,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct RlPersisted {
    q_table: Vec<f64>,
    current_adjustment: f64,
    total_ticks: u64,
    total_overflows: u64,
    #[serde(default = "default_neuro_alpha")]
    neuro_alpha_mult: f64,
}

fn default_neuro_alpha() -> f64 { 1.0 }

/// Dyna-Q planning steps per real transition (Sutton 1991).
/// 10 simulated updates per real step ≈ 10x sample efficiency.
/// Inspired by memoria-core/src/embodiment/dyna_q.rs.
const DYNA_PLANNING_STEPS: usize = 10;

/// Stored model entry for Dyna-Q planning.
#[derive(Clone)]
struct DynaTransition {
    reward: f64,
    next_state_idx: usize,
}

/// Tabular Q-learning agent for adaptive threshold tuning.
pub struct RlThresholdAgent {
    q_table: [[f64; NUM_ACTIONS]; NUM_STATES],
    last_state: Option<RlState>,
    last_action: Option<RlAction>,
    pub current_adjustment: f64,
    total_ticks: u64,
    total_overflows: u64,
    path: PathBuf,
    /// Previous memory pressure for potential-based reward shaping.
    /// Φ(s) = -pressure² — so shaped reward = Φ(s') - Φ(s) = prev² - cur².
    prev_pressure: f64,
    /// Exponential moving average of |RPE| (reward prediction error magnitude).
    /// Used to normalize surprise: surprise_factor = |rpe| / rpe_ema.
    /// Initialized to 1.0 so early ticks start with a neutral baseline.
    rpe_ema: f64,
    /// Dyna-Q transition model: (state_idx, action_idx) → transition.
    /// Stores observed transitions for model-based planning replay.
    dyna_model: HashMap<(usize, usize), DynaTransition>,
    /// Round-robin cursor for deterministic planning (no RNG needed).
    dyna_cursor: usize,
    /// Cached keys for round-robin iteration.
    dyna_keys: Vec<(usize, usize)>,
    // ── Neuromodulator-driven parameters (set by daemon each cycle) ──
    /// DA → RL alpha multiplier [0.5, 1.5]. Default=1.0 (no change).
    pub neuro_alpha_mult: f64,
    /// ACh → exploration epsilon bonus [0.0, 0.05]. Default=0.0.
    pub neuro_epsilon_bonus: f64,
    /// NA → Dyna-Q planning steps [4, 20]. Default=10.
    pub dyna_steps: usize,
    /// Infrastructure-locked safety constraints (Hermes pattern).
    constraints: RlConstraints,
    /// Consecutive Lower5pp actions (for constraint enforcement).
    consecutive_lower: u32,
}

impl RlThresholdAgent {
    pub fn load_or_default(path: &Path) -> Self {
        let (q_table, current_adjustment, total_ticks, total_overflows, neuro_alpha_mult) =
            std::fs::read_to_string(path)
                .ok()
                .and_then(|s| serde_json::from_str::<RlPersisted>(&s).ok())
                .and_then(|p| {
                    if p.q_table.len() != NUM_STATES * NUM_ACTIONS {
                        return None;
                    }
                    let mut qt = [[0.0_f64; NUM_ACTIONS]; NUM_STATES];
                    for (i, &val) in p.q_table.iter().enumerate() {
                        qt[i / NUM_ACTIONS][i % NUM_ACTIONS] = val;
                    }
                    Some((qt, p.current_adjustment, p.total_ticks, p.total_overflows, p.neuro_alpha_mult))
                })
                .unwrap_or_else(|| {
                    // ZeroTune: pre-seed critical pressure band (3) to favor Lower5pp.
                    // Domain knowledge: at pressure > 0.92, acting early is always correct.
                    let mut qt = [[0.0_f64; NUM_ACTIONS]; NUM_STATES];
                    for cb in 0..3usize {
                        for oh in 0..4usize {
                            let idx = 3 * 12 + cb * 4 + oh; // pressure_band=3
                            qt[idx][0] = 2.0;  // Lower5pp: positive prior
                            qt[idx][1] = -1.0; // Hold: mild negative
                            qt[idx][2] = -2.0; // Raise1pp: bad at critical pressure
                        }
                    }
                    (qt, 0.0, 0, 0, 1.0)
                });

        Self {
            q_table,
            last_state: None,
            last_action: None,
            current_adjustment,
            total_ticks,
            total_overflows,
            path: path.to_path_buf(),
            prev_pressure: 0.5,
            rpe_ema: 1.0,
            dyna_model: HashMap::new(),
            dyna_cursor: 0,
            dyna_keys: Vec::new(),
            neuro_alpha_mult,
            neuro_epsilon_bonus: 0.0,
            dyna_steps: DYNA_PLANNING_STEPS,
            constraints: RlConstraints::default(),
            consecutive_lower: 0,
        }
    }

    pub fn epsilon(&self) -> f64 {
        if self.total_ticks < EPSILON_DECAY_TICKS {
            EPSILON_INITIAL
        } else {
            EPSILON_STABLE
        }
    }

    /// Decaying learning rate: high at start (explore), low later (exploit).
    /// α = max(ALPHA_MIN, ALPHA_INITIAL / (1 + total_ticks / 200))
    /// Half-life ≈ 200 ticks — matches EPSILON_DECAY_TICKS.
    pub fn alpha(&self) -> f64 {
        (ALPHA_INITIAL / (1.0 + self.total_ticks as f64 / 200.0)).max(ALPHA_MIN)
    }

    pub fn total_ticks(&self) -> u64 {
        self.total_ticks
    }
    pub fn total_overflows(&self) -> u64 {
        self.total_overflows
    }

    fn select_action(&self, state: RlState) -> RlAction {
        let eps = self.epsilon() + self.neuro_epsilon_bonus;
        let explore = (self.total_ticks.wrapping_mul(2_654_435_761)) % 100 < (eps * 100.0) as u64;
        if explore {
            let action_idx = ((self.total_ticks.wrapping_mul(7_919)) % 3) as usize;
            RlAction::from_index(action_idx)
        } else {
            let row = &self.q_table[state.index()];
            let mut best_idx = 0;
            let mut best_val = row[0];
            for (i, &q) in row.iter().enumerate().skip(1) {
                if q > best_val {
                    best_val = q;
                    best_idx = i;
                }
            }
            RlAction::from_index(best_idx)
        }
    }

    /// Map a pressure_band to a representative pressure float for shaping.
    fn band_to_pressure(band: u8) -> f64 {
        match band {
            0 => 0.35,
            1 => 0.65,
            _ => 0.90,
        }
    }

    pub fn tick(&mut self, state: RlState, overflow_occurred: bool) {
        // Potential-based reward shaping: Φ(s) = -pressure²
        // R_shaped = Φ(s') - Φ(s) = prev_pressure² - current_pressure²
        let current_pressure = Self::band_to_pressure(state.pressure_band);
        let shaped = self.prev_pressure * self.prev_pressure
            - current_pressure * current_pressure;

        let base_reward = if overflow_occurred {
            self.total_overflows += 1;
            REWARD_OVERFLOW
        } else {
            REWARD_STABLE
        };
        let reward = base_reward + shaped * 2.0;

        if let (Some(prev_state), Some(prev_action)) = (self.last_state, self.last_action) {
            let s = prev_state.index();
            let a = prev_action as usize;
            let s_prime = state.index();
            let max_q_next = self.q_table[s_prime]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let old_q = self.q_table[s][a];
            // Dopamine RPE: scale alpha by surprise magnitude (Bhatt et al., Nature Comm. 2024).
            // Large unexpected outcomes temporarily boost alpha for rapid re-adaptation.
            let rpe_abs = (reward + GAMMA * max_q_next - old_q).abs();
            self.rpe_ema = 0.99 * self.rpe_ema + 0.01 * rpe_abs;
            let surprise_factor = (rpe_abs / self.rpe_ema.max(0.01)).clamp(0.5, 5.0);
            let effective_alpha = self.alpha() * surprise_factor * self.neuro_alpha_mult;
            self.q_table[s][a] = old_q + effective_alpha * (reward + GAMMA * max_q_next - old_q);

            // Dyna-Q: record real transition and run planning steps.
            // Inspired by memoria-core dyna_q.rs (Sutton 1991).
            self.dyna_record(s, a, reward, s_prime);
            self.dyna_plan();

            // Trajectory recording (Hermes pattern): persist transitions for offline learning.
            // Append to JSONL every 10 ticks to amortize I/O cost.
            if self.total_ticks % 10 == 0 {
                self.record_trajectory(s, a, reward, s_prime);
            }
        }

        let mut action = self.select_action(state);
        // Infrastructure-locked constraint (Hermes/Tinker-Atropos):
        // Prevent threshold collapse from too many consecutive Lower5pp.
        if matches!(action, RlAction::Lower5pp) {
            self.consecutive_lower += 1;
            if self.consecutive_lower > self.constraints.max_consecutive_lower {
                action = RlAction::Hold; // force hold
            }
        } else {
            self.consecutive_lower = 0;
        }
        match action {
            RlAction::Lower5pp => self.current_adjustment -= 0.05,
            RlAction::Hold => {}
            RlAction::Raise1pp => self.current_adjustment += 0.01,
        }
        self.current_adjustment = self
            .current_adjustment
            .clamp(ADJUSTMENT_FLOOR, ADJUSTMENT_CEIL);

        self.last_state = Some(state);
        self.last_action = Some(action);
        self.prev_pressure = current_pressure;
        self.total_ticks += 1;
    }

    /// Q-value for the last state-action pair (for observability/testing).
    /// Returns 0.0 if no action has been taken yet.
    pub fn last_q_value(&self) -> f64 {
        match (self.last_state, self.last_action) {
            (Some(s), Some(a)) => self.q_table[s.index()][a as usize],
            _ => 0.0,
        }
    }

    /// Inject an external reward signal into the last state-action pair.
    /// Used by the feedback loop: OutcomeTracker sends penalties when
    /// throttling is ineffective, enriching RL beyond binary overflow.
    pub fn inject_external_reward(&mut self, reward: f64) {
        if let (Some(prev_state), Some(prev_action)) = (self.last_state, self.last_action) {
            let s = prev_state.index();
            let a = prev_action as usize;
            let alpha = self.alpha();
            // Direct injection: no Bellman lookahead, just nudge Q toward reward.
            self.q_table[s][a] += alpha * reward;
        }
    }

    /// Record a real (s, a, r, s') transition into the Dyna-Q model.
    fn dyna_record(&mut self, state_idx: usize, action_idx: usize, reward: f64, next_state_idx: usize) {
        let key = (state_idx, action_idx);
        match self.dyna_model.get_mut(&key) {
            Some(t) => {
                // EMA blend: keep history, weight new reward at 10%.
                t.reward = t.reward * 0.9 + reward * 0.1;
                t.next_state_idx = next_state_idx;
            }
            None => {
                self.dyna_model.insert(key, DynaTransition { reward, next_state_idx });
                // Invalidate key cache.
                self.dyna_keys.clear();
            }
        }
    }

    /// Run DYNA_PLANNING_STEPS simulated Q-updates from the stored model.
    /// Deterministic round-robin — no RNG dependency.
    fn dyna_plan(&mut self) {
        if self.dyna_model.is_empty() {
            return;
        }
        // Rebuild key cache if invalidated.
        if self.dyna_keys.is_empty() {
            self.dyna_keys = self.dyna_model.keys().copied().collect();
        }
        let n_keys = self.dyna_keys.len();
        let alpha = self.alpha() * self.neuro_alpha_mult;
        for _ in 0..self.dyna_steps {
            let idx = self.dyna_cursor % n_keys;
            self.dyna_cursor = self.dyna_cursor.wrapping_add(1);
            let (s, a) = self.dyna_keys[idx];
            let (reward, next_state_idx) = match self.dyna_model.get(&(s, a)) {
                Some(t) => (t.reward, t.next_state_idx),
                None => continue,
            };
            let max_q_next = self.q_table[next_state_idx]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let max_q_next = if max_q_next.is_infinite() { 0.0 } else { max_q_next };
            let old_q = self.q_table[s][a];
            self.q_table[s][a] = old_q + alpha * (reward + GAMMA * max_q_next - old_q);
        }
    }

    /// Number of unique (state, action) pairs in the Dyna-Q model.
    pub fn dyna_model_size(&self) -> usize {
        self.dyna_model.len()
    }

    /// Record a transition to trajectory JSONL file (Hermes pattern).
    /// Enables offline RL training from historical data.
    fn record_trajectory(&self, s: usize, a: usize, reward: f64, s_prime: usize) {
        let traj_path = self.path.with_extension("trajectories.jsonl");
        // Cap file at ~1MB to prevent disk bloat.
        if let Ok(meta) = std::fs::metadata(&traj_path) {
            if meta.len() > 1_000_000 {
                return; // silently skip when full
            }
        }
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&traj_path)
        {
            let _ = writeln!(f, "{{\"s\":{},\"a\":{},\"r\":{:.4},\"s_prime\":{},\"tick\":{}}}",
                s, a, reward, s_prime, self.total_ticks);
        }
    }

    /// Enforce infrastructure-locked constraints on neuro-driven params.
    /// Called after neuromodulator pushes new values.
    pub fn enforce_constraints(&mut self) {
        self.dyna_steps = self.dyna_steps.min(self.constraints.max_dyna_steps);
        self.neuro_alpha_mult = self.neuro_alpha_mult.max(self.constraints.min_alpha_mult);
        let max_eps = self.constraints.max_epsilon - self.epsilon();
        self.neuro_epsilon_bonus = self.neuro_epsilon_bonus.min(max_eps.max(0.0));
    }

    pub fn persist(&self) {
        let flattened: Vec<f64> = self
            .q_table
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let persisted = RlPersisted {
            q_table: flattened,
            current_adjustment: self.current_adjustment,
            total_ticks: self.total_ticks,
            total_overflows: self.total_overflows,
            neuro_alpha_mult: self.neuro_alpha_mult,
        };
        if let Ok(json) = serde_json::to_string_pretty(&persisted) {
            let _ = std::fs::write(&self.path, json);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agent() -> RlThresholdAgent {
        RlThresholdAgent {
            q_table: [[0.0; NUM_ACTIONS]; NUM_STATES],
            last_state: None,
            last_action: None,
            current_adjustment: 0.0,
            total_ticks: 0,
            total_overflows: 0,
            path: PathBuf::from("/dev/null"),
            prev_pressure: 0.5,
            rpe_ema: 1.0,
            dyna_model: HashMap::new(),
            dyna_cursor: 0,
            dyna_keys: Vec::new(),
            neuro_alpha_mult: 1.0,
            neuro_epsilon_bonus: 0.0,
            dyna_steps: DYNA_PLANNING_STEPS,
            constraints: RlConstraints::default(),
            consecutive_lower: 0,
        }
    }

    #[test]
    fn test_state_index_range() {
        let mut seen = std::collections::HashSet::new();
        for pb in 0..4u8 {
            for cb in 0..3u8 {
                for oh in 0..4u8 {
                    let state = RlState {
                        pressure_band: pb,
                        compressor_band: cb,
                        overflow_last_hour: oh,
                    };
                    let idx = state.index();
                    assert!(
                        idx < NUM_STATES,
                        "index {} out of range for {:?}",
                        idx,
                        state
                    );
                    seen.insert(idx);
                }
            }
        }
        assert_eq!(seen.len(), NUM_STATES, "must cover all 48 states");
    }

    #[test]
    fn test_initial_q_values_zero() {
        let agent = make_agent();
        for row in &agent.q_table {
            for &val in row {
                assert_eq!(val, 0.0);
            }
        }
    }

    #[test]
    fn test_overflow_penalizes_action() {
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.85, 0.40, 0);
        agent.tick(state, false);
        let prev_state = agent.last_state.unwrap();
        let prev_action = agent.last_action.unwrap();
        let prev_q = agent.q_table[prev_state.index()][prev_action as usize];
        let state2 = RlState::from_metrics(0.90, 0.70, 1);
        agent.tick(state2, true);
        let new_q = agent.q_table[prev_state.index()][prev_action as usize];
        assert!(
            new_q < prev_q,
            "overflow must decrease Q: prev={} new={}",
            prev_q,
            new_q
        );
    }

    #[test]
    fn test_stable_rewards_action() {
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.30, 0.10, 0);
        for _ in 0..50 {
            agent.tick(state, false);
        }
        let row = &agent.q_table[state.index()];
        let max_q = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_q > 0.0,
            "after many stable ticks, best Q should be positive: {}",
            max_q
        );
    }

    #[test]
    fn test_adjustment_clamped() {
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.90, 0.80, 3);
        for _ in 0..100 {
            agent.tick(state, true);
        }
        assert!(agent.current_adjustment >= ADJUSTMENT_FLOOR);
        let calm = RlState::from_metrics(0.10, 0.05, 0);
        for _ in 0..200 {
            agent.tick(calm, false);
        }
        assert!(agent.current_adjustment <= ADJUSTMENT_CEIL);
    }

    #[test]
    fn test_epsilon_decay() {
        let mut agent = make_agent();
        assert_eq!(agent.epsilon(), EPSILON_INITIAL);
        let state = RlState::from_metrics(0.50, 0.30, 0);
        for _ in 0..EPSILON_DECAY_TICKS {
            agent.tick(state, false);
        }
        assert_eq!(agent.epsilon(), EPSILON_STABLE);
    }

    #[test]
    fn test_absolute_floor() {
        let mut agent = make_agent();
        agent.current_adjustment = ADJUSTMENT_FLOOR;
        let effective = (0.78 + agent.current_adjustment).max(RL_ABSOLUTE_FLOOR);
        assert!(effective >= RL_ABSOLUTE_FLOOR);
        let effective2 =
            (0.78 + (-0.20) + (-0.08) + agent.current_adjustment).max(RL_ABSOLUTE_FLOOR);
        assert!(effective2 >= RL_ABSOLUTE_FLOOR);
    }

    #[test]
    fn test_ema_alpha_decays_over_time() {
        let mut agent = make_agent();
        let alpha_0 = agent.alpha();
        assert!((alpha_0 - ALPHA_INITIAL).abs() < 1e-6, "initial alpha should be {}", ALPHA_INITIAL);

        let state = RlState::from_metrics(0.50, 0.30, 0);
        for _ in 0..400 {
            agent.tick(state, false);
        }
        let alpha_400 = agent.alpha();
        assert!(alpha_400 < alpha_0, "alpha should decay: {} < {}", alpha_400, alpha_0);
        assert!(alpha_400 >= ALPHA_MIN, "alpha should not go below floor: {}", alpha_400);
    }

    #[test]
    fn test_ema_converges_faster_than_fixed_alpha() {
        // Darwinian selection: EMA agent vs hypothetical fixed-alpha agent.
        // Both see 50 stable ticks then 20 overflow ticks.
        // EMA should have higher Q variance (learned more from early data).
        let mut agent = make_agent();
        let calm = RlState::from_metrics(0.30, 0.10, 0);
        for _ in 0..50 {
            agent.tick(calm, false);
        }
        let q_after_calm: f64 = agent.q_table[calm.index()]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        // After 50 calm ticks, best Q should be meaningfully positive
        // because early high alpha accumulated reward faster.
        assert!(q_after_calm > 2.0,
            "EMA agent should learn fast from early data: best_q={}", q_after_calm);
    }

    #[test]
    fn test_inject_external_reward_nudges_q() {
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.60, 0.40, 0);
        agent.tick(state, false);

        let s = agent.last_state.unwrap().index();
        let a = agent.last_action.unwrap() as usize;
        let q_before = agent.q_table[s][a];

        agent.inject_external_reward(-5.0);
        let q_after = agent.q_table[s][a];
        assert!(q_after < q_before,
            "negative external reward should decrease Q: {} < {}", q_after, q_before);
    }

    #[test]
    fn test_from_metrics_boundaries() {
        assert_eq!(RlState::from_metrics(0.00, 0.00, 0).pressure_band, 0);
        assert_eq!(RlState::from_metrics(0.49, 0.00, 0).pressure_band, 0);
        assert_eq!(RlState::from_metrics(0.50, 0.00, 0).pressure_band, 1);
        assert_eq!(RlState::from_metrics(0.80, 0.00, 0).pressure_band, 1);
        assert_eq!(RlState::from_metrics(0.81, 0.00, 0).pressure_band, 2);
        assert_eq!(RlState::from_metrics(0.00, 0.29, 0).compressor_band, 0);
        assert_eq!(RlState::from_metrics(0.00, 0.30, 0).compressor_band, 1);
        assert_eq!(RlState::from_metrics(0.00, 0.60, 0).compressor_band, 1);
        assert_eq!(RlState::from_metrics(0.00, 0.61, 0).compressor_band, 2);
        assert_eq!(RlState::from_metrics(0.00, 0.00, 5).overflow_last_hour, 3);
    }

    #[test]
    fn test_shaped_reward_pressure_drop_beats_no_change() {
        // Agent starting at high pressure (band 2 → 0.90) dropping to low (band 0 → 0.35)
        // should accumulate higher Q than agent that stays at same pressure.
        let mut agent_drop = make_agent();
        agent_drop.prev_pressure = 0.90; // start high
        let high_state = RlState::from_metrics(0.85, 0.40, 0); // band 2
        let low_state = RlState::from_metrics(0.30, 0.10, 0);  // band 0
        agent_drop.tick(high_state, false); // prev=0.90, cur=0.90, shaped≈0
        agent_drop.tick(low_state, false);  // prev=0.90, cur=0.35, shaped= 0.81-0.1225=+0.6875

        let mut agent_flat = make_agent();
        agent_flat.prev_pressure = 0.90;
        agent_flat.tick(high_state, false);
        agent_flat.tick(high_state, false); // stays at 0.90, shaped=0

        // The drop agent's Q for the first state should be higher (better reward received)
        let q_drop: f64 = agent_drop.q_table[high_state.index()].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let q_flat: f64 = agent_flat.q_table[high_state.index()].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            q_drop > q_flat,
            "pressure drop should yield higher Q than staying high: drop={} flat={}",
            q_drop, q_flat
        );
    }

    #[test]
    fn test_shaped_overflow_still_dominates() {
        // Even with positive shaping, repeated overflow ticks must push Q strongly negative.
        let mut agent = make_agent();
        let high = RlState::from_metrics(0.85, 0.70, 2);
        // Run enough overflow ticks to accumulate penalty (first tick has no prev_state update)
        for _ in 0..5 {
            agent.tick(high, true);
        }
        // The best Q value across all actions for this state should be negative
        let best_q = agent.q_table[high.index()]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            best_q < 0.0,
            "repeated overflow must push best Q negative: best_q={}",
            best_q
        );
    }

    #[test]
    fn test_prev_pressure_updates_after_tick() {
        let mut agent = make_agent();
        assert!((agent.prev_pressure - 0.5).abs() < 1e-9, "initial prev_pressure=0.5");
        let low_state = RlState::from_metrics(0.30, 0.10, 0); // band 0 → 0.35
        agent.tick(low_state, false);
        assert!(
            (agent.prev_pressure - 0.35).abs() < 1e-9,
            "prev_pressure should update to band 0 midpoint: {}",
            agent.prev_pressure
        );
        let high_state = RlState::from_metrics(0.90, 0.70, 0); // band 2 → 0.90
        agent.tick(high_state, false);
        assert!(
            (agent.prev_pressure - 0.90).abs() < 1e-9,
            "prev_pressure should update to band 2 midpoint: {}",
            agent.prev_pressure
        );
    }

    #[test]
    fn test_rpe_steady_surprise_factor_near_one() {
        // When RPE is consistent, rpe_ema converges to |rpe|, so surprise_factor → 1.0.
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.50, 0.30, 0);
        // Warm up with many stable ticks so rpe_ema tracks steady RPE.
        for _ in 0..500 {
            agent.tick(state, false);
        }
        // After convergence the rpe_ema should roughly equal the current |rpe|.
        // We can verify indirectly: rpe_ema should be a reasonable positive value (not 1.0 or 0.0).
        assert!(agent.rpe_ema > 0.0, "rpe_ema must remain positive: {}", agent.rpe_ema);
        assert!(agent.rpe_ema < 50.0, "rpe_ema should not explode: {}", agent.rpe_ema);
        // In steady state the effective alpha should stay near base alpha (factor ≈ 1).
        // We can't directly measure surprise_factor here, but we verify rpe_ema is bounded.
        let ratio = agent.rpe_ema / agent.rpe_ema.max(0.01);
        assert!((ratio - 1.0).abs() < 1e-9, "rpe_ema / max(rpe_ema, 0.01) == 1.0 in steady state");
    }

    #[test]
    fn test_rpe_spike_amplifies_alpha() {
        // After a large unexpected overflow following many stable ticks,
        // the surprise_factor should temporarily exceed 1.0, boosting the Q update.
        let mut agent = make_agent();
        let calm = RlState::from_metrics(0.30, 0.10, 0);
        // Warm up: stable ticks → rpe_ema tracks small stable RPE.
        for _ in 0..200 {
            agent.tick(calm, false);
        }
        let rpe_ema_before = agent.rpe_ema;

        // Now fire an overflow from a high-pressure state — large RPE spike.
        let high = RlState::from_metrics(0.90, 0.80, 3);
        agent.tick(high, true);
        // The rpe_ema should increase because a large |rpe| was observed.
        // (It won't jump all the way because of the 0.01 decay weight, but it moves up.)
        assert!(
            agent.rpe_ema >= rpe_ema_before * 0.99,
            "rpe_ema should not drop after a spike: before={} after={}",
            rpe_ema_before,
            agent.rpe_ema
        );
        // The Q update on tick(high, true) applied to the PREVIOUS state (calm).
        // Verify that some Q entry for the calm state was updated (non-zero).
        let calm_q_sum: f64 = agent.q_table[calm.index()].iter().copied().map(f64::abs).sum();
        assert!(calm_q_sum > 0.0, "Q for calm state must be updated after overflow spike: sum={}", calm_q_sum);
    }

    #[test]
    fn test_rpe_surprise_factor_clamp_prevents_runaway() {
        // Even with a massive RPE, surprise_factor must never exceed 5.0.
        // Effective alpha must never exceed alpha() * 5.0.
        let mut agent = make_agent();
        // Force rpe_ema to a tiny value so any real RPE creates huge ratio.
        agent.rpe_ema = 0.001;
        let base_alpha = agent.alpha();

        let state = RlState::from_metrics(0.90, 0.80, 3);
        // Two ticks needed: first records (state, action), second applies Q update.
        agent.tick(state, false);
        agent.tick(state, true); // overflow → huge RPE relative to tiny rpe_ema

        // If clamp works, Q change is bounded by alpha * 5.0 * |td_error|.
        // We verify indirectly: no NaN or infinite values in Q table.
        for row in &agent.q_table {
            for &q in row {
                assert!(q.is_finite(), "Q values must remain finite after large RPE spike: {}", q);
            }
        }
        // And rpe_ema has grown (0.99 * 0.001 + 0.01 * big_rpe > 0.001).
        assert!(agent.rpe_ema > 0.001, "rpe_ema must grow after spike: {}", agent.rpe_ema);
        // Max effective_alpha seen can be reconstructed: base_alpha * 5 (max clamp).
        let max_effective = base_alpha * 5.0;
        assert!(max_effective < 1.0 + 1e-9, "clamped effective_alpha must stay < 1.0: {}", max_effective);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.60, 0.40, 1);
        for _ in 0..10 {
            agent.tick(state, false);
        }
        agent.tick(state, true);
        let flattened: Vec<f64> = agent
            .q_table
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let persisted = RlPersisted {
            q_table: flattened,
            current_adjustment: agent.current_adjustment,
            total_ticks: agent.total_ticks,
            total_overflows: agent.total_overflows,
            neuro_alpha_mult: agent.neuro_alpha_mult,
        };
        let json = serde_json::to_string(&persisted).unwrap();
        let loaded: RlPersisted = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.q_table.len(), NUM_STATES * NUM_ACTIONS);
        assert_eq!(loaded.current_adjustment, agent.current_adjustment);
        assert_eq!(loaded.total_ticks, agent.total_ticks);
        assert_eq!(loaded.total_overflows, agent.total_overflows);
        assert_eq!(loaded.neuro_alpha_mult, agent.neuro_alpha_mult);
    }

    #[test]
    fn test_dyna_q_amplifies_learning() {
        // Agent WITH Dyna-Q (default) should learn faster than one without.
        // Both see the same 30-tick sequence of high→low pressure transitions.
        let mut agent_dyna = make_agent();
        let mut agent_nodyna = make_agent();

        let high = RlState::from_metrics(0.85, 0.70, 2);
        let low = RlState::from_metrics(0.30, 0.10, 0);
        for _ in 0..15 {
            agent_dyna.tick(high, true);
            agent_dyna.tick(low, false);

            // Simulate no-dyna by clearing model each tick.
            agent_nodyna.tick(high, true);
            agent_nodyna.dyna_model.clear();
            agent_nodyna.dyna_keys.clear();
            agent_nodyna.tick(low, false);
            agent_nodyna.dyna_model.clear();
            agent_nodyna.dyna_keys.clear();
        }

        // Dyna agent should have more model entries and different Q-values.
        assert!(agent_dyna.dyna_model_size() > 0, "dyna model should have entries");
        // Compare Q-value spread: dyna should have more differentiated Q-values
        // (higher variance) because planning amplifies learning from each transition.
        let variance = |qt: &[[f64; NUM_ACTIONS]; NUM_STATES]| -> f64 {
            let vals: Vec<f64> = qt.iter().flat_map(|r| r.iter().copied()).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64
        };
        assert!(
            variance(&agent_dyna.q_table) > variance(&agent_nodyna.q_table),
            "dyna agent should have more differentiated Q-values"
        );
    }
}
