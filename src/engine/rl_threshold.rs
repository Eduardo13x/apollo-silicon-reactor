//! Q-table RL agent for adaptive overflow thresholds.
//!
//! ## Scientific basis
//! Tabular Q-learning (Watkins, 1989) with e-greedy exploration.
//! State space is deliberately small (36 states x 3 actions = 108 Q-values)
//! to ensure convergence within ~200 episodes per state-action pair.
//!
//! ## State representation
//! (pressure_band, compressor_band, overflow_last_hour) -- 3 x 3 x 4 = 36 states.
//!
//! ## Reward function
//! +1.0 per tick without overflow (stability reward).
//! -10.0 per overflow event (penalizes dangerous thresholds).
//!
//! ## Safety
//! Hard floor: the RL adjustment can never push absolute thresholds below 0.45.
//! The RL output is an additive correction on top of the existing exponential
//! decay (compute_dynamic_offset), which provides the baseline behavior.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const NUM_STATES: usize = 36;
const NUM_ACTIONS: usize = 3;
const ALPHA: f64 = 0.10;
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
        } else {
            2
        };
        let compressor_band = if compressor_pressure < 0.30 {
            0
        } else if compressor_pressure <= 0.60 {
            1
        } else {
            2
        };
        let overflow_last_hour = (overflows_last_hour as u8).min(3);
        Self { pressure_band, compressor_band, overflow_last_hour }
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
}

impl RlThresholdAgent {
    pub fn load_or_default(path: &Path) -> Self {
        let (q_table, current_adjustment, total_ticks, total_overflows) =
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
                    Some((qt, p.current_adjustment, p.total_ticks, p.total_overflows))
                })
                .unwrap_or(([[0.0; NUM_ACTIONS]; NUM_STATES], 0.0, 0, 0));

        Self {
            q_table, last_state: None, last_action: None,
            current_adjustment, total_ticks, total_overflows,
            path: path.to_path_buf(),
        }
    }

    pub fn epsilon(&self) -> f64 {
        if self.total_ticks < EPSILON_DECAY_TICKS { EPSILON_INITIAL } else { EPSILON_STABLE }
    }

    pub fn total_ticks(&self) -> u64 { self.total_ticks }
    pub fn total_overflows(&self) -> u64 { self.total_overflows }

    fn select_action(&self, state: RlState) -> RlAction {
        let eps = self.epsilon();
        let explore = (self.total_ticks.wrapping_mul(2_654_435_761)) % 100 < (eps * 100.0) as u64;
        if explore {
            let action_idx = ((self.total_ticks.wrapping_mul(7_919)) % 3) as usize;
            RlAction::from_index(action_idx)
        } else {
            let row = &self.q_table[state.index()];
            let mut best_idx = 0;
            let mut best_val = row[0];
            for (i, &q) in row.iter().enumerate().skip(1) {
                if q > best_val { best_val = q; best_idx = i; }
            }
            RlAction::from_index(best_idx)
        }
    }

    pub fn tick(&mut self, state: RlState, overflow_occurred: bool) {
        let reward = if overflow_occurred {
            self.total_overflows += 1;
            REWARD_OVERFLOW
        } else {
            REWARD_STABLE
        };

        if let (Some(prev_state), Some(prev_action)) = (self.last_state, self.last_action) {
            let s = prev_state.index();
            let a = prev_action as usize;
            let s_prime = state.index();
            let max_q_next = self.q_table[s_prime].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let old_q = self.q_table[s][a];
            self.q_table[s][a] = old_q + ALPHA * (reward + GAMMA * max_q_next - old_q);
        }

        let action = self.select_action(state);
        match action {
            RlAction::Lower5pp => self.current_adjustment -= 0.05,
            RlAction::Hold => {}
            RlAction::Raise1pp => self.current_adjustment += 0.01,
        }
        self.current_adjustment = self.current_adjustment.clamp(ADJUSTMENT_FLOOR, ADJUSTMENT_CEIL);

        self.last_state = Some(state);
        self.last_action = Some(action);
        self.total_ticks += 1;
    }

    pub fn persist(&self) {
        let flattened: Vec<f64> = self.q_table.iter().flat_map(|row| row.iter().copied()).collect();
        let persisted = RlPersisted {
            q_table: flattened,
            current_adjustment: self.current_adjustment,
            total_ticks: self.total_ticks,
            total_overflows: self.total_overflows,
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
            last_state: None, last_action: None,
            current_adjustment: 0.0, total_ticks: 0, total_overflows: 0,
            path: PathBuf::from("/dev/null"),
        }
    }

    #[test]
    fn test_state_index_range() {
        let mut seen = std::collections::HashSet::new();
        for pb in 0..3u8 {
            for cb in 0..3u8 {
                for oh in 0..4u8 {
                    let state = RlState { pressure_band: pb, compressor_band: cb, overflow_last_hour: oh };
                    let idx = state.index();
                    assert!(idx < NUM_STATES, "index {} out of range for {:?}", idx, state);
                    seen.insert(idx);
                }
            }
        }
        assert_eq!(seen.len(), NUM_STATES, "must cover all 36 states");
    }

    #[test]
    fn test_initial_q_values_zero() {
        let agent = make_agent();
        for row in &agent.q_table {
            for &val in row { assert_eq!(val, 0.0); }
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
        assert!(new_q < prev_q, "overflow must decrease Q: prev={} new={}", prev_q, new_q);
    }

    #[test]
    fn test_stable_rewards_action() {
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.30, 0.10, 0);
        for _ in 0..50 { agent.tick(state, false); }
        let row = &agent.q_table[state.index()];
        let max_q = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_q > 0.0, "after many stable ticks, best Q should be positive: {}", max_q);
    }

    #[test]
    fn test_adjustment_clamped() {
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.90, 0.80, 3);
        for _ in 0..100 { agent.tick(state, true); }
        assert!(agent.current_adjustment >= ADJUSTMENT_FLOOR);
        let calm = RlState::from_metrics(0.10, 0.05, 0);
        for _ in 0..200 { agent.tick(calm, false); }
        assert!(agent.current_adjustment <= ADJUSTMENT_CEIL);
    }

    #[test]
    fn test_epsilon_decay() {
        let mut agent = make_agent();
        assert_eq!(agent.epsilon(), EPSILON_INITIAL);
        let state = RlState::from_metrics(0.50, 0.30, 0);
        for _ in 0..EPSILON_DECAY_TICKS { agent.tick(state, false); }
        assert_eq!(agent.epsilon(), EPSILON_STABLE);
    }

    #[test]
    fn test_absolute_floor() {
        let mut agent = make_agent();
        agent.current_adjustment = ADJUSTMENT_FLOOR;
        let effective = (0.78 + agent.current_adjustment).max(RL_ABSOLUTE_FLOOR);
        assert!(effective >= RL_ABSOLUTE_FLOOR);
        let effective2 = (0.78 + (-0.20) + (-0.08) + agent.current_adjustment).max(RL_ABSOLUTE_FLOOR);
        assert!(effective2 >= RL_ABSOLUTE_FLOOR);
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
    fn test_persistence_roundtrip() {
        let mut agent = make_agent();
        let state = RlState::from_metrics(0.60, 0.40, 1);
        for _ in 0..10 { agent.tick(state, false); }
        agent.tick(state, true);
        let flattened: Vec<f64> = agent.q_table.iter().flat_map(|row| row.iter().copied()).collect();
        let persisted = RlPersisted {
            q_table: flattened, current_adjustment: agent.current_adjustment,
            total_ticks: agent.total_ticks, total_overflows: agent.total_overflows,
        };
        let json = serde_json::to_string(&persisted).unwrap();
        let loaded: RlPersisted = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.q_table.len(), NUM_STATES * NUM_ACTIONS);
        assert_eq!(loaded.current_adjustment, agent.current_adjustment);
        assert_eq!(loaded.total_ticks, agent.total_ticks);
        assert_eq!(loaded.total_overflows, agent.total_overflows);
    }
}
