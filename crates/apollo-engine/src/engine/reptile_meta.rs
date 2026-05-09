//! Reptile Meta-Learning — fast adaptation across workload fingerprints.
//!
//! ## Problem solved
//! When the workload type changes (dev → LLM, browser → build), all learning
//! agents start from scratch. Previously learned parameters for that workload
//! type are lost.
//!
//! ## Design: Reptile [Nichol 2018]
//! Maintain θ_slow (global) + θ_fast (per workload fingerprint).
//! On workload change: interpolate θ_current = θ_slow + 0.5×(θ_fast[new] - θ_slow).
//! After learning: θ_slow ← θ_slow + ε×(θ_current - θ_slow).
//!
//! This is first-order meta-learning (no second derivatives like MAML),
//! making it feasible for our 48-state Q-table + 5-arm LinUCB.
//!
//! ## References
//! - [Nichol 2018] "On First-Order Meta-Learning Algorithms" arXiv:1803.02999
//! - [Finn 2017] "Model-Agnostic Meta-Learning" ICML (MAML — Reptile simplifies this)

use std::collections::HashMap;

use super::neon_ema::ema_f32;

use serde::{Deserialize, Serialize};

/// Reptile outer-loop learning rate.
const META_LR: f64 = 0.01;

/// Interpolation factor when switching to a known workload fingerprint.
const INTERPOLATION_FACTOR: f64 = 0.5;

/// Maximum number of workload-specific parameter sets to cache.
const MAX_WORKLOAD_CACHE: usize = 16;

/// Stale threshold: if a workload's params haven't been updated in this many cycles,
/// decay them toward θ_slow.
const STALE_THRESHOLD: u64 = 10_000;

/// Number of RL Q-table states.
const RL_STATES: usize = 48;

/// Number of LinUCB arms.
const LINUCB_ARMS: usize = 5;

// ── Types ──────────────────────────────────────────────────────────────────────

/// Compact meta-parameters that can be quickly adapted per workload.
///
/// These are *biases* applied on top of the base agent parameters,
/// not full parameter copies (keeps memory small).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaParams {
    /// Bias on RL Q-table values per state (additive correction).
    pub rl_q_bias: Vec<f64>,
    /// Bias on LinUCB arm scores (additive correction).
    pub linucb_arm_biases: Vec<f64>,
    /// NARS confidence floor adjustment (additive, clamped to [0, 0.30]).
    pub nars_confidence_adj: f32,
    /// Last cycle this was updated.
    pub last_updated: u64,
}

impl Default for MetaParams {
    fn default() -> Self {
        Self {
            rl_q_bias: vec![0.0; RL_STATES],
            linucb_arm_biases: vec![0.0; LINUCB_ARMS],
            nars_confidence_adj: 0.0,
            last_updated: 0,
        }
    }
}

impl MetaParams {
    /// L2 distance between two parameter sets (for measuring adaptation).
    fn distance(&self, other: &MetaParams) -> f64 {
        let rl_dist: f64 = self
            .rl_q_bias
            .iter()
            .zip(other.rl_q_bias.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let linucb_dist: f64 = self
            .linucb_arm_biases
            .iter()
            .zip(other.linucb_arm_biases.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let nars_dist = (self.nars_confidence_adj - other.nars_confidence_adj).powi(2) as f64;
        (rl_dist + linucb_dist + nars_dist).sqrt()
    }

    /// Interpolate between self and other: result = self + factor * (other - self).
    fn interpolate(&self, other: &MetaParams, factor: f64) -> MetaParams {
        let rl_q_bias: Vec<f64> = self
            .rl_q_bias
            .iter()
            .zip(other.rl_q_bias.iter())
            .map(|(a, b)| a + factor * (b - a))
            .collect();
        let linucb_arm_biases: Vec<f64> = self
            .linucb_arm_biases
            .iter()
            .zip(other.linucb_arm_biases.iter())
            .map(|(a, b)| a + factor * (b - a))
            .collect();
        let nars_confidence_adj = self.nars_confidence_adj
            + (factor as f32) * (other.nars_confidence_adj - self.nars_confidence_adj);
        MetaParams {
            rl_q_bias,
            linucb_arm_biases,
            nars_confidence_adj: nars_confidence_adj.clamp(-0.15, 0.30),
            last_updated: other.last_updated,
        }
    }
}

/// Reptile meta-learner — manages slow/fast parameter adaptation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReptileMeta {
    /// θ_slow: global meta-parameters learned across all workloads.
    pub global_params: MetaParams,
    /// θ_fast: per-workload parameter cache.
    workload_params: HashMap<u64, MetaParams>,
    /// Current active workload fingerprint.
    current_fingerprint: u64,
    /// Current working parameters (what's actually applied).
    pub current_params: MetaParams,
    /// Number of Reptile outer-loop updates performed.
    pub adaptation_steps: u64,
    /// EMA of adaptation distance (how much θ_current moves per switch).
    pub adaptation_quality: f32,
}

impl Default for ReptileMeta {
    fn default() -> Self {
        Self::new()
    }
}

impl ReptileMeta {
    pub fn new() -> Self {
        Self {
            global_params: MetaParams::default(),
            workload_params: HashMap::new(),
            current_fingerprint: 0,
            current_params: MetaParams::default(),
            adaptation_steps: 0,
            adaptation_quality: 0.0,
        }
    }

    /// Notify that the workload fingerprint has changed.
    ///
    /// Saves current params for old workload, loads (or interpolates) params
    /// for new workload, performs Reptile outer-loop update on θ_slow.
    pub fn on_fingerprint_change(&mut self, new_fingerprint: u64, current_cycle: u64) {
        if new_fingerprint == self.current_fingerprint {
            return;
        }

        // Save current params for old workload
        let mut saved = self.current_params.clone();
        saved.last_updated = current_cycle;
        self.workload_params.insert(self.current_fingerprint, saved);

        // Reptile outer-loop: θ_slow ← θ_slow + ε×(θ_current - θ_slow)
        // [Nichol 2018] Algorithm 1
        self.global_params = self
            .global_params
            .interpolate(&self.current_params, META_LR);
        self.adaptation_steps += 1;

        // Load or interpolate params for new workload
        self.current_params = if let Some(cached) = self.workload_params.get(&new_fingerprint) {
            // Known workload: interpolate between θ_slow and θ_fast[new]
            let distance_before = self.global_params.distance(cached);
            let result = self.global_params.interpolate(cached, INTERPOLATION_FACTOR);
            self.adaptation_quality = ema_f32(
                self.adaptation_quality,
                (1.0 - (distance_before as f32 / 5.0).min(1.0)).max(0.0),
                0.1,
            );
            result
        } else {
            // Unknown workload: use θ_slow (warm start from global experience)
            self.adaptation_quality = ema_f32(self.adaptation_quality, 0.5, 0.1);
            self.global_params.clone()
        };
        self.current_params.last_updated = current_cycle;

        self.current_fingerprint = new_fingerprint;

        // Enforce cache size limit (evict least recently updated)
        while self.workload_params.len() > MAX_WORKLOAD_CACHE {
            if let Some((&oldest_key, _)) = self
                .workload_params
                .iter()
                .min_by_key(|(_, v)| v.last_updated)
            {
                self.workload_params.remove(&oldest_key);
            } else {
                break;
            }
        }
    }

    /// Update current_params with new learning from this cycle.
    ///
    /// Called each cycle with the delta from RL/LinUCB updates.
    pub fn apply_learning_delta(
        &mut self,
        rl_state_idx: usize,
        rl_q_delta: f64,
        linucb_arm_idx: usize,
        linucb_delta: f64,
        current_cycle: u64,
    ) {
        if rl_state_idx < self.current_params.rl_q_bias.len() {
            self.current_params.rl_q_bias[rl_state_idx] += rl_q_delta * 0.1;
        }
        if linucb_arm_idx < self.current_params.linucb_arm_biases.len() {
            self.current_params.linucb_arm_biases[linucb_arm_idx] += linucb_delta * 0.1;
        }
        self.current_params.last_updated = current_cycle;
    }

    /// Get RL Q-bias for a specific state.
    pub fn rl_bias(&self, state_idx: usize) -> f64 {
        self.current_params
            .rl_q_bias
            .get(state_idx)
            .copied()
            .unwrap_or(0.0)
    }

    /// Get LinUCB arm bias for a specific arm.
    pub fn linucb_bias(&self, arm_idx: usize) -> f64 {
        self.current_params
            .linucb_arm_biases
            .get(arm_idx)
            .copied()
            .unwrap_or(0.0)
    }

    /// Get NARS confidence floor adjustment.
    pub fn nars_confidence_adjustment(&self) -> f32 {
        self.current_params.nars_confidence_adj
    }

    /// Prune stale workload params that haven't been used recently.
    pub fn prune_stale(&mut self, current_cycle: u64) {
        self.workload_params.retain(|_, params| {
            if current_cycle.saturating_sub(params.last_updated) > STALE_THRESHOLD {
                // Decay toward global before removing (preserve some signal)
                false
            } else {
                true
            }
        });
    }

    /// Number of cached workload fingerprints.
    pub fn cached_workloads(&self) -> usize {
        self.workload_params.len()
    }

    /// Current active fingerprint.
    pub fn current_fingerprint(&self) -> u64 {
        self.current_fingerprint
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_defaults() {
        let rm = ReptileMeta::new();
        assert_eq!(rm.adaptation_steps, 0);
        assert_eq!(rm.cached_workloads(), 0);
        assert_eq!(rm.current_fingerprint(), 0);
    }

    #[test]
    fn test_fingerprint_change_saves_old() {
        let mut rm = ReptileMeta::new();
        rm.on_fingerprint_change(42, 100);
        // Old fingerprint 0 should be cached
        assert_eq!(rm.cached_workloads(), 1);
        assert_eq!(rm.current_fingerprint(), 42);
        assert_eq!(rm.adaptation_steps, 1);
    }

    #[test]
    fn test_same_fingerprint_is_noop() {
        let mut rm = ReptileMeta::new();
        rm.on_fingerprint_change(42, 100);
        let steps_before = rm.adaptation_steps;
        rm.on_fingerprint_change(42, 200);
        assert_eq!(
            rm.adaptation_steps, steps_before,
            "Same fingerprint = no update"
        );
    }

    #[test]
    fn test_known_workload_interpolates() {
        let mut rm = ReptileMeta::new();
        // Visit workload A, learn something
        rm.on_fingerprint_change(1, 10);
        rm.apply_learning_delta(0, 5.0, 0, 3.0, 20);

        // Switch to B
        rm.on_fingerprint_change(2, 30);

        // Switch back to A — should interpolate with cached params
        rm.on_fingerprint_change(1, 50);
        assert!(
            rm.rl_bias(0).abs() > 0.0,
            "Should have nonzero bias from cache"
        );
    }

    #[test]
    fn test_unknown_workload_uses_global() {
        let mut rm = ReptileMeta::new();
        rm.on_fingerprint_change(1, 10);
        rm.apply_learning_delta(5, 10.0, 2, 5.0, 20);
        rm.on_fingerprint_change(2, 30);

        // Visit brand new workload 999
        rm.on_fingerprint_change(999, 50);
        // Should use global params (warm start), which has some signal from Reptile update
        // Global is updated via META_LR=0.01, so it will have tiny bias
        assert!(rm.rl_bias(5).abs() < 1.0, "Global has small signal");
    }

    #[test]
    fn test_reptile_outer_loop_updates_global() {
        let mut rm = ReptileMeta::new();
        // Learn a lot in workload 1
        rm.on_fingerprint_change(1, 10);
        for i in 0..RL_STATES {
            rm.current_params.rl_q_bias[i] = 10.0;
        }

        // Switch workload — triggers Reptile update
        rm.on_fingerprint_change(2, 100);

        // Global should have moved slightly toward 10.0 (by META_LR=0.01)
        let global_bias = rm.global_params.rl_q_bias[0];
        assert!(
            global_bias > 0.05 && global_bias < 0.5,
            "Global should move toward 10.0 by META_LR: {global_bias}"
        );
    }

    #[test]
    fn test_apply_learning_delta() {
        let mut rm = ReptileMeta::new();
        rm.apply_learning_delta(10, 2.0, 3, 1.5, 50);
        assert!((rm.rl_bias(10) - 0.2).abs() < 0.001, "delta * 0.1 = 0.2");
        assert!(
            (rm.linucb_bias(3) - 0.15).abs() < 0.001,
            "delta * 0.1 = 0.15"
        );
    }

    #[test]
    fn test_rl_bias_out_of_bounds() {
        let rm = ReptileMeta::new();
        assert_eq!(rm.rl_bias(9999), 0.0, "Out of bounds = 0.0");
    }

    #[test]
    fn test_linucb_bias_out_of_bounds() {
        let rm = ReptileMeta::new();
        assert_eq!(rm.linucb_bias(9999), 0.0, "Out of bounds = 0.0");
    }

    #[test]
    fn test_cache_eviction() {
        let mut rm = ReptileMeta::new();
        for i in 0..MAX_WORKLOAD_CACHE + 5 {
            rm.on_fingerprint_change(i as u64 + 1, i as u64 * 10);
        }
        assert!(rm.cached_workloads() <= MAX_WORKLOAD_CACHE);
    }

    #[test]
    fn test_prune_stale() {
        let mut rm = ReptileMeta::new();
        rm.on_fingerprint_change(1, 10);
        rm.on_fingerprint_change(2, 20);
        assert_eq!(rm.cached_workloads(), 2);

        // Prune with current_cycle way beyond stale threshold
        rm.prune_stale(STALE_THRESHOLD + 100);
        assert_eq!(rm.cached_workloads(), 0, "All should be pruned");
    }

    #[test]
    fn test_meta_params_distance() {
        let a = MetaParams::default();
        let mut b = MetaParams::default();
        b.rl_q_bias[0] = 3.0;
        b.rl_q_bias[1] = 4.0;
        let dist = a.distance(&b);
        assert!((dist - 5.0).abs() < 0.01, "3-4-5 triangle: {dist}");
    }

    #[test]
    fn test_meta_params_interpolate() {
        let a = MetaParams::default();
        let mut b = MetaParams::default();
        b.rl_q_bias[0] = 10.0;
        b.linucb_arm_biases[0] = 4.0;

        let mid = a.interpolate(&b, 0.5);
        assert!((mid.rl_q_bias[0] - 5.0).abs() < 0.01);
        assert!((mid.linucb_arm_biases[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_nars_confidence_adj_clamped() {
        let mut a = MetaParams::default();
        a.nars_confidence_adj = -0.10;
        let mut b = MetaParams::default();
        b.nars_confidence_adj = 1.0;

        let result = a.interpolate(&b, 1.0);
        assert!(result.nars_confidence_adj <= 0.30, "Should clamp to 0.30");
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut rm = ReptileMeta::new();
        rm.on_fingerprint_change(42, 100);
        rm.apply_learning_delta(5, 3.0, 2, 1.0, 110);

        let json = serde_json::to_string(&rm).expect("serialize");
        let restored: ReptileMeta = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.adaptation_steps, rm.adaptation_steps);
        assert_eq!(restored.current_fingerprint(), rm.current_fingerprint());
        assert!((restored.rl_bias(5) - rm.rl_bias(5)).abs() < 1e-10);
    }
}
