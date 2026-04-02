//! Causal Graph — learn cause-effect relationships between actions and outcomes.
//!
//! Adapted from memoria-core/src/cognitive_core/causal_inference.rs.
//! Original: DAG-based causal learning with DashMap for concurrent access.
//! Apollo version: single-threaded HashMap, optimized for daemon hot path.
//!
//! Key insight: correlation ≠ causation. Apollo throttles hundreds of processes
//! but only some actually reduce memory pressure. The causal graph tracks:
//!   "throttle:Firefox" → "pressure_drop" with confidence 0.85 (47 observations)
//!   "throttle:contactsd" → "pressure_drop" with confidence 0.12 (30 observations)
//!
//! This feeds back into decide_actions: processes with solid causal links to
//! pressure reduction get throttled first, wasting fewer cycles on ineffective actions.
//!
//! References:
//! - Pearl (2009) "Causality: Models, Reasoning and Inference"
//! - memoria-core causal_inference.rs (constraint-based inference)

use std::collections::HashMap;

/// A causal edge: action X caused outcome Y with measured confidence.
#[derive(Clone, Debug)]
pub struct CausalEdge {
    /// Cause (e.g., "throttle:Safari", "freeze:Dropbox").
    pub cause: String,
    /// Effect (e.g., "pressure_drop", "pressure_unchanged").
    pub effect: String,
    /// Bayesian confidence [0, 1]. Updated with each observation.
    pub confidence: f32,
    /// Total observations supporting or refuting this edge.
    pub evidence_count: u32,
    /// Typical latency in cycles between cause and observed effect.
    pub latency_cycles: u8,
    /// EMA of actual pressure delta when this edge fired (effective observations only).
    /// Captures HOW MUCH pressure dropped, not just WHETHER it dropped.
    /// Range: 0.0–1.0. Init: 0.0 (no observations yet).
    pub avg_delta: f32,
}

impl CausalEdge {
    fn new(cause: &str, effect: &str) -> Self {
        Self {
            cause: cause.to_string(),
            effect: effect.to_string(),
            confidence: 0.5, // uninformed prior
            evidence_count: 0,
            latency_cycles: 3, // default: expect effect within 3 cycles
            avg_delta: 0.0,
        }
    }

    /// Bayesian update: blend new evidence into confidence.
    /// When effective, also track the magnitude of the pressure delta.
    #[allow(dead_code)]
    fn update(&mut self, was_effective: bool) {
        self.update_with_delta(was_effective, 0.0);
    }

    /// Bayesian update with observed pressure delta magnitude.
    fn update_with_delta(&mut self, was_effective: bool, delta: f32) {
        self.evidence_count += 1;
        let target = if was_effective { 1.0 } else { 0.0 };
        self.confidence = self.confidence * 0.9 + target * 0.1;
        // Track average delta only when effective (delta > 0).
        if was_effective && delta > 0.0 {
            // EMA alpha=0.15: adapts to changing workload patterns.
            self.avg_delta = self.avg_delta * 0.85 + delta * 0.15;
        }
    }

    /// Impact score: confidence × avg_delta. Ranks edges by real-world effect.
    /// A solid edge with 0.80 confidence and 0.10 avg drop scores higher
    /// than one with 0.90 confidence but only 0.02 avg drop.
    pub fn impact_score(&self) -> f32 {
        self.confidence * self.avg_delta
    }

    /// Edge is solid: high confidence with sufficient evidence.
    pub fn is_solid(&self) -> bool {
        self.confidence > 0.7 && self.evidence_count >= 5
    }

    /// Edge is weak: low confidence despite sufficient evidence.
    pub fn is_weak(&self) -> bool {
        self.confidence < 0.25 && self.evidence_count >= 5
    }
}

/// Pending action waiting for outcome observation.
#[derive(Clone)]
struct PendingAction {
    /// Process or group that was acted on.
    action_key: String,
    /// Memory pressure at the time of action.
    pressure_at_action: f32,
    /// Cycle when the action was taken.
    cycle: u64,
}

/// Causal graph tracking action → outcome relationships.
pub struct CausalGraph {
    /// Directed edges: (cause, effect) → CausalEdge.
    edges: HashMap<(String, String), CausalEdge>,
    /// Actions waiting for outcome evaluation.
    pending: Vec<PendingAction>,
    /// Cycles to wait before evaluating outcome.
    eval_delay: u8,
}

const EFFECT_PRESSURE_DROP: &str = "pressure_drop";
const EFFECT_PRESSURE_UNCHANGED: &str = "pressure_no_change";
/// Minimum pressure delta to count as a "drop".
const MIN_DELTA: f32 = 0.02;

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            pending: Vec::new(),
            eval_delay: 3,
        }
    }

    /// Record that an action was taken on a process/group.
    /// Called after execute_actions with the names of throttled/frozen processes.
    pub fn record_action(&mut self, action_key: &str, pressure: f32, cycle: u64) {
        self.pending.push(PendingAction {
            action_key: action_key.to_string(),
            pressure_at_action: pressure,
            cycle,
        });
        // Cap pending queue to avoid unbounded growth.
        if self.pending.len() > 200 {
            self.pending.drain(..100);
        }
    }

    /// Evaluate pending actions against current pressure.
    /// Called each cycle — checks actions that are old enough for evaluation.
    pub fn evaluate(&mut self, current_pressure: f32, current_cycle: u64) {
        let delay = self.eval_delay as u64;
        let mut i = 0;
        while i < self.pending.len() {
            if current_cycle.saturating_sub(self.pending[i].cycle) >= delay {
                // Move the item out (swap_remove avoids shifting elements).
                let pending = self.pending.swap_remove(i);
                let delta = pending.pressure_at_action - current_pressure;
                let was_effective = delta >= MIN_DELTA;

                let (effect, anti_effect) = if was_effective {
                    (EFFECT_PRESSURE_DROP, EFFECT_PRESSURE_UNCHANGED)
                } else {
                    (EFFECT_PRESSURE_UNCHANGED, EFFECT_PRESSURE_DROP)
                };

                // Update causal edge for this action → pressure outcome.
                let key = (pending.action_key.clone(), effect.to_string());
                self.edges
                    .entry(key)
                    .or_insert_with(|| CausalEdge::new(&pending.action_key, effect))
                    .update_with_delta(true, delta.max(0.0));

                // Also record the complementary edge (non-event).
                // Move pending.action_key into anti_key — no second clone.
                // or_insert_with_key passes &key to the closure when a new edge is needed.
                let anti_key = (pending.action_key, anti_effect.to_string());
                self.edges
                    .entry(anti_key)
                    .or_insert_with_key(|k| CausalEdge::new(&k.0, anti_effect))
                    .update_with_delta(false, 0.0);
                // Don't increment i: swap_remove placed a new element at position i.
            } else {
                i += 1;
            }
        }
    }

    /// Get a specific causal edge if it exists.
    pub fn get_edge(&self, cause: &str, effect: &str) -> Option<&CausalEdge> {
        let key = (cause.to_string(), effect.to_string());
        self.edges.get(&key)
    }

    /// Get the causal effectiveness of an action (confidence in causing pressure_drop).
    /// Returns None if not enough evidence.
    pub fn effectiveness(&self, action_key: &str) -> Option<f32> {
        let key = (action_key.to_string(), EFFECT_PRESSURE_DROP.to_string());
        self.edges.get(&key).and_then(|e| {
            if e.evidence_count >= 3 {
                Some(e.confidence)
            } else {
                None
            }
        })
    }

    /// Get all solid edges (high confidence, sufficient evidence).
    pub fn solid_edges(&self) -> Vec<&CausalEdge> {
        self.edges.values().filter(|e| e.is_solid()).collect()
    }

    /// Solid edges sorted by impact_score (confidence × avg_delta), highest first.
    /// Use this when prioritizing which actions to try — prefers actions that
    /// both reliably work AND produce large pressure reductions.
    pub fn solid_edges_by_impact(&self) -> Vec<&CausalEdge> {
        let mut edges: Vec<&CausalEdge> = self.edges.values().filter(|e| e.is_solid()).collect();
        edges.sort_by(|a, b| b.impact_score().partial_cmp(&a.impact_score()).unwrap_or(std::cmp::Ordering::Equal));
        edges
    }

    /// Get all weak edges (low confidence despite evidence).
    pub fn weak_edges(&self) -> Vec<&CausalEdge> {
        self.edges.values().filter(|e| e.is_weak()).collect()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of solid causal links discovered.
    pub fn solid_count(&self) -> usize {
        self.edges.values().filter(|e| e.is_solid()).count()
    }

    /// Build a map of action_key → causal_confidence for use in decide_actions.
    /// Only includes actions with ≥5 evidence observations.
    pub fn confidence_map(&self) -> HashMap<String, f32> {
        let mut map = HashMap::new();
        for ((action_key, effect), edge) in &self.edges {
            if effect == EFFECT_PRESSURE_DROP && edge.evidence_count >= 5 {
                map.insert(action_key.clone(), edge.confidence);
            }
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_edge_uninformed() {
        let e = CausalEdge::new("throttle:Safari", "pressure_drop");
        assert_eq!(e.confidence, 0.5);
        assert_eq!(e.evidence_count, 0);
        assert!(!e.is_solid());
    }

    #[test]
    fn test_edge_becomes_solid() {
        let mut e = CausalEdge::new("throttle:Dropbox", "pressure_drop");
        for _ in 0..20 {
            e.update(true);
        }
        assert!(e.confidence > 0.7);
        assert!(e.is_solid());
    }

    #[test]
    fn test_edge_becomes_weak() {
        let mut e = CausalEdge::new("throttle:contactsd", "pressure_drop");
        for _ in 0..20 {
            e.update(false);
        }
        assert!(e.confidence < 0.25);
        assert!(e.is_weak());
    }

    #[test]
    fn test_record_and_evaluate_effective() {
        let mut g = CausalGraph::new();
        g.record_action("throttle:Safari", 0.75, 10);
        // 3 cycles later, pressure dropped.
        g.evaluate(0.70, 13);
        let eff = g.effectiveness("throttle:Safari");
        assert!(eff.is_none()); // only 1 observation, need ≥3
        // Add more observations.
        g.record_action("throttle:Safari", 0.75, 14);
        g.evaluate(0.70, 17);
        g.record_action("throttle:Safari", 0.75, 18);
        g.evaluate(0.70, 21);
        let eff = g.effectiveness("throttle:Safari").unwrap();
        assert!(eff > 0.5, "should be effective: {}", eff);
    }

    #[test]
    fn test_record_and_evaluate_ineffective() {
        let mut g = CausalGraph::new();
        for cycle in 0..10 {
            g.record_action("throttle:contactsd", 0.75, cycle * 4);
            g.evaluate(0.74, cycle * 4 + 3); // pressure barely changed
        }
        let eff = g.effectiveness("throttle:contactsd").unwrap();
        assert!(eff < 0.4, "should be ineffective: {}", eff);
    }

    #[test]
    fn test_confidence_map() {
        let mut g = CausalGraph::new();
        for cycle in 0..10 {
            g.record_action("throttle:Safari", 0.80, cycle * 4);
            g.evaluate(0.70, cycle * 4 + 3);
        }
        let map = g.confidence_map();
        assert!(map.contains_key("throttle:Safari"));
        assert!(*map.get("throttle:Safari").unwrap() > 0.5);
    }

    #[test]
    fn test_pending_cap() {
        let mut g = CausalGraph::new();
        for i in 0..250 {
            g.record_action(&format!("action:{}", i), 0.7, i);
        }
        assert!(g.pending.len() <= 200);
    }
}
