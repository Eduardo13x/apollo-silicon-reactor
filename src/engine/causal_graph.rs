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
    /// Slow-horizon confidence [0, 1]. Evaluated at 15 cycles (~7.5s at 2Hz).
    /// Captures delayed causal effects: page decompression, swap writeback,
    /// memory compaction. [Granger 1969] longer windows for delayed causation.
    pub slow_confidence: f32,
    /// EMA of pressure delta at slow horizon. Separate from fast avg_delta
    /// because memory reclaim often produces larger delayed drops.
    pub slow_avg_delta: f32,
    /// Mechanism attribution: which resource channel carried the causal effect.
    /// Tracks EMA of RSS delta, CPU delta, and swap delta per edge.
    /// [Pearl 2009] Ch.3 — mediation analysis: identify causal pathways.
    pub mechanism: MechanismAttribution,
}

/// Tracks WHICH resource changed when an action was effective.
/// Answers "WHY did throttling X reduce pressure?" — was it RSS release,
/// CPU reduction, or swap avoidance?
#[derive(Clone, Debug, Default)]
pub struct MechanismAttribution {
    /// EMA of RSS delta (MB) when action was effective. Positive = RSS freed.
    pub rss_delta_mb: f32,
    /// EMA of CPU delta (%) when action was effective. Positive = CPU freed.
    pub cpu_delta_pct: f32,
    /// EMA of swap delta (MB) when action was effective. Positive = swap avoided/freed.
    pub swap_delta_mb: f32,
    /// Observation count for mechanism data.
    pub observations: u32,
}

impl MechanismAttribution {
    /// Update mechanism EMAs with observed deltas.
    fn observe(&mut self, rss_mb: f32, cpu_pct: f32, swap_mb: f32) {
        const ALPHA: f32 = 0.15;
        self.rss_delta_mb = self.rss_delta_mb * (1.0 - ALPHA) + rss_mb * ALPHA;
        self.cpu_delta_pct = self.cpu_delta_pct * (1.0 - ALPHA) + cpu_pct * ALPHA;
        self.swap_delta_mb = self.swap_delta_mb * (1.0 - ALPHA) + swap_mb * ALPHA;
        self.observations += 1;
    }

    /// Primary mechanism: which resource channel explains the most effect.
    pub fn primary(&self) -> &'static str {
        if self.observations < 3 {
            return "unknown";
        }
        let rss = self.rss_delta_mb.abs();
        let cpu = self.cpu_delta_pct.abs();
        let swap = self.swap_delta_mb.abs();
        if rss >= cpu && rss >= swap {
            "rss"
        } else if cpu >= swap {
            "cpu"
        } else {
            "swap"
        }
    }
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
            slow_confidence: 0.5,
            slow_avg_delta: 0.0,
            mechanism: MechanismAttribution::default(),
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

    /// Update slow-horizon confidence (15-cycle eval window).
    fn update_slow(&mut self, was_effective: bool, delta: f32) {
        let target = if was_effective { 1.0 } else { 0.0 };
        self.slow_confidence = self.slow_confidence * 0.9 + target * 0.1;
        if was_effective && delta > 0.0 {
            self.slow_avg_delta = self.slow_avg_delta * 0.85 + delta * 0.15;
        }
    }

    /// Impact score: confidence × avg_delta. Ranks edges by real-world effect.
    /// A solid edge with 0.80 confidence and 0.10 avg drop scores higher
    /// than one with 0.90 confidence but only 0.02 avg drop.
    /// [Granger 1969] Blends fast (3-cycle) and slow (15-cycle) horizons.
    pub fn impact_score(&self) -> f32 {
        let fast = self.confidence * self.avg_delta;
        let slow = self.slow_confidence * self.slow_avg_delta;
        // Take the max: if slow horizon shows bigger effect, use it.
        // This captures delayed effects like memory reclaim.
        fast.max(slow)
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

/// Snapshot of process resource state at action time — for mechanism attribution.
#[derive(Clone, Default)]
pub struct ResourceSnapshot {
    /// RSS in MB at action time.
    pub rss_mb: f32,
    /// CPU % at action time.
    pub cpu_pct: f32,
    /// Swap used in MB at action time.
    pub swap_mb: f32,
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
    /// Resource snapshot at action time — for mechanism attribution.
    resources: ResourceSnapshot,
}

/// Causal graph tracking action → outcome relationships.
pub struct CausalGraph {
    /// Directed edges: (cause, effect) → CausalEdge.
    edges: HashMap<(String, String), CausalEdge>,
    /// Actions waiting for fast outcome evaluation (3 cycles).
    pending: Vec<PendingAction>,
    /// Actions waiting for slow outcome evaluation (15 cycles).
    /// [Granger 1969] Captures delayed causal effects: page decompression,
    /// swap writeback, compaction. Separate queue to avoid inflating fast eval.
    pending_slow: Vec<PendingAction>,
    /// Cycles to wait before evaluating outcome (fast horizon).
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
            pending_slow: Vec::new(),
            eval_delay: 3,
        }
    }

    /// Record that an action was taken on a process/group.
    /// Called after execute_actions with the names of throttled/frozen processes.
    pub fn record_action(&mut self, action_key: &str, pressure: f32, cycle: u64) {
        self.record_action_with_resources(action_key, pressure, cycle, ResourceSnapshot::default());
    }

    /// Record action with resource snapshot for mechanism attribution.
    /// [Pearl 2009] Ch.3 mediation: track resource channels (RSS, CPU, swap)
    /// to learn WHY an action was effective, not just WHETHER.
    pub fn record_action_with_resources(
        &mut self,
        action_key: &str,
        pressure: f32,
        cycle: u64,
        resources: ResourceSnapshot,
    ) {
        let action = PendingAction {
            action_key: action_key.to_string(),
            pressure_at_action: pressure,
            cycle,
            resources: resources.clone(),
        };
        self.pending.push(action.clone());
        self.pending_slow.push(action);
        // Cap pending queues to avoid unbounded growth.
        if self.pending.len() > 200 {
            self.pending.drain(..100);
        }
        if self.pending_slow.len() > 200 {
            self.pending_slow.drain(..100);
        }
    }

    /// Evaluate pending actions against current pressure.
    /// Called each cycle — checks actions that are old enough for evaluation.
    /// Now also accepts current resource snapshot for mechanism attribution.
    pub fn evaluate(&mut self, current_pressure: f32, current_cycle: u64) {
        self.evaluate_with_resources(current_pressure, current_cycle, &ResourceSnapshot::default());
    }

    /// Evaluate with resource snapshots for mechanism attribution.
    /// [Pearl 2009] Ch.3 mediation analysis + [Granger 1969] multi-horizon.
    pub fn evaluate_with_resources(
        &mut self,
        current_pressure: f32,
        current_cycle: u64,
        current_resources: &ResourceSnapshot,
    ) {
        // ── Fast horizon: 3 cycles (~1.5s) ──────────────────────────────────
        let delay = self.eval_delay as u64;
        let mut i = 0;
        while i < self.pending.len() {
            if current_cycle.saturating_sub(self.pending[i].cycle) >= delay {
                let pending = self.pending.swap_remove(i);
                let delta = pending.pressure_at_action - current_pressure;
                let was_effective = delta >= MIN_DELTA;

                let (effect, anti_effect) = if was_effective {
                    (EFFECT_PRESSURE_DROP, EFFECT_PRESSURE_UNCHANGED)
                } else {
                    (EFFECT_PRESSURE_UNCHANGED, EFFECT_PRESSURE_DROP)
                };

                let key = (pending.action_key.clone(), effect.to_string());
                let edge = self.edges
                    .entry(key)
                    .or_insert_with(|| CausalEdge::new(&pending.action_key, effect));
                edge.update_with_delta(true, delta.max(0.0));

                // Mechanism attribution: what resource channel changed?
                if was_effective {
                    let rss_d = pending.resources.rss_mb - current_resources.rss_mb;
                    let cpu_d = pending.resources.cpu_pct - current_resources.cpu_pct;
                    let swap_d = pending.resources.swap_mb - current_resources.swap_mb;
                    edge.mechanism.observe(rss_d.max(0.0), cpu_d.max(0.0), swap_d.max(0.0));
                }

                let anti_key = (pending.action_key, anti_effect.to_string());
                self.edges
                    .entry(anti_key)
                    .or_insert_with_key(|k| CausalEdge::new(&k.0, anti_effect))
                    .update_with_delta(false, 0.0);
            } else {
                i += 1;
            }
        }

        // ── Slow horizon: 15 cycles (~7.5s) — captures memory reclaim ───────
        // [Granger 1969] Delayed causation: page decompression, swap writeback,
        // VM compaction happen 3-10s after a throttle/freeze. The fast 3-cycle
        // window misses these entirely. Slow horizon catches them.
        const SLOW_DELAY: u64 = 15;
        let mut j = 0;
        while j < self.pending_slow.len() {
            if current_cycle.saturating_sub(self.pending_slow[j].cycle) >= SLOW_DELAY {
                let pending = self.pending_slow.swap_remove(j);
                let delta = pending.pressure_at_action - current_pressure;
                let was_effective = delta >= MIN_DELTA;

                // Update slow-horizon confidence on the pressure_drop edge.
                let drop_key = (pending.action_key.clone(), EFFECT_PRESSURE_DROP.to_string());
                let edge = self.edges
                    .entry(drop_key)
                    .or_insert_with(|| CausalEdge::new(&pending.action_key, EFFECT_PRESSURE_DROP));
                edge.update_slow(was_effective, delta.max(0.0));
            } else {
                j += 1;
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
    /// [Granger 1969] Blends fast and slow horizons: takes the max of both,
    /// so delayed-effect processes aren't penalized by the fast window.
    pub fn confidence_map(&self) -> HashMap<String, f32> {
        let mut map = HashMap::new();
        for ((action_key, effect), edge) in &self.edges {
            if effect == EFFECT_PRESSURE_DROP && edge.evidence_count >= 5 {
                // Blend: use the better of fast and slow confidence.
                // A process that only shows effect at 7.5s still gets credit.
                let blended = edge.confidence.max(edge.slow_confidence);
                map.insert(action_key.clone(), blended);
            }
        }
        map
    }

    /// Build an impact-ranked map: action_key → impact_score for prioritization.
    /// Higher = more effective AND larger pressure drops.
    pub fn impact_map(&self) -> HashMap<String, f32> {
        let mut map = HashMap::new();
        for ((action_key, effect), edge) in &self.edges {
            if effect == EFFECT_PRESSURE_DROP && edge.evidence_count >= 5 {
                map.insert(action_key.clone(), edge.impact_score());
            }
        }
        map
    }

    /// Get mechanism attribution for an action.
    /// Returns (primary_mechanism, rss_delta, cpu_delta, swap_delta) or None.
    pub fn mechanism(&self, action_key: &str) -> Option<(&str, f32, f32, f32)> {
        let key = (action_key.to_string(), EFFECT_PRESSURE_DROP.to_string());
        self.edges.get(&key).and_then(|e| {
            if e.mechanism.observations >= 3 {
                Some((
                    e.mechanism.primary(),
                    e.mechanism.rss_delta_mb,
                    e.mechanism.cpu_delta_pct,
                    e.mechanism.swap_delta_mb,
                ))
            } else {
                None
            }
        })
    }

    /// Count of edges with slow-horizon data (slow_confidence != 0.5 prior).
    pub fn slow_horizon_count(&self) -> usize {
        self.edges
            .values()
            .filter(|e| (e.slow_confidence - 0.5).abs() > 0.01)
            .count()
    }

    /// Count of edges with mechanism attribution data.
    pub fn mechanism_count(&self) -> usize {
        self.edges
            .values()
            .filter(|e| e.mechanism.observations >= 3)
            .count()
    }

    /// Experience-informed confidence: for processes with insufficient causal data
    /// (< 5 observations), fall back to experience memory as a Bayesian prior.
    /// [Kahneman & Tversky 1973] Availability heuristic: similar past episodes
    /// inform current prediction. [Pearl 2009] §3.6 priors from observational data.
    ///
    /// Returns a blended confidence map where cold processes get priors from
    /// experience memory, and warm processes use their causal graph confidence.
    pub fn confidence_map_with_experience(
        &self,
        experience: &crate::engine::outcome_tracker::ExperienceMemory,
        current_pressure: f64,
    ) -> HashMap<String, f32> {
        let mut map = self.confidence_map();

        // For each process in experience that isn't in the causal map yet,
        // compute a prior from similar episodes.
        let mut seen: std::collections::HashSet<String> = map.keys().cloned().collect();

        for record in experience.records() {
            let key = format!("throttle:{}", record.process_name);
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key.clone());

            // Query experience for this process at current pressure.
            if let Some((avg_drop, confidence)) =
                experience.query_similar(&record.process_name, current_pressure)
            {
                // Convert experience effectiveness to causal prior.
                // avg_drop > 0.02 and confidence > 0.15 → warm prior.
                // Scale: a 0.05 average drop at 0.5 confidence → 0.65 prior.
                if confidence >= 0.15 {
                    let prior = if avg_drop >= 0.02 {
                        // Effective in similar conditions: prior 0.5 + scaled by drop magnitude.
                        (0.5 + (avg_drop * 3.0).min(0.4) as f32).min(0.85)
                    } else {
                        // Ineffective: low prior.
                        0.25
                    };
                    map.insert(key, prior);
                }
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
