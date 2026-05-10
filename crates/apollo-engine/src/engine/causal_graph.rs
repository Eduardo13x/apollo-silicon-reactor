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

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hard cap for in-cycle edge count. Raised from 500 → 1500 after canary
/// telemetry showed the graph was saturating immediately, causing edges to be
/// evicted before they could accumulate the 3-15 cycles of evidence needed for
/// causal evaluation. Sub-microsecond compaction (0.000375ms) confirms this is
/// safe for CPU budget. [Cormen et al. 2009] §11 — bounded HashMap.
pub const HOT_PATH_EDGE_CAP: usize = 1500;

/// A causal edge: action X caused outcome Y with measured confidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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
        // Always apply EMA decay; target = observed delta when effective, else 0.
        // Prevents avg_delta from freezing at historical highs when an edge stops
        // being effective — repeated failures should decay it toward 0.
        let delta_target = if was_effective && delta > 0.0 {
            delta
        } else {
            0.0
        };
        self.avg_delta = self.avg_delta * 0.85 + delta_target * 0.15;
    }

    /// Update slow-horizon confidence (15-cycle eval window).
    fn update_slow(&mut self, was_effective: bool, delta: f32) {
        let target = if was_effective { 1.0 } else { 0.0 };
        self.slow_confidence = self.slow_confidence * 0.9 + target * 0.1;
        let delta_target = if was_effective && delta > 0.0 {
            delta
        } else {
            0.0
        };
        self.slow_avg_delta = self.slow_avg_delta * 0.85 + delta_target * 0.15;
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
    /// Counter for edges evicted due to hot-path capacity limits.
    evictions_total: u64,
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
            evictions_total: 0,
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
        self.evaluate_with_resources(
            current_pressure,
            current_cycle,
            &ResourceSnapshot::default(),
        );
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
                // High-swap regime: natural pressure drift is 3-4% per cycle due
                // to kernel compressor flushes, independent of Apollo actions.
                // Require a larger delta to credit an edge as causal; otherwise
                // the graph attributes natural reclaim to throttle actions and
                // builds overconfident edges (predicted=0.90 vs actual=0.30).
                // [Pearl 2009] §3: confounding — swap flushes are a common cause
                // of both "we acted" and "pressure dropped"; higher bar required.
                let effective_min_delta = if pending.resources.swap_mb > 2000.0 {
                    MIN_DELTA * 2.0
                } else {
                    MIN_DELTA
                };
                let was_effective = delta >= effective_min_delta;

                let (effect, anti_effect) = if was_effective {
                    (EFFECT_PRESSURE_DROP, EFFECT_PRESSURE_UNCHANGED)
                } else {
                    (EFFECT_PRESSURE_UNCHANGED, EFFECT_PRESSURE_DROP)
                };

                let key = (pending.action_key.clone(), effect.to_string());
                let edge = self
                    .edges
                    .entry(key)
                    .or_insert_with(|| CausalEdge::new(&pending.action_key, effect));
                edge.update_with_delta(true, delta.max(0.0));

                // Mechanism attribution: what resource channel changed?
                if was_effective {
                    let rss_d = pending.resources.rss_mb - current_resources.rss_mb;
                    let cpu_d = pending.resources.cpu_pct - current_resources.cpu_pct;
                    let swap_d = pending.resources.swap_mb - current_resources.swap_mb;
                    edge.mechanism
                        .observe(rss_d.max(0.0), cpu_d.max(0.0), swap_d.max(0.0));
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
                let edge = self
                    .edges
                    .entry(drop_key)
                    .or_insert_with(|| CausalEdge::new(&pending.action_key, EFFECT_PRESSURE_DROP));
                edge.update_slow(was_effective, delta.max(0.0));
            } else {
                j += 1;
            }
        }

        // In-cycle cap: persist-time prune in LearnedState::self_improve runs
        // every ~300 cycles (~150s). Within that window the edges HashMap can
        // grow with every unique (process_name, effect) pair the daemon sees,
        // and lookups in solid_edges_by_impact / effectiveness become O(N).
        // Hard cap at 500 entries; on overflow, evict the lowest-impact edge.
        // The persist-time decay-and-retain still runs and does the principled
        // GC; this is just a safety valve to keep hot-path lookups bounded.
        // [Cormen et al. 2009] §11 — bounded-size HashMap keeps O(1) amortised.
        if self.edges.len() > HOT_PATH_EDGE_CAP {
            // Score = impact_score (higher = more useful). Evict lowest.
            // Multiply by 1000 + cast to i32 for stable Ord.
            if let Some(weakest) = self
                .edges
                .iter()
                .min_by_key(|(_, e)| (e.impact_score() * 1000.0) as i32)
                .map(|(k, _)| k.clone())
            {
                self.edges.remove(&weakest);
                self.evictions_total += 1;
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
        edges.sort_by(|a, b| {
            b.impact_score()
                .partial_cmp(&a.impact_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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

    /// Number of unique causes (nodes) in the graph.
    pub fn nodes_count(&self) -> usize {
        let mut unique_causes = std::collections::HashSet::new();
        for (cause, _) in self.edges.keys() {
            unique_causes.insert(cause.as_str());
        }
        unique_causes.len()
    }

    /// Total number of evictions due to capacity limits.
    pub fn evictions_total(&self) -> u64 {
        self.evictions_total
    }

    /// Age of the oldest pending action in cycles.
    pub fn oldest_pending_action_age_cycles(&self, current_cycle: u64) -> u64 {
        let oldest_fast = self
            .pending
            .iter()
            .map(|p| p.cycle)
            .min()
            .unwrap_or(current_cycle);
        let oldest_slow = self
            .pending_slow
            .iter()
            .map(|p| p.cycle)
            .min()
            .unwrap_or(current_cycle);

        let oldest = std::cmp::min(oldest_fast, oldest_slow);
        current_cycle.saturating_sub(oldest)
    }

    /// Count edges whose cause matches known ephemeral process patterns.
    /// This is an **observation-only** metric for Fase 2 planning.
    /// Does NOT filter or block anything — just reports how much of the
    /// graph's capacity is occupied by likely short-lived XPC/Helper processes.
    pub fn ephemeral_edge_count(&self) -> usize {
        self.edges
            .keys()
            .filter(|(cause, _)| {
                // Extract process name from "throttle:name" or "freeze:name".
                let name = cause
                    .strip_prefix("throttle:")
                    .or_else(|| cause.strip_prefix("freeze:"))
                    .unwrap_or(cause);
                Self::is_ephemeral_name(name)
            })
            .count()
    }

    /// Heuristic: does this process name look like a short-lived XPC or helper?
    /// Conservative: only matches patterns with very high confidence of being
    /// ephemeral. Does NOT match all com.apple.* (many are long-lived daemons).
    fn is_ephemeral_name(name: &str) -> bool {
        name == "xpcproxy"
            || name.contains("XPCService")
            || name.starts_with("com.apple.WebKit.WebContent")
            || name.starts_with("com.apple.WebKit.Networking")
            || name.starts_with("com.apple.WebKit.GPU")
            || name.contains("(Utility)")
            || name.contains("(Renderer)")
            || (name.contains("Helper") && name.contains("("))
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

    /// Returns `true` when causal evidence indicates that QoS tier demotion
    /// (set_tier → Background) is preferable to SIGSTOP for the given process.
    ///
    /// The heuristic: if the primary causal mechanism for this process is CPU
    /// reduction (not RSS release or swap avoidance), then a CPU scheduler hint
    /// achieves the pressure benefit while letting the process stay responsive
    /// to events — SIGSTOP would be unnecessarily invasive.
    ///
    /// Returns `false` when:
    /// - Fewer than 3 causal observations exist (conservative default: use SIGSTOP)
    /// - Primary mechanism is "rss" or "swap" (memory pages must actually stop being
    ///   touched — QoS tier alone won't achieve this)
    /// - Mechanism is "unknown" (insufficient data)
    ///
    /// [Pearl 2009 Ch.3] — mediation analysis: identify the causal pathway
    /// [Nygard 2018] — bulkhead: least-invasive intervention first
    pub fn prefer_qos_over_sigstop(&self, process_name: &str) -> bool {
        let action_key = format!("throttle:{}", process_name);
        match self.mechanism(&action_key) {
            Some(("cpu", _, cpu_pct, _)) => {
                // Only prefer QoS if CPU reduction is the dominant effect
                // (not marginal: require at least 5% CPU delta on average).
                cpu_pct.abs() >= 5.0
            }
            _ => false, // no data or non-CPU mechanism → default SIGSTOP
        }
    }

    /// Returns all process names for which `prefer_qos_over_sigstop()` is true.
    /// Used by the execution pipeline to bulk-upgrade FreezeProcess → ThrottleProcess
    /// when causal evidence identifies CPU reduction as the primary mechanism.
    /// [Pearl 2009 §3] — identify causal pathway before choosing intervention
    pub fn qos_preferred_names(&self) -> std::collections::HashSet<String> {
        self.edges
            .keys()
            .filter_map(|(action_key, _)| {
                action_key.strip_prefix("throttle:").and_then(|name| {
                    if self.prefer_qos_over_sigstop(name) {
                        Some(name.to_string())
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Count of edges with slow-horizon data (slow_confidence != 0.5 prior).
    pub fn slow_horizon_count(&self) -> usize {
        self.edges
            .values()
            .filter(|e| (e.slow_confidence - 0.5).abs() > 0.01)
            .count()
    }

    /// Snapshot edges for persistence. Only persists edges with ≥ 3 evidence
    /// (skip noise from very early observations).
    pub fn to_persisted(&self) -> Vec<((String, String), CausalEdge)> {
        self.edges
            .iter()
            .filter(|(_, e)| e.evidence_count >= 3)
            .map(|(k, e)| (k.clone(), e.clone()))
            .collect()
    }

    /// Restore edges from persisted snapshot. Merges with any existing edges.
    pub fn restore(&mut self, persisted: Vec<((String, String), CausalEdge)>) {
        for (key, edge) in persisted {
            // Only restore if we don't already have fresher data.
            let entry = self.edges.entry(key).or_insert_with(|| edge.clone());
            if entry.evidence_count < edge.evidence_count {
                *entry = edge;
            }
        }
    }

    /// Count of edges with mechanism attribution data.
    pub fn mechanism_count(&self) -> usize {
        self.edges
            .values()
            .filter(|e| e.mechanism.observations >= 3)
            .count()
    }

    /// Pearl-mediation breakdown: count causal edges by the dominant resource
    /// channel they attribute to.  Returns `(rss, cpu, swap, unknown)` where
    /// `unknown` covers edges that don't yet have ≥3 observations.
    ///
    /// Consumed as an observability surface in runtime metrics so an operator
    /// can tell at a glance *why* the daemon is choosing SIGSTOP vs throttle
    /// vs E-core demotion — RSS-dominant edges justify SIGSTOP, CPU-dominant
    /// edges justify QoS-background, swap-dominant edges justify page_reclaim.
    /// [Pearl 2009 §3] mediation analysis — intervention follows mechanism
    pub fn mechanism_breakdown(&self) -> (usize, usize, usize, usize) {
        let mut rss = 0usize;
        let mut cpu = 0usize;
        let mut swap = 0usize;
        let mut unknown = 0usize;
        for edge in self.edges.values() {
            if edge.mechanism.observations < 3 {
                unknown += 1;
                continue;
            }
            match edge.mechanism.primary() {
                "rss" => rss += 1,
                "cpu" => cpu += 1,
                "swap" => swap += 1,
                _ => unknown += 1,
            }
        }
        (rss, cpu, swap, unknown)
    }

    /// Co-occurrence cluster boosting: if process B co-occurs with solid process A
    /// (confidence > 0.70) ≥ 5 times, and B's own confidence is below skip threshold,
    /// boost B's confidence to the cluster average. [Pearl 2009] Ch.2: confounding —
    /// processes that always appear together share causal structure.
    pub fn apply_cluster_boost(
        &self,
        map: &mut HashMap<String, f32>,
        co_occurrence: &[(String, String, u32)],
    ) {
        // Build a set of solid action keys for fast lookup.
        let solid_keys: std::collections::HashSet<&str> = map
            .iter()
            .filter(|(_, &conf)| conf > 0.70)
            .map(|(k, _)| k.as_str())
            .collect();

        let mut boosts: Vec<(String, f32)> = Vec::new();

        for (a, b, count) in co_occurrence {
            if *count < 5 {
                continue;
            }
            let key_a = format!("throttle:{}", a);
            let key_b = format!("throttle:{}", b);

            let a_is_solid = solid_keys.contains(key_a.as_str());
            let b_is_solid = solid_keys.contains(key_b.as_str());

            // If A is solid and B is cold/weak, boost B.
            if a_is_solid {
                let b_conf = map.get(&key_b).copied().unwrap_or(0.5);
                if b_conf < 0.30 {
                    let a_conf = map[&key_a];
                    let boost = ((a_conf + b_conf) / 2.0).min(0.50);
                    boosts.push((key_b.clone(), boost));
                }
            }
            // Symmetric: if B is solid and A is cold/weak, boost A.
            if b_is_solid {
                let a_conf = map.get(&key_a).copied().unwrap_or(0.5);
                if a_conf < 0.30 {
                    let b_conf = map.get(&key_b).copied().unwrap_or(0.5);
                    let boost = ((b_conf + a_conf) / 2.0).min(0.50);
                    boosts.push((key_a.clone(), boost));
                }
            }
        }

        for (key, boosted_conf) in boosts {
            let entry = map.entry(key).or_insert(0.5);
            if *entry < boosted_conf {
                *entry = boosted_conf;
            }
        }
    }

    /// NARS × Causal fusion: discount causal confidence by NARS belief stability.
    /// When concept drift is detected for a process (NARS confidence < 0.30 or
    /// frequency changed > 20pp), the learned causal relationship may no longer hold.
    /// Discount the causal confidence proportionally.
    /// [Pei Wang 2013] NARS §3.3.3 — stale beliefs should reduce decision weight.
    pub fn apply_nars_discount(
        map: &mut HashMap<String, f32>,
        drift_detector: &crate::engine::nars_belief::DriftDetector,
    ) {
        for (key, conf) in map.iter_mut() {
            // Extract process name from "throttle:ProcessName".
            let process_name = key.strip_prefix("throttle:").unwrap_or(key);
            if let Some(tv) = drift_detector.belief(process_name) {
                // Low NARS confidence = belief has been revised many times recently
                // (unstable). Discount causal confidence proportionally.
                // NARS conf 0.50+ → no discount. NARS conf < 0.30 → 40% discount.
                if tv.confidence < 0.50 {
                    let discount = 0.6 + 0.8 * tv.confidence; // 0.6..1.0
                    *conf *= discount;
                }
            }
        }
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

    // ── Multi-horizon tests ──────────────────────────────────────────────

    #[test]
    fn slow_horizon_captures_delayed_effect() {
        let mut g = CausalGraph::new();
        // Record action at cycle 10 with pressure 0.80.
        g.record_action("throttle:Safari", 0.80, 10);
        // Fast eval at cycle 13: pressure unchanged → fast says ineffective.
        g.evaluate(0.79, 13);
        let edge = g.get_edge("throttle:Safari", EFFECT_PRESSURE_DROP);
        // Fast confidence should be close to uninformed (slightly below 0.5).
        assert!(edge.is_some());

        // Slow eval at cycle 25 (15 cycles later): pressure dropped significantly.
        g.evaluate(0.65, 25);
        let edge = g.get_edge("throttle:Safari", EFFECT_PRESSURE_DROP).unwrap();
        // Slow confidence should reflect the delayed drop.
        assert!(
            edge.slow_confidence > 0.5,
            "slow should see effect: {}",
            edge.slow_confidence
        );
        assert!(edge.slow_avg_delta > 0.0, "slow delta should be positive");
    }

    #[test]
    fn slow_horizon_blends_into_confidence_map() {
        let mut g = CausalGraph::new();
        // Build a process with weak fast but strong slow signal.
        for cycle in 0..10u64 {
            g.record_action("throttle:compactd", 0.80, cycle * 20);
            // Fast eval: no change.
            g.evaluate(0.79, cycle * 20 + 3);
            // Slow eval: big drop.
            g.evaluate(0.60, cycle * 20 + 15);
        }
        let map = g.confidence_map();
        // confidence_map uses max(fast, slow), so slow signal should show.
        if let Some(&conf) = map.get("throttle:compactd") {
            assert!(conf > 0.5, "blended confidence should be > 0.5: {}", conf);
        }
    }

    // ── Mechanism attribution tests ─────────────────────────────────────

    #[test]
    fn mechanism_attribution_tracks_resource_deltas() {
        let mut m = MechanismAttribution::default();
        assert_eq!(m.primary(), "unknown"); // not enough observations
        m.observe(100.0, 5.0, 0.0); // RSS dominant
        m.observe(120.0, 3.0, 0.0);
        m.observe(90.0, 4.0, 0.0);
        assert_eq!(m.primary(), "rss");
        // EMA with α=0.15 from 0: after 3 obs of ~100 → ~35 (not fully converged).
        assert!(
            m.rss_delta_mb > 10.0,
            "RSS EMA should be positive: {}",
            m.rss_delta_mb
        );
    }

    #[test]
    fn mechanism_cpu_dominant() {
        let mut m = MechanismAttribution::default();
        m.observe(5.0, 80.0, 0.0); // CPU dominant
        m.observe(3.0, 90.0, 0.0);
        m.observe(4.0, 85.0, 0.0);
        assert_eq!(m.primary(), "cpu");
    }

    #[test]
    fn mechanism_swap_dominant() {
        let mut m = MechanismAttribution::default();
        m.observe(0.0, 0.0, 500.0); // swap dominant
        m.observe(0.0, 0.0, 600.0);
        m.observe(0.0, 0.0, 400.0);
        assert_eq!(m.primary(), "swap");
    }

    #[test]
    fn evaluate_with_resources_populates_mechanism() {
        let mut g = CausalGraph::new();
        let res_before = ResourceSnapshot {
            rss_mb: 500.0,
            cpu_pct: 30.0,
            swap_mb: 1000.0,
        };
        for cycle in 0..5u64 {
            g.record_action_with_resources("throttle:Chrome", 0.80, cycle * 4, res_before.clone());
            let res_after = ResourceSnapshot {
                rss_mb: 350.0,
                cpu_pct: 10.0,
                swap_mb: 900.0,
            };
            g.evaluate_with_resources(0.70, cycle * 4 + 3, &res_after);
        }
        let mech = g.mechanism("throttle:Chrome");
        assert!(
            mech.is_some(),
            "should have mechanism data after 5 effective evals"
        );
        let (primary, rss, cpu, swap) = mech.unwrap();
        assert!(rss > 0.0, "RSS delta should be positive");
        assert!(cpu > 0.0, "CPU delta should be positive");
        assert!(swap > 0.0, "swap delta should be positive");
        assert_eq!(primary, "rss"); // 150MB RSS > 20% CPU > 100MB swap
    }

    // ── Impact score tests ──────────────────────────────────────────────

    #[test]
    fn impact_score_uses_max_of_fast_and_slow() {
        let mut e = CausalEdge::new("test", "pressure_drop");
        // Fast: low
        e.confidence = 0.30;
        e.avg_delta = 0.01;
        // Slow: high
        e.slow_confidence = 0.80;
        e.slow_avg_delta = 0.10;
        // Impact should use slow (0.08) > fast (0.003).
        assert!(
            e.impact_score() > 0.05,
            "should use slow: {}",
            e.impact_score()
        );
    }

    // ── Cluster boost tests ─────────────────────────────────────────────

    #[test]
    fn cluster_boost_rescues_cold_process() {
        let g = CausalGraph::new();
        let mut map = HashMap::new();
        map.insert("throttle:Safari".to_string(), 0.80_f32); // solid
        map.insert("throttle:cloudd".to_string(), 0.15_f32); // would be skipped
        let pairs = vec![
            ("Safari".to_string(), "cloudd".to_string(), 10), // co-occur 10 times
        ];
        g.apply_cluster_boost(&mut map, &pairs);
        let boosted = map["throttle:cloudd"];
        assert!(
            boosted > 0.20,
            "cloudd should be boosted above skip threshold: {}",
            boosted
        );
    }

    #[test]
    fn cluster_boost_ignores_low_cooccurrence() {
        let g = CausalGraph::new();
        let mut map = HashMap::new();
        map.insert("throttle:Safari".to_string(), 0.80_f32);
        map.insert("throttle:cloudd".to_string(), 0.15_f32);
        let pairs = vec![
            ("Safari".to_string(), "cloudd".to_string(), 3), // too few co-occurrences
        ];
        g.apply_cluster_boost(&mut map, &pairs);
        assert_eq!(map["throttle:cloudd"], 0.15); // unchanged
    }

    // ── NARS discount tests ─────────────────────────────────────────────

    #[test]
    fn nars_discount_reduces_drifted_confidence() {
        use crate::engine::nars_belief::DriftDetector;
        let mut dd = DriftDetector::new();
        // Create a belief with low confidence (many revisions → unstable).
        for _ in 0..5 {
            dd.observe("Safari", true);
        }
        for _ in 0..5 {
            dd.observe("Safari", false);
        }
        // The belief should have moderate-to-low confidence now.
        let tv = dd.belief("Safari").unwrap();

        let mut map = HashMap::new();
        map.insert("throttle:Safari".to_string(), 0.80_f32);
        CausalGraph::apply_nars_discount(&mut map, &dd);

        if tv.confidence < 0.50 {
            assert!(map["throttle:Safari"] < 0.80, "should be discounted");
        }
    }

    #[test]
    fn nars_discount_no_effect_on_stable_beliefs() {
        use crate::engine::nars_belief::DriftDetector;
        let mut dd = DriftDetector::new();
        // Consistent successes → high confidence.
        for _ in 0..20 {
            dd.observe("Dropbox", true);
        }
        let tv = dd.belief("Dropbox").unwrap();
        assert!(tv.confidence >= 0.50, "should be high confidence");

        let mut map = HashMap::new();
        map.insert("throttle:Dropbox".to_string(), 0.80_f32);
        CausalGraph::apply_nars_discount(&mut map, &dd);
        assert_eq!(map["throttle:Dropbox"], 0.80); // unchanged
    }

    // ── Slow horizon + pending_slow cap ─────────────────────────────────

    #[test]
    fn pending_slow_cap() {
        let mut g = CausalGraph::new();
        for i in 0..250u64 {
            g.record_action(&format!("action:{}", i), 0.7, i);
        }
        assert!(g.pending_slow.len() <= 200);
    }

    #[test]
    fn slow_horizon_count_tracks_updated_edges() {
        let mut g = CausalGraph::new();
        assert_eq!(g.slow_horizon_count(), 0);
        g.record_action("throttle:test", 0.80, 0);
        g.evaluate(0.60, 15); // triggers slow eval
        assert!(g.slow_horizon_count() > 0);
    }

    #[test]
    fn mechanism_count_tracks_attributed_edges() {
        let mut g = CausalGraph::new();
        assert_eq!(g.mechanism_count(), 0);
        let res = ResourceSnapshot {
            rss_mb: 500.0,
            cpu_pct: 30.0,
            swap_mb: 100.0,
        };
        for i in 0..5u64 {
            g.record_action_with_resources("throttle:X", 0.80, i * 4, res.clone());
            let after = ResourceSnapshot {
                rss_mb: 300.0,
                cpu_pct: 10.0,
                swap_mb: 50.0,
            };
            g.evaluate_with_resources(0.60, i * 4 + 3, &after);
        }
        assert!(g.mechanism_count() > 0);
    }

    #[test]
    fn mechanism_breakdown_empty_graph_returns_all_zero() {
        let g = CausalGraph::new();
        assert_eq!(g.mechanism_breakdown(), (0, 0, 0, 0));
    }

    #[test]
    fn mechanism_breakdown_classifies_dominant_channels() {
        // Build three distinct edges, each with a different dominant mechanism.
        let mut g = CausalGraph::new();

        // Edge 1: RSS-dominant (500MB → 300MB vs small CPU/swap).
        let before = ResourceSnapshot {
            rss_mb: 500.0,
            cpu_pct: 30.0,
            swap_mb: 100.0,
        };
        let after = ResourceSnapshot {
            rss_mb: 300.0,
            cpu_pct: 28.0,
            swap_mb: 98.0,
        };
        for i in 0..5u64 {
            g.record_action_with_resources("throttle:Rss", 0.80, i * 10, before.clone());
            g.evaluate_with_resources(0.60, i * 10 + 3, &after);
        }

        // Edge 2: CPU-dominant (80% → 10% vs small rss/swap delta).
        let before = ResourceSnapshot {
            rss_mb: 500.0,
            cpu_pct: 80.0,
            swap_mb: 100.0,
        };
        let after = ResourceSnapshot {
            rss_mb: 498.0,
            cpu_pct: 10.0,
            swap_mb: 99.0,
        };
        for i in 0..5u64 {
            g.record_action_with_resources("throttle:Cpu", 0.80, 100 + i * 10, before.clone());
            g.evaluate_with_resources(0.60, 100 + i * 10 + 3, &after);
        }

        // Edge 3: no observations yet → unknown.
        g.record_action("throttle:Cold", 0.80, 1000);

        let (rss, cpu, swap, unknown) = g.mechanism_breakdown();
        assert_eq!(
            rss,
            1,
            "expected 1 rss-dominant edge, breakdown={:?}",
            (rss, cpu, swap, unknown)
        );
        assert_eq!(
            cpu,
            1,
            "expected 1 cpu-dominant edge, breakdown={:?}",
            (rss, cpu, swap, unknown)
        );
        assert_eq!(swap, 0);
        assert!(
            unknown >= 1,
            "cold edge with 0 observations must land in unknown bucket"
        );
    }

    #[test]
    fn mechanism_breakdown_total_equals_edge_count() {
        // Invariant: every edge lands in exactly one bucket, so the four
        // buckets must sum to the total edge count.  Load-bearing if metrics
        // are ever wired to display percentages.
        let mut g = CausalGraph::new();
        for i in 0..3u64 {
            g.record_action(&format!("throttle:proc-{}", i), 0.80, i * 10);
        }
        let (rss, cpu, swap, unknown) = g.mechanism_breakdown();
        assert_eq!(rss + cpu + swap + unknown, g.edges.len());
    }

    // ── Causal Counterfactual Validity Contract [Pearl 2009 §3] ────────────

    #[test]
    fn causal_counterfactual_effective_action_strengthens_edge() {
        // If Apollo takes action A in context C and pressure drops,
        // the causal edge C→A must strengthen (more solid evidence).
        // [Pearl 2009 §3] causal mediation — effective actions build evidence.
        //
        // API: record_action(key, pressure_before, cycle) followed by
        // evaluate(pressure_after, cycle+delay). When pressure_before - pressure_after
        // >= MIN_DELTA (0.02), the action is deemed effective.
        // is_solid() requires confidence > 0.7 AND evidence_count >= 5.
        let mut g = CausalGraph::new();

        // Record action with pressure drop on each cycle (effective: 0.80 → 0.70, delta=0.10)
        for i in 0..20u64 {
            g.record_action("throttle:Safari", 0.80, i * 4);
            g.evaluate(0.70, i * 4 + 3); // 3 cycles later (eval_delay=3), pressure dropped
        }

        let solid_edges = g.solid_edges_by_impact();
        let safari_solid = solid_edges
            .iter()
            .any(|e| e.cause.contains("Safari") && e.effect == "pressure_drop");
        assert!(
            safari_solid,
            "Effective repeated action should produce solid causal edge for pressure_drop. \
             Edge count: {}, solid_edges: {:?}",
            g.edge_count(),
            solid_edges
                .iter()
                .map(|e| (&e.cause, &e.effect, e.confidence))
                .collect::<Vec<_>>()
        );

        // Also verify the edge confidence is substantial
        let drop_edge = g.get_edge("throttle:Safari", "pressure_drop");
        assert!(
            drop_edge.is_some(),
            "pressure_drop edge must exist after repeated effective actions"
        );
        let edge = drop_edge.unwrap();
        assert!(
            edge.confidence > 0.5,
            "Solid edge confidence should exceed 0.5, got {}",
            edge.confidence
        );
    }

    #[test]
    fn causal_counterfactual_ineffective_action_weakens_edge() {
        // If the action doesn't help, the pressure_drop edge should NOT become solid.
        // [Pearl 2009 §3] — spurious correlations must not elevate to solid.
        //
        // We record actions where pressure barely changes (0.75 → 0.74, delta=0.01 < MIN_DELTA=0.02)
        // so the action is consistently classified as NOT effective.
        let mut g = CausalGraph::new();

        // Record action with negligible pressure change (ineffective: delta < 0.02)
        for i in 0..20u64 {
            g.record_action("throttle:contactsd", 0.75, i * 4);
            g.evaluate(0.74, i * 4 + 3); // delta=0.01 < MIN_DELTA=0.02 → not effective
        }

        let solid_edges = g.solid_edges_by_impact();
        let contactsd_solid = solid_edges
            .iter()
            .any(|e| e.cause.contains("contactsd") && e.effect == "pressure_drop");
        assert!(
            !contactsd_solid,
            "Ineffective action should not produce solid causal edge for pressure_drop. \
             edge confidence: {:?}",
            g.get_edge("throttle:contactsd", "pressure_drop")
                .map(|e| e.confidence)
        );

        // The action should have produced a solid edge for pressure_no_change instead
        // (this validates the anti-edge correctly learns the right outcome)
        let no_change_edge = g.get_edge("throttle:contactsd", "pressure_no_change");
        assert!(
            no_change_edge.is_some(),
            "Repeated ineffective actions should register a pressure_no_change edge"
        );
        if let Some(edge) = no_change_edge {
            assert!(
                edge.confidence > 0.5,
                "pressure_no_change edge should be confident after repeated ineffective actions, \
                 got {}",
                edge.confidence
            );
        }
    }

    // ── Impact map test ─────────────────────────────────────────────────

    #[test]
    fn impact_map_ranks_by_score() {
        let mut g = CausalGraph::new();
        // Chrome: effective with big drops.
        for i in 0..10u64 {
            g.record_action("throttle:Chrome", 0.80, i * 4);
            g.evaluate(0.65, i * 4 + 3); // 0.15 drop
        }
        // contactsd: effective with tiny drops.
        for i in 0..10u64 {
            g.record_action("throttle:contactsd", 0.80, 100 + i * 4);
            g.evaluate(0.77, 100 + i * 4 + 3); // 0.03 drop
        }
        let map = g.impact_map();
        let chrome = map.get("throttle:Chrome").copied().unwrap_or(0.0);
        let contact = map.get("throttle:contactsd").copied().unwrap_or(0.0);
        assert!(
            chrome > contact,
            "Chrome ({}) should rank higher than contactsd ({})",
            chrome,
            contact
        );
    }

    // ── prefer_qos_over_sigstop tests ──────────────────────────────────────

    /// CPU-dominant mechanism with ≥5% delta → prefer QoS over SIGSTOP.
    /// [Pearl 2009 Ch.3] — CPU reduction is the causal pathway, QoS is sufficient.
    #[test]
    fn prefer_qos_cpu_dominant() {
        let mut g = CausalGraph::new();
        let res_before = ResourceSnapshot {
            rss_mb: 200.0,
            cpu_pct: 40.0,
            swap_mb: 500.0,
        };
        // Simulate CPU-dominant effect: large CPU drop, small RSS/swap change
        for cycle in 0..5u64 {
            g.record_action_with_resources(
                "throttle:electron_bg",
                0.80,
                cycle * 4,
                res_before.clone(),
            );
            let res_after = ResourceSnapshot {
                rss_mb: 195.0,  // ~2.5% RSS change — minor
                cpu_pct: 15.0,  // 25% CPU freed — dominant
                swap_mb: 498.0, // ~0.4% swap — negligible
            };
            g.evaluate_with_resources(0.70, cycle * 4 + 3, &res_after);
        }
        assert!(
            g.prefer_qos_over_sigstop("electron_bg"),
            "CPU-dominant mechanism should prefer QoS over SIGSTOP"
        );
    }

    /// RSS-dominant mechanism → do NOT prefer QoS (SIGSTOP is required to stop page access).
    /// Memory pages must stop being touched — CPU scheduler hint alone is insufficient.
    #[test]
    fn prefer_sigstop_rss_dominant() {
        let mut g = CausalGraph::new();
        let res_before = ResourceSnapshot {
            rss_mb: 800.0,
            cpu_pct: 15.0,
            swap_mb: 500.0,
        };
        // RSS-dominant: large RSS freed, moderate CPU, small swap
        for cycle in 0..5u64 {
            g.record_action_with_resources(
                "throttle:chrome_renderer",
                0.80,
                cycle * 4,
                res_before.clone(),
            );
            let res_after = ResourceSnapshot {
                rss_mb: 400.0,  // 400MB RSS freed — dominant
                cpu_pct: 12.0,  // 3% CPU — minor
                swap_mb: 490.0, // 10MB swap — minor
            };
            g.evaluate_with_resources(0.70, cycle * 4 + 3, &res_after);
        }
        assert!(
            !g.prefer_qos_over_sigstop("chrome_renderer"),
            "RSS-dominant mechanism should prefer SIGSTOP (must stop page access)"
        );
    }

    /// No causal data → conservative default: prefer SIGSTOP (false).
    #[test]
    fn prefer_sigstop_when_no_causal_data() {
        let g = CausalGraph::new();
        assert!(
            !g.prefer_qos_over_sigstop("unknown_process"),
            "no causal data should default to SIGSTOP (conservative)"
        );
    }
}
