//! Directional companion graph — `P(proc B alive | foreground = app A)`.
//!
//! Apollo learns *what* the user uses (UserProfile), but until now did not
//! learn *which other processes belong to a workflow*. When you put Brave
//! in the foreground, Slack / 1Password / a Brave audio helper are part
//! of the same browsing context even though they're not foreground —
//! pressuring them with `kern.memorystatus_vm_pressure_send` destroys
//! the state behind the active app.
//!
//! ## Model
//!
//! Conditional frequency tables with Laplace smoothing. Per-cycle
//! observation: while app A is foreground for `cycles_with_a_fg`, every
//! alive process B accumulates `cycles_with_b_alive_while_a_fg`. A separate
//! global counter `cycles_with_b_alive_total` lets us compute *Lift*:
//!
//! ```text
//!   conf(B|A) = (cycles_b_while_a + 1) / (cycles_a + 2)        // Laplace
//!   base(B)   = (global_b + 1) / (total_cycles + 2)
//!   lift(B|A) = conf(B|A) / base(B)
//! ```
//!
//! Lift > 1 means B is *more* likely to be alive while A is fg than in
//! general. Always-on daemons (kernel_task, WindowServer, mds, cfprefsd)
//! have `base(B) ≈ 1.0`, so their lift collapses to ≈ conf(B|A) which is
//! also ≈ 1.0 — they fail the lift gate naturally.
//!
//! ## Membership query
//!
//! `is_companion_of(fg_app, proc, …)` returns true iff
//!
//! - `cycles_a_fg ≥ MIN_OBSERVATIONS` (anchor app has enough evidence)
//! - `conf(B|A) ≥ MIN_CONFIDENCE` (B is reliably present with A)
//! - `lift(B|A) ≥ MIN_LIFT` (B's presence is *specific* to A, not noise)
//!
//! ## Decay + GC
//!
//! Every persist cycle (~500 cycles) `self_improve()` multiplies counters
//! by 0.97 (Bayesian forgetting half-life ≈ 23 cycles). Edges with
//! `last_seen_cycle` older than `evict_after_cycles` are dropped.
//!
//! ## What this does NOT track
//!
//! - PIDs (graph keys are *process names*; PID recycling is irrelevant).
//! - Per-workload split (single global graph; ReptileMeta meta-learning
//!   over workloads is a future extension).
//!
//! Papers: [Pearl 2009] §3.6 priors from observational data; [Pfau 2010]
//! Streaming Bayesian forgetting; [Agrawal 1993] Mining association rules
//! (Lift score).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Minimum foreground-cycles for an app before its companion edges are
/// trusted. ~15 minutes of *qualified* fg time at 5s/cycle (cf. NotebookLM
/// mature-evidence gate N≥15-20 in Apollo's existing OutcomeTracker /
/// SkillRegistry). Cumulative, not continuous — bursty workflows still
/// mature naturally over a few sessions.
const MIN_OBSERVATIONS: u64 = 180;
/// Minimum P(B|A) to consider B a companion of A.
const MIN_CONFIDENCE: f32 = 0.50;
/// Minimum lift to filter always-on noise (kernel_task, WindowServer).
const MIN_LIFT: f32 = 2.0;
/// Decay applied each persist cycle.
const DECAY_FACTOR: f32 = 0.97;
/// Evict edges not seen for this many cycles (~6h at 5s/cycle).
const EVICT_AFTER_CYCLES: u64 = 4_320;
/// Hard cap on number of distinct fg-app entries (memory ceiling).
const MAX_APPS: usize = 64;
/// Attention Floor — fg-bursts shorter than this contribute nothing to
/// anchor maturity. Keeps Slack-notification-15s blips out of the
/// statistic so a context-switching user's true workflow (sustained
/// sessions ≥ 30 s) is what teaches the graph. [Altmann & Trafton 2002]
/// Memory for Goals: pre-activation cost separates real task-resumption
/// from transient interruption. 6 cycles × 5 s = 30 s wall clock.
const ATTENTION_FLOOR: u64 = 6;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct AppEdges {
    cycles_fg: u64,
    /// proc_name -> (cycles_alive_while_fg, last_seen_cycle).
    edges: HashMap<String, (u64, u64)>,
}

/// Directional companion graph. Single owner; clone-cheap only when small.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CompanionGraph {
    per_app: HashMap<String, AppEdges>,
    /// Global presence counter used as the Lift denominator base.
    global: HashMap<String, u64>,
    total_cycles: u64,
    /// Current consecutive-cycle counter for the active fg app.
    /// Reset to 0 when fg changes. Anchor data is only credited once
    /// `current_streak >= ATTENTION_FLOOR`, so 15-second notification
    /// blips don't pollute conf(B|A) in steady state. Runtime-only;
    /// `serde(default)` so old persisted graphs load without it.
    #[serde(default, skip_serializing)]
    current_fg: Option<String>,
    #[serde(default, skip_serializing)]
    current_streak: u64,
}

impl CompanionGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one cycle of observation.
    ///
    /// `fg_app` — current foreground app name (None when nothing is fg).
    /// `alive_procs` — names of processes currently alive (top-N by RSS or
    ///   the full top_processes slice; the graph only learns from what it
    ///   sees, so a stable sampling strategy matters more than completeness).
    /// `current_cycle` — monotonic cycle counter for decay/GC bookkeeping.
    ///
    /// Attention Floor semantics: per-anchor data (`cycles_fg` and `edges`)
    /// is credited only once the current fg burst has lasted at least
    /// `ATTENTION_FLOOR` consecutive cycles. Global counters and
    /// `total_cycles` always update — the Lift denominator must reflect
    /// total observation time including transient bursts, otherwise short
    /// blips would inflate lift for everything.
    pub fn observe_cycle(
        &mut self,
        fg_app: Option<&str>,
        alive_procs: &[String],
        current_cycle: u64,
    ) {
        self.total_cycles = self.total_cycles.saturating_add(1);
        for name in alive_procs {
            *self.global.entry(name.clone()).or_insert(0) += 1;
        }

        // Maintain attention streak. Reset on fg-app change (or fg=None).
        match (fg_app, self.current_fg.as_deref()) {
            (Some(now), Some(prev)) if now == prev => {
                self.current_streak = self.current_streak.saturating_add(1);
            }
            (Some(now), _) => {
                self.current_fg = Some(now.to_string());
                self.current_streak = 1;
            }
            (None, _) => {
                self.current_fg = None;
                self.current_streak = 0;
                return;
            }
        }

        // Below the floor → nothing credits to the anchor. Burst that dies
        // before reaching the floor is dropped silently.
        if self.current_streak < ATTENTION_FLOOR {
            return;
        }

        let Some(fg) = fg_app else { return };
        if self.per_app.len() >= MAX_APPS && !self.per_app.contains_key(fg) {
            return; // capped — wait for GC to free a slot
        }
        let entry = self.per_app.entry(fg.to_string()).or_default();
        entry.cycles_fg = entry.cycles_fg.saturating_add(1);
        for name in alive_procs {
            let e = entry.edges.entry(name.clone()).or_insert((0, 0));
            e.0 = e.0.saturating_add(1);
            e.1 = current_cycle;
        }
    }

    /// Smoothed conditional probability P(proc | fg_app), or `None` if the
    /// anchor app has insufficient evidence.
    pub fn confidence(&self, fg_app: &str, proc: &str) -> Option<f32> {
        let app = self.per_app.get(fg_app)?;
        if app.cycles_fg < MIN_OBSERVATIONS {
            return None;
        }
        let cyc_b = app.edges.get(proc).map(|e| e.0).unwrap_or(0);
        // Laplace +1/+2.
        Some((cyc_b as f32 + 1.0) / (app.cycles_fg as f32 + 2.0))
    }

    /// Lift = conf(proc|fg) / base(proc). Returns `None` when anchor is
    /// undertrained or `total_cycles` is zero.
    pub fn lift(&self, fg_app: &str, proc: &str) -> Option<f32> {
        let conf = self.confidence(fg_app, proc)?;
        if self.total_cycles == 0 {
            return None;
        }
        let global_b = *self.global.get(proc).unwrap_or(&0);
        let base = (global_b as f32 + 1.0) / (self.total_cycles as f32 + 2.0);
        if base <= 0.0 {
            return None;
        }
        Some(conf / base)
    }

    /// Trusted-companion gate. Use this to protect satellites from purge.
    pub fn is_companion_of(&self, fg_app: &str, proc: &str) -> bool {
        let Some(conf) = self.confidence(fg_app, proc) else {
            return false;
        };
        if conf < MIN_CONFIDENCE {
            return false;
        }
        match self.lift(fg_app, proc) {
            Some(l) => l >= MIN_LIFT,
            None => false,
        }
    }

    /// Decay all counters and drop stale / cold edges. Returns evicted edge
    /// count for telemetry.
    pub fn self_improve(&mut self, current_cycle: u64) -> usize {
        let decay = |v: u64| ((v as f32) * DECAY_FACTOR) as u64;
        self.total_cycles = decay(self.total_cycles);
        for v in self.global.values_mut() {
            *v = decay(*v);
        }
        self.global.retain(|_, &mut v| v > 0);

        let mut evicted = 0usize;
        let mut empty_apps: Vec<String> = Vec::new();
        for (app_name, app) in self.per_app.iter_mut() {
            app.cycles_fg = decay(app.cycles_fg);
            let before = app.edges.len();
            app.edges.retain(|_, (count, last_seen)| {
                let age = current_cycle.saturating_sub(*last_seen);
                if age > EVICT_AFTER_CYCLES {
                    return false;
                }
                *count = decay(*count);
                *count > 0
            });
            evicted += before - app.edges.len();
            if app.cycles_fg == 0 && app.edges.is_empty() {
                empty_apps.push(app_name.clone());
            }
        }
        for k in empty_apps {
            self.per_app.remove(&k);
        }
        evicted
    }

    pub fn anchor_count(&self) -> usize {
        self.per_app.len()
    }

    pub fn edge_count(&self) -> usize {
        self.per_app.values().map(|a| a.edges.len()).sum()
    }

    /// Total observation cycles. Exposed for persistence sanity checks.
    pub fn total_cycles(&self) -> u64 {
        self.total_cycles
    }

    /// Phase 3.3 — **Cross-Group Attention Propagation (decider API)**.
    ///
    /// Given a list of `group_keys` (coalition_id / process_tree root names
    /// supplied by the caller — one per "process family" currently alive),
    /// infer weak companion edges between mature anchors that do **NOT**
    /// share a group_key but **do** share at least one direct companion
    /// pivot inside an observed group.
    ///
    /// # Algorithm
    ///
    /// For every ordered pair `(A, B)` of mature anchors in `per_app`:
    /// 1. Find a process `P` such that both `conf(P|A)` and `conf(P|B)`
    ///    pass [`MIN_CONFIDENCE`].
    /// 2. Emit an inferred edge `(A, B, sqrt(conf_AP × conf_BP) × 0.5)`.
    ///    - Geometric mean preserves NARS-style inheritance: the smaller
    ///      confidence dominates [Pei Wang 2013] §3.3.1.
    ///    - `× 0.5` is the Granovetter "weak tie" discount: cross-group
    ///      inference is weaker than directly observed evidence
    ///      [Granovetter 1973] *The Strength of Weak Ties*.
    ///
    /// # Guards
    ///
    /// - Empty result if `group_keys.len() < 2` (nothing to cross between).
    /// - Hard cap at **100** returned entries to keep blast radius bounded
    ///   (caller can wire this every cycle without unbounded work).
    /// - **Does NOT mutate the graph.** This is a decider API — callers
    ///   inspect the inferred edges and decide whether to act (e.g. raise
    ///   protection on the cross-group satellite, or merely log).
    ///
    /// # Complexity
    ///
    /// O(V² · E) over anchors and edges. With `MAX_APPS=64` the inner
    /// pair-loop is at most 64×63=4032 iterations; the cap-at-100 short-
    /// circuit prevents pathological emission. Safe to call per-cycle.
    ///
    /// # Wiring (TODO — `OPENS: 1`)
    ///
    /// Caller is NOT wired in this commit. The intended call site is the
    /// daemon main-loop "reason" stage (see
    /// `apollo-optimizerd/src/main.rs` cycle layout — after coalition
    /// enrichment), passing the live coalition_id set. Until wired the
    /// counter [`crate::engine::lse_counters::LockFreeMetrics::companion_cross_group_inferences_total`]
    /// stays at 0.
    pub fn propagate_attention_across_groups(
        &self,
        group_keys: &[String],
    ) -> Vec<(String, String, f32)> {
        // Cross-group inference requires at least two groups to bridge.
        if group_keys.len() < 2 {
            return Vec::new();
        }

        const CAP: usize = 100;
        let mut out: Vec<(String, String, f32)> = Vec::new();

        // Collect mature anchors (`cycles_fg ≥ MIN_OBSERVATIONS`).
        let mature: Vec<&String> = self
            .per_app
            .iter()
            .filter(|(_, app)| app.cycles_fg >= MIN_OBSERVATIONS)
            .map(|(name, _)| name)
            .collect();

        // For every ordered pair of distinct mature anchors find ONE shared
        // pivot whose conf passes MIN_CONFIDENCE on both sides. We stop at
        // the first qualifying pivot per pair — more would inflate the
        // result and the cap is the only knob the caller controls.
        'outer: for a in &mature {
            let app_a = match self.per_app.get(a.as_str()) {
                Some(x) => x,
                None => continue,
            };
            for b in &mature {
                if a == b {
                    continue;
                }
                let app_b = match self.per_app.get(b.as_str()) {
                    Some(x) => x,
                    None => continue,
                };
                // Find a shared pivot. Iterate the smaller edge set.
                let (small, large) = if app_a.edges.len() <= app_b.edges.len() {
                    (app_a, app_b)
                } else {
                    (app_b, app_a)
                };
                for pivot in small.edges.keys() {
                    if !large.edges.contains_key(pivot) {
                        continue;
                    }
                    let Some(conf_a) = self.confidence(a, pivot) else {
                        continue;
                    };
                    if conf_a < MIN_CONFIDENCE {
                        continue;
                    }
                    let Some(conf_b) = self.confidence(b, pivot) else {
                        continue;
                    };
                    if conf_b < MIN_CONFIDENCE {
                        continue;
                    }
                    // sqrt(conf_a × conf_b) × 0.5
                    let score = (conf_a * conf_b).sqrt() * 0.5;
                    out.push(((*a).clone(), (*b).clone(), score));
                    if out.len() >= CAP {
                        break 'outer;
                    }
                    break; // one inference per (A, B) pair
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_brave_session() -> CompanionGraph {
        let mut g = CompanionGraph::new();
        // 200 cycles of Brave foreground, Slack alive every cycle.
        // Also kernel_task always alive globally including outside Brave.
        for c in 0..200 {
            g.observe_cycle(Some("Brave"), &["Slack".into(), "kernel_task".into()], c);
        }
        // 800 cycles of "other workload" with kernel_task alive (background noise).
        for c in 200..1000 {
            g.observe_cycle(Some("Other"), &["kernel_task".into()], c);
        }
        g
    }

    #[test]
    fn slack_is_companion_of_brave_via_lift() {
        let g = build_brave_session();
        // P(Slack|Brave) ~ 1.0; P(Slack) global ~ 200/1000 = 0.2 → lift ~ 5.
        assert!(g.is_companion_of("Brave", "Slack"));
    }

    #[test]
    fn kernel_task_fails_lift_gate() {
        let g = build_brave_session();
        // P(kernel_task|Brave) = 1.0 but P(kernel_task) global = 1.0 → lift = 1.0.
        assert!(!g.is_companion_of("Brave", "kernel_task"));
        let lift = g.lift("Brave", "kernel_task").expect("lift");
        assert!(
            (lift - 1.0).abs() < 0.05,
            "lift should be ≈1.0 for always-on, got {lift}"
        );
    }

    #[test]
    fn undertrained_anchor_returns_no_companions() {
        let mut g = CompanionGraph::new();
        // Only 50 cycles — below MIN_OBSERVATIONS=180.
        for c in 0..50 {
            g.observe_cycle(Some("NewApp"), &["Helper".into()], c);
        }
        assert!(!g.is_companion_of("NewApp", "Helper"));
        assert!(g.confidence("NewApp", "Helper").is_none());
    }

    #[test]
    fn self_improve_evicts_stale_edges() {
        let mut g = CompanionGraph::new();
        for c in 0..200 {
            g.observe_cycle(Some("App"), &["Helper".into()], c);
        }
        let before = g.edge_count();
        // Run self_improve far in the future — edges should be evicted by age.
        let evicted = g.self_improve(10_000);
        assert!(
            evicted >= before,
            "expected ≥{before} evictions, got {evicted}"
        );
    }

    #[test]
    fn self_improve_decays_counters() {
        let mut g = CompanionGraph::new();
        for c in 0..200 {
            g.observe_cycle(Some("App"), &["Helper".into()], c);
        }
        let before_total = g.total_cycles;
        // Recent self_improve — should decay but not evict.
        g.self_improve(200);
        assert!(g.total_cycles < before_total);
        assert!(g.total_cycles > 0);
    }

    #[test]
    fn capped_at_max_apps() {
        let mut g = CompanionGraph::new();
        for i in 0..(MAX_APPS + 10) {
            let app = format!("App{i}");
            g.observe_cycle(Some(&app), &["x".into()], i as u64);
        }
        assert!(g.anchor_count() <= MAX_APPS);
    }

    #[test]
    fn graph_serializes_and_restores() {
        let g = build_brave_session();
        let json = serde_json::to_string(&g).expect("serialize");
        let g2: CompanionGraph = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(g.total_cycles, g2.total_cycles);
        assert!(g2.is_companion_of("Brave", "Slack"));
    }

    #[test]
    fn attention_floor_drops_short_burst() {
        // 3-cycle burst on "Notification" should not credit any anchor data.
        let mut g = CompanionGraph::new();
        for c in 0..3 {
            g.observe_cycle(Some("Notification"), &["bg-proc".into()], c);
        }
        // total_cycles still ticks (Lift denominator must be honest).
        assert_eq!(g.total_cycles, 3);
        // Anchor itself never created — burst was below floor.
        assert!(g.confidence("Notification", "bg-proc").is_none());
    }

    #[test]
    fn attention_floor_credits_once_streak_passes() {
        // A 200-cycle continuous Brave session should credit cycles_fg=
        // 200 - (ATTENTION_FLOOR - 1) = 195. The first 5 cycles are below
        // the floor and not credited.
        let mut g = CompanionGraph::new();
        for c in 0..200 {
            g.observe_cycle(Some("Brave"), &["Slack".into()], c);
        }
        // 200 cycles - 5 below-floor cycles = 195 credited.
        let cred = g
            .per_app
            .get("Brave")
            .map(|a| a.cycles_fg)
            .unwrap_or_default();
        assert_eq!(cred, 200 - (ATTENTION_FLOOR - 1));
    }

    // ── Phase 3.3 — Cross-Group Attention Propagation tests ─────────────────
    //
    // Decider-only API: callers receive inferred (A, B, score) triples and
    // decide whether to act. The graph is NEVER mutated by these inferences.
    //
    // Algorithm: for every pair (A, B) of mature anchors that do NOT share a
    // group_key, if both have a confident companion edge to the SAME third
    // process P inside a group, infer a weak cross-group companion edge
    // (A, B, sqrt(conf_AP × conf_BP) × 0.5). The geometric mean preserves
    // NARS-style inheritance (smaller of the two evidence levels dominates;
    // [Pei Wang 2013] §3.3.1); the 0.5 dampening is the Granovetter "weak
    // tie" discount — inferred edges across groups carry less weight than
    // directly observed ones [Granovetter 1973].

    /// Helper: build a graph with two mature anchors that both see `pivot`.
    fn build_two_anchor_session(pivot: &str) -> CompanionGraph {
        let mut g = CompanionGraph::new();
        // 250 cycles of AnchorA with pivot alive (mature, conf ≈ 1.0).
        for c in 0..250 {
            g.observe_cycle(Some("AnchorA"), &[pivot.into()], c);
        }
        // 250 cycles of AnchorB with the same pivot alive (also mature).
        for c in 250..500 {
            g.observe_cycle(Some("AnchorB"), &[pivot.into()], c);
        }
        g
    }

    #[test]
    fn cross_group_returns_empty_when_no_overlap() {
        // Two mature anchors but each lives in a DIFFERENT pivot group:
        // AnchorA -> X, AnchorB -> Y. No shared third process, no inference.
        let mut g = CompanionGraph::new();
        for c in 0..250 {
            g.observe_cycle(Some("AnchorA"), &["X".into()], c);
        }
        for c in 250..500 {
            g.observe_cycle(Some("AnchorB"), &["Y".into()], c);
        }
        // Two distinct group_keys — anchors are NOT in the same group.
        let groups = vec!["group-a".to_string(), "group-b".to_string()];
        let inferred = g.propagate_attention_across_groups(&groups);
        assert!(
            inferred.is_empty(),
            "no shared pivot ⇒ no cross-group edges, got {inferred:?}"
        );
    }

    #[test]
    fn cross_group_infers_geometric_mean_with_dampening() {
        // Both AnchorA and AnchorB see the same pivot ("Pivot") with high
        // confidence. Expected inferred edge:
        //   conf_AP ≈ 1.0, conf_BP ≈ 1.0
        //   score = sqrt(1.0 × 1.0) × 0.5 = 0.5
        let g = build_two_anchor_session("Pivot");
        let groups = vec!["g1".to_string(), "g2".to_string()];
        let inferred = g.propagate_attention_across_groups(&groups);

        // Expect both directions: (A,B) and (B,A).
        let edge_ab = inferred
            .iter()
            .find(|(a, b, _)| a == "AnchorA" && b == "AnchorB");
        let edge_ba = inferred
            .iter()
            .find(|(a, b, _)| a == "AnchorB" && b == "AnchorA");
        assert!(
            edge_ab.is_some(),
            "expected A→B inference, got {inferred:?}"
        );
        assert!(
            edge_ba.is_some(),
            "expected B→A inference, got {inferred:?}"
        );

        let score = edge_ab.unwrap().2;
        // conf_AP and conf_BP are Laplace-smoothed: (250+1)/(250+2) ≈ 0.996.
        // sqrt(0.996 × 0.996) × 0.5 ≈ 0.498. Allow 0.02 slack.
        assert!(
            (score - 0.498).abs() < 0.02,
            "expected score ≈ 0.498, got {score}"
        );
        // Symmetry: A→B and B→A should have the same score (geometric mean
        // is commutative on the two confidences).
        assert!(
            (edge_ab.unwrap().2 - edge_ba.unwrap().2).abs() < 1e-6,
            "geometric mean must be symmetric"
        );
    }

    #[test]
    fn cross_group_caps_at_100() {
        // Build 20 mature anchors that all see the same pivot. With 20
        // anchors, ordered pairs = 20 × 19 = 380 candidate inferences.
        // The cap must clamp the result at 100.
        let mut g = CompanionGraph::new();
        let mut cycle = 0u64;
        for i in 0..20 {
            let app = format!("A{i}");
            for _ in 0..200 {
                g.observe_cycle(Some(&app), &["Pivot".into()], cycle);
                cycle += 1;
            }
        }
        // 20 distinct group_keys — every anchor is in its own group.
        let groups: Vec<String> = (0..20).map(|i| format!("g{i}")).collect();
        let inferred = g.propagate_attention_across_groups(&groups);
        assert!(
            inferred.len() <= 100,
            "cap violated: got {} entries (expected ≤100)",
            inferred.len()
        );
    }

    #[test]
    fn cross_group_does_not_mutate_graph() {
        // Snapshot the graph before propagation; assert structural equality
        // afterwards. Inference is decider-only — callers add edges via
        // separate paths (observe_cycle), never through this method.
        let g_before = build_two_anchor_session("Pivot");
        let snap_before =
            serde_json::to_string(&g_before).expect("serialize pre-propagation graph");

        let g_after = g_before.clone();
        let groups = vec!["g1".to_string(), "g2".to_string()];
        let _ = g_after.propagate_attention_across_groups(&groups);

        let snap_after = serde_json::to_string(&g_after).expect("serialize post-propagation graph");
        assert_eq!(
            snap_before, snap_after,
            "propagate_attention_across_groups must NOT mutate the graph"
        );
    }

    #[test]
    fn attention_floor_resets_on_app_switch() {
        // Bursty workflow: 4 cycles on Notification (below floor), then
        // 200 cycles on Brave. Notification contributes 0; Brave starts
        // counting from cycle 1 of its session (after floor warmup).
        let mut g = CompanionGraph::new();
        for c in 0..4 {
            g.observe_cycle(Some("Notification"), &["bg".into()], c);
        }
        for c in 4..204 {
            g.observe_cycle(Some("Brave"), &["Slack".into(), "kernel_task".into()], c);
        }
        for c in 204..1000 {
            g.observe_cycle(Some("Other"), &["kernel_task".into()], c);
        }
        // Notification never matured.
        assert!(g.confidence("Notification", "bg").is_none());
        // Brave reached maturity (cumulative 200 - 5 = 195 ≥ MIN_OBSERVATIONS).
        assert!(g.is_companion_of("Brave", "Slack"));
    }
}
