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
/// trusted. ~15 minutes at 5s/cycle (cf. NotebookLM mature-evidence gate
/// N≥15-20 in Apollo's existing OutcomeTracker / SkillRegistry).
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
        assert!((lift - 1.0).abs() < 0.05, "lift should be ≈1.0 for always-on, got {lift}");
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
        assert!(evicted >= before, "expected ≥{before} evictions, got {evicted}");
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
}
