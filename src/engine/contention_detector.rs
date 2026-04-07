//! Cache Contention Detection for Apple Silicon (M1+)
//!
//! On Apple Silicon, each core cluster (P-cluster, E-cluster) shares its own
//! L2 cache. Multiple CPU-heavy processes on the same cluster compete for L2,
//! causing cache thrashing and elevated miss rates — felt as system "choppiness".
//!
//! # Why IPC as proxy
//!
//! macOS does not expose per-process L2 miss counters publicly (KPC configurable
//! event IDs are private and chip-generation specific). Instead, we use
//! system-wide IPC as a contention signal:
//!
//!   Low IPC + multiple concurrent high-CPU processes → likely L2 contention
//!
//! This is conservative: IPC can be low for reasons other than cache contention
//! (I/O wait, lock contention). We require ≥3 consecutive detections before
//! promoting a pair to "confirmed", reducing false positives.
//!
//! # Action
//!
//! When a confirmed pair is detected, the caller can separate them to different
//! QoS tiers: heavy → P-cores (Foreground), light → E-cores (Background).
//! On M1, P-cluster and E-cluster have separate L2 caches → no competition.
//!
//! # References
//!
//! - Hennessy & Patterson (2017) "Computer Architecture: A Quantitative Approach"
//!   Ch.2 — memory hierarchy effects on IPC.
//! - Apple "Optimizing for Apple Silicon" WWDC21 — P-core / E-core semantics.

use std::collections::HashMap;

/// A pair of processes detected as likely competing for the same L2 cache cluster.
#[derive(Debug, Clone)]
pub struct ContentionPair {
    /// Process that should stay on P-cores (higher CPU, latency-sensitive).
    pub heavy_pid: u32,
    pub heavy_name: String,
    /// Process that can tolerate E-core routing without user-visible impact.
    pub light_pid: u32,
    pub light_name: String,
    /// How many consecutive cycles this pair was detected (max 10, used for confidence).
    pub consecutive_cycles: u32,
}

impl ContentionPair {
    /// Confidence score 0..1. Requires ≥3 cycles; saturates at 10.
    pub fn confidence(&self) -> f64 {
        ((self.consecutive_cycles as f64 - 2.0) / 8.0).clamp(0.0, 1.0)
    }
}

/// Per-cycle contention summary.
#[derive(Debug, Default)]
pub struct ContentionState {
    /// System-wide contention likelihood 0..1.
    /// Combines IPC factor (low IPC → high contention risk) and concurrent heavy process count.
    pub score: f64,
    /// Confirmed contention pairs (≥3 consecutive detection cycles).
    pub pairs: Vec<ContentionPair>,
    /// Number of processes with CPU above the detection threshold this cycle.
    pub heavy_count: usize,
}

/// Tracks cache contention between co-executing high-CPU processes.
///
/// Designed to live in the daemon state across cycles. `tick()` is called once
/// per main loop iteration with the current IPC and process snapshot.
pub struct ContentionDetector {
    /// How many consecutive cycles each (pid_lo, pid_hi) pair appeared as co-executing heavy.
    co_exec_cycles: HashMap<(u32, u32), u32>,
    /// PIDs that were heavy in the previous cycle (for co-execution detection).
    prev_heavy_pids: Vec<u32>,
}

impl ContentionDetector {
    pub fn new() -> Self {
        Self {
            co_exec_cycles: HashMap::new(),
            prev_heavy_pids: Vec::new(),
        }
    }

    /// Evaluate contention for this cycle.
    ///
    /// # Arguments
    /// - `system_ipc`: system-wide IPC from `KpcSnapshot.ipc` (0.0 if unavailable → neutral)
    /// - `processes`: (pid, name, cpu_percent) for all visible processes this cycle
    /// - `pressure`: memory pressure 0..1 (used to gate action threshold)
    /// - `min_cpu`: CPU% floor to classify a process as "heavy" (default: 15.0)
    pub fn tick(
        &mut self,
        system_ipc: f64,
        processes: &[(u32, String, f32)],
        pressure: f64,
        min_cpu: f32,
    ) -> ContentionState {
        // Identify high-CPU processes this cycle.
        let heavy: Vec<(u32, String, f32)> = processes
            .iter()
            .filter(|(_, _, cpu)| *cpu >= min_cpu)
            .cloned()
            .collect();

        // Contention score:
        //   ipc_factor:   low IPC → memory-bound → higher contention risk.
        //                 IPC 2.0 = compute-efficient (no contention) → factor 0.0
        //                 IPC 0.5 = memory-bound (contention likely) → factor 0.75
        //   heavy_factor: more concurrent heavy processes → more competition.
        //                 1 process: 0.0, 2: 0.33, 3: 0.67, 4+: 1.0
        let ipc_norm = if system_ipc > 0.0 {
            (1.0 - (system_ipc / 2.0).clamp(0.0, 1.0)).max(0.0)
        } else {
            0.3 // neutral when IPC unavailable
        };
        let heavy_factor = ((heavy.len().saturating_sub(1)) as f64 / 3.0).clamp(0.0, 1.0);
        let score = (ipc_factor(ipc_norm) * 0.65 + heavy_factor * 0.35).clamp(0.0, 1.0);

        // Update co-execution counters.
        // A pair increments only if BOTH processes were heavy in the *previous* cycle too
        // (stability requirement — avoids counting a single-cycle spike as contention).
        let prev_set: std::collections::HashSet<u32> =
            self.prev_heavy_pids.iter().copied().collect();

        let mut active_pairs: Vec<(u32, u32)> = Vec::new();
        for i in 0..heavy.len() {
            for j in (i + 1)..heavy.len() {
                let (pid_a, pid_b) = canonical_pair(heavy[i].0, heavy[j].0);
                if prev_set.contains(&heavy[i].0) && prev_set.contains(&heavy[j].0) {
                    let count = self.co_exec_cycles.entry((pid_a, pid_b)).or_insert(0);
                    *count = count.saturating_add(1).min(10);
                    active_pairs.push((pid_a, pid_b));
                }
            }
        }

        // Decay counts for pairs no longer co-executing.
        self.co_exec_cycles.retain(|(a, b), count| {
            if !active_pairs.contains(&(*a, *b)) {
                *count = count.saturating_sub(1);
                *count > 0
            } else {
                true
            }
        });

        // Emit confirmed pairs (≥3 consecutive cycles, score > 0.40, pressure > 0.25).
        let mut pairs = Vec::new();
        if score > 0.40 && pressure > 0.25 {
            for &(pid_a, pid_b) in &active_pairs {
                let cycles = self
                    .co_exec_cycles
                    .get(&(pid_a, pid_b))
                    .copied()
                    .unwrap_or(0);
                if cycles >= 3 {
                    // heavy_pid = whichever has higher CPU (keep on P-cores).
                    let cpu_a = heavy
                        .iter()
                        .find(|(p, _, _)| *p == pid_a)
                        .map(|(_, _, c)| *c)
                        .unwrap_or(0.0);
                    let cpu_b = heavy
                        .iter()
                        .find(|(p, _, _)| *p == pid_b)
                        .map(|(_, _, c)| *c)
                        .unwrap_or(0.0);
                    let (heavy_pid, light_pid) = if cpu_a >= cpu_b {
                        (pid_a, pid_b)
                    } else {
                        (pid_b, pid_a)
                    };
                    let heavy_name = heavy
                        .iter()
                        .find(|(p, _, _)| *p == heavy_pid)
                        .map(|(_, n, _)| n.clone())
                        .unwrap_or_default();
                    let light_name = heavy
                        .iter()
                        .find(|(p, _, _)| *p == light_pid)
                        .map(|(_, n, _)| n.clone())
                        .unwrap_or_default();
                    pairs.push(ContentionPair {
                        heavy_pid,
                        heavy_name,
                        light_pid,
                        light_name,
                        consecutive_cycles: cycles,
                    });
                }
            }
        }

        self.prev_heavy_pids = heavy.iter().map(|(pid, _, _)| *pid).collect();
        let heavy_count = heavy.len();
        ContentionState {
            score,
            pairs,
            heavy_count,
        }
    }

    /// Remove dead PIDs from tracking state.
    pub fn gc(&mut self, alive_pids: &std::collections::HashSet<u32>) {
        self.co_exec_cycles
            .retain(|(a, b), _| alive_pids.contains(a) && alive_pids.contains(b));
        self.prev_heavy_pids.retain(|pid| alive_pids.contains(pid));
    }
}

/// Returns (min, max) pair for stable HashMap keys regardless of argument order.
#[inline]
fn canonical_pair(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// Soft nonlinear scaling: low IPC punished harder than mid-range.
#[inline]
fn ipc_factor(norm: f64) -> f64 {
    // Quadratic: norm=0 → 0.0, norm=0.5 → 0.25, norm=1.0 → 1.0 (flipped → 0.0..1.0)
    norm * norm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn procs(list: &[(u32, &str, f32)]) -> Vec<(u32, String, f32)> {
        list.iter()
            .map(|(p, n, c)| (*p, n.to_string(), *c))
            .collect()
    }

    #[test]
    fn no_contention_with_single_heavy_process() {
        let mut det = ContentionDetector::new();
        let p = procs(&[(1, "rustc", 80.0), (2, "launchd", 0.1)]);
        let state = det.tick(0.4, &p, 0.6, 15.0);
        assert!(
            state.pairs.is_empty(),
            "single heavy process cannot form a pair"
        );
    }

    #[test]
    fn no_pair_before_3_consecutive_cycles() {
        let mut det = ContentionDetector::new();
        let p = procs(&[(1, "rustc", 60.0), (2, "ollama", 40.0)]);
        for i in 0..2 {
            let state = det.tick(0.3, &p, 0.7, 15.0);
            assert!(
                state.pairs.is_empty(),
                "cycle {i}: pair should not fire before 3 cycles"
            );
        }
    }

    #[test]
    fn pair_fires_after_3_cycles() {
        let mut det = ContentionDetector::new();
        let p = procs(&[(1, "rustc", 60.0), (2, "ollama", 40.0)]);
        let mut last_state = ContentionState::default();
        for _ in 0..4 {
            last_state = det.tick(0.3, &p, 0.7, 15.0);
        }
        assert!(
            !last_state.pairs.is_empty(),
            "pair should fire after 3+ cycles"
        );
        assert_eq!(
            last_state.pairs[0].heavy_pid, 1,
            "rustc has higher CPU → heavy"
        );
        assert_eq!(last_state.pairs[0].light_pid, 2);
    }

    #[test]
    fn high_ipc_suppresses_score() {
        let mut det = ContentionDetector::new();
        let p = procs(&[(1, "rustc", 60.0), (2, "ollama", 40.0)]);
        // IPC = 3.0 (above saturation) → ipc_factor = 0, score should be low
        let state = det.tick(3.0, &p, 0.7, 15.0);
        assert!(
            state.score < 0.40,
            "high IPC means compute-bound, not cache-bound"
        );
    }

    #[test]
    fn pair_decays_when_process_disappears() {
        let mut det = ContentionDetector::new();
        let p2 = procs(&[(1, "rustc", 60.0), (2, "ollama", 40.0)]);
        let p1 = procs(&[(1, "rustc", 60.0)]); // ollama gone
        for _ in 0..4 {
            det.tick(0.3, &p2, 0.7, 15.0);
        }
        // Now ollama gone — pair should not fire
        for _ in 0..3 {
            det.tick(0.3, &p1, 0.7, 15.0);
        }
        let state = det.tick(0.3, &p1, 0.7, 15.0);
        assert!(
            state.pairs.is_empty(),
            "pair should decay when one process disappears"
        );
    }

    #[test]
    fn confidence_saturates_at_10_cycles() {
        let pair = ContentionPair {
            heavy_pid: 1,
            heavy_name: "a".into(),
            light_pid: 2,
            light_name: "b".into(),
            consecutive_cycles: 10,
        };
        assert!((pair.confidence() - 1.0).abs() < 0.01);
    }

    #[test]
    fn gc_removes_dead_pids() {
        let mut det = ContentionDetector::new();
        let p = procs(&[(1, "rustc", 60.0), (2, "ollama", 40.0)]);
        for _ in 0..4 {
            det.tick(0.3, &p, 0.7, 15.0);
        }
        let alive: std::collections::HashSet<u32> = [1].into_iter().collect();
        det.gc(&alive);
        assert!(
            det.co_exec_cycles.is_empty(),
            "dead pair should be removed by gc"
        );
    }
}
