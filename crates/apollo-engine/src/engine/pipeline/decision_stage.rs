//! # Decision Stage
//!
//! Wraps the call to [`decide_actions`] and its immediate inputs into a single
//! typed boundary, reducing the raw parameter count visible at the call site.
//!
//! ## Extraction status
//!
//! **PARTIAL** — the `decide_actions` call and its grouping inputs are
//! extracted.  Post-processing that touches binary-local state (`SharedState`,
//! `llm_daemon`, `skill_registry` trials, coordinated-cluster logic) remains
//! in the main daemon loop and operates on the returned [`DecisionStageOutput`].
//!
//! ## What is extracted
//!
//! 1. Locking `MachQoSManager` and calling `decide_actions`.
//! 2. Returning the full [`DecisionOutput`] wrapped in [`DecisionStageOutput`]
//!    so the caller can spread the results without re-importing `DecisionOutput`.
//!
//! ## What is NOT extracted (and why)
//!
//! - `state.last_blockers` / `state.thermal_state` mutations: require
//!   `SharedState`, which is binary-local and cannot be imported from a library
//!   crate module.
//! - `llm_daemon::apply_learned_policy_actions`: binary-local module.
//! - Skill-registry trial loop: depends on `pending_trial_skill` (binary-local
//!   mutable state), `foreground_pid`, and `collector` — too many disparate
//!   binary-local borrows for a clean library extraction.
//! - Coordinated causal-cluster pass: `OutcomeTracker::top_causal_pairs` result
//!   is used together with `collector.system().processes()` iteration — the same
//!   borrow-checker concern as the trial loop.
//!
//! ## Parameters grouped by [`PolicyContext`]
//!
//! The 7 Bayesian/pattern parameters from `decide_actions` are collapsed into
//! [`PolicyContext`], reducing the call site from 16 positional arguments to 11.
//!
//! ## Next step
//!
//! Once [`super::learning_context::LearningContext`] is wired into the main
//! loop, `qos_mgr` can be added to it and the remaining parameters can be
//! further collapsed.

use std::collections::{HashMap, HashSet};

use sysinfo::System;

use crate::collector::SystemSnapshot;
use crate::engine::decide_actions::{decide_actions, DecisionOutput};
use crate::engine::mach_qos::MachQoSManager;
use crate::engine::outcome_tracker::{HopGroupWeight, PatternWeight, WorkloadHop};
use crate::engine::overflow_guard::OverflowThresholds;
use crate::engine::types::{LatencyTarget, OptimizationProfile, RootAction};
use crate::engine::user_context::UserContext;

/// Policy-level inputs that control *which* processes are targeted and with
/// what learned weights.
///
/// These 6 parameters map 1-to-1 onto the `learned_*`, `outcome_*`, and
/// `*_pids` parameters of [`decide_actions`].
pub struct PolicyContext<'a> {
    /// Names of processes learned to be interactive (from [`crate::engine::llm::LearnedPolicy`]).
    pub decide_interactive: &'a [String],

    /// Names of processes learned to be low-value background noise.
    pub decide_noise: &'a [String],

    /// Bayesian per-pattern throttle weights from [`crate::engine::outcome_tracker::OutcomeTracker`].
    pub decide_weights: &'a HashMap<String, PatternWeight>,

    /// Counterfactual baseline: natural pressure-drop rate without actions.
    /// Processes are skipped if their observed effectiveness is <90% of this baseline.
    pub outcome_baseline: f64,

    /// PIDs detected as behaviorally interactive via cpu_wall_ratio EMA (<0.05).
    pub behavior_interactive_pids: &'a HashSet<u32>,

    /// Per-process IPC hints (ri_instructions/ri_cycles).
    /// Low IPC → memory-bound (safe to throttle); high IPC → compute-bound (avoid).
    pub ipc_hints: &'a HashMap<u32, f64>,

    /// HRPO group effectiveness from OutcomeTracker; groups with <15% effectiveness
    /// after sufficient observations are skipped.
    pub hop_groups: &'a HashMap<WorkloadHop, HopGroupWeight>,

    /// PIDs whose (cpu_bucket, rss_bucket) are stable for ≥N cycles.
    /// Their last decision is maintained without re-computing this cycle.
    pub habituated_pids: &'a HashSet<u32>,

    /// Pearl-style causal confidence map: `"throttle:ProcessName"` → [0,1].
    /// Processes with confidence <0.20 after ≥5 observations are skipped.
    pub causal_confidence: &'a HashMap<String, f32>,

    /// Pearl-style causal impact map: `"throttle:ProcessName"` → impact_score.
    /// Used for throttle ordering; confidence remains the skip/filter input.
    pub causal_impact: &'a HashMap<String, f32>,

    /// Current user context: idle time, sleep assertions, call detection, audio.
    /// Drives freeze gating and throttle conservatism based on user activity.
    pub user_ctx: &'a UserContext,

    /// Per-process wakeup rate (idle + interrupt wakeups/sec) from proc_pid_rusage.
    /// Battery vampire detection: >100/s = priority throttle target.
    pub wakeup_hints: &'a HashMap<u32, f64>,

    /// Per-process physical footprint (MB) from ri_phys_footprint.
    /// Used for freeze ranking instead of RSS (more accurate).
    pub footprint_hints: &'a HashMap<u32, f64>,

    /// DRAM bandwidth utilization 0.0–1.0 from IOReport AMC stats.
    pub dram_bandwidth_pct: f64,

    /// Per-process disk write rate (MB/s) from ri_disk_write_bytes delta.
    /// Throttle background I/O abusers (>5 MB/s) to protect LLM inference bandwidth.
    pub io_burst_hints: &'a HashMap<u32, f64>,

    /// Per-process behavioral anomaly score vs learned hardware counter baseline.
    /// ≥ 3.0 MADs from {ipc, wakeup_rate, disk_mbps} baseline = priority throttle.
    pub anomaly_hints: &'a HashMap<u32, f64>,

    /// Per-PID post-thaw cooldown set. gate_e ("swap-pct") freeze candidates
    /// in cooldown are skipped to prevent freeze→thaw→freeze oscillation.
    /// Other gates (a/b/c/d) ignore the cooldown.
    pub freeze_cooldown: &'a crate::engine::freeze_cooldown::FreezeCooldown,

    /// Sprint 12 Convergence #1 (2026-05-17). PIDs the CompanionGraph
    /// classifies as companions of the current foreground app, used by
    /// the cold-thread router to keep them on the same P-cluster as
    /// the foreground hot threads under low DRAM bandwidth.
    pub companion_of_foreground_pids: &'a HashSet<u32>,
}

/// The output returned by [`DecisionStage::run`].
///
/// Carries the full [`DecisionOutput`] so callers can access both the action
/// list and the metadata (blockers, context, low-value skips) without
/// re-importing the inner type.
pub struct DecisionStageOutput {
    /// The raw output from `decide_actions`, including actions and metadata.
    pub decision: DecisionOutput,
}

impl DecisionStageOutput {
    /// Consume self and return just the actions list.
    ///
    /// Convenience helper for call sites that only need actions and will
    /// read the other fields directly from [`Self::decision`] before calling this.
    pub fn into_actions(self) -> Vec<RootAction> {
        self.decision.actions
    }
}

/// The decision pipeline stage.
///
/// Stateless — all inputs are passed per-call via [`DecisionStage::run`].
/// Constructed once and reused across cycles (zero allocation on `new`).
pub struct DecisionStage;

impl DecisionStage {
    pub fn new() -> Self {
        Self
    }

    /// Execute one decision cycle.
    ///
    /// Calls `decide_actions` with the provided inputs and returns the full
    /// output.  The caller is responsible for:
    ///
    /// 1. Locking `MachQoSManager` and passing it as `qos_mgr`.
    /// 2. Updating `SharedState` fields (`last_blockers`, `thermal_state`,
    ///    `top_skipped_processes`) from `output.decision`.
    /// 3. Applying learned policy and skill-registry passes to `output.decision.actions`.
    ///
    /// # Parameters
    ///
    /// - `snapshot`: current system metrics snapshot.
    /// - `sys`: `sysinfo::System` process table (must be pre-refreshed).
    /// - `profile`: active optimization profile.
    /// - `latency_target`: latency SLO for the current workload.
    /// - `reactor_weight`: event weight from the reactive daemon (0.0–1.0).
    /// - `overflow_thresholds`: per-profile memory-pressure gate thresholds.
    /// - `qos_mgr`: optional mutable reference to the QoS manager (None in
    ///   non-root or test contexts).
    /// - `policy`: grouped Bayesian/pattern policy inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn run<'a>(
        &mut self,
        snapshot: &SystemSnapshot,
        sys: &System,
        profile: OptimizationProfile,
        latency_target: LatencyTarget,
        reactor_weight: f64,
        overflow_thresholds: OverflowThresholds,
        // S4 cutover (2026-06-06): shared Arc<Mutex<_>> per execute_actions
        // signature change.
        qos_mgr: Option<&std::sync::Arc<std::sync::Mutex<MachQoSManager>>>,
        policy: &PolicyContext<'a>,
    ) -> DecisionStageOutput {
        let decision = decide_actions(
            snapshot,
            sys,
            profile,
            latency_target,
            reactor_weight,
            policy.decide_interactive,
            policy.decide_noise,
            overflow_thresholds,
            qos_mgr,
            policy.decide_weights,
            policy.outcome_baseline,
            policy.behavior_interactive_pids,
            policy.ipc_hints,
            policy.hop_groups,
            policy.habituated_pids,
            policy.causal_confidence,
            policy.causal_impact,
            policy.user_ctx,
            policy.wakeup_hints,
            policy.footprint_hints,
            policy.dram_bandwidth_pct,
            policy.io_burst_hints,
            policy.anomaly_hints,
            policy.freeze_cooldown,
            policy.companion_of_foreground_pids,
        );

        DecisionStageOutput { decision }
    }
}

impl Default for DecisionStage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::{CpuStats, MemoryStats, PressureStats, SystemSnapshot};
    use crate::engine::overflow_guard::OverflowThresholds;
    use crate::engine::types::{LatencyTarget, OptimizationProfile};

    fn make_snapshot() -> SystemSnapshot {
        SystemSnapshot {
            timestamp: chrono::Utc::now(),
            cpu: CpuStats {
                global_usage: 0.0,
                core_count: 4,
            },
            memory: MemoryStats {
                total_ram: 8 * 1024 * 1024 * 1024,
                used_ram: 0,
                free_ram: 8 * 1024 * 1024 * 1024,
                total_swap: 0,
                used_swap: 0,
            },
            pressure: PressureStats {
                memory_pressure: 0.0,
                swap_used_bytes: 0,
                swap_total_bytes: 0,
                swap_delta_bytes_per_sec: 0.0,
                thermal_level: "nominal".to_string(),
                compressor_pressure: 0.0,
                thrashing_score: 0.0,
                memory_pressure_raw: 0.0,
            },
            disks: vec![],
            networks: vec![],
            top_processes: vec![],
        }
    }

    fn make_policy<'a>(
        interactive: &'a Vec<String>,
        noise: &'a Vec<String>,
        weights: &'a HashMap<String, PatternWeight>,
        pids: &'a HashSet<u32>,
        ipc: &'a HashMap<u32, f64>,
        hops: &'a HashMap<WorkloadHop, HopGroupWeight>,
        causal: &'a HashMap<String, f32>,
        impact: &'a HashMap<String, f32>,
        user_ctx: &'a UserContext,
        cooldown: &'a crate::engine::freeze_cooldown::FreezeCooldown,
        companions: &'a HashSet<u32>,
    ) -> PolicyContext<'a> {
        PolicyContext {
            decide_interactive: interactive,
            decide_noise: noise,
            decide_weights: weights,
            outcome_baseline: 0.0,
            behavior_interactive_pids: pids,
            ipc_hints: ipc,
            hop_groups: hops,
            habituated_pids: pids,
            causal_confidence: causal,
            causal_impact: impact,
            user_ctx,
            wakeup_hints: ipc, // reuse empty map (same type)
            footprint_hints: ipc,
            dram_bandwidth_pct: 0.0,
            io_burst_hints: ipc,
            anomaly_hints: ipc,
            freeze_cooldown: cooldown,
            companion_of_foreground_pids: companions,
        }
    }

    /// Verify that `DecisionStage::run` compiles and returns output with the
    /// expected structural fields, using default/empty inputs.
    #[test]
    fn decision_stage_runs_with_empty_inputs() {
        let mut stage = DecisionStage::new();
        let snapshot = make_snapshot();
        let sys = System::new();

        let empty_interactive: Vec<String> = Vec::new();
        let empty_noise: Vec<String> = Vec::new();
        let empty_weights: HashMap<String, PatternWeight> = HashMap::new();
        let empty_pids: HashSet<u32> = HashSet::new();
        let empty_ipc: HashMap<u32, f64> = HashMap::new();
        let empty_hops: HashMap<WorkloadHop, HopGroupWeight> = HashMap::new();
        let empty_causal: HashMap<String, f32> = HashMap::new();
        let empty_impact: HashMap<String, f32> = HashMap::new();
        let user_ctx = UserContext::default();
        let cooldown = crate::engine::freeze_cooldown::FreezeCooldown::new();
        let policy = make_policy(
            &empty_interactive,
            &empty_noise,
            &empty_weights,
            &empty_pids,
            &empty_ipc,
            &empty_hops,
            &empty_causal,
            &empty_impact,
            &user_ctx,
            &cooldown,
            &empty_pids,
        );

        let output = stage.run(
            &snapshot,
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.0,
            OverflowThresholds::default(),
            None, // no QoS manager in tests
            &policy,
        );

        // With zero pressure and no processes, actions should be empty.
        assert!(
            output.decision.actions.is_empty(),
            "no actions expected with zero pressure and empty process list"
        );
    }

    /// Verify `DecisionStageOutput::into_actions` consumes the output correctly.
    #[test]
    fn decision_stage_output_into_actions() {
        let mut stage = DecisionStage::new();
        let snapshot = make_snapshot();
        let sys = System::new();

        let empty_interactive: Vec<String> = Vec::new();
        let empty_noise: Vec<String> = Vec::new();
        let empty_weights: HashMap<String, PatternWeight> = HashMap::new();
        let empty_pids: HashSet<u32> = HashSet::new();
        let empty_ipc: HashMap<u32, f64> = HashMap::new();
        let empty_hops: HashMap<WorkloadHop, HopGroupWeight> = HashMap::new();
        let empty_causal: HashMap<String, f32> = HashMap::new();
        let empty_impact: HashMap<String, f32> = HashMap::new();
        let user_ctx = UserContext::default();
        let cooldown = crate::engine::freeze_cooldown::FreezeCooldown::new();
        let policy = make_policy(
            &empty_interactive,
            &empty_noise,
            &empty_weights,
            &empty_pids,
            &empty_ipc,
            &empty_hops,
            &empty_causal,
            &empty_impact,
            &user_ctx,
            &cooldown,
            &empty_pids,
        );

        let output = stage.run(
            &snapshot,
            &sys,
            OptimizationProfile::BalancedRoot,
            LatencyTarget::Normal,
            0.0,
            OverflowThresholds::default(),
            None,
            &policy,
        );

        // Metadata fields are accessible before consuming.
        let _blocker_count = output.decision.blockers.len();
        let actions = output.into_actions();
        assert!(actions.is_empty());
    }
}
