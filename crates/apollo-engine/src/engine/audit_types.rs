use crate::engine::types::RootAction;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecisionReason {
    /// Action prompted by wait-graph contention score.
    WaitGraphBlocker,
    /// Proactive boost for known interactive applications.
    InteractiveFocus,
    /// Action justified by high causal confidence (Pearl 2009).
    CausalInference,
    /// Action skipped because OutcomeTracker predicts low effectiveness.
    OutcomeIneffective,
    /// Throttling avoided because process is compute-bound (High IPC).
    IpcProtected,
    /// Throttling escalation due to behavioral anomaly (MAD > 3.0).
    AnomalyDetected,
    /// Throttling escalation due to high disk write rate (> 5MB/s).
    IoBurst,
    /// Throttling escalation due to high wakeup rate (> 100/s).
    WakeupVampire,
    /// Throttling escalation due to high DRAM bandwidth utilization.
    DramBandwidth,
    /// Proactive P-core routing for ML/AMX inference workloads.
    MLWorkload,
    /// Proactive boost for display pipeline daemons during swap growth.
    DisplayPipeline,
    /// Proactive boost for WindowServer during high compositor load.
    CompositorPriority,
    /// Action prompted by general system pressure context.
    PressureContext,
    /// Action prompted by memory budget enforcement (8GB/16GB constraints).
    MemoryBudget,
    /// Action triggered by entering Critical zone (bypassing rate-limit).
    CriticalBypass,
    /// Action allowed after hysteresis recovery (exiting high pressure).
    HysteresisRecovery,
    /// Heuristic skip: user is recently active and pressure is low.
    UserActiveSkip,
    /// Heuristic skip: HRPO group effectiveness is too low (Dr. Zero).
    HrpoSkip,
    /// Adaptive governor swarm rule (>30 procs competing, waste >= 0.30, no GUI).
    /// [Saltzer & Schroeder 1975] economy of mechanism — system-wide signal,
    /// not per-process pressure attribution.
    SwarmThrottling,
    /// Adaptive governor graduated-idle rules: 6h+ idle no GUI → Throttle;
    /// 12h+ idle → Freeze; 24h+ GUI abandonment → Freeze.
    /// [Denning 1968] working-set decay; cold-set reclamation.
    GraduatedIdle,
    /// Per-thread Mach QoS routing decision (interactive vs background tier,
    /// or P-cluster/E-cluster affinity hint on Apple Silicon).
    /// [ARM big.LITTLE 2013 §3] thread-level affinity reduces migration cost.
    ThreadQoSRouting,
    /// Other — free-form reason.
    Other(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BlockReason {
    /// Process is in the hardcoded L0 protection list.
    ProtectedProcess,
    /// Process is an active ML/AMX workload.
    MlProtected,
    /// PID was recycled or process died between snapshot and execution.
    PidRecycled,
    /// Process is a signed Apple platform binary (cs_platform_binary).
    ApplePlatform,
    /// Process is critical infrastructure (docker, postgres, etc.).
    CriticalBackground,
    /// Process holds a power assertion (audio, download, sleep-prevent).
    AssertionActive,
    /// Process is in a zombie state.
    Zombie,
    /// Sysctl key not found or read failed.
    InvalidSysctl,
    /// Sysctl value out of allowed range.
    SysctlOutOfRange,
    /// Sysctl write failed (e.g. permission or kernel state).
    SysctlFailed,
    /// Memorystatus/Jetsam priority change failed.
    MemorystatusFailed,
    /// Per-cycle action budget exceeded for this category.
    BudgetExhausted,
    /// Process-level cooldown is still active.
    CooldownActive,
    /// Action blocked by dry-run mode.
    DryRun,
    /// Action blocked by global safety circuit-breaker.
    CircuitBreakerActive,
    /// Epistemic uncertainty too high.
    EpistemicHigh,
    /// Process belongs to a coalition that was foreground within the
    /// active grace window — subprocess of an active user workflow.
    ActiveCoalition,
    /// CPU pegged across cores while memory headroom adequate — freeze
    /// would worsen scheduler contention without easing the real bottleneck.
    /// Per `cpu_saturation::CpuSaturation::pegged_fraction` >= 0.80 AND
    /// memory_pressure < 0.75. Sensor introduced 2026-04-08; wired into
    /// decision path 2026-05-10.
    CpuSaturated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecisionTrace {
    pub t: DateTime<Utc>,
    pub cycle: u64,

    /// The action the policy *intended* to take.
    pub intended_action: RootAction,

    /// The reason why the policy chose this action.
    pub decision_reason: DecisionReason,

    /// Whether the action was actually applied to the system.
    pub applied: bool,

    /// If not applied, the reason why it was blocked.
    pub block_reason: Option<BlockReason>,

    /// Snapshot of relevant pressure indicators.
    pub pressure: f32,
    pub swap_gb: f32,
    pub thrashing: f32,
}
