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

/// Phase 5.3 — structured, machine-readable explanation of why an action was
/// taken. Attached to [`crate::engine::types::JournalEntry`] as an optional
/// field so old journal lines (no `rationale` key) still parse cleanly via
/// `#[serde(default)]` on the consumer.
///
/// The rationale is intentionally bounded and cheap to construct so it can be
/// emitted on every action in the daemon hot path without violating the
/// per-cycle work budget documented in `CLAUDE.md`. No heap-heavy debug
/// formatting, no allocator churn — three small strings + an `Option`.
///
/// References:
/// - [Doshi-Velez & Kim 2017] "Towards a Rigorous Science of Interpretable
///   Machine Learning" — structured local explanations enable post-hoc
///   audit and disagreement-driven model improvement.
/// - [Ribeiro et al. 2016] "Why Should I Trust You? Explaining the
///   Predictions of Any Classifier" (LIME) — per-decision explanations
///   built from the same evidence the model uses, not free text.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rationale {
    /// One-word action class, e.g. `"throttle"`, `"freeze"`, `"unfreeze"`,
    /// `"boost"`, `"sysctl"`, `"memorystatus"`. Stored as
    /// [`std::borrow::Cow<'static, str>`] so the common case (compile-time
    /// literal) pays zero allocation via `Cow::Borrowed`, while serde
    /// can still round-trip through `Cow::Owned(String)` on deserialise.
    pub action_class: std::borrow::Cow<'static, str>,

    /// The pattern, specialist, or rule that triggered the action. Example:
    /// `"PressureSpecialist:critical-zone"`, `"chromium_manager:long_idle"`,
    /// `"swarm-throttle"`. Free-form but should be stable enough that
    /// downstream tooling can group by it.
    pub trigger: String,

    /// Brief, machine-parseable evidence. Convention: `k=v` pairs separated
    /// by commas, e.g. `"p_oom_30s=0.62,pressure=0.78,swap_gb=2.1"`. Keep
    /// short — this rides on every journal line.
    pub evidence: String,

    /// Optional expected-outcome metric for closed-loop attribution.
    /// Convention: `metric=delta`, e.g. `"expected_pressure_drop=0.05"` or
    /// `"expected_swap_release_mb=180"`. When wired through to the
    /// outcome tracker (follow-up commit), this turns the rationale into
    /// a falsifiable prediction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_outcome: Option<String>,
}

impl Rationale {
    /// Construct a rationale with the three mandatory fields. The
    /// expected-outcome slot defaults to `None`; chain
    /// [`Rationale::with_expected_outcome`] to populate it.
    ///
    /// `action_class` accepts a `&'static str` and is stored as
    /// `Cow::Borrowed` — the common case (compile-time literal) pays zero
    /// allocation. `trigger`/`evidence` accept anything that
    /// `Into<String>` so callers can pass either `&str` or `String`
    /// without ceremony.
    pub fn new(
        action_class: &'static str,
        trigger: impl Into<String>,
        evidence: impl Into<String>,
    ) -> Self {
        Self {
            action_class: std::borrow::Cow::Borrowed(action_class),
            trigger: trigger.into(),
            evidence: evidence.into(),
            expected_outcome: None,
        }
    }

    /// Attach an expected-outcome prediction. Consumers (outcome tracker,
    /// counterfactual baseline) can later score the rationale by comparing
    /// the prediction against measured deltas.
    pub fn with_expected_outcome(mut self, expected: impl Into<String>) -> Self {
        self.expected_outcome = Some(expected.into());
        self
    }
}

#[cfg(test)]
mod rationale_tests {
    use super::*;

    #[test]
    fn rationale_serializes_minimal_fields() {
        let r = Rationale::new(
            "throttle",
            "PressureSpecialist:critical-zone",
            "pressure=0.82,swap_gb=2.1",
        );
        let json = serde_json::to_string(&r).expect("serialize Rationale");
        assert!(
            json.contains("\"action_class\":\"throttle\""),
            "expected action_class field, got: {json}"
        );
        assert!(
            json.contains("\"trigger\":\"PressureSpecialist:critical-zone\""),
            "expected trigger field, got: {json}"
        );
        assert!(
            json.contains("\"evidence\":\"pressure=0.82,swap_gb=2.1\""),
            "expected evidence field, got: {json}"
        );
        // `expected_outcome` MUST be omitted (skip_serializing_if) when None
        // so the journal stays small on the common path.
        assert!(
            !json.contains("expected_outcome"),
            "expected_outcome should be omitted when None, got: {json}"
        );
    }

    #[test]
    fn rationale_serializes_with_expected_outcome() {
        let r = Rationale::new("freeze", "chromium_manager:long_idle", "idle_cycles=32")
            .with_expected_outcome("expected_swap_release_mb=180");
        let json = serde_json::to_string(&r).expect("serialize Rationale");
        assert!(
            json.contains("\"expected_outcome\":\"expected_swap_release_mb=180\""),
            "expected_outcome should be serialized when Some, got: {json}"
        );
    }

    #[test]
    fn rationale_builder_default_no_expected_outcome() {
        let r = Rationale::new("boost", "interactive_focus", "fg_pid=4242");
        assert!(
            r.expected_outcome.is_none(),
            "default rationale must have None expected_outcome"
        );
        assert_eq!(r.action_class, "boost");
        assert_eq!(r.trigger, "interactive_focus");
        assert_eq!(r.evidence, "fg_pid=4242");
    }

    #[test]
    fn rationale_round_trips_through_json() {
        let r = Rationale::new("throttle", "swarm-throttle", "procs=42,waste=0.32")
            .with_expected_outcome("expected_pressure_drop=0.05");
        let json = serde_json::to_string(&r).expect("serialize");
        let back: Rationale = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(r, back);
    }
}
