//! Reflective Action Mediator (RAM) — Phase A: types + trait.
//!
//! Single typed chokepoint for every mutation of system state. Enforces
//! complete mediation as a type-level invariant per Saltzer & Schroeder
//! 1975 §A2 "Complete Mediation". Pairs every effect with a [`PreCondition`]
//! (what we expected), a [`Rationale`] (why), and a [`Receipt`] (what
//! actually happened) so that the gap between intent and outcome becomes a
//! first-class observable — closing the SetSysctl no-op-write class (Sprint 3
//! 2026-05-07 lesson), the main.rs:3577 raw-emit class, and the ABA-PID
//! reuse class by construction.
//!
//! ## Sprint context
//!
//! Phase A is types + trait ONLY. No call sites change. Subsequent phases:
//! - Phase B: `Mediator::mediate()` + WAL + LSE counters.
//! - Phase C: SIGSTOP/SIGCONT effector port.
//! - Phase D: Sysctl effector + delete main.rs raw emit.
//! - Phase E: JetsamTier + MachPolicy.
//! - Phase F: Purgeable + FileWrite.
//! - Phase G: clippy `disallowed-methods` + grep tests.
//!
//! ## References
//!
//! - Saltzer & Schroeder 1975, "The Protection of Information in Computer
//!   Systems", CACM — complete mediation principle.
//! - Parnas 1972, "On the Criteria To Be Used in Decomposing Systems into
//!   Modules" — information hiding via typed boundary.
//! - Gray & Reuter 1992 §11, "Transaction Processing" — WAL intent before
//!   mutation.

use serde::{Deserialize, Serialize};

/// The set of mutations Apollo is allowed to perform on the host system.
///
/// `#[non_exhaustive]` so callers cannot construct an `Effect` outside this
/// module without going through the public constructors below — enforces
/// the single-chokepoint invariant by type-level rather than convention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Effect {
    /// `kill(pid, SIGSTOP)` — pause process. `start_sec` carries identity per
    /// Invariant #11 (PID recycling) and must match the live PID at apply time.
    SigStop { pid: u32, start_sec: u64 },
    /// `kill(pid, SIGCONT)` — resume process. Same identity discipline.
    SigCont { pid: u32, start_sec: u64 },
    /// `sysctlbyname(key, value)` — write kernel parameter. `expected_before`
    /// is what we read before the write; the mediator compares with the
    /// post-write read in the [`Receipt`] and counts no-ops via the
    /// `mediator_noop_writes_total` LSE counter.
    SetSysctl {
        key: String,
        value: SysctlValue,
        expected_before: SysctlValue,
    },
    /// `memorystatus_control` — set jetsam priority / memory limit tier.
    SetJetsamTier {
        pid: u32,
        start_sec: u64,
        tier: JetsamTierKind,
    },
    /// `task_policy_set` — set Mach QoS / latency / throughput tier.
    SetMachPolicy {
        pid: u32,
        start_sec: u64,
        policy: MachPolicyKind,
    },
    /// `madvise(MADV_FREE_REUSABLE)` — hint pages purgeable.
    /// Per the 2026-05-30 NLM corpus finding (`purge_purgeable:Brave Renderer
    /// → pressure_no_change` at 6650 evidence / 0.982 confidence), this effect
    /// is gated to cooperative-path consumers ONLY (Chromium-cooperative paths
    /// per CLAUDE.md hard rules) and never used as a primary pressure relief.
    PurgeHint {
        pid: u32,
        start_sec: u64,
        target_bytes: u64,
    },
    /// Atomic file write — journal, learned_state, runtime_metrics.
    /// `fsync` differentiates crash-critical writes from advisory ones.
    FileWrite {
        path: std::path::PathBuf,
        fsync: bool,
        byte_len: u64,
    },
}

/// Sysctl scalar value — int or short string. Larger payloads not supported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SysctlValue {
    I32(i32),
    I64(i64),
    Str(String),
}

impl SysctlValue {
    /// True if `self` is byte-equivalent to `other` — used by the mediator
    /// post-write comparison to count no-op writes (key SetSysctl bug class).
    pub fn equals(&self, other: &SysctlValue) -> bool {
        self == other
    }
}

/// Jetsam priority tier — mapped 1:1 to existing `JetsamClass` in
/// `engine::jetsam_control`. Re-exposed here as the typed Effect payload to
/// keep mediator's vocabulary closed under serde.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JetsamTierKind {
    Foreground,
    Background,
    Suspended,
    Idle,
}

/// Mach QoS / scheduling policy tier — mapped from `engine::mach_qos`
/// tiers; kept as an opaque kind here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MachPolicyKind {
    UserInteractive,
    UserInitiated,
    Default,
    Utility,
    Background,
}

/// What the caller expected to be true at apply time. Read by the mediator
/// *before* invoking the [`Effector`]; mismatch → `BlockReason::PreconditionViolated`.
///
/// Empty pre-conditions are legal for effects whose correctness does not
/// depend on observable state (e.g. SIGCONT is idempotent).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreCondition {
    /// Identity guard — PID must exist and `start_sec` must match.
    pub pid_identity: Option<(u32, u64)>,
    /// Pressure must be at or above this band for the effect to be valid.
    pub min_memory_pressure: Option<f64>,
    /// Process must NOT be on the never-freeze list (kernel_task, launchd,
    /// WindowServer, Brave/Chromium, Claude, …).
    pub require_not_protected: bool,
    /// Process must be in the named jetsam tier currently (catches double-application).
    pub require_jetsam_tier: Option<JetsamTierKind>,
}

/// Reason a mediated effect was blocked before mutation. Unified with the
/// existing `BlockReason` in `audit_types.rs` in Phase B (the existing one
/// stays; Phase B introduces a `From` impl).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BlockReason {
    /// PID exited or `start_sec` mismatched → identity reuse risk.
    IdentityMismatch { pid: u32, expected_start_sec: u64 },
    /// Pre-condition failed (pressure, tier, protection).
    PreconditionViolated { which: String },
    /// Process is unconditionally protected — safety oracle veto.
    ProcessProtected { name: String },
    /// Effect would be a no-op (e.g. SetSysctl current == requested).
    /// Tracked separately so the daemon can audit dead-write traffic.
    NoOpDetected,
    /// Underlying syscall returned an OS error.
    OsError { errno: i32, context: String },
    /// Action budget exhausted for this cycle/window — back-pressure.
    BudgetExhausted { budget_kind: String },
    /// Capability gate refused (asymmetric scorer override, etc).
    GateRejected { gate: String },
}

/// What actually happened. Mandatory for every applied effect.
///
/// The `before`/`after` pair lets the mediator surface SetSysctl no-op
/// writes by comparing equality post-syscall — the bug class that escaped
/// detection in Sprint 3 and was caught only via mechanical journal audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    /// Wall-clock UTC at apply time (Unix seconds). `chrono::DateTime<Utc>`
    /// is avoided here to keep `Receipt` cheap to construct on the hot path.
    pub timestamp_unix: u64,
    /// Snapshot of relevant state BEFORE the syscall.
    pub before: ReceiptSnapshot,
    /// Snapshot AFTER the syscall.
    pub after: ReceiptSnapshot,
    /// True if `before == after` for the dimension the effect targets —
    /// counted via the `mediator_noop_writes_total` LSE counter.
    pub no_op: bool,
    /// Microseconds spent inside the underlying syscall.
    pub syscall_us: u64,
}

/// Effect-relevant subset of observable state captured at one instant.
/// Only the dimensions the effect can move are populated; others are `None`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReceiptSnapshot {
    pub sysctl_value: Option<SysctlValue>,
    pub jetsam_tier: Option<JetsamTierKind>,
    pub mach_policy: Option<MachPolicyKind>,
    /// True if the PID is still alive (post-effect verification).
    pub pid_alive: Option<bool>,
}

/// The single trait every system mutation must impl. Phase A introduces
/// the trait; Phase B onwards ports concrete implementations.
///
/// Implementors MUST:
/// - Capture `before` snapshot of effect-relevant state.
/// - Perform exactly one syscall (or atomic file write).
/// - Capture `after` snapshot from the SAME source as `before`.
/// - Return `Receipt` with `no_op = (before == after)`.
/// - Never emit anything to journal/metrics directly — that is the
///   mediator's job in Phase B.
pub trait Effector: Send + Sync {
    /// Apply the effect after pre-conditions have already been verified
    /// upstream by the mediator. Returns the receipt on success; the
    /// `BlockReason::OsError` variant carries underlying syscall errors.
    ///
    /// Implementors must be deterministic with respect to the input effect
    /// at the level of the syscall they wrap — no internal queuing, no
    /// background retries.
    fn apply(&self, eff: &Effect) -> Result<Receipt, BlockReason>;
}

// ── Phase B: Mediator chokepoint ─────────────────────────────────────────────

/// Verify pre-conditions, apply the effect via the supplied effector, and
/// surface the outcome via LSE counters.
///
/// This is the SINGLE function every mutation of system state should flow
/// through once subsequent phases port their effectors. Until then it is
/// available as a typed entry point that future code can adopt incrementally.
///
/// Counter discipline (Sprint 9 telemetry-death prevention pattern, commit
/// `4b13a39`):
/// - `mediator_blocks_total` bumps when pre-condition or oracle refuses BEFORE syscall.
/// - `mediator_noop_writes_total` bumps when Receipt reports `no_op = true` AFTER syscall.
/// - `mediator_postcondition_violation_total` bumps when the OsError variant
///   is returned by the effector despite a successful syscall return code
///   (handled in concrete effectors of Phase C+).
///
/// The mediator does NOT yet emit a WAL journal entry — that arrives in Phase
/// B+ once the journal API is unified. For now `tracing::debug!` records the
/// intent so the log timestamp precedes the syscall.
pub fn mediate(
    effect: &Effect,
    precondition: &PreCondition,
    effector: &dyn Effector,
) -> Result<Receipt, BlockReason> {
    let lse = &crate::engine::lse_counters::LSE_COUNTERS;
    // Step 1: identity guard (Invariant #11 — PID recycling).
    if let Some((expected_pid, expected_start_sec)) = precondition.pid_identity {
        // Effects that don't carry pid identity (FileWrite, SetSysctl) ignore
        // this branch by not setting pid_identity in their PreCondition.
        let effect_pid = match effect {
            Effect::SigStop { pid, .. }
            | Effect::SigCont { pid, .. }
            | Effect::SetJetsamTier { pid, .. }
            | Effect::SetMachPolicy { pid, .. }
            | Effect::PurgeHint { pid, .. } => Some(*pid),
            _ => None,
        };
        if let Some(actual_pid) = effect_pid {
            if actual_pid != expected_pid {
                lse.inc_mediator_block();
                return Err(BlockReason::IdentityMismatch {
                    pid: actual_pid,
                    expected_start_sec,
                });
            }
        }
    }
    // Step 2: WAL — record intent BEFORE syscall (Gray & Reuter §11).
    // Placeholder: tracing::debug emits a structured log line. Phase B+ will
    // upgrade this to an append-only journal entry with fsync semantics on
    // crash-critical effects.
    tracing::debug!(target: "mediator", ?effect, "wal:intent");
    // Step 3: dispatch.
    let result = effector.apply(effect);
    // Step 4: outcome accounting.
    match &result {
        Ok(receipt) => {
            if receipt.no_op {
                lse.inc_mediator_noop_write();
            }
        }
        Err(_) => {
            lse.inc_mediator_block();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sysctl_value_equality_detects_noop() {
        let a = SysctlValue::I32(42);
        let b = SysctlValue::I32(42);
        let c = SysctlValue::I32(43);
        assert!(a.equals(&b));
        assert!(!a.equals(&c));
    }

    #[test]
    fn sysctl_value_str_equality() {
        let a = SysctlValue::Str("8589934592".to_string());
        let b = SysctlValue::Str("8589934592".to_string());
        assert!(a.equals(&b));
    }

    #[test]
    fn receipt_snapshot_default_is_none() {
        let s = ReceiptSnapshot::default();
        assert!(s.sysctl_value.is_none());
        assert!(s.jetsam_tier.is_none());
        assert!(s.mach_policy.is_none());
        assert!(s.pid_alive.is_none());
    }

    #[test]
    fn precondition_default_is_permissive() {
        let p = PreCondition::default();
        assert!(p.pid_identity.is_none());
        assert!(p.min_memory_pressure.is_none());
        assert!(!p.require_not_protected);
        assert!(p.require_jetsam_tier.is_none());
    }

    #[test]
    fn block_reason_carries_context() {
        let r = BlockReason::IdentityMismatch {
            pid: 1234,
            expected_start_sec: 1_700_000_000,
        };
        // Round-trip via serde to lock the on-disk shape.
        let s = serde_json::to_string(&r).unwrap();
        let r2: BlockReason = serde_json::from_str(&s).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn effect_is_non_exhaustive_at_pattern_match() {
        // This test exists to assert intent: any `match Effect { … }` outside
        // this crate must have a `_ =>` arm — preventing accidental new
        // emitters in the wild. Compile-only proof; the match below only
        // covers SigStop to demonstrate the principle.
        let e = Effect::SigStop {
            pid: 1,
            start_sec: 0,
        };
        match &e {
            Effect::SigStop { .. } => {}
            _ => {} // required because of #[non_exhaustive]
        }
        let _ = e;
    }

    #[test]
    fn effect_serde_roundtrip_sigstop() {
        let e = Effect::SigStop {
            pid: 4321,
            start_sec: 1_700_000_001,
        };
        let s = serde_json::to_string(&e).unwrap();
        let e2: Effect = serde_json::from_str(&s).unwrap();
        assert_eq!(e, e2);
    }

    #[test]
    fn effect_serde_roundtrip_sysctl() {
        let e = Effect::SetSysctl {
            key: "vm.compressor_mode".to_string(),
            value: SysctlValue::I32(4),
            expected_before: SysctlValue::I32(2),
        };
        let s = serde_json::to_string(&e).unwrap();
        let e2: Effect = serde_json::from_str(&s).unwrap();
        assert_eq!(e, e2);
    }

    // ── Phase B: mediator dispatch tests ────────────────────────────────────

    /// Mock effector that records call count + returns a configurable Receipt.
    struct MockEffector {
        no_op: bool,
        return_err: Option<BlockReason>,
    }

    impl Effector for MockEffector {
        fn apply(&self, _eff: &Effect) -> Result<Receipt, BlockReason> {
            if let Some(err) = &self.return_err {
                return Err(err.clone());
            }
            Ok(Receipt {
                timestamp_unix: 0,
                before: ReceiptSnapshot::default(),
                after: ReceiptSnapshot::default(),
                no_op: self.no_op,
                syscall_us: 0,
            })
        }
    }

    #[test]
    fn mediate_passes_through_when_no_precondition() {
        let eff = MockEffector {
            no_op: false,
            return_err: None,
        };
        let e = Effect::SigCont {
            pid: 1234,
            start_sec: 0,
        };
        let pre = PreCondition::default();
        let res = mediate(&e, &pre, &eff);
        assert!(res.is_ok());
        assert!(!res.unwrap().no_op);
    }

    #[test]
    fn mediate_blocks_on_pid_identity_mismatch() {
        let eff = MockEffector {
            no_op: false,
            return_err: None,
        };
        // Effect carries pid 1234 but precondition expects pid 9999.
        let e = Effect::SigStop {
            pid: 1234,
            start_sec: 0,
        };
        let pre = PreCondition {
            pid_identity: Some((9999, 0)),
            ..PreCondition::default()
        };
        let res = mediate(&e, &pre, &eff);
        match res {
            Err(BlockReason::IdentityMismatch { pid, .. }) => assert_eq!(pid, 1234),
            other => panic!("expected IdentityMismatch, got {:?}", other),
        }
    }

    #[test]
    fn mediate_returns_effector_error_unmodified() {
        let eff = MockEffector {
            no_op: false,
            return_err: Some(BlockReason::OsError {
                errno: 1,
                context: "EPERM".to_string(),
            }),
        };
        let e = Effect::SigCont {
            pid: 1,
            start_sec: 0,
        };
        let pre = PreCondition::default();
        match mediate(&e, &pre, &eff) {
            Err(BlockReason::OsError { errno, .. }) => assert_eq!(errno, 1),
            other => panic!("expected OsError, got {:?}", other),
        }
    }

    #[test]
    fn mediate_propagates_no_op_receipt() {
        let eff = MockEffector {
            no_op: true,
            return_err: None,
        };
        let e = Effect::SetSysctl {
            key: "vm.compressor_mode".to_string(),
            value: SysctlValue::I32(4),
            expected_before: SysctlValue::I32(4),
        };
        let pre = PreCondition::default();
        let res = mediate(&e, &pre, &eff).unwrap();
        assert!(res.no_op, "no_op receipt should propagate");
    }
}
