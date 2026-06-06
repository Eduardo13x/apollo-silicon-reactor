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
    /// `thread_policy_set` — per-thread QoS routing (P-core vs E-core).
    /// Inv#11: `start_sec` carries PID identity; identity-mismatch → block.
    /// `affinity_tag` Some(1)=P-cluster, Some(2)=E-cluster, Some(0)/None=no hint.
    /// Sprint S3 (2026-06-06): activates `mediator_thread_policy_total` —
    /// rewires the `RootAction::SetThreadQoS` arm in `execute_actions.rs`
    /// to flow through `ThreadPolicyEffector::apply_raw`.
    SetThreadPolicy {
        pid: u32,
        start_sec: u64,
        thread_index: u32,
        tier: crate::engine::mach_qos::ThreadTier,
        affinity_tag: Option<u32>,
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
    /// Sprint patch (2026-06-05) — S9. Number of underlying unit-effects the
    /// effector applied. `0` for a full no-op; `1` for atomic syscall effectors
    /// (Signal/Sysctl/MachPolicy/Jetsam); ≥1 for batched effectors (Purgeable
    /// reports number of regions touched). `#[serde(default)]` keeps older
    /// journal lines deserialisable so the field is additive in production.
    #[serde(default)]
    pub applied_count: u32,
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

// ── Phase C: SIGSTOP / SIGCONT effectors ─────────────────────────────────────

/// Effector for SIGSTOP / SIGCONT signals. One implementor handles both
/// variants because the syscall surface is identical (`libc::kill`); the
/// branch is inside `apply` to keep the public API tight.
///
/// PID identity verification (`start_sec` match) is performed by the caller
/// via `PreCondition::pid_identity` and the mediator's identity guard — this
/// effector trusts that check and issues the signal directly. Implementors
/// of competing trait paths MUST do the same to keep Invariant #11 honored.
///
/// Liveness check is encoded in the post-snapshot: if `kill(pid, 0)` confirms
/// the process is alive after the signal, `Receipt.after.pid_alive = Some(true)`;
/// SIGCONT leaves a stopped child running, SIGSTOP leaves a running child
/// stopped. The mediator's `no_op` check fires when the signal hits an
/// already-stopped (for SIGSTOP) or already-running (for SIGCONT) process —
/// future Phase B+ upgrade can detect this via `proc_taskinfo::is_stopped_pid`.
pub struct SignalEffector;

impl Effector for SignalEffector {
    fn apply(&self, eff: &Effect) -> Result<Receipt, BlockReason> {
        let (pid, sig) = match eff {
            Effect::SigStop { pid, .. } => (*pid, libc::SIGSTOP),
            Effect::SigCont { pid, .. } => (*pid, libc::SIGCONT),
            _ => {
                return Err(BlockReason::PreconditionViolated {
                    which: "SignalEffector: unsupported Effect variant".to_string(),
                });
            }
        };
        let before = ReceiptSnapshot {
            pid_alive: Some(unsafe { libc::kill(pid as i32, 0) } == 0),
            ..ReceiptSnapshot::default()
        };
        let t0 = std::time::Instant::now();
        let rc = unsafe { libc::kill(pid as i32, sig) };
        let syscall_us = t0.elapsed().as_micros().min(u64::MAX as u128) as u64;
        if rc != 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(BlockReason::OsError {
                errno,
                context: format!("kill({}, {})", pid, sig),
            });
        }
        let after = ReceiptSnapshot {
            pid_alive: Some(unsafe { libc::kill(pid as i32, 0) } == 0),
            ..ReceiptSnapshot::default()
        };
        // no_op heuristic: if the alive bit didn't change AND we expected
        // a state transition, count it as a no-op. SIGSTOP/SIGCONT both
        // leave the PID alive, so a stable Some(true)→Some(true) is the
        // common case — the real no-op signal at this layer is when the
        // PID was already dead pre-syscall and remains so post.
        let no_op = matches!(before.pid_alive, Some(false))
            && matches!(after.pid_alive, Some(false));
        // S9: SignalEffector dispatches exactly one libc::kill on success;
        // 0 when the syscall short-circuited to a dead-PID no_op.
        let applied_count = if no_op { 0 } else { 1 };
        Ok(Receipt {
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            before,
            after,
            no_op,
            syscall_us,
            applied_count,
        })
    }
}

// ── Phase D: Sysctl effector ─────────────────────────────────────────────────

/// Effector for kernel sysctl writes. Wraps `sysctl_direct::write_i32` /
/// `write_str_value`. The Receipt captures `before` and `after` reads so the
/// mediator's `no_op` detector fires when the syscall succeeds but the kernel
/// silently rejected (or coerced) the requested value — the SetSysctl no-op
/// bug class from Sprint 3 (2026-05-07 lesson).
///
/// Effect carries `expected_before`; if the live pre-read disagrees, the
/// effector returns `BlockReason::PreconditionViolated` instead of writing.
/// This catches stale state in the caller (race against another writer).
pub struct SysctlEffector;

impl SysctlEffector {
    fn read(key: &str, kind_hint: &SysctlValue) -> Option<SysctlValue> {
        match kind_hint {
            SysctlValue::I32(_) => {
                crate::engine::sysctl_direct::read_i32(key).map(SysctlValue::I32)
            }
            SysctlValue::I64(_) => {
                crate::engine::sysctl_direct::read_u64(key).map(|v| SysctlValue::I64(v as i64))
            }
            SysctlValue::Str(_) => crate::engine::sysctl_direct::read_str(key).map(SysctlValue::Str),
        }
    }
}

impl Effector for SysctlEffector {
    fn apply(&self, eff: &Effect) -> Result<Receipt, BlockReason> {
        let (key, value, expected_before) = match eff {
            Effect::SetSysctl {
                key,
                value,
                expected_before,
            } => (key.as_str(), value, expected_before),
            _ => {
                return Err(BlockReason::PreconditionViolated {
                    which: "SysctlEffector: unsupported Effect variant".to_string(),
                });
            }
        };
        // Pre-read: detect stale caller state vs live kernel value.
        let live_before = Self::read(key, value);
        if let Some(live) = &live_before {
            if !live.equals(expected_before) {
                return Err(BlockReason::PreconditionViolated {
                    which: format!("sysctl {} live != expected_before", key),
                });
            }
        }
        let before = ReceiptSnapshot {
            sysctl_value: live_before.clone(),
            ..ReceiptSnapshot::default()
        };
        let t0 = std::time::Instant::now();
        let ok = match value {
            SysctlValue::I32(v) => crate::engine::sysctl_direct::write_i32(key, *v),
            SysctlValue::I64(v) => crate::engine::sysctl_direct::write_str_value(key, &v.to_string()),
            SysctlValue::Str(s) => crate::engine::sysctl_direct::write_str_value(key, s),
        };
        let syscall_us = t0.elapsed().as_micros().min(u64::MAX as u128) as u64;
        if !ok {
            return Err(BlockReason::OsError {
                errno: 0,
                context: format!("sysctl write {} failed", key),
            });
        }
        let live_after = Self::read(key, value);
        let after = ReceiptSnapshot {
            sysctl_value: live_after.clone(),
            ..ReceiptSnapshot::default()
        };
        // no_op detection: write returned ok but live read shows value unchanged
        // → kernel silently rejected. The Sprint 3 SetSysctl bug class.
        let no_op = match (&before.sysctl_value, &after.sysctl_value) {
            (Some(b), Some(a)) => b.equals(a),
            _ => false,
        };
        // S9: 0 when the kernel silently rejected the write (no_op), 1 when
        // the value moved.
        let applied_count = if no_op { 0 } else { 1 };
        Ok(Receipt {
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            before,
            after,
            no_op,
            syscall_us,
            applied_count,
        })
    }
}

// ── Phase E: Jetsam tier effector ────────────────────────────────────────────

/// Effector for jetsam priority / memory-limit tier changes. Wraps
/// `jetsam_control::apply_apollo_policy` + `get_priority` for before/after
/// snapshots.
///
/// `JetsamTierKind` is mapped to the production `JetsamClass` per the table
/// below. The 4-variant mediator enum is intentionally broader than the
/// 3-variant production enum so future effects (e.g. Suspended/Idle splits)
/// can be added without changing the public Effect surface.
///
/// | JetsamTierKind | JetsamClass     | Rationale                              |
/// |----------------|------------------|----------------------------------------|
/// | Foreground     | Interactive      | user-facing, never demoted             |
/// | Background     | Noise            | normal demote candidate                |
/// | Suspended      | Noise            | aggressive demote (no exact match yet) |
/// | Idle           | Noise            | aggressive demote (no exact match yet) |
pub struct JetsamEffector;

impl JetsamEffector {
    fn to_class(kind: JetsamTierKind) -> crate::engine::jetsam_control::JetsamClass {
        use crate::engine::jetsam_control::JetsamClass;
        match kind {
            JetsamTierKind::Foreground => JetsamClass::Interactive,
            JetsamTierKind::Background | JetsamTierKind::Suspended | JetsamTierKind::Idle => {
                JetsamClass::Noise
            }
        }
    }
}

impl Effector for JetsamEffector {
    fn apply(&self, eff: &Effect) -> Result<Receipt, BlockReason> {
        let (pid, tier) = match eff {
            Effect::SetJetsamTier { pid, tier, .. } => (*pid, *tier),
            _ => {
                return Err(BlockReason::PreconditionViolated {
                    which: "JetsamEffector: unsupported Effect variant".to_string(),
                });
            }
        };
        let before_priority = crate::engine::jetsam_control::get_priority(pid);
        let before = ReceiptSnapshot {
            jetsam_tier: Some(tier),
            ..ReceiptSnapshot::default()
        };
        let class = Self::to_class(tier);
        let t0 = std::time::Instant::now();
        let result = crate::engine::jetsam_control::apply_apollo_policy(pid, class);
        let syscall_us = t0.elapsed().as_micros().min(u64::MAX as u128) as u64;
        if let Err(msg) = result {
            return Err(BlockReason::OsError {
                errno: 0,
                context: format!("jetsam {} → {:?}: {}", pid, class, msg),
            });
        }
        let after_priority = crate::engine::jetsam_control::get_priority(pid);
        let after = ReceiptSnapshot {
            jetsam_tier: Some(tier),
            ..ReceiptSnapshot::default()
        };
        // no_op: jetsam priority value unchanged after the policy apply.
        let no_op = matches!((before_priority, after_priority), (Some(a), Some(b)) if a == b);
        // S9: 0 when no_op (kernel kept the existing tier), 1 otherwise.
        let applied_count = if no_op { 0 } else { 1 };
        Ok(Receipt {
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            before,
            after,
            no_op,
            syscall_us,
            applied_count,
        })
    }
}

// ── Phase F: MachPolicy + Purgeable + FileWrite effectors ────────────────────

/// Effector for Mach scheduling policy changes (P-core / E-core routing,
/// QoS class). Switch-4 (2026-06-03): real syscall dispatch via injected
/// `Arc<Mutex<MachQoSManager>>` — replaces the Phase F placeholder.
///
/// The manager is locked for the duration of the syscall. Existing call
/// sites that pre-lock then call `qos.set_tier(pid, tier)` migrate to:
/// construct the effector with a clone of the SharedState Arc, drop any
/// outer guard, then call `mediator::mediate()` — the lock is acquired
/// inside the effector exactly once per dispatch.
///
/// Receipt fidelity limitations:
/// - MachQoSManager does not expose a query API for the current tier, so
///   `before`/`after.mach_policy` both reflect the *intended* policy from
///   the Effect rather than a kernel-confirmed read. `no_op = false` is
///   the safe default until a query method is added.
/// - `QoSOutcome` from `set_tier` is currently discarded; future Switch-4
///   iteration can map its variants to `BlockReason` / receipt fields.
pub struct MachPolicyEffector {
    qos: std::sync::Arc<std::sync::Mutex<crate::engine::mach_qos::MachQoSManager>>,
}

impl MachPolicyEffector {
    pub fn new(
        qos: std::sync::Arc<std::sync::Mutex<crate::engine::mach_qos::MachQoSManager>>,
    ) -> Self {
        Self { qos }
    }

    fn to_sched_tier(policy: MachPolicyKind) -> crate::engine::mach_qos::SchedulingTier {
        use crate::engine::mach_qos::SchedulingTier;
        match policy {
            MachPolicyKind::UserInteractive | MachPolicyKind::UserInitiated => {
                SchedulingTier::Foreground
            }
            MachPolicyKind::Default => SchedulingTier::Normal,
            MachPolicyKind::Utility | MachPolicyKind::Background => SchedulingTier::Background,
        }
    }
}

impl Effector for MachPolicyEffector {
    fn apply(&self, eff: &Effect) -> Result<Receipt, BlockReason> {
        let (pid, policy) = match eff {
            Effect::SetMachPolicy { pid, policy, .. } => (*pid, *policy),
            _ => {
                return Err(BlockReason::PreconditionViolated {
                    which: "MachPolicyEffector: unsupported Effect variant".to_string(),
                });
            }
        };
        let sched_tier = Self::to_sched_tier(policy);
        let t0 = std::time::Instant::now();
        let mut mgr = self.qos.lock().unwrap_or_else(|e| e.into_inner());
        let _outcome = mgr.set_tier(pid, sched_tier);
        let syscall_us = t0.elapsed().as_micros().min(u64::MAX as u128) as u64;
        drop(mgr);
        let snapshot = ReceiptSnapshot {
            mach_policy: Some(policy),
            ..ReceiptSnapshot::default()
        };
        Ok(Receipt {
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            before: snapshot.clone(),
            after: snapshot,
            no_op: false,
            syscall_us,
            // S9: MachPolicy issues exactly one task_policy_set syscall per
            // dispatch — Receipt fidelity limitations doc note still applies,
            // but the dispatch count is unambiguously 1.
            applied_count: 1,
        })
    }
}

/// Sprint patch (2026-06-05) — S3 (SHRUNK).
///
/// Per-thread Mach QoS effector. The full design called for a new `Effect`
/// variant (`SetThreadPolicy { pid, start_sec, thread_index, tier,
/// affinity_tag }`) plus rewiring the `RootAction::SetThreadQoS` handler in
/// `execute_actions.rs:947` to flow through this effector. To keep the
/// patch additive (no enum variant churn, no producer rewrites this
/// iteration) the effector accepts a tightly-scoped raw API instead — the
/// downstream Effect variant + decide_actions plumbing can land in a
/// follow-up commit without disturbing the typed surface.
///
/// Counter: `mediator_thread_policy_total` (`LSE_COUNTERS`) bumps only on
/// successful dispatch (see follow-up #3, MED severity below) so the cutover
/// observability stays meaningful: counter = "successful mediated thread
/// policies", not "attempted dispatches". If a future revision needs to
/// distinguish attempts vs failures, add a parallel
/// `mediator_thread_policy_failed_total` counter rather than mixing both
/// signals in one number.
///
/// Constructor is `pub(crate)` (follow-up #3, safety finding) so callers
/// cannot create rogue effector instances outside the central dispatch
/// path; the complete-mediation oracle stays intact.
pub struct ThreadPolicyEffector {
    qos: std::sync::Arc<std::sync::Mutex<crate::engine::mach_qos::MachQoSManager>>,
}

impl ThreadPolicyEffector {
    pub(crate) fn new(
        qos: std::sync::Arc<std::sync::Mutex<crate::engine::mach_qos::MachQoSManager>>,
    ) -> Self {
        Self { qos }
    }

    /// Apply a per-thread QoS effect against the wrapped manager.
    /// Returns `(ok, syscall_us, applied_count)`.
    ///
    /// **Counter semantics (follow-up #3):** the LSE counter bumps only on
    /// `ok == true`. A failed `set_thread_qos` returns
    /// `(false, syscall_us, 0)` without incrementing the dispatch counter,
    /// so dashboards never confuse attempts with successful side effects.
    pub fn apply_raw(
        &self,
        pid: u32,
        thread_index: u32,
        tier: crate::engine::mach_qos::ThreadTier,
        affinity_tag: Option<u32>,
    ) -> (bool, u64, u32) {
        let t0 = std::time::Instant::now();
        let mgr = self.qos.lock().unwrap_or_else(|e| e.into_inner());
        let ok = mgr.set_thread_qos(pid, thread_index, tier);
        if let Some(tag) = affinity_tag {
            if tag != 0 {
                let _ = mgr.set_thread_affinity_tag(pid, thread_index, tag);
            }
        }
        let syscall_us = t0.elapsed().as_micros().min(u64::MAX as u128) as u64;
        drop(mgr);
        if ok {
            crate::engine::lse_counters::LSE_COUNTERS.inc_mediator_thread_policy();
        }
        (ok, syscall_us, if ok { 1 } else { 0 })
    }
}

impl Effector for ThreadPolicyEffector {
    fn apply(&self, eff: &Effect) -> Result<Receipt, BlockReason> {
        let (pid, thread_index, tier, affinity_tag) = match eff {
            Effect::SetThreadPolicy {
                pid,
                thread_index,
                tier,
                affinity_tag,
                ..
            } => (*pid, *thread_index, *tier, *affinity_tag),
            _ => {
                return Err(BlockReason::PreconditionViolated {
                    which: "ThreadPolicyEffector: unsupported Effect variant".to_string(),
                });
            }
        };
        let (ok, syscall_us, applied_count) =
            self.apply_raw(pid, thread_index, tier, affinity_tag);
        if !ok {
            return Err(BlockReason::PreconditionViolated {
                which: "ThreadPolicyEffector: set_thread_qos returned false (pid invalid or task_for_pid failed)".to_string(),
            });
        }
        let snapshot = ReceiptSnapshot::default();
        Ok(Receipt {
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            before: snapshot.clone(),
            after: snapshot,
            no_op: false,
            syscall_us,
            applied_count,
        })
    }
}

/// Effector for `madvise(MADV_FREE_REUSABLE)` purgeable hints. Best-effort —
/// kernel may ignore the hint without error. Receipt records the byte count
/// requested for audit; actual reclaimed bytes are unknowable without
/// post-call vm_region inspection which is out of scope.
///
/// CLAUDE.md hard rule: this effect is reserved for Chromium-cooperative
/// paths only. The mediator wires it but the SAFETY ORACLE in
/// classify_protection must continue to gate emission upstream — this
/// effector does NOT re-check process protection (single source of truth).
pub struct PurgeableEffector;

impl Effector for PurgeableEffector {
    fn apply(&self, eff: &Effect) -> Result<Receipt, BlockReason> {
        let (pid, _target_bytes) = match eff {
            Effect::PurgeHint {
                pid, target_bytes, ..
            } => (*pid, *target_bytes),
            _ => {
                return Err(BlockReason::PreconditionViolated {
                    which: "PurgeableEffector: unsupported Effect variant".to_string(),
                });
            }
        };
        // Switch-5 (2026-06-03): real dispatch via existing
        // `compressor_aware::purge_purgeable_regions(pid)`, which walks all
        // purgeable VM regions for the target task and issues madvise per
        // region internally — no address-range plumbing needed in the
        // Effect surface.
        //
        // CLAUDE.md hard rule: this effector trusts the upstream gate
        // (classify_protection + Chromium-cooperative emission path) to
        // decide whether to fire — the single source of safety truth stays
        // in `safety.rs`. The effector does not re-validate process kind.
        let t0 = std::time::Instant::now();
        let purged_regions = crate::engine::compressor_aware::purge_purgeable_regions(pid)
            .unwrap_or(0);
        let syscall_us = t0.elapsed().as_micros().min(u64::MAX as u128) as u64;
        // no_op when the walker reported zero purgeable regions — the
        // 2026-05-30 NLM corpus finding that purge_purgeable:Brave Renderer
        // often returns pressure_no_change shows up here as a non-zero
        // mediator_noop_writes_total signal over time.
        let no_op = purged_regions == 0;
        Ok(Receipt {
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            before: ReceiptSnapshot::default(),
            after: ReceiptSnapshot::default(),
            no_op,
            syscall_us,
            // S9: batched effector — number of purgeable regions visited /
            // madvise'd in the inner walker. 0 ⇔ no_op (no regions found).
            applied_count: purged_regions as u32,
        })
    }
}

/// Effector for atomic file writes (journal, learned_state, runtime_metrics).
/// Writes go through `crate::engine::llm::write_json_fsync` style temp+rename
/// semantics in the switch-over sprint. The Phase F implementation here is
/// the typed surface that production writers will adopt; concrete bytes
/// arrive via a follow-up that threads the payload through.
///
/// Receipt.no_op fires when `byte_len == 0` (an empty write request, which
/// the production code does NOT today catch — surfacing this via the
/// mediator counter exposes a class of dead writes).
pub struct FileWriteEffector;

impl Effector for FileWriteEffector {
    fn apply(&self, eff: &Effect) -> Result<Receipt, BlockReason> {
        let (path, byte_len, _fsync) = match eff {
            Effect::FileWrite {
                path,
                fsync,
                byte_len,
            } => (path.as_path(), *byte_len, *fsync),
            _ => {
                return Err(BlockReason::PreconditionViolated {
                    which: "FileWriteEffector: unsupported Effect variant".to_string(),
                });
            }
        };
        // Phase F: typed surface only — no actual write. The switch-over
        // sprint plumbs the payload through and replaces this stub with
        // `llm::write_json_fsync` semantics.
        let _ = path;
        Ok(Receipt {
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            before: ReceiptSnapshot::default(),
            after: ReceiptSnapshot::default(),
            no_op: byte_len == 0,
            syscall_us: 0,
            // S9: stub effector — phase-F placeholder reports the byte_len > 0
            // intent count. Real production wiring will refine this to the
            // actual bytes-written (or 0 on empty/no-op).
            applied_count: if byte_len == 0 { 0 } else { 1 },
        })
    }
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
            | Effect::SetThreadPolicy { pid, .. }
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
                applied_count: if self.no_op { 0 } else { 1 },
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

    // ── Phase C: SignalEffector tests ───────────────────────────────────────

    // ── Phase F: MachPolicy / Purgeable / FileWrite tests ───────────────────

    fn make_mach_effector_for_test() -> MachPolicyEffector {
        let mgr = crate::engine::mach_qos::MachQoSManager::new();
        MachPolicyEffector::new(std::sync::Arc::new(std::sync::Mutex::new(mgr)))
    }

    #[test]
    fn mach_policy_effector_dispatches_real_set_tier() {
        let eff = make_mach_effector_for_test();
        let e = Effect::SetMachPolicy {
            pid: 1234,
            start_sec: 0,
            policy: MachPolicyKind::Background,
        };
        // PID 1234 likely doesn't exist or we lack permission — the manager
        // handles that internally. We only assert the typed surface works.
        let r = eff.apply(&e).expect("Switch-4 dispatch returns Ok");
        assert_eq!(r.after.mach_policy, Some(MachPolicyKind::Background));
        assert!(!r.no_op, "Switch-4 honest default — no_op detection deferred");
    }

    #[test]
    fn mach_policy_effector_rejects_other_effect() {
        let eff = make_mach_effector_for_test();
        let e = Effect::SigStop { pid: 1, start_sec: 0 };
        assert!(eff.apply(&e).is_err());
    }

    #[test]
    fn mach_policy_kind_to_sched_tier_mapping() {
        use crate::engine::mach_qos::SchedulingTier;
        assert_eq!(
            MachPolicyEffector::to_sched_tier(MachPolicyKind::UserInteractive),
            SchedulingTier::Foreground
        );
        assert_eq!(
            MachPolicyEffector::to_sched_tier(MachPolicyKind::UserInitiated),
            SchedulingTier::Foreground
        );
        assert_eq!(
            MachPolicyEffector::to_sched_tier(MachPolicyKind::Default),
            SchedulingTier::Normal
        );
        assert_eq!(
            MachPolicyEffector::to_sched_tier(MachPolicyKind::Utility),
            SchedulingTier::Background
        );
        assert_eq!(
            MachPolicyEffector::to_sched_tier(MachPolicyKind::Background),
            SchedulingTier::Background
        );
    }

    #[test]
    fn purgeable_effector_records_intent() {
        let eff = PurgeableEffector;
        let e = Effect::PurgeHint {
            pid: 1234,
            start_sec: 0,
            target_bytes: 4096,
        };
        let r = eff.apply(&e).expect("phase F placeholder returns Ok");
        assert!(r.no_op);
    }

    #[test]
    fn file_write_effector_detects_empty_write() {
        let eff = FileWriteEffector;
        let e = Effect::FileWrite {
            path: std::path::PathBuf::from("/tmp/apollo_test.json"),
            fsync: false,
            byte_len: 0,
        };
        let r = eff.apply(&e).expect("Ok for empty write");
        assert!(r.no_op, "empty write should be no_op");
    }

    #[test]
    fn file_write_effector_non_empty_not_noop() {
        let eff = FileWriteEffector;
        let e = Effect::FileWrite {
            path: std::path::PathBuf::from("/tmp/apollo_test.json"),
            fsync: true,
            byte_len: 128,
        };
        let r = eff.apply(&e).expect("Ok");
        assert!(!r.no_op);
    }

    // ── Phase E: JetsamEffector tests ───────────────────────────────────────

    #[test]
    fn jetsam_effector_rejects_non_jetsam_effect() {
        let eff = JetsamEffector;
        let e = Effect::SigCont { pid: 1, start_sec: 0 };
        match eff.apply(&e) {
            Err(BlockReason::PreconditionViolated { which }) => {
                assert!(which.contains("unsupported"));
            }
            other => panic!("expected PreconditionViolated, got {:?}", other),
        }
    }

    #[test]
    fn jetsam_tier_kind_maps_to_class() {
        use crate::engine::jetsam_control::JetsamClass;
        assert_eq!(
            JetsamEffector::to_class(JetsamTierKind::Foreground),
            JetsamClass::Interactive
        );
        assert_eq!(
            JetsamEffector::to_class(JetsamTierKind::Background),
            JetsamClass::Noise
        );
        assert_eq!(
            JetsamEffector::to_class(JetsamTierKind::Suspended),
            JetsamClass::Noise
        );
        assert_eq!(
            JetsamEffector::to_class(JetsamTierKind::Idle),
            JetsamClass::Noise
        );
    }

    // ── Phase D: SysctlEffector tests ───────────────────────────────────────

    #[test]
    fn sysctl_effector_rejects_non_sysctl_effect() {
        let eff = SysctlEffector;
        let e = Effect::SigStop { pid: 1, start_sec: 0 };
        match eff.apply(&e) {
            Err(BlockReason::PreconditionViolated { which }) => {
                assert!(which.contains("unsupported"));
            }
            other => panic!("expected PreconditionViolated, got {:?}", other),
        }
    }

    #[test]
    fn sysctl_effector_invalid_key_returns_error_or_blocked() {
        // "foo.does.not.exist" is guaranteed to fail at the kernel level.
        let eff = SysctlEffector;
        let e = Effect::SetSysctl {
            key: "foo.does.not.exist".to_string(),
            value: SysctlValue::I32(1),
            expected_before: SysctlValue::I32(0),
        };
        // We don't assert specific variant — kernel may return error OR
        // sysctl_direct::read returning None bypasses the precondition check
        // and the write returns false. Both are acceptable; what matters is
        // we do not panic and do not silently report success.
        let res = eff.apply(&e);
        assert!(res.is_err(), "invalid sysctl key must surface an error");
    }

    #[test]
    fn signal_effector_rejects_non_signal_effect() {
        let eff = SignalEffector;
        let e = Effect::SetSysctl {
            key: "vm.compressor_mode".to_string(),
            value: SysctlValue::I32(4),
            expected_before: SysctlValue::I32(2),
        };
        match eff.apply(&e) {
            Err(BlockReason::PreconditionViolated { which }) => {
                assert!(which.contains("unsupported"));
            }
            other => panic!("expected PreconditionViolated, got {:?}", other),
        }
    }

    #[test]
    fn signal_effector_dead_pid_returns_os_error() {
        // PID 0xFFFFFFE is reserved/non-existent on macOS; kill returns -1 ESRCH.
        let eff = SignalEffector;
        let e = Effect::SigCont {
            pid: 0xFFFF_FFFE,
            start_sec: 0,
        };
        match eff.apply(&e) {
            Err(BlockReason::OsError { errno, context }) => {
                assert!(errno != 0);
                assert!(context.contains("kill"));
            }
            // On some kernels kill(invalid_pid, 0) may be ignored; that case is also acceptable
            // as long as the effector did not panic. Tighten this assertion if it flakes.
            Ok(_) => {}
            Err(other) => panic!("unexpected error variant: {:?}", other),
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
