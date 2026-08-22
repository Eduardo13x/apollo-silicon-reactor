use crate::engine::sysctl_direct;
use std::collections::{HashMap, HashSet};

use chrono::Utc;

use crate::engine::active_coalition_envelope::CoalitionGuard;
use crate::engine::activity_sensor::pids_with_assertions;
use crate::engine::amx_detector;
use crate::engine::audit_types::{BlockReason, PolicyDecisionTrace};
use crate::engine::decision_ledger::{
    ActuatorDecisionEvent, ActuatorDecisionOutcome, CycleDecisionEvents,
};
use crate::engine::io_tiering::{apply_io_tier, io_tier_for_throttle};
// Switch-3: jetsam_control imports retired — production path now routes
// through mediator::JetsamEffector. Direct apply_apollo_policy/JetsamClass
// usage remains allowed only inside the typed effector + jetsam_control
// module itself.
use crate::engine::journal::append_journal_batch;
use crate::engine::mach_qos::{LatencyTier, MachQoSManager, ThreadTier, ThroughputTier};
use crate::engine::proc_taskinfo;
use crate::engine::process_identity::{self, ProcessIdentity};
use crate::engine::safety::{
    allowlisted_sysctls, allowlisted_sysctls_with_ranges, infrastructure_processes_cached,
    protected_processes_cached, ProtectionLevel,
};
use crate::engine::types::{CapabilityReport, JournalEntry, RootAction};

const BOOST_EFFECT_TTL: std::time::Duration = std::time::Duration::from_secs(12);
const THREAD_QOS_NICE_FALLBACK: i32 = -2;
const THREAD_QOS_FALLBACK_TTL: std::time::Duration = std::time::Duration::from_secs(12);

fn should_use_thread_qos_nice_fallback(
    tier: ThreadTier,
    outcome: &crate::engine::mach_qos::ThreadQoSOutcome,
) -> bool {
    tier == ThreadTier::Interactive
        && !outcome.mutated
        && outcome.failure_kind() == Some(crate::engine::mach_qos::MachQoSFailureKind::Restricted)
        && outcome.diagnostics.terminal
            == crate::engine::mach_qos::MachQoSTerminal::TaskAccessFailed
}

fn record_boost_qos_effects(pid: u32, start_sec: u64, tier_applied: bool, task_qos_applied: bool) {
    if tier_applied {
        crate::engine::effect_ledger::record_global(
            crate::engine::effect_ledger::AppliedEffect::MachTier { pid },
            BOOST_EFFECT_TTL,
            start_sec,
            "boost: Foreground tier",
        );
    }
    if task_qos_applied {
        crate::engine::effect_ledger::record_global(
            crate::engine::effect_ledger::AppliedEffect::TaskQoS { pid },
            BOOST_EFFECT_TTL,
            start_sec,
            "boost: interactive task QoS",
        );
    }
}

/// Set the nice value for a process via `setpriority(2)`.
/// Returns the prior nice value when the process priority actually changed.
/// `Ok(None)` means the target value was already present.
fn set_nice(pid: u32, nice: i32) -> anyhow::Result<Option<i32>> {
    // A2 fix (round-3): skip zombies before setpriority. setpriority on a
    // zombie returns ESRCH which was previously silenced but still wasted a
    // syscall and polluted the error log path.
    if proc_taskinfo::is_zombie_pid(pid) {
        anyhow::bail!("setpriority({}, {}) skipped: zombie", pid, nice);
    }
    let current;
    unsafe {
        // getpriority may legitimately return -1; errno disambiguates it.
        *libc::__error() = 0;
        current = libc::getpriority(libc::PRIO_PROCESS, pid);
        if current == -1 && *libc::__error() != 0 {
            anyhow::bail!(
                "getpriority({}) failed: {}",
                pid,
                std::io::Error::last_os_error()
            );
        }
        if current == nice {
            return Ok(None);
        }

        // errno must be cleared before setpriority — a return of -1 is
        // ambiguous because -1 is a valid priority.
        *libc::__error() = 0;
        let rc = libc::setpriority(libc::PRIO_PROCESS, pid, nice);
        if rc == -1 && *libc::__error() != 0 {
            anyhow::bail!(
                "setpriority({}, {}) failed: {}",
                pid,
                nice,
                std::io::Error::last_os_error()
            );
        }
    }
    Ok(Some(current))
}

/// Send a signal to all processes whose name matches `daemon` exactly.
/// Equivalent to `/usr/bin/killall <signal> <daemon>` but without fork/exec.
fn killall_by_name(daemon: &str, signal: i32) -> anyhow::Result<()> {
    let pids = proc_taskinfo::list_all_pids();
    let mut matched = 0u32;
    for pid in pids {
        if let Some(name) = process_identity::proc_name_for_pid(pid) {
            if name == daemon {
                let rc = unsafe { libc::kill(pid as i32, signal) };
                if rc == 0 {
                    matched += 1;
                }
            }
        }
    }
    if matched == 0 {
        anyhow::bail!("no process found matching '{}'", daemon);
    }
    Ok(())
}

/// Toggle Spotlight indexing via `mdutil -a -i on/off`.
///
/// mdutil communicates with the Spotlight server via XPC (com.apple.spotlightserver).
/// Spawned on a detached worker thread to (a) not block the daemon hot path
/// and (b) actually reap the child instead of leaking a zombie. Previous
/// `let _ = spawn()` left the Child to drop without `wait()`, accumulating
/// zombies across the daemon's lifetime (xnu does NOT auto-reap dropped
/// Child handles — Drop on `std::process::Child` is a no-op by design).
// ── Timeout wrappers for kernel syscalls that can block as root ──────────
//
// A1 fix (round-3): the previous implementation spawned one `thread::spawn`
// per timeout call and leaked it on timeout.  Over hours that produced
// thousands of detached zombies.  Replace with a single dedicated worker
// thread, spawned lazily on first use and fed via a mpsc request queue.
// On timeout, the caller abandons the response channel. The worker completes
// the request and compensates a late numeric mutation before accepting more
// work — only one worker total, no matter how many requests.

enum SysctlRequest {
    ReadNumeric {
        key: String,
        reply: std::sync::mpsc::Sender<Option<sysctl_direct::NumericSysctlValue>>,
    },
    TransactNumeric {
        key: String,
        before: sysctl_direct::NumericSysctlValue,
        requested: i64,
        reply: std::sync::mpsc::Sender<NumericSysctlTransaction>,
    },
    WriteI32 {
        key: String,
        value: i32,
        reply: std::sync::mpsc::Sender<bool>,
    },
}

fn sysctl_request_tx() -> &'static std::sync::mpsc::Sender<SysctlRequest> {
    use std::sync::OnceLock;
    static TX: OnceLock<std::sync::mpsc::Sender<SysctlRequest>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<SysctlRequest>();
        std::thread::Builder::new()
            .name("apollo-sysctl-worker".to_string())
            .spawn(move || {
                // Dedicated serial worker. A stuck syscall only blocks this
                // single thread — subsequent requests queue up but the main
                // loop is never blocked because callers recv_timeout().
                while let Ok(req) = rx.recv() {
                    match req {
                        SysctlRequest::ReadNumeric { key, reply } => {
                            let _ = reply.send(sysctl_direct::read_numeric(&key));
                        }
                        SysctlRequest::TransactNumeric {
                            key,
                            before,
                            requested,
                            reply,
                        } => {
                            let outcome = run_numeric_sysctl_transaction_with(
                                before,
                                requested,
                                || sysctl_direct::read_numeric(&key),
                                |value, width| sysctl_direct::write_numeric(&key, value, width),
                            );
                            deliver_numeric_sysctl_transaction_with(
                                reply,
                                outcome,
                                before,
                                || sysctl_direct::read_numeric(&key),
                                |value, width| sysctl_direct::write_numeric(&key, value, width),
                            );
                        }
                        SysctlRequest::WriteI32 { key, value, reply } => {
                            let _ = reply.send(sysctl_direct::write_i32(&key, value));
                        }
                    }
                }
            })
            .expect("failed to spawn apollo-sysctl-worker");
        tx
    })
}

/// Read a sysctl with 500ms timeout. Prevents `sysctlbyname` from blocking
/// the daemon loop indefinitely under kernel lock contention.
fn sysctl_read_numeric_with_timeout(key: &str) -> Option<sysctl_direct::NumericSysctlValue> {
    let (reply_tx, reply_rx) = std::sync::mpsc::channel();
    if sysctl_request_tx()
        .send(SysctlRequest::ReadNumeric {
            key: key.to_string(),
            reply: reply_tx,
        })
        .is_err()
    {
        return None;
    }
    reply_rx
        .recv_timeout(std::time::Duration::from_millis(500))
        .ok()
        .flatten()
}

fn sysctl_numeric_transaction_with_timeout(
    key: &str,
    before: sysctl_direct::NumericSysctlValue,
    requested: i64,
) -> Option<NumericSysctlTransaction> {
    let (reply_tx, reply_rx) = std::sync::mpsc::channel();
    if sysctl_request_tx()
        .send(SysctlRequest::TransactNumeric {
            key: key.to_string(),
            before,
            requested,
            reply: reply_tx,
        })
        .is_err()
    {
        return None;
    }
    reply_rx
        .recv_timeout(std::time::Duration::from_millis(500))
        .ok()
}

/// Write an i32 sysctl with 500ms timeout.
fn sysctl_write_i32_with_timeout(key: &str, value: i32) -> bool {
    let (reply_tx, reply_rx) = std::sync::mpsc::channel();
    if sysctl_request_tx()
        .send(SysctlRequest::WriteI32 {
            key: key.to_string(),
            value,
            reply: reply_tx,
        })
        .is_err()
    {
        return false;
    }
    reply_rx
        .recv_timeout(std::time::Duration::from_millis(500))
        .ok()
        .unwrap_or(false)
}

#[inline]
#[cfg(test)]
fn sysctl_postcondition_matches(
    requested: i64,
    observed: Option<sysctl_direct::NumericSysctlValue>,
) -> bool {
    observed.is_some_and(|observed| observed.value == requested)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericSysctlTransaction {
    Applied(sysctl_direct::NumericSysctlValue),
    NoOp,
    ReadFailed,
    PreconditionChanged,
    WriteFailed,
    Uncertain(Option<sysctl_direct::NumericSysctlValue>),
}

#[inline]
fn numeric_sysctl_matches(
    expected: sysctl_direct::NumericSysctlValue,
    observed: Option<sysctl_direct::NumericSysctlValue>,
) -> bool {
    observed == Some(expected)
}

fn restore_numeric_sysctl_if_owned_with<Read, Write>(
    before: sysctl_direct::NumericSysctlValue,
    owned: sysctl_direct::NumericSysctlValue,
    mut read: Read,
    mut write: Write,
) -> bool
where
    Read: FnMut() -> Option<sysctl_direct::NumericSysctlValue>,
    Write: FnMut(i64, sysctl_direct::NumericSysctlWidth) -> bool,
{
    // There is no kernel CAS primitive for sysctl. Re-read immediately before
    // compensation and abstain if another writer has taken ownership.
    if !numeric_sysctl_matches(owned, read()) {
        return true;
    }
    write(before.value, before.width) && numeric_sysctl_matches(before, read())
}

fn deliver_numeric_sysctl_transaction_with<Read, Write>(
    reply: std::sync::mpsc::Sender<NumericSysctlTransaction>,
    outcome: NumericSysctlTransaction,
    before: sysctl_direct::NumericSysctlValue,
    read: Read,
    write: Write,
) where
    Read: FnMut() -> Option<sysctl_direct::NumericSysctlValue>,
    Write: FnMut(i64, sysctl_direct::NumericSysctlWidth) -> bool,
{
    let Err(undelivered) = reply.send(outcome) else {
        return;
    };
    if let NumericSysctlTransaction::Applied(owned) = undelivered.0 {
        if !restore_numeric_sysctl_if_owned_with(before, owned, read, write) {
            crate::engine::lse_counters::LSE_COUNTERS.inc_mediator_postcondition_violation();
        }
    }
}

/// Execute a numeric sysctl as one compare/write/verify transaction. The
/// dedicated worker owns compensation, so timeout cannot leave a late write
/// silently applied after the caller has abandoned its receipt.
fn run_numeric_sysctl_transaction_with<Read, Write>(
    before: sysctl_direct::NumericSysctlValue,
    requested: i64,
    mut read: Read,
    mut write: Write,
) -> NumericSysctlTransaction
where
    Read: FnMut() -> Option<sysctl_direct::NumericSysctlValue>,
    Write: FnMut(i64, sysctl_direct::NumericSysctlWidth) -> bool,
{
    let Some(live_before) = read() else {
        return NumericSysctlTransaction::ReadFailed;
    };
    if live_before != before {
        return NumericSysctlTransaction::PreconditionChanged;
    }
    if requested == before.value {
        return NumericSysctlTransaction::NoOp;
    }
    if !write(requested, before.width) {
        return NumericSysctlTransaction::WriteFailed;
    }

    let observed = read();
    let expected = sysctl_direct::NumericSysctlValue {
        value: requested,
        width: before.width,
    };
    if numeric_sysctl_matches(expected, observed) {
        return NumericSysctlTransaction::Applied(expected);
    }

    // A mismatching value may belong to another system component. Without a
    // kernel CAS primitive Apollo cannot prove ownership, so it must not write
    // `before` over that value.
    NumericSysctlTransaction::Uncertain(observed)
}

/// Aggregate counters returned by execute_actions so callers do not need to
/// hold a RuntimeMetrics lock during blocking I/O.
#[derive(Debug, Default)]
pub struct ExecuteOutcomes {
    pub boosts_applied: u64,
    pub throttles_applied: u64,
    pub freezes_applied: u64,
    pub unfreezes_applied: u64,
    pub paging_hints_applied: u64,
    pub sysctl_applied: u64,
    pub failures: u64,
    pub last_error: Option<String>,
    pub critical_background_skips: u64,
    pub invalid_sysctl_denied: u64,
    pub top_skipped: Vec<String>,
    pub throttle_reverted: u64,
    pub thread_qos_applied: u64,
    pub thread_qos_hot_routes: u64,
    pub thread_qos_cold_routes: u64,
    pub journal_rotations: u64,
    pub journal_rotation_failures: u64,
    /// PIDs that were successfully frozen (SIGSTOP sent) this cycle.
    /// Used by causal graph to record only new freeze actions, not all active frozen PIDs.
    pub newly_frozen_pids: Vec<u32>,
    /// PIDs that were successfully thawed (SIGCONT sent) this cycle.
    /// Consumed by `UnfreezeDecayModel::record_thaw` — the model needs exactly
    /// the set of pids whose post-thaw RSS should start being tracked.
    pub newly_unfrozen_pids: Vec<u32>,
    /// A3 + A5/D1 fix (round-3): per-PID identity snapshot captured at the
    /// moment of SIGSTOP.  Parallel to `newly_frozen_pids`.
    /// `(start_sec, original_jetsam_priority)` — either may be 0/None if
    /// the lookup failed.
    pub newly_frozen_identity: Vec<(u32, u64, Option<i32>)>,
    /// Per-action skip reason channel — set by `push_skip`, drained by the
    /// outer journal-write code so the journal entry records `success=false`
    /// with the actual skip reason instead of falsely claiming success.
    /// Reset to `None` at the start of every action iteration.
    pub last_skip: Option<String>,
    /// Audit traces for all intended actions.
    pub audit_traces: Vec<PolicyDecisionTrace>,
    /// Exact bounded proposal/receipt records for this execution batch.
    pub decision_events: CycleDecisionEvents,
}

impl ExecuteOutcomes {
    fn push_skip(&mut self, what: String) {
        // Channel the skip reason out to the per-action journal write.
        self.last_skip = Some(what.clone());
        if self.top_skipped.len() < 12 && !self.top_skipped.contains(&what) {
            self.top_skipped.push(what);
        }
    }
}

fn root_action_target(action: &RootAction) -> String {
    match action {
        RootAction::BoostProcess { pid, name, .. }
        | RootAction::ThrottleProcess { pid, name, .. }
        | RootAction::FreezeProcess { pid, name, .. }
        | RootAction::UnfreezeProcess { pid, name, .. }
        | RootAction::SetThreadQoS { pid, name, .. } => format!("{name}:pid:{pid}"),
        RootAction::SetMemorystatus { pid, priority, .. } => {
            format!("pid:{pid}:priority:{priority}")
        }
        RootAction::SetSysctl(action) => format!("{}={}", action.key(), action.value()),
        RootAction::ToggleSpotlight { enabled, .. } => {
            if *enabled {
                "enabled".to_string()
            } else {
                "disabled".to_string()
            }
        }
        RootAction::QuarantineDaemon { daemon, active, .. } => {
            format!("{daemon}:{}", if *active { "active" } else { "released" })
        }
    }
}

fn root_action_outcome(
    applied: bool,
    result: &anyhow::Result<()>,
    block_reason: Option<BlockReason>,
    skip: Option<&str>,
) -> ActuatorDecisionOutcome {
    if applied {
        return ActuatorDecisionOutcome::Applied;
    }
    if result.is_err() {
        return ActuatorDecisionOutcome::Failed;
    }
    if block_reason == Some(BlockReason::NoMutation)
        || skip.is_some_and(|detail| detail.contains("noop") || detail.contains("no-mutation"))
    {
        return ActuatorDecisionOutcome::NoOp;
    }
    if block_reason.is_some() || skip.is_some() {
        return ActuatorDecisionOutcome::Blocked;
    }
    ActuatorDecisionOutcome::NoOp
}

pub fn decision_event_for_root_action(
    action: &RootAction,
    outcome: ActuatorDecisionOutcome,
    detail: String,
) -> ActuatorDecisionEvent {
    decision_event_for_root_action_from(action, outcome, "actuation-broker", detail)
}

pub fn decision_event_for_root_action_from(
    action: &RootAction,
    outcome: ActuatorDecisionOutcome,
    source: &str,
    detail: String,
) -> ActuatorDecisionEvent {
    let action_key = crate::engine::telemetry_medallion::actuator_action_key(action)
        .unwrap_or_else(|| format!("root:{}", action.action_class()));
    ActuatorDecisionEvent::local(
        action_key,
        root_action_target(action),
        0,
        outcome,
        source,
        detail,
    )
}

#[derive(Debug)]
enum FastUnfreezeResult {
    Applied,
    Stale(BlockReason),
    Failed(crate::engine::mediator::BlockReason),
}

/// Coordinate the two mandatory ordering points of a freeze without heap
/// allocation: optional reversible preparation, then the authoritative
/// SIGSTOP. A failed stop compensates preparation before returning.
#[inline]
fn run_freeze_saga<Prepare, Stop, Restore>(
    mut prepare: Prepare,
    mut stop: Stop,
    mut restore: Restore,
) -> crate::engine::mediator::SagaReport<String>
where
    Prepare: FnMut() -> Result<bool, String>,
    Stop: FnMut() -> Result<(), String>,
    Restore: FnMut() -> Result<(), String>,
{
    crate::engine::mediator::run_saga(
        2,
        |step| match step {
            0 => prepare().map(|applied| {
                if applied {
                    crate::engine::mediator::SagaStep::Applied
                } else {
                    crate::engine::mediator::SagaStep::NoOp
                }
            }),
            1 => stop().map(|()| crate::engine::mediator::SagaStep::Applied),
            _ => unreachable!("freeze saga has exactly two steps"),
        },
        |step| match step {
            0 => restore(),
            _ => Ok(()),
        },
    )
}

/// Execute a list of actions. Returns an [ExecuteOutcomes] accumulator that
/// the caller can merge into RuntimeMetrics **after** releasing any locks,
/// eliminating the need to hold locks across blocking I/O.
///
/// `memory_pressure` is the current kernel/compressor pressure in [0.0, 1.0]; at
/// or above 0.75 the per-PID power-assertion gate is bypassed so OOM-pressure
/// freezes can land even when a background app holds `PreventUserIdleSleep`.
pub fn execute_actions(
    actions: Vec<RootAction>,
    caps: &CapabilityReport,
    journal_path: &std::path::Path,
    frozen: &mut HashSet<u32>,
    learned_protected: &[String],
    learned_interactive: &[String],
    // S4 cutover (2026-06-06): shared ownership via Arc<Mutex<_>> so
    // ThreadPolicyEffector / MachPolicyEffector can co-own the manager
    // through the mediator chokepoint. The 4 internal mgr.* sites below
    // lock under the short-guard discipline (CLAUDE.md "Mutex-guarded
    // sections must be short; drop guards before any syscall" — the
    // set_tier / set_thread_qos calls ARE the syscall, so each guard
    // wraps exactly 1-2 FFI calls and drops immediately).
    qos_mgr: Option<&std::sync::Arc<std::sync::Mutex<MachQoSManager>>>,
    async_commands: Option<&crate::engine::daemon_helpers::AsyncCommandQueue>,
    dry_run: bool,
    memory_pressure: f64,
    thrashing_score: f64,
    coalition_guard: Option<&CoalitionGuard<'_>>,
    cpu_pegged_fraction: f64,
) -> ExecuteOutcomes {
    let protected = protected_processes_cached();
    // Only infrastructure (docker, postgres, redis, etc.) gets unconditional protection
    // at execution time. Dev runtimes (python, node, etc.) are filtered upstream by
    // behavioral_protection_score in the daemon — if they reach execute_actions,
    // they've already lost their behavioral gate.
    let critical_bg = infrastructure_processes_cached();
    let allowlist = allowlisted_sysctls();
    // Self-protection: never freeze/throttle/kill the daemon itself.
    let my_pid = std::process::id();
    // ML/AMX workloads: final safety net — never throttle or freeze inference processes.
    let ml_pids = amx_detector::ml_protected_pids();
    // Lazy: computed only if we actually have a FreezeProcess action.
    let mut assertion_pids: Option<std::collections::HashSet<u32>> = None;

    // Unified policy list for classify_protection(): learned_protected + learned_interactive.
    //
    // At execute time there is no foreground context, so learned_interactive patterns are
    // treated as unconditional skips (same as learned_protected).  Both are passed as
    // `policy_protected` to classify_protection(), which maps them to ProtectionLevel::Unconditional.
    // This is behaviorally identical to the previous three-step explicit check.
    let policy_all: Vec<String> = learned_protected
        .iter()
        .chain(learned_interactive.iter())
        .cloned()
        .collect();
    // Pre-build the Aho-Corasick matcher once for the entire execute_actions
    // loop. classify_protection() called below for every candidate action;
    // shared AC eliminates per-call `p.to_ascii_lowercase()` allocation in
    // Tier 3 substring scan. Built once even if loop body iterates ~50-200 times.
    let policy_all_ac = crate::engine::safety::cached_policy_protected_ac(&policy_all);

    let mut out = ExecuteOutcomes::default();
    // Batched journal buffer: entries are flushed in a single open/write/close
    // AFTER the main loop exits, so journaling never queues between actions
    // on the user-visible latency path.
    let mut pending_journal: Vec<JournalEntry> = Vec::with_capacity(16);

    // ── Fast-path unfreeze pre-pass ─────────────────────────────────────────
    //
    // The main loop below does ~5 syscalls (SIGCONT + taskpolicy I/O tier +
    // mach_qos + memorystatus + journal fsync) per action, serially. With N
    // frozen Chromium renderers that's ~N × 10–30 ms, dominated by the
    // synchronous journal append. During that window the user perceives the
    // LATER pids in the list as "still frozen" — the browser grey-tabs a
    // renderer long after SIGCONT would have resumed it.
    //
    // Deliver one verified SIGCONT to every UnfreezeProcess action before
    // slower restoration/bookkeeping. The old pre-pass used a name-only
    // check, ignored the syscall result, then sent SIGCONT a second time in
    // the main loop. Carry the receipt forward instead: this preserves low
    // tail latency while closing PID-reuse and double-signal races.
    //
    // References:
    // - [Dean & Barroso 2013] "The Tail at Scale" CACM §3 — keep
    //   latency-critical work off the serialized path where slow
    //   operations queue ahead of it.
    // - [POSA2] "Half-Sync/Half-Async" — fast synchronous dispatch
    //   decoupled from slower async bookkeeping.
    // - [Gray & Reuter 1992] §10 — journaling must not gate user-visible
    //   state transitions; log-after-apply is correct here because the
    //   kernel already owns the authoritative frozen state.
    let mut fast_unfreezes = HashMap::new();
    if !dry_run {
        for action in &actions {
            let RootAction::UnfreezeProcess {
                pid,
                name,
                start_sec,
                start_usec,
                ..
            } = action
            else {
                continue;
            };

            let result = if proc_taskinfo::is_zombie_pid(*pid) {
                FastUnfreezeResult::Stale(BlockReason::Zombie)
            } else if !ProcessIdentity::verify(*pid, Some(name), *start_sec, *start_usec) {
                crate::engine::lse_counters::LSE_COUNTERS.inc_pid_recycle_block();
                FastUnfreezeResult::Stale(BlockReason::PidRecycled)
            } else {
                let effect = crate::engine::mediator::Effect::SigCont {
                    pid: *pid,
                    start_sec: *start_sec,
                };
                let precondition = crate::engine::mediator::PreCondition {
                    pid_identity: Some((*pid, *start_sec)),
                    ..Default::default()
                };
                match crate::engine::mediator::mediate(
                    &effect,
                    &precondition,
                    &crate::engine::mediator::SignalEffector,
                ) {
                    Ok(receipt) if receipt.applied_count > 0 => FastUnfreezeResult::Applied,
                    Ok(_) => FastUnfreezeResult::Stale(BlockReason::PidRecycled),
                    Err(crate::engine::mediator::BlockReason::IdentityMismatch { .. }) => {
                        FastUnfreezeResult::Stale(BlockReason::PidRecycled)
                    }
                    Err(crate::engine::mediator::BlockReason::OsError { errno, .. })
                        if errno == libc::ESRCH =>
                    {
                        FastUnfreezeResult::Stale(BlockReason::PidRecycled)
                    }
                    Err(error) => FastUnfreezeResult::Failed(error),
                }
            };
            fast_unfreezes.insert(*pid, result);
        }
    }

    for action in actions {
        // Drain any leftover skip reason from prior iteration before running.
        out.last_skip = None;
        let mut before = None;
        let mut after = None;
        // Separate real kernel mutation from policy/journal acceptance. Dry-run
        // remains a successful simulation in the journal, but it must not feed
        // learning or the cross-cycle recently-applied cache.
        let mut action_applied = false;
        let mut async_submission = None;
        let mut receipt_detail = None;

        let decision_reason = match &action {
            RootAction::BoostProcess {
                decision_reason, ..
            }
            | RootAction::ThrottleProcess {
                decision_reason, ..
            }
            | RootAction::FreezeProcess {
                decision_reason, ..
            }
            | RootAction::UnfreezeProcess {
                decision_reason, ..
            }
            | RootAction::SetMemorystatus {
                decision_reason, ..
            }
            | RootAction::ToggleSpotlight {
                decision_reason, ..
            }
            | RootAction::QuarantineDaemon {
                decision_reason, ..
            }
            | RootAction::SetThreadQoS {
                decision_reason, ..
            } => decision_reason.clone(),
            RootAction::SetSysctl(s) => s.decision_reason().clone(),
        };

        let reason = match &action {
            RootAction::BoostProcess { reason, .. }
            | RootAction::ThrottleProcess { reason, .. }
            | RootAction::FreezeProcess { reason, .. }
            | RootAction::SetMemorystatus { reason, .. }
            | RootAction::ToggleSpotlight { reason, .. }
            | RootAction::QuarantineDaemon { reason, .. }
            | RootAction::SetThreadQoS { reason, .. }
            | RootAction::UnfreezeProcess { reason, .. } => reason.clone(),
            RootAction::SetSysctl(s) => s.reason().to_string(),
        };

        let mut block_reason = None;
        if dry_run {
            block_reason = Some(BlockReason::DryRun);
        }

        let result: anyhow::Result<()> = (|| {
            match &action {
                RootAction::BoostProcess {
                    pid,
                    name,
                    start_sec,
                    start_usec,
                    ..
                } => {
                    // Self-protection only — display-critical daemons (coreaudiod, Dock,
                    // mediaserverd) are in protected_processes for freeze/throttle safety, but
                    // must be BOOSTABLE. True OS-kernel processes (WindowServer, kernel_task)
                    // fail gracefully via is_sip_protected() in set_tier().
                    if *pid == my_pid || name.contains("apollo-optimizer") {
                        out.push_skip(format!("boost-no-mutation:{name}:pid={pid}"));
                        block_reason = Some(BlockReason::NoMutation);
                        return Ok(());
                    }
                    // Inv#11 (2026-06-06): real start_sec verify closes the
                    // A-B-A window — previous `0,0` legacy fallback was a
                    // no-op tautology (verify always accepted, counter was
                    // perma-zero across 59 675 cycles). Producers populate
                    // start_sec at all Boost emit sites — see
                    // decide_actions.rs / local_policy_learning.rs sweep.
                    if !ProcessIdentity::verify(*pid, Some(name), *start_sec, *start_usec) {
                        crate::engine::lse_counters::LSE_COUNTERS.inc_pid_recycle_block();
                        block_reason = Some(BlockReason::PidRecycled);
                        return Ok(());
                    }
                    // FIX-4-v2 (2026-06-07): track whether the Mach
                    // `set_tier` syscall actually mutated state. The Round-2
                    // unconditional `record_global` call enrolled a
                    // PendingObservation even when caps.can_taskpolicy was
                    // false OR qos_mgr was None — a phantom-enrollment
                    // pathway that would assert
                    // SchedulingTier::Foreground post-state without any
                    // syscall having run. The consumer would then re-read
                    // the live tier, observe disagreement against a value
                    // we never set, and feed false-positive HP
                    // disagreements into the rollback trigger window.
                    let mut tier_syscall_ok = false;
                    let mut latency_syscall_ok = false;
                    if !dry_run {
                        if caps.can_taskpolicy {
                            // Phase 2: direct Mach syscalls (~50µs vs ~5ms fork/exec).
                            // S4 cutover: short-guard Mutex lock per CLAUDE.md doctrine.
                            if let Some(arc) = qos_mgr.as_ref() {
                                let mut mgr = arc.lock().unwrap_or_else(|e| e.into_inner());
                                let outcome = mgr.set_tier(
                                    *pid,
                                    crate::engine::mach_qos::SchedulingTier::Foreground,
                                );
                                let latency_outcome = mgr.set_latency_and_throughput(
                                    *pid,
                                    LatencyTier::Interactive,
                                    ThroughputTier::High,
                                );
                                latency_syscall_ok = latency_outcome.mutated;
                                drop(mgr);
                                // Round-4 (2026-06-07): use `mutated` not
                                // `success`. set_tier returns success=true on
                                // cache-hit, permanently-blocked, sip-skip
                                // paths where NO task_policy_set syscall ran.
                                // Phantom enrollment of cached Brave Boosts
                                // would feed FIX-3-v2 unconditional HP forward
                                // and spurious-rollback zone_alpha during
                                // crisis. `mutated` is true only after
                                // KERN_SUCCESS from apply_task_policy.
                                tier_syscall_ok = outcome.mutated;
                            }
                            // Boost I/O tier to Interactive.
                            action_applied |=
                                apply_io_tier(*pid, crate::engine::io_tiering::IOTier::Interactive);
                        }
                        let nice_prior = set_nice(*pid, -10).ok().flatten();
                        let nice_applied = nice_prior.is_some();
                        if nice_prior.is_none() {
                            crate::engine::effect_ledger::refresh_nice_global(
                                *pid,
                                BOOST_EFFECT_TTL,
                            );
                        }
                        action_applied |= tier_syscall_ok || latency_syscall_ok || nice_applied;
                        // Evolve iter-4 (2026-06-10): unified EffectLedger
                        // replaces the ad-hoc boost_ledger. Both side-effects
                        // of a boost (nice -10 + Foreground tier) are now
                        // recorded with their undo; reconcile_global reverts
                        // them once the process stops qualifying.
                        if let Some(prior) = nice_prior {
                            crate::engine::effect_ledger::record_global(
                                crate::engine::effect_ledger::AppliedEffect::Nice {
                                    pid: *pid,
                                    prior,
                                },
                                BOOST_EFFECT_TTL,
                                *start_sec,
                                "boost: renice -10",
                            );
                        }
                        record_boost_qos_effects(
                            *pid,
                            *start_sec,
                            tier_syscall_ok,
                            latency_syscall_ok,
                        );
                    }
                    out.boosts_applied += (action_applied || dry_run) as u64;
                    if !dry_run && !action_applied {
                        out.push_skip(format!("boost-no-mutation:{name}:pid={pid}"));
                        block_reason = Some(BlockReason::NoMutation);
                    }
                    // FIX-4-v2 (2026-06-07): phantom-enrollment guard.
                    // Only enroll a PendingObservation for the Hellerstein
                    // 2004 §9.3 effect-decay watchdog when the Mach
                    // mutation actually ran. Guard chain:
                    //   1. !dry_run               — no enrollment during simulate
                    //   2. caps.can_taskpolicy    — kernel API surface live
                    //   3. qos_mgr.is_some()      — manager wired into this path
                    //   4. tier_syscall_ok        — `set_tier` returned success
                    // Without (1-4) the consumer would re-read a tier we
                    // never set and emit a spurious effect_decay event,
                    // polluting the HP disagreement window that drives
                    // poke_rollback_guard_via_decay. The
                    // `effect_decay_phantom_enroll_skipped_total` counter
                    // surfaces every rejected enrollment so dashboards can
                    // separate "no signal because we didn't mutate" from
                    // "signal observed and accepted".
                    let phantom_guards_pass =
                        !dry_run && caps.can_taskpolicy && qos_mgr.is_some() && tier_syscall_ok;
                    if phantom_guards_pass {
                        // Round-4 (2026-06-07): gate `is_hp` on the carve-out
                        // DecisionReason. decide_actions has TWO intentional
                        // BoostProcess emissions targeting hard-protected
                        // names (DisplayPipeline → WindowServer/Dock/
                        // SystemUIServer; CompositorPriority → WindowServer
                        // high-CPU). These are NOT policy failures — they're
                        // cooperative carve-outs that the safety doctrine
                        // explicitly permits (see decide_actions.rs:888-891
                        // and :924-927). Marking them hard_protected would
                        // make FIX-3-v2 forward them to PolicyRollbackGuard
                        // as disagreements, inverting the signal: a
                        // successful intentional carve-out boost would log
                        // as "policy failure → revert zone_alpha".
                        let is_carveout = matches!(
                            decision_reason,
                            crate::engine::audit_types::DecisionReason::DisplayPipeline
                                | crate::engine::audit_types::DecisionReason::CompositorPriority
                        );
                        let is_hp = !is_carveout && crate::engine::safety::is_boost_forbidden(name);
                        crate::engine::effect_decay::record_global(
                            crate::engine::effect_decay::PendingObservation {
                                effect_id: 0,
                                pid: *pid,
                                kind: crate::engine::effect_decay::ObsKind::MachPolicy,
                                key: None,
                                // SchedulingTier::Foreground == 0 (first
                                // variant, see mach_qos.rs:271). Stable
                                // post-syscall encoding for the consumer's
                                // re-read via `MachQoSManager::current_tier`.
                                value_post: 0,
                                deadline: std::time::Instant::now()
                                    + crate::engine::effect_decay::DecayWatchdog::settle_window(),
                                hard_protected: is_hp,
                            },
                        );
                    } else if !dry_run {
                        // Only bump the skip counter for real (non-dry-run)
                        // cycles — dry-run skips are by-design and would
                        // otherwise saturate the counter.
                        crate::engine::lse_counters::LSE_COUNTERS
                            .inc_effect_decay_phantom_enroll_skipped();
                    }
                }
                RootAction::ThrottleProcess {
                    pid,
                    name,
                    aggressive,
                    start_sec,
                    start_usec,
                    ..
                } => {
                    if *pid == my_pid {
                        return Ok(());
                    }
                    // Coalition guard: never throttle a PID whose coalition
                    // is in the active fg envelope (current + 5-min grace).
                    // Subprocesses of the user's active workflow stay
                    // unthrottled even when names drift across versions.
                    if coalition_guard
                        .map(|g| g.is_protected(*pid))
                        .unwrap_or(false)
                    {
                        block_reason = Some(BlockReason::ActiveCoalition);
                        return Ok(());
                    }
                    // CPU-saturation guard: when ≥80% of cores are pegged
                    // and memory pressure is still healthy (<0.75), throttling
                    // adds scheduler contention without easing the real
                    // bottleneck. Threshold pair derived from cpu_saturation.rs
                    // pegged_fraction ≥0.80 (one core idle) and the survival
                    // threshold above which freezes are mandatory regardless.
                    if cpu_pegged_fraction >= 0.80 && memory_pressure < 0.75 {
                        block_reason = Some(BlockReason::CpuSaturated);
                        return Ok(());
                    }
                    // Unified protection check: hard OS names + policy-learned + interactive.
                    // learned_interactive is treated as Unconditional at execute time because
                    // no foreground context is available here (see policy_all pre-computation).
                    // infra (infrastructure_processes) is intentionally excluded: critical_bg
                    // below handles infra with soft-throttle semantics, not a full skip.
                    match crate::engine::safety::classify_protection_canonical(
                        name,
                        &policy_all,
                        policy_all_ac.as_deref(),
                        false,
                    ) {
                        ProtectionLevel::Unconditional => {
                            out.push_skip(format!("protected:{}", name));
                            block_reason = Some(BlockReason::ProtectedProcess);
                            return Ok(());
                        }
                        ProtectionLevel::ConditionalForeground | ProtectionLevel::Unprotected => {}
                    }
                    // ML/AMX protection: never throttle inference workloads.
                    if ml_pids.contains(pid) {
                        out.push_skip(format!("ml-protected:{}", name));
                        block_reason = Some(BlockReason::MlProtected);
                        return Ok(());
                    }
                    // Validate PID identity with start-time (prevents A-B-A recycling).
                    if !ProcessIdentity::verify(*pid, Some(name), *start_sec, *start_usec) {
                        out.push_skip(format!("pid-recycled:{}", name));
                        block_reason = Some(BlockReason::PidRecycled);
                        return Ok(());
                    }
                    // PID-level Apple platform check: csops CS_PLATFORM_BINARY + path prefix.
                    if process_identity::is_apple_platform_process(*pid) {
                        out.push_skip(format!("apple-platform:{}", name));
                        block_reason = Some(BlockReason::ApplePlatform);
                        return Ok(());
                    }
                    let is_critical_bg = critical_bg.iter().any(|p| name.contains(p));
                    let aggressive = if is_critical_bg { false } else { *aggressive };
                    if is_critical_bg {
                        out.critical_background_skips += 1;
                        out.push_skip(format!("critical-bg:{}", name));
                        block_reason = Some(BlockReason::CriticalBackground);
                    }
                    if !dry_run && caps.can_taskpolicy {
                        // Phase 2: direct Mach syscalls for CPU tier routing.
                        // S4 cutover: short-guard Mutex lock.
                        if let Some(arc) = qos_mgr.as_ref() {
                            let mut mgr = arc.lock().unwrap_or_else(|e| e.into_inner());
                            let sched_tier = if aggressive {
                                crate::engine::mach_qos::SchedulingTier::Background
                            // E-cores only
                            } else {
                                crate::engine::mach_qos::SchedulingTier::Normal
                                // scheduler decides, less invasive than E-cores-only
                            };
                            let tier_outcome = mgr.set_tier(*pid, sched_tier);
                            let lat = if aggressive {
                                LatencyTier::Background
                            } else {
                                LatencyTier::Default
                            };
                            let thr = if aggressive {
                                ThroughputTier::Low
                            } else {
                                ThroughputTier::Default
                            };
                            let latency_outcome = mgr.set_latency_and_throughput(*pid, lat, thr);
                            action_applied |= tier_outcome.mutated || latency_outcome.mutated;
                            drop(mgr);
                        }
                        // Granular I/O tiering based on aggressiveness.
                        // apply_io_tier uses PRIO_DARWIN_BG which is
                        // turnstile-compatible — do NOT also set nice=20
                        // via PRIO_PROCESS, as that breaks the Mach
                        // priority-inheritance chain (Finder/Settings hangs).
                        let io_tier = io_tier_for_throttle(aggressive);
                        action_applied |= apply_io_tier(*pid, io_tier);
                    }
                    out.throttles_applied += (action_applied || dry_run) as u64;
                }
                RootAction::FreezeProcess {
                    pid,
                    name,
                    start_sec,
                    start_usec,
                    ..
                } => {
                    if *pid == my_pid {
                        return Ok(());
                    }
                    // Coalition guard: never freeze a PID whose coalition is
                    // in the active fg envelope. Tabbing momentarily away
                    // from Antigravity to run `git status` does not strip
                    // its renderers of freeze immunity.
                    if coalition_guard
                        .map(|g| g.is_protected(*pid))
                        .unwrap_or(false)
                    {
                        out.push_skip(format!("active-coalition:{}", name));
                        block_reason = Some(BlockReason::ActiveCoalition);
                        return Ok(());
                    }
                    // CPU-saturation guard: when CPU is pegged but memory
                    // headroom is fine, freezing a background process moves
                    // its threads off the run queue but doesn't release any
                    // memory pressure (because there isn't any). The page
                    // residency stays, the freeze adds context-switch cost
                    // on resume, and the user perceives "system feels slow
                    // during CPU-heavy task". Skip with CpuSaturated.
                    if cpu_pegged_fraction >= 0.80 && memory_pressure < 0.75 {
                        out.push_skip(format!("cpu-saturated:{}", name));
                        block_reason = Some(BlockReason::CpuSaturated);
                        return Ok(());
                    }
                    // Unified protection check: hard OS names + infra + policy-learned + interactive.
                    // Unlike ThrottleProcess, infra (critical_bg) is included here because
                    // FreezeProcess treats infra as a full skip (not a soft-throttle path).
                    // learned_interactive is treated as Unconditional: no foreground context
                    // at execute time (see policy_all pre-computation above).
                    match crate::engine::safety::classify_protection_canonical(
                        name,
                        &policy_all,
                        policy_all_ac.as_deref(),
                        false,
                    ) {
                        ProtectionLevel::Unconditional => {
                            if critical_bg.iter().any(|p| name.contains(p)) {
                                out.critical_background_skips += 1;
                            }
                            out.push_skip(format!("protected:{}", name));
                            block_reason = Some(BlockReason::ProtectedProcess);
                            return Ok(());
                        }
                        ProtectionLevel::ConditionalForeground | ProtectionLevel::Unprotected => {}
                    }
                    // ML/AMX protection: never freeze inference workloads.
                    if ml_pids.contains(pid) {
                        out.push_skip(format!("ml-protected:{}", name));
                        block_reason = Some(BlockReason::MlProtected);
                        return Ok(());
                    }
                    // Validate PID identity with start-time (prevents A-B-A recycling).
                    if !ProcessIdentity::verify(*pid, Some(name), *start_sec, *start_usec) {
                        block_reason = Some(BlockReason::PidRecycled);
                        return Ok(());
                    }
                    // PID-level Apple platform check: csops CS_PLATFORM_BINARY + path prefix.
                    if process_identity::is_apple_platform_process(*pid) {
                        out.push_skip(format!("apple-platform:{}", name));
                        block_reason = Some(BlockReason::ApplePlatform);
                        return Ok(());
                    }
                    // Never freeze processes with active power assertions
                    // (audio playback, active downloads, background tasks).
                    //
                    // High-pressure bypass: at or above 0.70 kernel/compressor
                    // pressure the OOM risk outweighs interrupting a download
                    // or background task — without this, a single PID holding
                    // PreventUserIdleSleep blocks every freeze while swap climbs.
                    // Bypass per-PID assertion gate under physical crisis:
                    //   pressure ≥ 0.70 — RAM level critical
                    //   thrashing ≥ 10k — flow crisis (Gate C); compressor churning,
                    //                     OOM imminent regardless of assertion intent.
                    //   p_oom_30s ≥ 0.40 — hazard model predicts OOM within 30s.
                    // Mirror of UserContext::freeze_protected bypass conditions.
                    // [Nygard 2018] load shedding overrides politeness under overload;
                    // [Camacho 2007] predictive bypass catches crises before thrashing.
                    let p_oom_30s = crate::engine::shadow_signals::get_p_oom_30s().unwrap_or(0.0);
                    if memory_pressure < 0.70 && thrashing_score < 10_000.0 && p_oom_30s < 0.40 {
                        let busy = assertion_pids.get_or_insert_with(pids_with_assertions);
                        if busy.contains(pid) {
                            out.push_skip(format!("assertion-active:{}", name));
                            block_reason = Some(BlockReason::AssertionActive);
                            return Ok(());
                        }
                    }
                    if dry_run {
                        // Simulate success without touching the process.
                        frozen.insert(*pid);
                        out.freezes_applied += 1;
                        out.newly_frozen_pids.push(*pid);
                        out.newly_frozen_identity.push((*pid, *start_sec, None));
                    } else {
                        // A2/A4 fix (round-3): skip zombies before SIGSTOP. SIGSTOP on
                        // a zombie is a kernel no-op that still burns a syscall.
                        if proc_taskinfo::is_zombie_pid(*pid) {
                            out.push_skip(format!("zombie:{}", name));
                            block_reason = Some(BlockReason::Zombie);
                            return Ok(());
                        }
                        // A5/D1: capture the original jetsam priority BEFORE we demote
                        // the PID to BACKGROUND.  Saved on the FrozenEntry (propagated
                        // via ExecuteOutcomes::newly_frozen_identity) so unfreeze can
                        // restore the exact original value instead of blanket-setting
                        // Interactive (which previously lost AUDIO / VITAL).
                        let captured_priority = if caps.can_memorystatus {
                            crate::engine::jetsam_control::get_priority(*pid)
                        } else {
                            None
                        };
                        // Freeze is Apollo's only destructive multi-step action:
                        // prepare the jetsam tier, then issue SIGSTOP. Run both
                        // through the mediator's allocation-free saga harness so
                        // a failed SIGSTOP restores the exact prior jetsam value.
                        // Advisory I/O demotion happens only after commit and
                        // therefore cannot leave a partial action behind.
                        let freeze_saga = run_freeze_saga(
                            || -> Result<bool, String> {
                                // Without an exact prior value there is no safe
                                // compensation, so leave this optional step a no-op.
                                if !caps.can_memorystatus || captured_priority.is_none() {
                                    return Ok(false);
                                }
                                let effect = crate::engine::mediator::Effect::SetJetsamTier {
                                    pid: *pid,
                                    start_sec: *start_sec,
                                    tier: crate::engine::mediator::JetsamTierKind::Background,
                                };
                                let precondition = crate::engine::mediator::PreCondition {
                                    pid_identity: Some((*pid, *start_sec)),
                                    ..Default::default()
                                };
                                match crate::engine::mediator::mediate(
                                    &effect,
                                    &precondition,
                                    &crate::engine::mediator::JetsamEffector,
                                ) {
                                    Ok(receipt) => Ok(receipt.applied_count > 0),
                                    // Jetsam preparation is opportunistic. A
                                    // blocked preparation must not prevent the
                                    // independently-safe SIGSTOP operation.
                                    Err(_) => Ok(false),
                                }
                            },
                            || -> Result<(), String> {
                                let effect = crate::engine::mediator::Effect::SigStop {
                                    pid: *pid,
                                    start_sec: *start_sec,
                                };
                                let precondition = crate::engine::mediator::PreCondition {
                                    pid_identity: Some((*pid, *start_sec)),
                                    ..Default::default()
                                };
                                match crate::engine::mediator::mediate(
                                    &effect,
                                    &precondition,
                                    &crate::engine::mediator::SignalEffector,
                                ) {
                                    Ok(receipt) if receipt.applied_count > 0 => Ok(()),
                                    Ok(_) => {
                                        Err(format!("SIGSTOP produced no mutation for pid {}", pid))
                                    }
                                    Err(error) => {
                                        Err(format!("SIGSTOP failed for pid {}: {:?}", pid, error))
                                    }
                                }
                            },
                            || -> Result<(), String> {
                                if let Some(priority) = captured_priority {
                                    let effect = crate::engine::mediator::Effect::SetJetsamTier {
                                        pid: *pid,
                                        start_sec: *start_sec,
                                        tier: crate::engine::mediator::JetsamTierKind::Exact(
                                            priority,
                                        ),
                                    };
                                    let precondition = crate::engine::mediator::PreCondition {
                                        pid_identity: Some((*pid, *start_sec)),
                                        ..Default::default()
                                    };
                                    crate::engine::mediator::mediate(
                                        &effect,
                                        &precondition,
                                        &crate::engine::mediator::JetsamEffector,
                                    )
                                    .map(|_| ())
                                    .map_err(|error| {
                                        format!(
                                            "restore jetsam priority {} for pid {} failed: {:?}",
                                            priority, pid, error
                                        )
                                    })
                                } else {
                                    Ok(())
                                }
                            },
                        );

                        if !freeze_saga.committed() {
                            let (failed_step, apply_error) = freeze_saga
                                .apply_failure()
                                .map(|(step, error)| (step, error.as_str()))
                                .unwrap_or((usize::MAX, "invalid saga configuration"));
                            if let Some((compensation_step, compensation_error)) =
                                freeze_saga.compensation_failure()
                            {
                                if let Some(prior) = captured_priority {
                                    crate::engine::effect_ledger::record_global(
                                        crate::engine::effect_ledger::AppliedEffect::JetsamPriority {
                                            pid: *pid,
                                            prior,
                                        },
                                        std::time::Duration::ZERO,
                                        *start_sec,
                                        "freeze-saga: retry jetsam compensation",
                                    );
                                }
                                anyhow::bail!(
                                    "freeze saga {:?}: step {} failed: {}; compensation step {} failed: {}",
                                    freeze_saga.state(),
                                    failed_step,
                                    apply_error,
                                    compensation_step,
                                    compensation_error
                                );
                            }
                            anyhow::bail!(
                                "freeze saga {:?}: step {} failed: {}",
                                freeze_saga.state(),
                                failed_step,
                                apply_error
                            );
                        }

                        action_applied = freeze_saga.applied_step(1);
                        if action_applied {
                            // Post-commit advisory: this affects only resume
                            // behavior and is safe to retry or omit.
                            if caps.can_taskpolicy {
                                apply_io_tier(*pid, crate::engine::io_tiering::IOTier::Passive);
                            }
                            if freeze_saga.applied_step(0) {
                                let is_hp = crate::engine::safety::hard_protected_contains(name);
                                crate::engine::effect_decay::record_global(
                                    crate::engine::effect_decay::PendingObservation {
                                        effect_id: 0,
                                        pid: *pid,
                                        kind: crate::engine::effect_decay::ObsKind::JetsamTier,
                                        key: None,
                                        value_post: crate::engine::jetsam_control::priority::BACKGROUND as i64,
                                        deadline: std::time::Instant::now()
                                            + crate::engine::effect_decay::DecayWatchdog::settle_window(),
                                        hard_protected: is_hp,
                                    },
                                );
                            }
                            frozen.insert(*pid);
                            out.freezes_applied += 1;
                            out.newly_frozen_pids.push(*pid);
                            out.newly_frozen_identity
                                .push((*pid, *start_sec, captured_priority));
                        }
                    }
                }
                RootAction::UnfreezeProcess { pid, .. } => {
                    if dry_run {
                        // Simulate success without touching the process.
                        frozen.remove(pid);
                        out.unfreezes_applied += 1;
                        out.throttle_reverted += 1;
                        out.newly_unfrozen_pids.push(*pid);
                    } else {
                        match fast_unfreezes.remove(pid) {
                            Some(FastUnfreezeResult::Applied) => {}
                            Some(FastUnfreezeResult::Stale(reason)) => {
                                // The frozen entry belongs to a process that exited
                                // or a prior occupant of this numeric PID. Remove the
                                // stale ownership without signalling the live process.
                                frozen.remove(pid);
                                block_reason = Some(reason);
                                return Ok(());
                            }
                            Some(FastUnfreezeResult::Failed(error)) => {
                                anyhow::bail!(
                                    "verified SIGCONT failed for pid {}: {:?}",
                                    pid,
                                    error
                                )
                            }
                            None => {
                                anyhow::bail!("missing fast-unfreeze receipt for pid {}", pid)
                            }
                        }

                        action_applied = true;
                        // Restore I/O tier to Standard on unfreeze.
                        if caps.can_taskpolicy {
                            apply_io_tier(*pid, crate::engine::io_tiering::IOTier::Standard);
                            // Warmup boost: temporary Foreground QoS burst accelerates
                            // working-set reload from the compressor on resume.
                            // Next cycle re-evaluates and may demote back.
                            // [Ousterhout 2013 "Scheduling for Reduced Tail Latency" OSDI;
                            //  iOS app resume — foreground pulse for fast working-set reload]
                            // S4 cutover: short-guard Mutex lock.
                            if let Some(arc) = qos_mgr.as_ref() {
                                let mut mgr = arc.lock().unwrap_or_else(|e| e.into_inner());
                                mgr.set_tier(
                                    *pid,
                                    crate::engine::mach_qos::SchedulingTier::Foreground,
                                );
                                drop(mgr);
                            }
                        }
                        // A5/D1 fix (round-3): previously we blanket-set
                        // JetsamClass::Interactive (FOREGROUND=9), which clobbered
                        // AUDIO (18), AUDIO_AND_ACCESSORY (10), VITAL (12), etc.
                        // The correct restoration path runs from
                        // daemon_helpers::unfreeze_pids_verified(), which has
                        // access to `FrozenEntry::original_jetsam_priority`. Here
                        // we leave jetsam priority untouched when we don't know
                        // the original value.
                        frozen.remove(pid);
                        out.unfreezes_applied += 1;
                        out.throttle_reverted += 1;
                        out.newly_unfrozen_pids.push(*pid);
                    }
                }
                RootAction::SetSysctl(s) => {
                    let key = s.key();
                    let value = s.value();
                    let requested = match value.parse::<i64>() {
                        Ok(value) => value,
                        Err(_) => {
                            out.invalid_sysctl_denied += 1;
                            out.push_skip(format!("sysctl-nonnumeric:{}={}", key, value));
                            block_reason = Some(BlockReason::InvalidSysctl);
                            return Ok(());
                        }
                    };
                    if !allowlist.contains(key) {
                        out.invalid_sysctl_denied += 1;
                        out.push_skip(format!("sysctl-not-allowlisted:{key}"));
                        block_reason = Some(BlockReason::InvalidSysctl);
                        return Ok(());
                    }
                    if !caps.can_sysctl {
                        out.push_skip(format!("sysctl-capability-unavailable:{key}"));
                        block_reason = Some(BlockReason::SysctlFailed);
                        return Ok(());
                    }
                    // Defense-in-depth range check. The
                    // `SetSysctlAction::new_clamped` factory already clamps
                    // numeric values, but we re-validate here to catch:
                    //   1. Type-system escape via deserialization from a
                    //      hostile journal/socket payload (Sprint 4 Phase 4
                    //      seal protects construction in-process only).
                    //   2. Kernel-rejected ranges the safety allowlist
                    //      doesn't model fully.
                    let ranges = allowlisted_sysctls_with_ranges();
                    if let Some(range) = ranges.iter().find(|r| r.key == key) {
                        if let Ok(numeric_val) = value.parse::<i64>() {
                            if numeric_val < range.min || numeric_val > range.max {
                                out.invalid_sysctl_denied += 1;
                                out.push_skip(format!("sysctl-out-of-range:{}={}", key, value));
                                block_reason = Some(BlockReason::SysctlOutOfRange);
                                return Ok(());
                            }
                        }
                    }
                    // Read current value — doubles as existence check.
                    // Uses timeout wrapper: sysctlbyname can block as root.
                    let observed_before = match sysctl_read_numeric_with_timeout(key) {
                        Some(val) => {
                            before = Some(val.value.to_string());
                            val
                        }
                        None => {
                            // Read timed out (worker thread saturated) or key
                            // unreadable. Without push_skip, journal records
                            // success=true with before=null/after=null —
                            // 146 phantom entries observed in 7h prod soak
                            // (fix 2026-05-07).
                            out.invalid_sysctl_denied += 1;
                            out.push_skip(format!("sysctl-read-failed:{}", key));
                            block_reason = Some(BlockReason::InvalidSysctl);
                            return Ok(());
                        }
                    };
                    // Skip no-op writes: if current value already equals the
                    // proposed value, don't issue the write nor emit a journal
                    // entry. After the Phase C clamp landed, governor began
                    // emitting clamped-to-current writes (e.g. delayed_ack=3
                    // when sysctl already reads 3), inflating the journal
                    // with success-but-unchanged entries (fix 2026-05-07).
                    if observed_before.value == requested {
                        out.push_skip(format!("sysctl-noop:{}={}", key, value));
                        return Ok(());
                    }
                    if !dry_run {
                        let transaction = sysctl_numeric_transaction_with_timeout(
                            key,
                            observed_before,
                            requested,
                        );
                        let observed = match transaction {
                            Some(NumericSysctlTransaction::Applied(observed)) => observed,
                            Some(NumericSysctlTransaction::NoOp) => {
                                out.push_skip(format!("sysctl-noop:{}={}", key, value));
                                return Ok(());
                            }
                            Some(NumericSysctlTransaction::ReadFailed) => {
                                out.push_skip(format!("sysctl-transaction-read-failed:{key}"));
                                block_reason = Some(BlockReason::SysctlFailed);
                                return Ok(());
                            }
                            Some(NumericSysctlTransaction::PreconditionChanged) => {
                                out.push_skip(format!(
                                    "sysctl-precondition-changed:{} expected={}",
                                    key, observed_before.value
                                ));
                                block_reason = Some(BlockReason::SysctlFailed);
                                return Ok(());
                            }
                            Some(NumericSysctlTransaction::WriteFailed) => {
                                anyhow::bail!("sysctl write failed: {}={}", key, value);
                            }
                            Some(NumericSysctlTransaction::Uncertain(observed)) => {
                                crate::engine::lse_counters::LSE_COUNTERS
                                    .inc_mediator_postcondition_violation();
                                after = observed.map(|value| value.value.to_string());
                                action_applied = true;
                                anyhow::bail!(
                                    "sysctl state uncertain after failed compensation: {}={} observed={}",
                                    key,
                                    requested,
                                    after.as_deref().unwrap_or("unreadable")
                                );
                            }
                            None => {
                                anyhow::bail!(
                                    "sysctl transaction timed out; worker owns compensation: {}={}",
                                    key,
                                    value
                                );
                            }
                        };
                        after = Some(observed.value.to_string());
                        action_applied = true;
                        out.sysctl_applied += 1;
                        // S10 producer: enroll post-Receipt observation. The
                        // consumer re-reads the native i32/i64 width after the
                        // 5 s settle window and bumps effect_decay_detected_total on
                        // mismatch (kernel reverted, sysctl saturated to a
                        // different value, etc).
                        if let Some(post_str) = after.as_ref() {
                            if let Ok(post_val) = post_str.parse::<i64>() {
                                crate::engine::effect_decay::record_global(
                                    crate::engine::effect_decay::PendingObservation {
                                        effect_id: 0,
                                        pid: 0,
                                        kind:
                                            crate::engine::effect_decay::ObsKind::Sysctl,
                                        key: Some(key.to_string()),
                                        value_post: post_val,
                                        deadline: std::time::Instant::now()
                                            + crate::engine::effect_decay::DecayWatchdog::settle_window(),
                                        // Build-shim (see jetsam site above). SetSysctl has
                                        // no per-process name in scope; defaulting to false
                                        // is correct — sysctl keys are global, not per-PID.
                                        hard_protected: false,
                                    },
                                );
                            }
                        }
                    }
                }
                RootAction::SetMemorystatus { pid, .. } => {
                    // Coalition guard: never pressure a PID whose coalition
                    // is in the active fg envelope. memorystatus_vm_pressure_send
                    // forces the target to drop caches; doing this to a
                    // helper of the user's active app produces stutter.
                    if coalition_guard
                        .map(|g| g.is_protected(*pid))
                        .unwrap_or(false)
                    {
                        out.push_skip(format!("active-coalition:pid={}", *pid));
                        block_reason = Some(BlockReason::ActiveCoalition);
                        return Ok(());
                    }
                    if !dry_run && !caps.can_memory_pressure_send {
                        out.push_skip(format!("memorystatus-send-unsupported:pid={}", *pid));
                        block_reason = Some(BlockReason::MemorystatusFailed);
                        return Ok(());
                    }
                    if !dry_run {
                        // Guard: never send memory pressure to protected/critical processes.
                        let is_protected = crate::engine::process_identity::proc_name_for_pid(*pid)
                            .map(|name| {
                                let nl = name.to_ascii_lowercase();
                                protected
                                    .iter()
                                    .any(|p| nl.contains(&p.to_ascii_lowercase()))
                                    || critical_bg
                                        .iter()
                                        .any(|c| nl.contains(&c.to_ascii_lowercase()))
                                    // 2026-06-21 (P2, playback-easing Wave 1):
                                    // complete-mediation chokepoint — never send
                                    // memorystatus pressure to a Chromium/Brave
                                    // helper. A live 4K renderer dropping caches
                                    // mid-decode = frame drop. Last line of
                                    // defense; do not trust the upstream nominate
                                    // guard (03472d7/a98b33a rule).
                                    || crate::engine::safety::is_chromium_family(&name)
                            })
                            .unwrap_or(false);
                        if !is_protected {
                            // Capture sysctl result so a failed write doesn't
                            // get silently logged as success in the journal.
                            // Observed 2026-04-30: sysctl writes failed under
                            // OOM crisis but `paging_hints_applied` still
                            // incremented, masking the broken signal path.
                            let ok = sysctl_write_i32_with_timeout(
                                "kern.memorystatus_vm_pressure_send",
                                *pid as i32,
                            );
                            if ok {
                                action_applied = true;
                                out.paging_hints_applied += 1;
                            } else {
                                out.push_skip(format!("memorystatus-send-failed:pid={}", *pid));
                                anyhow::bail!("memorystatus pressure send failed for pid={pid}");
                            }
                        }
                    }
                }
                RootAction::ToggleSpotlight { enabled, .. } => {
                    if !dry_run && caps.can_mdutil {
                        let commands = async_commands.ok_or_else(|| {
                            anyhow::anyhow!("asynchronous command queue unavailable")
                        })?;
                        async_submission = Some(
                            commands
                                .submit_spotlight(*enabled, "actuation-broker")
                                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
                        );
                    }
                }
                RootAction::QuarantineDaemon { daemon, active, .. } => {
                    // Guard: never quarantine protected/critical daemons.
                    let dl = daemon.to_ascii_lowercase();
                    let is_protected = protected
                        .iter()
                        .any(|p| dl.contains(&p.to_ascii_lowercase()))
                        || critical_bg
                            .iter()
                            .any(|c| dl.contains(&c.to_ascii_lowercase()));
                    // Validate daemon name: only alphanumeric, dots, hyphens, underscores.
                    let name_valid = !daemon.is_empty()
                        && daemon.len() <= 128
                        && daemon
                            .chars()
                            .all(|c| c.is_alphanumeric() || c == '.' || c == '-' || c == '_');
                    if !dry_run && !is_protected && name_valid {
                        let signal = if *active {
                            libc::SIGSTOP
                        } else {
                            libc::SIGCONT
                        };
                        killall_by_name(daemon, signal)?;
                        action_applied = true;
                    }
                }
                RootAction::SetThreadQoS {
                    pid,
                    name,
                    thread_index,
                    tier,
                    affinity_tag,
                    start_sec,
                    start_usec,
                    ..
                } => {
                    if protected.iter().any(|p| name.contains(p)) {
                        return Ok(());
                    }
                    // Coalition guard: only skip when the requested QoS would
                    // demote (Background / Utility). Boosting (Interactive)
                    // toward an active-coalition helper is desirable.
                    let demotes = !matches!(tier.as_str(), "interactive");
                    if demotes
                        && coalition_guard
                            .map(|g| g.is_protected(*pid))
                            .unwrap_or(false)
                    {
                        block_reason = Some(BlockReason::ActiveCoalition);
                        return Ok(());
                    }
                    // Inv#11 (2026-06-06): real start_sec verify; previously
                    // `0,0` legacy fallback. Adds explicit
                    // BlockReason::PidRecycled (was silent skip before this
                    // sprint — see audit trace consumers in dashboards).
                    if !ProcessIdentity::verify(*pid, Some(name), *start_sec, *start_usec) {
                        crate::engine::lse_counters::LSE_COUNTERS.inc_pid_recycle_block();
                        block_reason = Some(BlockReason::PidRecycled);
                        return Ok(());
                    }
                    let thread_tier = match tier.as_str() {
                        "interactive" => ThreadTier::Interactive,
                        "background" => ThreadTier::Background,
                        _ => ThreadTier::Utility,
                    };
                    if !dry_run {
                        // FIX-4-v2 (2026-06-07): phantom-enrollment guard
                        // chain mirrors the Boost arm. Pre-syscall checks
                        // (caps.can_taskpolicy + qos_mgr.is_some()) are
                        // load-bearing because the Round-2 enrollment was
                        // gated only on the `ok` boolean returned by
                        // apply_raw — but on a system where the qos_mgr is
                        // None we never even entered the inner block, so
                        // no enrollment could happen and the counter would
                        // skew. Explicitly bump the skip counter when the
                        // outer caps/qos_mgr guards fail so the metric
                        // captures BOTH "no manager" and "syscall failed"
                        // pathways.
                        if caps.can_taskpolicy {
                            if let Some(arc) = qos_mgr.as_ref() {
                                // S4 cutover (2026-06-06 cont.): route through
                                // ThreadPolicyEffector::apply_raw so the typed
                                // chokepoint is the SOLE writer of thread QoS state.
                                // Counter `mediator_thread_policy_total` increments
                                // only on syscall success — see effector counter
                                // semantics doc-comment for the attempts-vs-applies
                                // distinction. Identity guard already verified
                                // above (Inv#11 early-return); apply_raw is the
                                // post-verification dispatch path.
                                let effector = crate::engine::mediator::ThreadPolicyEffector::new(
                                    std::sync::Arc::clone(arc),
                                );
                                let expected_generation = arc
                                    .lock()
                                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                                    .thread_qos_generation()
                                    .wrapping_add(1);
                                let (_legacy_ok, _syscall_us, _legacy_applied) = effector
                                    .apply_raw(*pid, *thread_index, thread_tier, *affinity_tag);
                                let qos_outcome = arc
                                    .lock()
                                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                                    .thread_qos_outcome_for(
                                        expected_generation,
                                        *pid,
                                        *thread_index,
                                        thread_tier,
                                    );
                                let qos_diagnostic = qos_outcome
                                    .as_ref()
                                    .map(|outcome| outcome.diagnostics.compact())
                                    .unwrap_or_else(|| "diagnostic=unavailable".to_string());
                                let qos_mutated =
                                    qos_outcome.as_ref().is_some_and(|outcome| outcome.mutated);
                                if qos_mutated {
                                    action_applied = true;
                                    out.thread_qos_applied += 1;
                                    if tier == "background" {
                                        out.thread_qos_cold_routes += 1;
                                    } else {
                                        out.thread_qos_hot_routes += 1;
                                    }
                                    receipt_detail = Some(format!("thread-qos:{qos_diagnostic}"));
                                    // FIX-4-v2 (2026-06-07): enrollment
                                    // strictly post-syscall — `ok=true`
                                    // means ThreadPolicyEffector observed
                                    // a successful set_thread_qos return.
                                    // The four-part guard chain
                                    // (!dry_run + caps.can_taskpolicy +
                                    // qos_mgr.is_some() + ok) is enforced
                                    // by the surrounding `if` ladder; no
                                    // PendingObservation is recorded for
                                    // would-be no-op or failed mutations.
                                    // is_hp routes through
                                    // `safety::is_boost_forbidden` so
                                    // family-root demotions/promotions on
                                    // structurally-misclassified targets
                                    // (Brave/Chromium) feed the HP
                                    // disagreement window. value_post
                                    // encodes the post-syscall ThreadTier
                                    // discriminant
                                    // (Interactive=0/Utility=1/Background=2
                                    // by declaration order, see
                                    // mach_qos.rs:314).
                                    let is_hp = crate::engine::safety::is_boost_forbidden(name);
                                    let tier_post: i64 = match thread_tier {
                                        ThreadTier::Interactive => 0,
                                        ThreadTier::Utility => 1,
                                        ThreadTier::Background => 2,
                                    };
                                    crate::engine::effect_decay::record_global(
                                        crate::engine::effect_decay::PendingObservation {
                                            effect_id: 0,
                                            pid: *pid,
                                            kind:
                                                crate::engine::effect_decay::ObsKind::MachPolicy,
                                            key: None,
                                            value_post: tier_post,
                                            deadline: std::time::Instant::now()
                                                + crate::engine::effect_decay::DecayWatchdog::settle_window(),
                                            hard_protected: is_hp,
                                        },
                                    );
                                } else if qos_outcome.as_ref().is_some_and(|outcome| {
                                    should_use_thread_qos_nice_fallback(thread_tier, outcome)
                                }) {
                                    match set_nice(*pid, THREAD_QOS_NICE_FALLBACK) {
                                        Ok(Some(prior)) => {
                                            crate::engine::effect_ledger::record_global(
                                                crate::engine::effect_ledger::AppliedEffect::Nice {
                                                    pid: *pid,
                                                    prior,
                                                },
                                                THREAD_QOS_FALLBACK_TTL,
                                                *start_sec,
                                                "thread-qos fallback: nice -2",
                                            );
                                            action_applied = true;
                                            receipt_detail = Some(format!(
                                                "thread-qos:{qos_diagnostic},fallback=nice--2"
                                            ));
                                        }
                                        Ok(None) => {
                                            let refreshed =
                                                crate::engine::effect_ledger::refresh_nice_global(
                                                    *pid,
                                                    THREAD_QOS_FALLBACK_TTL,
                                                );
                                            receipt_detail = Some(format!(
                                                "thread-qos:{qos_diagnostic},fallback=nice--2,no-op,owned={refreshed}"
                                            ));
                                        }
                                        Err(error) => {
                                            anyhow::bail!(
                                                "thread QoS failed ({qos_diagnostic}); nice -2 fallback failed: {error}"
                                            );
                                        }
                                    }
                                } else {
                                    // No thread policy mutation was confirmed. Bump
                                    // the phantom-skip counter so the
                                    // delta between "would have enrolled
                                    // pre-fix" and "did not enroll
                                    // post-fix" is visible in
                                    // runtime_metrics.json.
                                    crate::engine::lse_counters::LSE_COUNTERS
                                        .inc_effect_decay_phantom_enroll_skipped();
                                    anyhow::bail!(
                                        "thread QoS failed for pid={pid},thread={thread_index}: {qos_diagnostic}"
                                    );
                                }
                                // affinity_tag fallback handled inside
                                // ThreadPolicyEffector::apply_raw — caller no
                                // longer needs to drive it separately.
                            } else {
                                // qos_mgr unwired — surface the missed
                                // enrollment opportunity.
                                crate::engine::lse_counters::LSE_COUNTERS
                                    .inc_effect_decay_phantom_enroll_skipped();
                            }
                        } else {
                            // task_policy unavailable on this build/host
                            // — same skip semantics as the qos_mgr=None
                            // branch.
                            crate::engine::lse_counters::LSE_COUNTERS
                                .inc_effect_decay_phantom_enroll_skipped();
                        }
                    }
                }
            }
            Ok(())
        })();

        // Journal success preserves dry-run simulation semantics. The audit
        // trace below deliberately uses `action_applied` instead: learning and
        // cross-cycle suppression require a confirmed system mutation.
        let success = result.is_ok() && out.last_skip.is_none() && (dry_run || action_applied);

        let decision_outcome = root_action_outcome(
            action_applied,
            &result,
            block_reason,
            out.last_skip.as_deref(),
        );
        let decision_detail = result
            .as_ref()
            .err()
            .map(ToString::to_string)
            .or_else(|| out.last_skip.clone())
            .or_else(|| block_reason.map(|reason| format!("{reason:?}")))
            .or(receipt_detail)
            .unwrap_or_else(|| reason.clone());
        if let Some(submission) = async_submission {
            out.decision_events.push(submission.pending_event(0));
        } else {
            out.decision_events.push(decision_event_for_root_action(
                &action,
                decision_outcome,
                decision_detail,
            ));
        }

        out.audit_traces.push(PolicyDecisionTrace {
            t: Utc::now(),
            cycle: 0, // Filled by caller
            intended_action: action.clone(),
            decision_reason,
            applied: action_applied,
            block_reason,
            pressure: memory_pressure as f32,
            swap_gb: (crate::engine::host_vm_info::get_swap_used_bytes() as f32
                / (1024.0 * 1024.0 * 1024.0)),
            thrashing: thrashing_score as f32,
        });

        if let Err(e) = result {
            out.failures += 1;
            out.last_error = Some(e.to_string());
        }

        let journal_reason = match out.last_skip.take() {
            Some(s) => format!("skip:{s}"),
            None => reason,
        };

        // 2026-05-14: suppress sysctl-noop entries from journal flood.
        // network_optimizer (main.rs:3726) emits 4 sysctls every 30 cycles
        // without consulting the live kernel value; execute detects noop
        // and would otherwise write a `skip:sysctl-noop:KEY=VAL` line on
        // every cycle. These entries are non-actionable telemetry noise
        // — the journal is for OUTCOMES, not for "we tried but the kernel
        // already had the right value". Drop them at the journal boundary.
        if journal_reason.starts_with("skip:sysctl-noop:") {
            continue;
        }
        // A non-mutating boost is expected for an already-foreground or
        // SIP-protected process. Keep it in PolicyDecisionTrace for runtime
        // observability, but avoid persistent journal I/O for a retryable no-op.
        if journal_reason.starts_with("skip:boost-no-mutation:") {
            continue;
        }

        // Phase 5.3 wiring (2026-05-16): cycle-wide journal chokepoint.
        // Build a structured `Rationale` from the action's own
        // (action_class, decision_reason, reason) tuple. Attach only when
        // the action actually executed — skipped actions already carry
        // their skip reason in `journal_reason` and a structured rationale
        // would be misleading ("we threw a Throttle action with the
        // following Rationale" when the system never threw it).
        //
        // NotebookLM 2026-05-16 monitor target:
        //   `journal_rationales_attached_total / actions_pushed_total >= 0.90`
        // over 1000 cycles. Lower than 0.90 means a non-skip path is
        // bypassing this site — investigate.
        let rationale = if success && !journal_reason.starts_with("skip:") {
            let r = crate::engine::audit_types::Rationale::new(
                action.action_class(),
                format!("{:?}", action.decision_reason()),
                action.reason().to_string(),
            );
            crate::engine::lse_counters::LSE_COUNTERS.inc_journal_rationale_attached();
            Some(r)
        } else {
            None
        };

        pending_journal.push(JournalEntry {
            timestamp: Utc::now(),
            action,
            before,
            after,
            success,
            reason: journal_reason,
            rationale,
        });
    }

    // Flush the entire cycle's journal in a single batched append. Failures
    // here are logged via eprintln! (diagnostic-only) and never affect the
    // outcomes counters — the kernel already owns the authoritative state.
    if !pending_journal.is_empty() {
        match append_journal_batch(journal_path, &pending_journal) {
            Ok(journal_outcome) => {
                out.journal_rotations += journal_outcome.rotated as u64;
                out.journal_rotation_failures += journal_outcome.rotation_failed as u64;
            }
            Err(e) => eprintln!("[execute_actions] batched journal append failed: {e}"),
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::audit_types::DecisionReason;
    use std::cell::Cell;
    use std::collections::HashSet;

    #[test]
    fn freeze_saga_restores_preparation_when_stop_fails() {
        let restore_called = Cell::new(false);
        let report = run_freeze_saga(
            || Ok(true),
            || Err("stop failed".to_string()),
            || {
                restore_called.set(true);
                Ok(())
            },
        );

        assert_eq!(
            report.state(),
            crate::engine::mediator::SagaState::Compensated
        );
        assert!(report.applied_step(0));
        assert!(report.compensated_step(0));
        assert!(restore_called.get());
    }

    #[test]
    fn freeze_saga_commit_never_runs_restore() {
        let report = run_freeze_saga(
            || Ok(true),
            || Ok(()),
            || -> Result<(), String> { panic!("committed freeze must not restore") },
        );

        assert!(report.committed());
        assert!(report.applied_step(0));
        assert!(report.applied_step(1));
    }

    #[test]
    fn freeze_saga_skips_restore_when_preparation_was_a_noop() {
        let report = run_freeze_saga(
            || Ok(false),
            || Err("stop failed".to_string()),
            || -> Result<(), String> { panic!("no-op preparation has nothing to restore") },
        );

        assert_eq!(
            report.state(),
            crate::engine::mediator::SagaState::Compensated
        );
        assert!(!report.applied_step(0));
        assert!(!report.compensated_step(0));
    }

    #[test]
    fn freeze_saga_marks_recovery_when_restore_fails() {
        let report = run_freeze_saga(
            || Ok(true),
            || Err("stop failed".to_string()),
            || Err("restore failed".to_string()),
        );

        assert_eq!(
            report.state(),
            crate::engine::mediator::SagaState::RecoveryRequired
        );
        assert_eq!(
            report
                .compensation_failure()
                .map(|(step, error)| (step, error.as_str())),
            Some((0, "restore failed"))
        );
    }

    fn make_caps() -> CapabilityReport {
        CapabilityReport {
            can_taskpolicy: false,
            can_sysctl: false,
            can_memorystatus: false,
            can_memory_pressure_send: false,
            can_mdutil: false,
            can_tmutil: false,
            is_root: false,
            p_core_count: Some(8),
            e_core_count: Some(4),
            unavailable: vec![],
            memorystatus_probe: None,
            task_for_pid_probe: None,
        }
    }

    /// Helper: run execute_actions with a temp journal and return outcomes.
    fn run(
        actions: Vec<RootAction>,
        learned_protected: &[String],
        learned_interactive: &[String],
    ) -> ExecuteOutcomes {
        let journal = std::env::temp_dir().join("apollo-test-execute-actions.jsonl");
        let mut frozen = HashSet::new();
        execute_actions(
            actions,
            &make_caps(),
            &journal,
            &mut frozen,
            learned_protected,
            learned_interactive,
            None,
            None,
            false,
            0.0,
            0.0,
            None,
            0.0,
        )
    }

    /// A PID unlikely to exist so SIGSTOP/setpriority don't land on a real process.
    /// Using PID 9_999_999 (exceeds typical macOS max PID of ~99_999).
    const GHOST_PID: u32 = 9_999_999;

    #[test]
    fn set_nice_reports_noop_at_existing_priority() {
        let pid = std::process::id();
        let current = unsafe {
            *libc::__error() = 0;
            libc::getpriority(libc::PRIO_PROCESS, pid)
        };
        assert_eq!(
            set_nice(pid, current).expect("same-priority set must be readable"),
            None
        );
    }

    #[test]
    fn boost_effects_are_short_lived() {
        assert_eq!(BOOST_EFFECT_TTL, std::time::Duration::from_secs(12));
    }

    #[test]
    fn boost_records_explicit_task_qos_for_ttl_rollback() {
        const PID: u32 = 8_901_113;
        const START_SEC: u64 = 123;
        let effect = crate::engine::effect_ledger::AppliedEffect::TaskQoS { pid: PID };

        record_boost_qos_effects(PID, START_SEC, false, true);

        assert!(crate::engine::effect_ledger::is_global_owner(
            &effect,
            "boost: interactive task QoS"
        ));
        crate::engine::effect_ledger::forget_global(&effect);
    }

    #[test]
    fn unsupported_memory_pressure_channel_is_not_counted_as_applied() {
        let pid = std::process::id();
        let outcomes = run(
            vec![RootAction::SetMemorystatus {
                pid,
                priority: -1,
                reason: "test unsupported kernel channel".to_string(),
                decision_reason: DecisionReason::PressureContext,
            }],
            &[],
            &[],
        );
        assert_eq!(outcomes.paging_hints_applied, 0);
        assert!(
            outcomes
                .top_skipped
                .iter()
                .any(|reason| reason.starts_with("memorystatus-send-unsupported:")),
            "unsupported channel must be visible as a blocked action"
        );
        assert_eq!(outcomes.decision_events.as_slice().len(), 1);
        assert_eq!(
            outcomes.decision_events.as_slice()[0].outcome,
            crate::engine::decision_ledger::ActuatorDecisionOutcome::Blocked
        );
    }

    #[test]
    fn unavailable_sysctl_capability_has_an_explicit_skip_reason() {
        let outcomes = run(
            vec![RootAction::set_sysctl(
                "vm.compressor_eval_period_in_msecs",
                "250",
                "test capability gate",
                DecisionReason::PressureContext,
            )],
            &[],
            &[],
        );

        assert_eq!(outcomes.sysctl_applied, 0);
        assert!(outcomes.top_skipped.iter().any(|reason| {
            reason == "sysctl-capability-unavailable:vm.compressor_eval_period_in_msecs"
        }));
        assert!(matches!(
            outcomes.audit_traces.as_slice(),
            [PolicyDecisionTrace {
                applied: false,
                block_reason: Some(BlockReason::SysctlFailed),
                ..
            }]
        ));
        assert_eq!(
            outcomes.decision_events.as_slice()[0].outcome,
            crate::engine::decision_ledger::ActuatorDecisionOutcome::Blocked
        );
    }

    #[test]
    fn numeric_sysctl_postcondition_requires_exact_value() {
        let i32_value = sysctl_direct::NumericSysctlValue {
            value: 100,
            width: sysctl_direct::NumericSysctlWidth::I32,
        };
        let i64_value = sysctl_direct::NumericSysctlValue {
            value: 100,
            width: sysctl_direct::NumericSysctlWidth::I64,
        };
        assert!(sysctl_postcondition_matches(100, Some(i32_value)));
        assert!(sysctl_postcondition_matches(100, Some(i64_value)));
        assert!(!sysctl_postcondition_matches(
            100,
            Some(sysctl_direct::NumericSysctlValue {
                value: 101,
                width: sysctl_direct::NumericSysctlWidth::I32,
            })
        ));
        assert!(!sysctl_postcondition_matches(100, None));
    }

    #[test]
    fn numeric_sysctl_transaction_does_not_overwrite_a_mismatching_live_value() {
        use std::cell::Cell;

        let live = Cell::new(100_i64);
        let width = sysctl_direct::NumericSysctlWidth::I64;
        let before = sysctl_direct::NumericSysctlValue { value: 100, width };
        let outcome = run_numeric_sysctl_transaction_with(
            before,
            200,
            || {
                Some(sysctl_direct::NumericSysctlValue {
                    value: live.get(),
                    width,
                })
            },
            |value, _| {
                live.set(if value == 200 { 201 } else { value });
                true
            },
        );

        assert_eq!(
            outcome,
            NumericSysctlTransaction::Uncertain(Some(sysctl_direct::NumericSysctlValue {
                value: 201,
                width,
            }))
        );
        assert_eq!(live.get(), 201);
    }

    #[test]
    fn numeric_sysctl_transaction_rejects_a_changed_precondition() {
        let width = sysctl_direct::NumericSysctlWidth::I32;
        let before = sysctl_direct::NumericSysctlValue { value: 100, width };
        let writes = std::cell::Cell::new(0_u32);
        let outcome = run_numeric_sysctl_transaction_with(
            before,
            200,
            || Some(sysctl_direct::NumericSysctlValue { value: 101, width }),
            |_, _| {
                writes.set(writes.get() + 1);
                true
            },
        );

        assert_eq!(outcome, NumericSysctlTransaction::PreconditionChanged);
        assert_eq!(writes.get(), 0);
    }

    #[test]
    fn abandoned_numeric_sysctl_receipt_restores_a_late_applied_write() {
        let width = sysctl_direct::NumericSysctlWidth::I64;
        let before = sysctl_direct::NumericSysctlValue { value: 100, width };
        let live = std::cell::Cell::new(200_i64);
        let (reply, abandoned) = std::sync::mpsc::channel();
        drop(abandoned);

        deliver_numeric_sysctl_transaction_with(
            reply,
            NumericSysctlTransaction::Applied(sysctl_direct::NumericSysctlValue {
                value: 200,
                width,
            }),
            before,
            || {
                Some(sysctl_direct::NumericSysctlValue {
                    value: live.get(),
                    width,
                })
            },
            |value, _| {
                live.set(value);
                true
            },
        );

        assert_eq!(live.get(), before.value);
    }

    #[test]
    fn abandoned_numeric_sysctl_receipt_never_overwrites_a_new_owner() {
        let width = sysctl_direct::NumericSysctlWidth::I64;
        let before = sysctl_direct::NumericSysctlValue { value: 100, width };
        let live = std::cell::Cell::new(300_i64);
        let writes = std::cell::Cell::new(0_u32);
        let (reply, abandoned) = std::sync::mpsc::channel();
        drop(abandoned);

        deliver_numeric_sysctl_transaction_with(
            reply,
            NumericSysctlTransaction::Applied(sysctl_direct::NumericSysctlValue {
                value: 200,
                width,
            }),
            before,
            || {
                Some(sysctl_direct::NumericSysctlValue {
                    value: live.get(),
                    width,
                })
            },
            |value, _| {
                writes.set(writes.get() + 1);
                live.set(value);
                true
            },
        );

        assert_eq!(live.get(), 300);
        assert_eq!(writes.get(), 0);
    }

    #[test]
    fn binary_i32_value_is_reported_as_decimal_not_ascii() {
        let raw = 100_i32.to_ne_bytes();
        let decoded = i32::from_ne_bytes(raw);

        assert_eq!(decoded.to_string(), "100");
        assert_ne!(decoded.to_string(), "d");
    }

    #[test]
    fn memorystatus_syscall_failure_is_an_exact_failed_receipt() {
        let mut caps = make_caps();
        caps.can_memory_pressure_send = true;
        let journal = std::env::temp_dir().join("apollo-test-memorystatus-failure.jsonl");
        let mut frozen = HashSet::new();

        let outcomes = execute_actions(
            vec![RootAction::SetMemorystatus {
                pid: GHOST_PID,
                priority: -1,
                reason: "force missing-pid write".to_string(),
                decision_reason: DecisionReason::PressureContext,
            }],
            &caps,
            &journal,
            &mut frozen,
            &[],
            &[],
            None,
            None,
            false,
            0.8,
            0.0,
            None,
            0.0,
        );

        assert_eq!(outcomes.failures, 1);
        assert_eq!(
            outcomes.decision_events.as_slice()[0].outcome,
            ActuatorDecisionOutcome::Failed
        );
    }

    #[test]
    fn quarantine_effector_failure_is_an_exact_failed_receipt() {
        let outcomes = run(
            vec![RootAction::QuarantineDaemon {
                daemon: "apollo-daemon-that-does-not-exist".to_string(),
                active: true,
                reason: "force missing daemon".to_string(),
                decision_reason: DecisionReason::PressureContext,
            }],
            &[],
            &[],
        );

        assert_eq!(outcomes.failures, 1);
        assert_eq!(
            outcomes.decision_events.as_slice()[0].outcome,
            ActuatorDecisionOutcome::Failed
        );
    }

    #[test]
    fn thread_qos_syscall_failure_is_an_exact_failed_receipt() {
        let pid = std::process::id();
        let identity = ProcessIdentity::from_pid(pid).expect("current process identity");
        let name = process_identity::proc_name_for_pid(pid).expect("current process name");
        let qos = std::sync::Arc::new(std::sync::Mutex::new(MachQoSManager::new()));
        let mut caps = make_caps();
        caps.can_taskpolicy = true;
        let journal = std::env::temp_dir().join("apollo-test-thread-qos-failure.jsonl");
        let mut frozen = HashSet::new();

        let outcomes = execute_actions(
            vec![RootAction::SetThreadQoS {
                pid,
                name,
                thread_index: u32::MAX,
                tier: "utility".to_string(),
                affinity_tag: None,
                reason: "force invalid thread index".to_string(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: identity.start_sec,
                start_usec: identity.start_usec,
            }],
            &caps,
            &journal,
            &mut frozen,
            &[],
            &[],
            Some(&qos),
            None,
            false,
            0.0,
            0.0,
            None,
            0.0,
        );

        assert_eq!(outcomes.failures, 1);
        assert_eq!(
            outcomes.decision_events.as_slice()[0].outcome,
            ActuatorDecisionOutcome::Failed
        );
        assert!(
            outcomes
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("task_for_pid=")
                    && error.contains("task_threads=")
                    && error.contains("thread_policy_set")),
            "thread QoS failure must retain stage diagnostics: {:?}",
            outcomes.last_error
        );
    }

    #[test]
    fn nice_fallback_is_limited_to_restricted_interactive_thread_qos() {
        let restricted = crate::engine::mach_qos::ThreadQoSOutcome::from_diagnostics(
            crate::engine::mach_qos::MachQoSDiagnostics {
                task_for_pid: Some(crate::engine::mach_qos::mach_sys::KERN_FAILURE),
                terminal: crate::engine::mach_qos::MachQoSTerminal::TaskAccessFailed,
                ..Default::default()
            },
        );
        let invalid_index = crate::engine::mach_qos::ThreadQoSOutcome::from_diagnostics(
            crate::engine::mach_qos::MachQoSDiagnostics {
                task_for_pid: Some(crate::engine::mach_qos::mach_sys::KERN_SUCCESS),
                thread_enumeration: Some(crate::engine::mach_qos::mach_sys::KERN_SUCCESS),
                terminal: crate::engine::mach_qos::MachQoSTerminal::ThreadIndexOutOfRange,
                ..Default::default()
            },
        );
        let policy_restricted = crate::engine::mach_qos::ThreadQoSOutcome::from_diagnostics(
            crate::engine::mach_qos::MachQoSDiagnostics {
                task_for_pid: Some(crate::engine::mach_qos::mach_sys::KERN_SUCCESS),
                thread_enumeration: Some(crate::engine::mach_qos::mach_sys::KERN_SUCCESS),
                thread_latency_policy_set: Some(
                    crate::engine::mach_qos::mach_sys::KERN_PROTECTION_FAILURE,
                ),
                thread_throughput_policy_set: Some(
                    crate::engine::mach_qos::mach_sys::KERN_PROTECTION_FAILURE,
                ),
                terminal: crate::engine::mach_qos::MachQoSTerminal::PolicySetCompleted,
            },
        );

        assert!(should_use_thread_qos_nice_fallback(
            ThreadTier::Interactive,
            &restricted
        ));
        assert!(!should_use_thread_qos_nice_fallback(
            ThreadTier::Background,
            &restricted
        ));
        assert!(!should_use_thread_qos_nice_fallback(
            ThreadTier::Interactive,
            &invalid_index
        ));
        assert!(!should_use_thread_qos_nice_fallback(
            ThreadTier::Interactive,
            &policy_restricted
        ));
        let cached_block = crate::engine::mach_qos::ThreadQoSOutcome::from_diagnostics(
            crate::engine::mach_qos::MachQoSDiagnostics {
                terminal: crate::engine::mach_qos::MachQoSTerminal::PermanentlyBlocked,
                ..Default::default()
            },
        );
        assert!(!should_use_thread_qos_nice_fallback(
            ThreadTier::Interactive,
            &cached_block
        ));
    }

    #[test]
    fn thread_qos_nice_fallback_stays_conservative_and_short_lived() {
        assert_eq!(THREAD_QOS_NICE_FALLBACK, -2);
        assert_eq!(THREAD_QOS_FALLBACK_TTL, std::time::Duration::from_secs(12));
    }

    #[test]
    fn spotlight_queueing_is_pending_and_never_locally_applied() {
        let commands = crate::engine::daemon_helpers::AsyncCommandQueue::new();
        let mut caps = make_caps();
        caps.can_mdutil = true;
        let journal = std::env::temp_dir().join("apollo-test-spotlight-pending.jsonl");
        let mut frozen = HashSet::new();

        let outcomes = execute_actions(
            vec![RootAction::ToggleSpotlight {
                enabled: false,
                reason: "test asynchronous completion".to_string(),
                decision_reason: DecisionReason::PressureContext,
            }],
            &caps,
            &journal,
            &mut frozen,
            &[],
            &[],
            None,
            Some(&commands),
            false,
            0.0,
            0.0,
            None,
            0.0,
        );

        assert_eq!(outcomes.decision_events.as_slice().len(), 1);
        assert_eq!(
            outcomes.decision_events.as_slice()[0].outcome,
            ActuatorDecisionOutcome::Pending
        );
        assert!(outcomes.decision_events.as_slice()[0]
            .correlation_id
            .is_some());
        assert_eq!(outcomes.failures, 0);
    }

    #[test]
    fn boost_without_mutation_stays_auditable_without_journal_flood() {
        let journal = std::env::temp_dir().join("apollo-test-boost-no-mutation.jsonl");
        let _ = std::fs::remove_file(&journal);
        let pid = std::process::id();
        let mut frozen = HashSet::new();

        let outcomes = execute_actions(
            vec![RootAction::BoostProcess {
                pid,
                name: "apollo-optimizer-test".to_string(),
                reason: "self-protection no-op".to_string(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: 0,
                start_usec: 0,
            }],
            &make_caps(),
            &journal,
            &mut frozen,
            &[],
            &[],
            None,
            None,
            false,
            0.0,
            0.0,
            None,
            0.0,
        );

        assert!(
            matches!(
                outcomes.audit_traces.as_slice(),
                [PolicyDecisionTrace {
                    applied: false,
                    block_reason: Some(BlockReason::NoMutation),
                    ..
                }]
            ),
            "unexpected traces: {:#?}",
            outcomes.audit_traces
        );
        assert_eq!(
            outcomes.decision_events.as_slice()[0].outcome,
            crate::engine::decision_ledger::ActuatorDecisionOutcome::NoOp
        );
        let entries = crate::engine::journal::read_journal(&journal).expect("read journal");
        assert!(
            entries.is_empty(),
            "expected no-op boost to skip journal I/O"
        );
    }

    #[test]
    fn batched_unfreeze_removes_dead_pids_from_frozen_set() {
        // Regression test for the fast-path unfreeze pre-pass: even with the
        // pre-pass sending SIGCONT first, the main loop must still run and
        // the frozen-set bookkeeping must still be correct for dead pids.
        // Dead pids should be removed from the frozen set; counters must match.
        let journal = std::env::temp_dir().join("apollo-test-batched-unfreeze.jsonl");
        let mut frozen: HashSet<u32> = (GHOST_PID..GHOST_PID + 5).collect();
        let actions: Vec<RootAction> = (GHOST_PID..GHOST_PID + 5)
            .map(|pid| RootAction::UnfreezeProcess {
                pid,
                name: format!("ghost-{pid}"),
                reason: "test".to_string(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: 0,
                start_usec: 0,
            })
            .collect();
        let outcomes = execute_actions(
            actions,
            &make_caps(),
            &journal,
            &mut frozen,
            &[],
            &[],
            None,
            None,
            false,
            0.0,
            0.0,
            None,
            0.0,
        );
        // All 5 ghost pids are dead → should be removed from frozen set.
        // unfreezes_applied stays 0 because the live-branch (which increments
        // the counter) never runs for dead pids — but the frozen set MUST be
        // cleaned up so the daemon doesn't get stuck thinking they're still
        // frozen forever.
        assert!(
            frozen.is_empty(),
            "dead pids must be removed from frozen set, still holds: {frozen:?}"
        );
        assert_eq!(outcomes.failures, 0);
    }

    #[test]
    fn unfreeze_with_recycled_identity_never_counts_as_applied() {
        let pid = std::process::id();
        let identity = ProcessIdentity::from_pid(pid).expect("test process identity");
        let name = process_identity::proc_name_for_pid(pid).expect("test process name");
        let journal = std::env::temp_dir().join("apollo-test-unfreeze-pid-recycle.jsonl");
        let _ = std::fs::remove_file(&journal);
        let mut frozen = HashSet::from([pid]);
        let wrong_start = identity.start_sec.saturating_add(1).max(1);

        let outcomes = execute_actions(
            vec![RootAction::unfreeze_full(
                pid,
                name,
                "recycled identity regression",
                wrong_start,
                identity.start_usec,
                DecisionReason::CriticalBypass,
            )],
            &make_caps(),
            &journal,
            &mut frozen,
            &[],
            &[],
            None,
            None,
            false,
            0.0,
            0.0,
            None,
            0.0,
        );

        assert!(frozen.is_empty(), "stale frozen ownership must be removed");
        assert_eq!(outcomes.unfreezes_applied, 0);
        assert_eq!(outcomes.failures, 0);
        assert!(matches!(
            outcomes.audit_traces.as_slice(),
            [PolicyDecisionTrace {
                applied: false,
                block_reason: Some(BlockReason::PidRecycled),
                ..
            }]
        ));
        let entries = crate::engine::journal::read_journal(&journal).expect("read journal");
        let entry = entries.last().expect("stale unfreeze journal entry");
        assert!(
            !entry.success,
            "blocked thaw must not be journaled as success"
        );
        assert!(entry.rationale.is_none());
    }

    /// Phase 5.3 wiring proof (TODO closeout 2026-06-11): a successful action
    /// carried through `execute_actions` must end up journaled WITH a
    /// structured `Rationale` attached at the cycle-wide chokepoint. We use
    /// `dry_run=true` so a ghost PID counts as a clean success without
    /// touching any real process, then read the journal back and assert the
    /// rationale is present and reflects the action's own metadata.
    #[test]
    fn successful_action_journals_a_rationale() {
        use crate::engine::journal::read_journal;

        let journal = std::env::temp_dir().join("apollo-test-rationale-wiring.jsonl");
        // Start from a clean slate so we only observe this cycle's entries.
        let _ = std::fs::remove_file(&journal);

        let attached_before = crate::engine::lse_counters::LSE_COUNTERS
            .journal_rationales_attached_total
            .load(std::sync::atomic::Ordering::Relaxed);

        let mut frozen: HashSet<u32> = [GHOST_PID].into_iter().collect();
        let outcomes = execute_actions(
            vec![RootAction::UnfreezeProcess {
                pid: GHOST_PID,
                name: "ghost-rationale".to_string(),
                reason: "pressure=0.81,swap_gb=2.1".to_string(),
                decision_reason: DecisionReason::CriticalBypass,
                start_sec: 0,
                start_usec: 0,
            }],
            &make_caps(),
            &journal,
            &mut frozen,
            &[],
            &[],
            None,
            None,
            true, // dry_run → ghost PID succeeds without a real process
            0.0,
            0.0,
            None,
            0.0,
        );
        assert_eq!(outcomes.failures, 0, "dry-run unfreeze must not fail");
        assert!(
            outcomes.audit_traces.iter().all(|trace| !trace.applied),
            "dry-run journal success must not masquerade as a real mutation"
        );

        let entries = read_journal(&journal).expect("read back journal");
        let entry = entries
            .iter()
            .find(|e| matches!(e.action, RootAction::UnfreezeProcess { .. }))
            .expect("the unfreeze action must be journaled");

        assert!(
            entry.success,
            "dry-run unfreeze must be recorded as success"
        );
        let rationale = entry
            .rationale
            .as_ref()
            .expect("successful action must carry a Rationale (Phase 5.3 wiring)");

        // The rationale's fields are built from the action's own metadata at
        // the chokepoint — verify the wiring threads them through faithfully.
        assert_eq!(
            rationale.action_class, "unfreeze",
            "action_class must match RootAction::action_class()"
        );
        assert!(
            rationale.trigger.contains("CriticalBypass"),
            "trigger must reflect the action's DecisionReason, got: {}",
            rationale.trigger
        );
        assert_eq!(
            rationale.evidence, "pressure=0.81,swap_gb=2.1",
            "evidence must be the action's reason payload"
        );

        // The LSE counter must have advanced for the attachment so telemetry
        // stays observable in runtime_metrics.json (silent-telemetry-death guard).
        let attached_after = crate::engine::lse_counters::LSE_COUNTERS
            .journal_rationales_attached_total
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            attached_after > attached_before,
            "journal_rationale_attached_total must increment on a rationale'd entry"
        );

        let _ = std::fs::remove_file(&journal);
    }

    // ── learned_interactive skips (BUG-07) ────────────────────────────────────

    #[test]
    fn throttle_skips_learned_interactive_process() {
        let interactive = vec!["MyInteractiveApp".to_string()];
        let outcomes = run(
            vec![RootAction::ThrottleProcess {
                pid: GHOST_PID,
                name: "MyInteractiveApp".to_string(),
                aggressive: false,
                reason: "test".to_string(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: 0,
                start_usec: 0,
            }],
            &[],
            &interactive,
        );
        assert_eq!(
            outcomes.throttles_applied, 0,
            "learned_interactive process must not be throttled"
        );
        assert!(
            outcomes
                .top_skipped
                .iter()
                .any(|s| s.contains("MyInteractiveApp")),
            "skip reason must mention the process name"
        );
    }

    #[test]
    fn freeze_skips_learned_interactive_process() {
        let interactive = vec!["MyInteractiveApp".to_string()];
        let outcomes = run(
            vec![RootAction::FreezeProcess {
                pid: GHOST_PID,
                name: "MyInteractiveApp".to_string(),
                reason: "test".to_string(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: 0,
                start_usec: 0,
            }],
            &[],
            &interactive,
        );
        assert_eq!(
            outcomes.freezes_applied, 0,
            "learned_interactive process must not be frozen"
        );
        assert!(
            outcomes
                .top_skipped
                .iter()
                .any(|s| s.contains("MyInteractiveApp")),
            "skip reason must mention the process name"
        );
    }

    #[test]
    fn throttle_skips_learned_interactive_case_insensitive() {
        // Pattern stored lowercase; process name has mixed case — must still skip.
        let interactive = vec!["myinteractiveapp".to_string()];
        let outcomes = run(
            vec![RootAction::ThrottleProcess {
                pid: GHOST_PID,
                name: "MyInteractiveApp".to_string(),
                aggressive: false,
                reason: "test".to_string(),
                start_sec: 0,
                start_usec: 0,
                decision_reason: DecisionReason::PressureContext,
            }],
            &[],
            &interactive,
        );
        assert_eq!(outcomes.throttles_applied, 0);
    }

    #[test]
    fn throttle_skips_learned_protected_process() {
        let protected = vec!["MyProtectedDaemon".to_string()];
        let outcomes = run(
            vec![RootAction::ThrottleProcess {
                pid: GHOST_PID,
                name: "MyProtectedDaemon".to_string(),
                aggressive: false,
                reason: "test".to_string(),
                start_sec: 0,
                start_usec: 0,
                decision_reason: DecisionReason::PressureContext,
            }],
            &protected,
            &[],
        );
        assert_eq!(outcomes.throttles_applied, 0);
    }
}
