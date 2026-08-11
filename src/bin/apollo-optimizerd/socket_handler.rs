//! Socket Handler — Unix domain socket server + request dispatch.
//!
//! Extracted from the daemon monolith. Contains:
//! - `run_socket_server()` — bind, listen, spawn per-client threads
//! - `handle_client()` — read request, auth, dispatch
//! - `process_request()` — the 22-arm command dispatcher
//! - `broadcast_current_status()` — push updates to subscribers
//! - `is_peer_root()` — peer credential check

use std::ffi::CString;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;

use anyhow::Context;
use chrono::Utc;

use apollo_engine::engine::capabilities::{
    detect_capabilities, detect_capabilities_with_write_probes,
};
use apollo_engine::engine::daemon_helpers::{
    frozen_state_path, kill_switch_path, metrics_path, socket_path, unfreeze_pids_verified_outcome,
    write_frozen_state,
};
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::policy_store::{append_jsonl, FeedbackEntry};
use apollo_engine::engine::protocol::{DaemonRequest, DaemonResponse};
use apollo_engine::engine::types::{
    DaemonStatus, FrozenProcessInfo, HardPath, HealthReport, RuntimeMetrics, UsageResponse,
};

use super::{SharedState, STOP_REQUESTED};

// ── Peer Authentication ────────────────────────────────────────────────────

pub fn is_peer_root(stream: &UnixStream) -> bool {
    // If we're not running as root, anyone who can connect is allowed (usually protected by dir perms)
    if unsafe { libc::geteuid() } != 0 {
        return true;
    }

    #[cfg(target_os = "macos")]
    {
        let mut euid: libc::uid_t = 0;
        let mut egid: libc::gid_t = 0;
        let res = unsafe { libc::getpeereid(stream.as_raw_fd(), &mut euid, &mut egid) };
        if res == 0 {
            return euid == 0;
        }
    }
    // Default to false for security if we can't verify
    false
}

// ── Client Handler ─────────────────────────────────────────────────────────

pub fn handle_client(mut stream: UnixStream, state: &SharedState) {
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(5)));
    let is_root = is_peer_root(&stream);

    // Lee y parsea la peticion (reader se libera al salir del bloque)
    let req_result = {
        let mut reader = BufReader::new(&stream);
        const MAX_REQUEST_BYTES: u64 = 65_536;
        let mut line = String::new();
        match reader.by_ref().take(MAX_REQUEST_BYTES).read_line(&mut line) {
            Ok(_) => serde_json::from_str::<DaemonRequest>(&line)
                .map_err(|e| format!("invalid request: {e}")),
            Err(e) => Err(format!("read error: {e}")),
        }
    };

    let mut req = match req_result {
        Ok(r) => r,
        Err(msg) => {
            if let Ok(text) = serde_json::to_string(&DaemonResponse::Error { message: msg }) {
                let _ = writeln!(stream, "{}", text);
            }
            return;
        }
    };
    req.sanitize();

    // Suscripcion push: conexion persistente, el daemon enviara StatusPush cada ciclo
    if let DaemonRequest::Subscribe = req {
        if let Ok(text) = serde_json::to_string(&DaemonResponse::Ok) {
            let _ = writeln!(stream, "{}", text);
        }
        if let Ok(write_clone) = stream.try_clone() {
            state.subscribers.lock_recover().push(write_clone);
        }
        // Bloquear hasta que el cliente desconecte; la limpieza es lazy (fallo de escritura)
        let _ = stream.set_read_timeout(None);
        let mut buf = [0u8; 1];
        loop {
            match Read::read(&mut stream, &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
        return;
    }

    if req.is_privileged() && !is_root {
        if let Ok(text) = serde_json::to_string(&DaemonResponse::Error {
            message: "privileged command requires root/sudo".to_string(),
        }) {
            let _ = writeln!(stream, "{}", text);
        }
        return;
    }

    let response = process_request(req, state);
    if let Ok(text) = serde_json::to_string(&response) {
        let _ = writeln!(stream, "{}", text);
    }
}

// ── Broadcast ──────────────────────────────────────────────────────────────

/// Broadcast del estado actual a todos los suscriptores.
/// Los streams que fallen (cliente desconectado) se eliminan automaticamente.
pub fn broadcast_current_status(state: &SharedState) {
    let mut subs = state.subscribers.lock_recover();
    if subs.is_empty() {
        return;
    }
    let DaemonResponse::Status(status) = process_request(DaemonRequest::GetStatus, state) else {
        return;
    };
    let Ok(text) = serde_json::to_string(&DaemonResponse::StatusPush(status)) else {
        return;
    };
    subs.retain_mut(|stream| writeln!(stream, "{}", text).is_ok());
}

// ── Request Dispatcher ─────────────────────────────────────────────────────

fn restore_frozen_processes(state: &SharedState) -> Result<u64, String> {
    let mut frozen_state = state.frozen_state.lock_recover();
    let outcome = unfreeze_pids_verified_outcome(&frozen_state);
    for pid in outcome.forgettable_pids() {
        frozen_state.remove(&pid);
    }
    write_frozen_state(Path::new(frozen_state_path()), &frozen_state);
    let applied = outcome.applied_count();
    let failed = outcome.failed_pids.clone();
    drop(frozen_state);

    state.metrics.lock_recover().metrics.unfreezes_applied += applied;
    if failed.is_empty() {
        Ok(applied)
    } else {
        Err(format!("SIGCONT failed for PIDs: {failed:?}"))
    }
}

pub fn process_request(req: DaemonRequest, state: &SharedState) -> DaemonResponse {
    match req {
        DaemonRequest::GetStatus => {
            let now = Utc::now();
            let profile = state.policy.lock_recover().profile;
            let latency_target = state.policy.lock_recover().latency_target;
            // Non-blocking metrics: try_lock avoids stalling when the main loop
            // holds the metrics lock during its end-of-cycle update (~100 lines).
            // Fall back to default metrics if busy — dashboard shows stale data
            // briefly, but never hangs.
            let metrics = match state.metrics.try_lock() {
                Ok(m) => m.metrics.clone(),
                Err(_) => {
                    // Lock held by main loop — read last-written snapshot from disk.
                    // This is always ≤1 cycle old (written at end of each cycle).
                    match std::fs::read_to_string(metrics_path()) {
                        Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
                        Err(_) => RuntimeMetrics::default(),
                    }
                }
            };
            let blockers = state.process.lock_recover().last_blockers.clone();
            let thermal_state = state.metrics.lock_recover().thermal_state.clone();
            let throttle_level = state.metrics.lock_recover().throttle_level.clone();
            // Snapshot governor + wake_state, then drop locks before I/O.
            let (
                auto_profile_enabled,
                base_profile,
                override_active,
                override_expires_at,
                transition_reason,
            ) = {
                let pg = state.policy.lock_recover();
                (
                    pg.governor.auto_profile_enabled,
                    pg.governor.base_profile,
                    pg.governor.manual_override.is_some(),
                    pg.governor.manual_override.as_ref().map(|o| o.expires_at),
                    pg.governor.transition_reason.clone(),
                )
            };
            let (grace_active, grace_remaining, last_wake_at, post_wake_policy) = {
                let proc = state.process.lock_recover();
                let ws = &proc.wake_state;
                let ga = ws.post_wake_grace_until.map(|t| t > now).unwrap_or(false);
                let gr = ws
                    .post_wake_grace_until
                    .and_then(|t| (t - now).to_std().ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                (ga, gr, ws.last_wake_at, ws.post_wake_policy.clone())
            };
            let (reactor_mode, reactor_health) = {
                let m = state.metrics.lock_recover();
                (
                    m.reactor_status.mode.clone(),
                    m.reactor_status.health.clone(),
                )
            };
            let frozen_processes: Vec<FrozenProcessInfo> = {
                let fs = state.frozen_state.lock_recover();
                fs.iter()
                    .map(|(&pid, entry)| FrozenProcessInfo {
                        pid,
                        name: entry
                            .process_name
                            .clone()
                            .unwrap_or_else(|| pid.to_string()),
                        frozen_seconds: now
                            .signed_duration_since(entry.frozen_at)
                            .num_seconds()
                            .max(0) as u64,
                        source: entry.source,
                        pressure_at_freeze: entry.pressure_at_freeze,
                    })
                    .collect()
            };
            let status = DaemonStatus {
                running: !state.stop.load(Ordering::Acquire),
                profile,
                latency_target,
                effective_profile: metrics.effective_profile,
                kill_switch: Path::new(kill_switch_path()).exists(),
                throttle_level,
                thermal_state,
                last_blockers: blockers,
                auto_profile_enabled,
                base_profile,
                override_active,
                override_expires_at,
                transition_reason,
                post_wake_grace_active: grace_active,
                post_wake_grace_remaining_secs: grace_remaining,
                last_wake_at,
                post_wake_policy,
                reactor_mode,
                reactor_health,
                metrics,
                frozen_processes,
            };
            DaemonResponse::Status(status)
        }
        DaemonRequest::GetMetrics => {
            DaemonResponse::Metrics(state.metrics.lock_recover().metrics.clone())
        }
        DaemonRequest::GetTopBlockers => {
            DaemonResponse::TopBlockers(state.process.lock_recover().last_blockers.clone())
        }
        DaemonRequest::GetProfileTimeline => DaemonResponse::ProfileTimeline(
            state
                .policy
                .lock_recover()
                .timeline
                .iter()
                .cloned()
                .collect(),
        ),
        DaemonRequest::GetCapabilities => DaemonResponse::Capabilities(detect_capabilities()),
        DaemonRequest::SetProfile {
            profile,
            ttl_minutes,
        } => {
            let ttl = ttl_minutes.unwrap_or(20).clamp(1, 1440);
            state.policy.lock_recover().governor.set_manual_override(
                profile,
                ttl,
                "cli-set-profile".to_string(),
            );
            DaemonResponse::Ok
        }
        DaemonRequest::SetLatencyTarget { target } => {
            state.policy.lock_recover().latency_target = target;
            DaemonResponse::Ok
        }
        DaemonRequest::SetAutoProfile { enabled } => {
            state
                .policy
                .lock_recover()
                .governor
                .set_auto_profile(enabled);
            DaemonResponse::Ok
        }
        DaemonRequest::ClearProfileOverride => {
            state.policy.lock_recover().governor.clear_manual_override();
            DaemonResponse::Ok
        }
        DaemonRequest::Restore => {
            // F3 — A-B-A defense: verify PID identity (name + start_sec) before
            // SIGCONT. If a frozen PID died and was recycled before this CLI-
            // triggered restore arrived, the new occupant must NOT receive our
            // SIGCONT. `unfreeze_pids_verified` re-reads each current PID's
            // kernel start-time and skips mismatches (or gone PIDs) silently.
            // [Gray & Reuter 1992 §11] crash recovery identity invariants.
            // NOTE: kill switch (/var/run/apollo.disable) is intentionally NOT
            // cleared here. Restore reverts Apollo's mutations (frozen PIDs,
            // sysctls) but does not override a manual operator pause.
            // PanicRestore is the correct path to toggle the kill switch.
            match restore_frozen_processes(state) {
                Ok(_) => DaemonResponse::Ok,
                Err(message) => DaemonResponse::Error { message },
            }
        }
        DaemonRequest::PanicRestore => {
            // Symlink protection: open with O_NOFOLLOW so the check and create
            // are atomic — no TOCTOU window for a symlink to be swapped in.
            let ks = kill_switch_path();
            let result = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .custom_flags(libc::O_NOFOLLOW)
                .open(ks);
            if let Err(e) = result {
                return DaemonResponse::Error {
                    message: format!("kill switch create failed (symlink?): {e}"),
                };
            }
            state.policy.lock_recover().governor.set_auto_profile(false);
            // F3 — A-B-A defense (PanicRestore path): same identity check as
            // Restore. PanicRestore is usually invoked during emergencies where
            // the system may have been under severe pressure — higher odds of
            // frozen PIDs having died + been recycled. `unfreeze_pids_verified`
            // skips any PID whose (name, start_sec) no longer matches.
            // [Gray & Reuter 1992 §11] crash recovery identity invariants.
            match restore_frozen_processes(state) {
                Ok(_) => DaemonResponse::Ok,
                Err(message) => DaemonResponse::Error { message },
            }
        }
        DaemonRequest::Doctor => {
            let caps = detect_capabilities_with_write_probes();
            // Doctor runs live write probes for memorystatus_control and
            // task_for_pid to confirm the kernel API write paths actually work.
            // Read-only checks (sysctl, swap, VM stats) remain passive.
            let (
                reactor_mode,
                reactor_health,
                reactor_pulses,
                iokit_snapshots,
                iokit_errors,
                qos_foreground,
                qos_background,
                qos_errors,
                kpc_available,
                kpc_ipc,
                kpc_memory_bound,
                memorystatus_runtime_failed,
            ) = {
                let m = state.metrics.lock_recover();
                (
                    m.reactor_status.mode.clone(),
                    m.reactor_status.health.clone(),
                    m.metrics.reactor_pulses,
                    m.metrics.iokit_snapshots,
                    m.metrics.iokit_errors,
                    m.metrics.qos_foreground_count,
                    m.metrics.qos_background_count,
                    m.metrics.qos_errors,
                    m.metrics.kpc_available,
                    m.metrics.kpc_ipc,
                    m.metrics.kpc_memory_bound_score,
                    m.metrics.top_skipped_processes.iter().any(|s| {
                        s.starts_with("memorystatus-send-failed:")
                            || s.starts_with("memorystatus-send-unsupported:")
                    }),
                )
            };
            let checks = vec![
                format!("is_root: {}", caps.is_root),
                format!("taskpolicy: {}", caps.can_taskpolicy),
                format!("sysctl: {}", caps.can_sysctl),
                format!(
                    "kernel_pressure_level_readable: {}",
                    apollo_engine::engine::sysctl_direct::read_i32(
                        "kern.memorystatus_vm_pressure_level"
                    )
                    .is_some()
                ),
                format!(
                    "memorystatus_control_write: {}",
                    caps.memorystatus_probe
                        .as_deref()
                        .unwrap_or("skipped (not root)")
                ),
                format!(
                    "memorystatus_pressure_send_runtime: {}",
                    if !caps.can_memory_pressure_send {
                        "unsupported by this kernel (disabled safely)"
                    } else if memorystatus_runtime_failed {
                        "degraded (runtime failures observed)"
                    } else {
                        "no failures observed"
                    }
                ),
                format!(
                    "task_for_pid: {}",
                    caps.task_for_pid_probe.as_deref().unwrap_or("unknown")
                ),
                format!("mdutil: {}", caps.can_mdutil),
                format!("tmutil: {}", caps.can_tmutil),
                format!("socket_exists: {}", Path::new(socket_path()).exists()),
                format!("kill_switch: {}", Path::new(kill_switch_path()).exists()),
                format!("reactor_mode: {}", reactor_mode),
                format!("reactor_health: {}", reactor_health),
                format!("reactor_pulses: {}", reactor_pulses),
                format!(
                    "swapusage_readable: {}",
                    apollo_engine::engine::sysctl_direct::read_swap_usage().is_some()
                ),
                format!(
                    "memory_pressure_readable: {}",
                    apollo_engine::engine::host_vm_info::read_vm_stats().is_some()
                ),
                format!(
                    "iokit_observed: snapshots={} errors={}",
                    iokit_snapshots, iokit_errors
                ),
                format!(
                    "qos_observed: foreground={} background={} errors={}",
                    qos_foreground, qos_background, qos_errors
                ),
                format!(
                    "kpc_observed: available={} ipc={:.3} memory_bound_proxy={:.3}",
                    kpc_available, kpc_ipc, kpc_memory_bound
                ),
            ];
            DaemonResponse::Doctor { checks }
        }
        DaemonRequest::UsageTop { limit } => {
            let limit = limit.unwrap_or(10).clamp(3, 30);
            let model = state.usage.lock_recover();
            let report = model.usage_model.top_report(limit);
            DaemonResponse::Usage(UsageResponse::Top(report))
        }
        DaemonRequest::UsageExplain { name } => {
            let model = state.usage.lock_recover();
            match model.usage_model.entry_summary(&name) {
                Some(s) => DaemonResponse::Usage(UsageResponse::Explain(s)),
                None => DaemonResponse::Error {
                    message: "usage entry not found".to_string(),
                },
            }
        }
        DaemonRequest::GetLearnedPolicy => {
            let policy = state.policy.lock_recover().learned_policy.clone();
            DaemonResponse::LearnedPolicy(policy)
        }
        DaemonRequest::Feedback { rating, note } => {
            if rating.len() > 256 {
                return DaemonResponse::Error {
                    message: "rating too long (max 256)".to_string(),
                };
            }
            if let Some(ref n) = note {
                if n.len() > 2048 {
                    return DaemonResponse::Error {
                        message: "note too long (max 2048)".to_string(),
                    };
                }
            }
            let entry = FeedbackEntry {
                at: Utc::now(),
                rating,
                note,
            };
            let feedback_path = state.policy.lock_recover().feedback_path.clone();
            append_jsonl(&feedback_path, &entry);
            DaemonResponse::Ok
        }
        DaemonRequest::GetSysctlGovernor => {
            let status = state.hardware.lock_recover().sysctl_governor_status.clone();
            DaemonResponse::SysctlGovernor(status)
        }
        DaemonRequest::RevertSysctls => {
            tracing::info!("RevertSysctls requested via RPC — flagging main loop");
            state
                .revert_sysctls_requested
                .store(true, std::sync::atomic::Ordering::Release);
            DaemonResponse::Ok
        }
        DaemonRequest::GetHealth => {
            use apollo_engine::engine::circuit_breaker::CircuitState;
            use apollo_engine::engine::degradation::OperationMode;

            let (cb_state_str, cb_trips) = {
                let pg = state.policy.lock_recover();
                (
                    pg.circuit_breaker.state().as_str().to_string(),
                    pg.circuit_breaker.trips_total,
                )
            };
            let (op_mode_str, failure_rate, deg_transitions) = {
                let pg = state.policy.lock_recover();
                (
                    pg.degradation.mode.as_str().to_string(),
                    pg.degradation.failure_rate_60s(),
                    pg.degradation.transitions_total,
                )
            };
            let (uptime_cycles, total_failures) = {
                let m = state.metrics.lock_recover();
                (m.metrics.cycles, m.metrics.failures)
            };
            let is_emergency = op_mode_str == OperationMode::Emergency.as_str();
            let is_degraded = op_mode_str != OperationMode::Full.as_str();
            let status = if is_emergency {
                "emergency"
            } else if is_degraded || cb_state_str != CircuitState::Closed.as_str() {
                "degraded"
            } else {
                "healthy"
            };
            DaemonResponse::Health(HealthReport {
                status: status.to_string(),
                circuit_breaker: cb_state_str,
                operation_mode: op_mode_str,
                failure_rate_60s: failure_rate,
                uptime_cycles,
                total_failures,
                cb_trips_total: cb_trips,
                degradation_transitions: deg_transitions,
            })
        }
        DaemonRequest::Purge => {
            use crate::main_loop_msg::{MainLoopMsg, MAIN_LOOP_TX};
            let tx = match MAIN_LOOP_TX.get() {
                Some(m) => m.lock().unwrap_or_else(|e| e.into_inner()).clone(),
                None => {
                    return DaemonResponse::PurgeResult {
                        fired: false,
                        reason: "main loop not ready".into(),
                    };
                }
            };
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            if tx.send(MainLoopMsg::CliPurge { response_tx }).is_err() {
                return DaemonResponse::PurgeResult {
                    fired: false,
                    reason: "main loop unreachable".into(),
                };
            }
            response_rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap_or(DaemonResponse::PurgeResult {
                    fired: false,
                    reason: "timeout".into(),
                })
        }
        // Subscribe es manejado antes de llegar aqui (en handle_client)
        DaemonRequest::Subscribe => DaemonResponse::Ok,
        DaemonRequest::GetVersion => DaemonResponse::VersionInfo {
            protocol: apollo_engine::engine::protocol::PROTOCOL_VERSION,
            build: env!("CARGO_PKG_VERSION").to_string(),
        },
    }
}

// ── Socket Server ──────────────────────────────────────────────────────────

/// Wrapper that signals bind success/failure via `tx` before entering the accept loop.
/// The main thread waits on `tx` to confirm binding before entering its hot loop,
/// so a bind failure causes an immediate exit(1) rather than a headless second instance.
///
/// Background: if socket bind fails (e.g., another instance is running), the previous
/// code logged an error and returned from the thread — but the daemon continued into its
/// main optimization loop with no socket, no control plane, and in conflict with the
/// other instance over frozen_state.json writes.
pub fn run_socket_server_with_notify(
    state: SharedState,
    tx: std::sync::mpsc::Sender<anyhow::Result<()>>,
) {
    let sp = socket_path();
    let socket_path = Path::new(sp);

    // Probe: can we set up and bind the socket?
    let bind_result = (|| -> anyhow::Result<()> {
        if let Some(parent) = socket_path.parent() {
            HardPath::secure_create_dir_all(parent)?;
        }
        HardPath::verify_no_symlink(socket_path)?;
        if socket_path.exists() {
            fs::remove_file(socket_path)?;
        }
        // A successful bind (and immediate close) confirms we can own the socket.
        // run_socket_server will rebind immediately after — the window is <1ms.
        let probe = UnixListener::bind(socket_path).context("bind socket")?;
        drop(probe);
        fs::remove_file(socket_path).ok();
        Ok(())
    })();

    let _ = tx.send(bind_result);
    // If bind_result was Err, main thread will exit(1) — this thread can return.
    // If bind_result was Ok, run the full server (which re-binds immediately).
    if let Err(e) = run_socket_server(state) {
        tracing::error!(err = ?e, "socket server exited with error");
    }
}

pub fn run_socket_server(state: SharedState) -> anyhow::Result<()> {
    let socket_path = Path::new(socket_path());
    println!("Socket server starting for path: {:?}", socket_path);
    if let Some(parent) = socket_path.parent() {
        HardPath::secure_create_dir_all(parent)?;
    }
    HardPath::verify_no_symlink(socket_path)?;
    if socket_path.exists() {
        println!("Stale socket found, removing: {:?}", socket_path);
        fs::remove_file(socket_path)?;
    }

    let listener = UnixListener::bind(socket_path).context("bind socket")?;
    println!("Socket server listening on: {:?}", socket_path);
    // Socket permissions: 0o660 root:staff — all human users (staff group, GID 20)
    // can connect for read-only queries (status, metrics, subscribe).
    // Mutating commands (SetProfile, Feedback, etc.) require root via getpeereid.
    if unsafe { libc::getuid() } == 0 {
        let _ = fs::set_permissions(socket_path, fs::Permissions::from_mode(0o660));
        if let Ok(c_path) = CString::new(socket_path.as_os_str().as_encoded_bytes()) {
            unsafe {
                const STAFF_GID: libc::gid_t = 20;
                libc::chown(c_path.as_ptr(), 0, STAFF_GID); // root:staff
            }
        }
    } else {
        // Non-root: restrict to owner only.
        let _ = fs::set_permissions(socket_path, fs::Permissions::from_mode(0o600));
    }

    // BUG 6 fix: spawn a thread per client so one slow/malicious client doesn't
    // block all others. The old synchronous loop also blocked indefinitely on
    // accept(), preventing clean shutdown when stop=true was set.
    let active_clients = Arc::new(std::sync::atomic::AtomicU32::new(0));
    const MAX_CONCURRENT_CLIENTS: u32 = 32;

    for conn in listener.incoming() {
        if state.stop.load(Ordering::Acquire) || STOP_REQUESTED.load(Ordering::Acquire) {
            break;
        }
        if let Ok(stream) = conn {
            let clients = active_clients.clone();
            // Atomically increment first, then check — prevents race where
            // multiple threads pass the limit check simultaneously.
            let prev = clients.fetch_add(1, Ordering::AcqRel);
            if prev >= MAX_CONCURRENT_CLIENTS {
                clients.fetch_sub(1, Ordering::Relaxed);
                drop(stream);
                continue;
            }
            let state_clone = state.clone();
            thread::spawn(move || {
                handle_client(stream, &state_clone);
                clients.fetch_sub(1, Ordering::Release);
            });
        }
    }

    Ok(())
}
