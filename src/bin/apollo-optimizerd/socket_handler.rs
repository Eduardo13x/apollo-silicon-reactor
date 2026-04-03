//! Socket Handler — Unix domain socket server + request dispatch.
//!
//! Extracted from the daemon monolith. Contains:
//! - `run_socket_server()` — bind, listen, spawn per-client threads
//! - `handle_client()` — read request, auth, dispatch
//! - `process_request()` — the 22-arm command dispatcher
//! - `build_llm_status()` — LLM status builder
//! - `broadcast_current_status()` — push updates to subscribers
//! - `is_peer_root()` — peer credential check

use std::ffi::CString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;

use anyhow::Context;
use chrono::{Duration as ChronoDuration, Local, Utc};

use apollo_optimizer::collector::SystemCollector;
use apollo_optimizer::engine::capabilities::detect_capabilities;
use apollo_optimizer::engine::daemon_helpers::{
    kill_switch_path, merge_seed_into, metrics_path, socket_path,
};
use apollo_optimizer::engine::llm::{
    append_jsonl, delete_file_best_effort, load_repo_config, write_json, write_secret, FeedbackEntry,
    LlmAdvisor,
};
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::protocol::{DaemonRequest, DaemonResponse};
use apollo_optimizer::engine::safety::pattern_conflicts_with_protected;
use apollo_optimizer::engine::types::{
    DaemonStatus, HardPath, HealthReport, LearnedPolicyStatus, LlmRunMode, LlmStatus,
    RuntimeMetrics, UsageResponse,
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

pub fn process_request(req: DaemonRequest, state: &SharedState) -> DaemonResponse {
    match req {
        DaemonRequest::GetStatus => {
            let now = Utc::now();
            let profile = *state.profile.lock_recover();
            let latency_target = *state.latency_target.lock_recover();
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
            // Snapshot governor + wake_state, then DROP locks before build_llm_status.
            let (auto_profile_enabled, base_profile, override_active, override_expires_at,
                 transition_reason) = {
                let gov = state.governor.lock_recover();
                (gov.auto_profile_enabled, gov.base_profile,
                 gov.manual_override.is_some(),
                 gov.manual_override.as_ref().map(|o| o.expires_at),
                 gov.transition_reason.clone())
            };
            let (grace_active, grace_remaining, last_wake_at, post_wake_policy) = {
                let proc = state.process.lock_recover();
                let ws = &proc.wake_state;
                let ga = ws.post_wake_grace_until.map(|t| t > now).unwrap_or(false);
                let gr = ws.post_wake_grace_until
                    .and_then(|t| (t - now).to_std().ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                (ga, gr, ws.last_wake_at, ws.post_wake_policy.clone())
            };
            let (reactor_mode, reactor_health) = {
                let m = state.metrics.lock_recover();
                (m.reactor_status.mode.clone(), m.reactor_status.health.clone())
            };
            let llm = build_llm_status(state);
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
                llm: Some(llm),
            };
            DaemonResponse::Status(status)
        }
        DaemonRequest::GetMetrics => DaemonResponse::Metrics(state.metrics.lock_recover().metrics.clone()),
        DaemonRequest::GetTopBlockers => {
            DaemonResponse::TopBlockers(state.process.lock_recover().last_blockers.clone())
        }
        DaemonRequest::GetProfileTimeline => {
            DaemonResponse::ProfileTimeline(state.timeline.lock_recover().iter().cloned().collect())
        }
        DaemonRequest::GetCapabilities => DaemonResponse::Capabilities(detect_capabilities()),
        DaemonRequest::SetProfile {
            profile,
            ttl_minutes,
        } => {
            let ttl = ttl_minutes.unwrap_or(20).clamp(1, 1440);
            state.governor.lock_recover().set_manual_override(
                profile,
                ttl,
                "cli-set-profile".to_string(),
            );
            DaemonResponse::Ok
        }
        DaemonRequest::SetLatencyTarget { target } => {
            *state.latency_target.lock_recover() = target;
            DaemonResponse::Ok
        }
        DaemonRequest::SetAutoProfile { enabled } => {
            state.governor.lock_recover().set_auto_profile(enabled);
            DaemonResponse::Ok
        }
        DaemonRequest::ClearProfileOverride => {
            state.governor.lock_recover().clear_manual_override();
            DaemonResponse::Ok
        }
        DaemonRequest::Restore => {
            let mut frozen_state = state.frozen_state.lock_recover();
            for pid in frozen_state.keys() {
                unsafe {
                    libc::kill(*pid as i32, libc::SIGCONT);
                }
            }
            frozen_state.clear();
            let _ = fs::remove_file(kill_switch_path());
            DaemonResponse::Ok
        }
        DaemonRequest::PanicRestore => {
            // Symlink protection: refuse to create kill switch through a symlink
            let ks = kill_switch_path();
            let ks_path = Path::new(ks);
            if ks_path.exists() {
                if let Ok(meta) = fs::symlink_metadata(ks_path) {
                    if meta.file_type().is_symlink() {
                        return DaemonResponse::Error {
                            message: "kill switch path is a symlink — refusing".to_string(),
                        };
                    }
                }
            }
            let _ = File::create(ks);
            state.governor.lock_recover().set_auto_profile(false);
            let mut frozen_state = state.frozen_state.lock_recover();
            for pid in frozen_state.keys() {
                unsafe {
                    libc::kill(*pid as i32, libc::SIGCONT);
                }
            }
            frozen_state.clear();
            DaemonResponse::Ok
        }
        DaemonRequest::Doctor => {
            let caps = detect_capabilities();
            let checks = vec![
                format!("is_root: {}", caps.is_root),
                format!("taskpolicy: {}", caps.can_taskpolicy),
                format!("sysctl: {}", caps.can_sysctl),
                format!("mdutil: {}", caps.can_mdutil),
                format!("tmutil: {}", caps.can_tmutil),
                format!("socket_exists: {}", Path::new(socket_path()).exists()),
                format!("kill_switch: {}", Path::new(kill_switch_path()).exists()),
                {
                    let m = state.metrics.lock_recover();
                    format!("reactor_mode: {}", m.reactor_status.mode)
                },
                {
                    let m = state.metrics.lock_recover();
                    format!("reactor_health: {}", m.reactor_status.health)
                },
                format!(
                    "swapusage_readable: {}",
                    apollo_optimizer::engine::sysctl_direct::read_swap_usage().is_some()
                ),
                format!(
                    "memory_pressure_readable: {}",
                    apollo_optimizer::engine::host_vm_info::read_vm_stats().is_some()
                ),
            ];
            DaemonResponse::Doctor { checks }
        }
        DaemonRequest::GetLlmStatus => DaemonResponse::LlmStatus(build_llm_status(state)),
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
        DaemonRequest::LlmSetKey { api_key, ttl_days } => {
            let now = Utc::now();
            let ttl_clamped = ttl_days.clamp(1, 365);
            let expires = now + ChronoDuration::days(ttl_clamped as i64);
            let (llm_key_path, llm_state_path) = {
                let llm = state.llm.lock_recover();
                (llm.llm_key_path.clone(), llm.llm_state_path.clone())
            };
            if write_secret(&llm_key_path, api_key.trim()).is_err() {
                return DaemonResponse::Error {
                    message: "failed to write llm key".to_string(),
                };
            }
            {
                let mut guard = state.llm.lock_recover();
                guard.llm_state.enabled = true;
                guard.llm_state.training_started_at = Some(now);
                guard.llm_state.training_expires_at = Some(expires);
                guard.llm_state.last_call_at = None;
                guard.llm_state.last_attempt_at = None;
                guard.llm_state.last_http_status = None;
                guard.llm_state.last_error = None;
                guard.llm_state.last_trigger_reason = None;
                guard.llm_state.consecutive_failures = 0;
                guard.llm_state.calls_in_window = 0;
                guard.llm_state.hour_window_started_at = Some(now);
                guard.llm_state.calls_today_day = None;
                guard.llm_state.calls_today = 0;
                guard.llm_state.mode = LlmRunMode::Sensitive;
                guard.llm_state.last_trigger_at = None;
                guard.llm_state.trigger_events.clear();
                guard.llm_state.no_trigger_since = Some(now);
                guard.llm_state.last_suggestion = None;
                guard.llm_state.policy_updates_day = None;
                guard.llm_state.policy_updates_today = 0;
                write_json(&llm_state_path, &guard.llm_state, Some(0o600));
            }
            DaemonResponse::Ok
        }
        DaemonRequest::LlmDisable => {
            let (llm_key_path, llm_state_path) = {
                let llm = state.llm.lock_recover();
                (llm.llm_key_path.clone(), llm.llm_state_path.clone())
            };
            delete_file_best_effort(&llm_key_path);
            {
                let mut guard = state.llm.lock_recover();
                guard.llm_state.enabled = false;
                guard.llm_state.training_expires_at = None;
                guard.llm_state.last_suggestion = None;
                write_json(&llm_state_path, &guard.llm_state, Some(0o600));
            }
            DaemonResponse::Ok
        }
        DaemonRequest::LlmTest => {
            let now = Utc::now();
            let (llm_key_path, llm_state_path, llm_cfg_default) = {
                let llm = state.llm.lock_recover();
                (llm.llm_key_path.clone(), llm.llm_state_path.clone(), llm.llm_cfg.clone())
            };
            let llm_cfg = load_repo_config(&state.config_path)
                .llm
                .unwrap_or(llm_cfg_default);
            if !llm_cfg.enabled() {
                return DaemonResponse::LlmTestResult {
                    ok: false,
                    http_status: None,
                    error: Some("llm disabled in config".to_string()),
                    suggestion: None,
                };
            }
            if !llm_key_path.exists() {
                return DaemonResponse::LlmTestResult {
                    ok: false,
                    http_status: None,
                    error: Some("missing llm api key".to_string()),
                    suggestion: None,
                };
            }
            {
                let guard = state.llm.lock_recover();
                if !guard.llm_state.training_active() {
                    return DaemonResponse::LlmTestResult {
                        ok: false,
                        http_status: None,
                        error: Some("training not active (enable + ttl)".to_string()),
                        suggestion: None,
                    };
                }
            }

            let api_key = match HardPath::read_to_string_limited(&llm_key_path, 4096) {
                Ok(v) => v,
                Err(_) => {
                    return DaemonResponse::LlmTestResult {
                        ok: false,
                        http_status: None,
                        error: Some("cannot read llm key".to_string()),
                        suggestion: None,
                    }
                }
            };

            // Collect a one-off snapshot for this test.
            let mut collector = SystemCollector::new();
            let mut snapshot = collector.collect_snapshot();
            snapshot.pressure.thermal_level = state.metrics.lock_recover().thermal_level_real.clone();

            // Record attempt immediately.
            {
                let mut guard = state.llm.lock_recover();
                if guard.llm_state.training_started_at.is_none() {
                    guard.llm_state.training_started_at = Some(now);
                }
                guard.llm_state.last_attempt_at = Some(now);
                guard.llm_state.last_trigger_reason = Some("manual-test".to_string());
                guard.llm_state.last_error = None;
                guard.llm_state.last_http_status = None;

                // Count this as a call attempt for observability/budget.
                let today = Local::now().date_naive().to_string();
                if guard.llm_state.calls_today_day.as_deref() != Some(&today) {
                    guard.llm_state.calls_today_day = Some(today);
                    guard.llm_state.calls_today = 0;
                }
                guard.llm_state.calls_today += 1;
                if guard.llm_state
                    .hour_window_started_at
                    .map(|t| now - t > ChronoDuration::hours(1))
                    .unwrap_or(true)
                {
                    guard.llm_state.hour_window_started_at = Some(now);
                    guard.llm_state.calls_in_window = 0;
                }
                guard.llm_state.calls_in_window += 1;

                write_json(&llm_state_path, &guard.llm_state, Some(0o600));
            }

            let mut advisor = LlmAdvisor::new(llm_cfg.clone());
            let current_policy = state.learned_policy.lock_recover().clone();
            match advisor.call_raw(&snapshot, &api_key, Some(&current_policy)) {
                Ok(suggestion) => {
                    {
                        let mut guard = state.llm.lock_recover();
                        guard.llm_state.last_call_at = Some(now);
                        guard.llm_state.last_http_status = Some(200);
                        guard.llm_state.last_suggestion = Some(suggestion.clone());
                        guard.llm_state.last_error = None;
                        write_json(&llm_state_path, &guard.llm_state, Some(0o600));
                    }
                    DaemonResponse::LlmTestResult {
                        ok: true,
                        http_status: Some(200),
                        error: None,
                        suggestion: Some(suggestion),
                    }
                }
                Err(err) => {
                    let (http_status, msg) = match err {
                        apollo_optimizer::engine::llm::LlmCallError::Cooldown => {
                            (None, "cooldown".to_string())
                        }
                        apollo_optimizer::engine::llm::LlmCallError::HttpStatus {
                            code,
                            body_excerpt,
                        } => (
                            Some(code),
                            format!("http {} {}", code, body_excerpt.unwrap_or_default()),
                        ),
                        apollo_optimizer::engine::llm::LlmCallError::Transport(e) => {
                            (None, format!("transport {}", e))
                        }
                        apollo_optimizer::engine::llm::LlmCallError::Parse(e) => {
                            (None, format!("parse {}", e))
                        }
                        apollo_optimizer::engine::llm::LlmCallError::Rejected(e) => {
                            (None, format!("rejected {}", e))
                        }
                    };
                    {
                        let mut guard = state.llm.lock_recover();
                        guard.llm_state.last_http_status = http_status;
                        guard.llm_state.last_error = Some(msg.clone());
                        write_json(&llm_state_path, &guard.llm_state, Some(0o600));
                    }
                    DaemonResponse::LlmTestResult {
                        ok: false,
                        http_status,
                        error: Some(msg),
                        suggestion: None,
                    }
                }
            }
        }
        DaemonRequest::GetLearnedPolicy => {
            let policy = state.learned_policy.lock_recover().clone();
            DaemonResponse::LearnedPolicy(policy)
        }
        DaemonRequest::SetLearnedPolicy { policy: new_policy } => {
            // Validate size limits to prevent OOM attacks
            const MAX_PATTERNS: usize = 500;
            if new_policy.interactive_patterns.len() > MAX_PATTERNS
                || new_policy.noise_patterns.len() > MAX_PATTERNS
                || new_policy.protected_patterns.len() > MAX_PATTERNS
            {
                DaemonResponse::Error {
                    message: format!(
                        "Policy too large: max {} patterns per category",
                        MAX_PATTERNS
                    ),
                }
            } else {
                // Validate individual pattern lengths.
                const MAX_PATTERN_LEN: usize = 256;
                const MIN_PATTERN_LEN: usize = 4;
                let has_invalid_pattern = new_policy
                    .interactive_patterns
                    .iter()
                    .chain(new_policy.noise_patterns.iter())
                    .chain(new_policy.protected_patterns.iter())
                    .any(|p| {
                        p.len() > MAX_PATTERN_LEN
                            || p.len() < MIN_PATTERN_LEN
                            || p.trim().is_empty()
                            || p.chars().any(|c| {
                                // Reject control chars and glob/regex metacharacters.
                                // Parentheses are intentionally allowed: macOS process
                                // names use them legitimately, e.g. "Helper (GPU)".
                                // Patterns are matched with str::contains(), not regex.
                                c.is_control()
                                    || c == '*'
                                    || c == '['
                                    || c == ']'
                                    || c == '|'
                                    || c == '\\'
                                    || c == '{'
                                    || c == '}'
                            })
                    });
                if has_invalid_pattern {
                    return DaemonResponse::Error {
                        message: format!(
                            "pattern length must be {}-{} chars, non-empty",
                            MIN_PATTERN_LEN, MAX_PATTERN_LEN
                        ),
                    };
                }

                // Sanitize: strip any patterns that could match a
                // hardcoded protected or critical-background process.
                // Uses bidirectional prefix/suffix overlap (75% threshold)
                // to block evasion attempts like "kernel_tas" for "kernel_task".
                let mut sanitized = new_policy;
                sanitized
                    .noise_patterns
                    .retain(|pat| !pattern_conflicts_with_protected(pat));
                sanitized
                    .interactive_patterns
                    .retain(|pat| !pattern_conflicts_with_protected(pat));
                sanitized
                    .protected_patterns
                    .retain(|pat| !pattern_conflicts_with_protected(pat));
                let learned_policy_path = state.llm.lock_recover().learned_policy_path.clone();
                let mut policy = state.learned_policy.lock_recover();
                *policy = sanitized;
                // Re-merge seed as floor — seed patterns can never be removed.
                merge_seed_into(&mut policy);
                policy.learned_at = Some(Utc::now());
                write_json(&learned_policy_path, &*policy, Some(0o600));
                // Propagate to ML classifier.
                {
                    let mut gov = state.adaptive_governor.lock_recover();
                    gov.update_learned_policy(&policy);
                }
                DaemonResponse::Ok
            }
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
            append_jsonl(&state.llm.lock_recover().feedback_path, &entry);
            DaemonResponse::Ok
        }
        DaemonRequest::GetSysctlGovernor => {
            let status = state.hardware.lock_recover().sysctl_governor_status.clone();
            DaemonResponse::SysctlGovernor(status)
        }
        DaemonRequest::GetHealth => {
            use apollo_optimizer::engine::circuit_breaker::CircuitState;
            use apollo_optimizer::engine::degradation::OperationMode;

            let (cb_state_str, cb_trips) = {
                let cb = state.circuit_breaker.lock_recover();
                (cb.state().as_str().to_string(), cb.trips_total)
            };
            let (op_mode_str, failure_rate, deg_transitions) = {
                let deg = state.degradation.lock_recover();
                (
                    deg.mode.as_str().to_string(),
                    deg.failure_rate_60s(),
                    deg.transitions_total,
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
        // Subscribe es manejado antes de llegar aqui (en handle_client)
        DaemonRequest::Subscribe => DaemonResponse::Ok,
        DaemonRequest::GetVersion => DaemonResponse::VersionInfo {
            protocol: apollo_optimizer::engine::protocol::PROTOCOL_VERSION,
            build: env!("CARGO_PKG_VERSION").to_string(),
        },
    }
}

// ── LLM Status Builder ─────────────────────────────────────────────────────

pub fn build_llm_status(state: &SharedState) -> LlmStatus {
    let (llm_cfg_default, llm_state, llm_key_path) = {
        let llm = state.llm.lock_recover();
        (llm.llm_cfg.clone(), llm.llm_state.clone(), llm.llm_key_path.clone())
    };
    let llm_cfg = load_repo_config(&state.config_path)
        .llm
        .unwrap_or(llm_cfg_default);
    let enabled_from_disk = llm_cfg.enabled();
    let policy = state.learned_policy.lock_recover().clone();

    let has_key = llm_key_path.exists();
    let enabled = enabled_from_disk && llm_state.enabled;
    let training_active = enabled && llm_state.training_active() && has_key;

    let now_local = Local::now();
    let today = now_local.date_naive().to_string();

    // Backward compatible: older persisted state may not have `training_started_at`.
    // Use the first observed call/attempt as a proxy.
    let training_started = llm_state
        .training_started_at
        .or(llm_state.last_call_at)
        .or(llm_state.last_attempt_at);
    let bootcamp = training_started
        .map(|t| Utc::now() - t < ChronoDuration::days(5))
        .unwrap_or(false);
    let daily_budget: u32 = if bootcamp { 24 } else { 8 };
    let calls_today = if llm_state.calls_today_day.as_deref() == Some(&today) {
        llm_state.calls_today
    } else {
        0
    };
    let daily_budget_remaining = daily_budget.saturating_sub(calls_today);

    LlmStatus {
        enabled,
        training_active,
        training_expires_at: llm_state.training_expires_at,
        has_api_key: has_key,
        mode: llm_state.mode,
        last_call_at: llm_state.last_call_at,
        last_attempt_at: llm_state.last_attempt_at,
        last_http_status: llm_state.last_http_status,
        last_error: llm_state.last_error.clone(),
        last_trigger_reason: llm_state.last_trigger_reason.clone(),
        calls_in_current_window: llm_state.calls_in_window,
        min_confidence: llm_cfg.min_confidence(),
        calls_today,
        daily_budget,
        daily_budget_remaining,
        last_suggestion_confidence: llm_state.last_suggestion.as_ref().map(|s| s.confidence),
        last_suggestion_rationale: llm_state
            .last_suggestion
            .as_ref()
            .map(|s| s.rationale.clone()),
        learned_policy: LearnedPolicyStatus {
            interactive_patterns: policy.interactive_patterns.len(),
            noise_patterns: policy.noise_patterns.len(),
            protected_patterns: policy.protected_patterns.len(),
            learned_at: policy.learned_at,
        },
    }
}

// ── Socket Server ──────────────────────────────────────────────────────────

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
    // Mutating commands (SetProfile, SetLearnedPolicy, etc.) require root via getpeereid.
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
