//! ══════════════════════════════════════════════════════════════════════════════
//! Apollo E2E Benchmark — Dry-Run Mode
//! ══════════════════════════════════════════════════════════════════════════════
//!
//! Spawns the real apollo-optimizerd binary with --dry-run so the full pipeline
//! runs (collect → decide → execute bookkeeping → learn → broadcast) without
//! touching real processes (no SIGSTOP/SIGCONT/taskpolicy/sysctl/mdutil).
//!
//! Three test layers:
//!   Layer 1 — Smoke   : daemon boots, protocol works, cycles run
//!   Layer 2 — State   : profile changes, kill-switch, rapid transitions
//!   Layer 3 — Adversarial : concurrent floods, malformed input, race conditions
//!
//! Each test gets its own tempdir + unique socket path → fully parallel.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

// ── Test harness ─────────────────────────────────────────────────────────────

const DAEMON_BIN: &str = env!("CARGO_BIN_EXE_apollo-optimizerd");
const BOOT_TIMEOUT: Duration = Duration::from_secs(10);
const CYCLE_TIMEOUT: Duration = Duration::from_secs(30);
const CYCLE_WAIT: Duration = Duration::from_millis(500);

struct DaemonGuard {
    child: Child,
    pub socket: String,
    pub kill_switch: String,
    _tmpdir: tempfile::TempDir,
}

impl Drop for DaemonGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// Spawn the daemon in dry-run mode with an isolated socket + kill-switch path.
fn spawn_daemon_guard() -> DaemonGuard {
    spawn_daemon_guard_with_profile("balanced")
}

fn spawn_daemon_guard_with_profile(profile: &str) -> DaemonGuard {
    let tmpdir = tempfile::tempdir().expect("create tempdir");
    let socket = tmpdir.path().join("apollo.sock").to_string_lossy().into_owned();
    let kill_switch = tmpdir.path().join("apollo.disable").to_string_lossy().into_owned();

    let child = Command::new(DAEMON_BIN)
        .args(["daemon", "--profile", profile, "--dry-run"])
        .env("APOLLO_SOCKET_PATH", &socket)
        .env("APOLLO_KILL_SWITCH_PATH", &kill_switch)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn daemon");

    // Wait for socket to appear (daemon is ready).
    let start = Instant::now();
    while start.elapsed() < BOOT_TIMEOUT {
        if PathBuf::from(&socket).exists() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        PathBuf::from(&socket).exists(),
        "daemon socket not created within {BOOT_TIMEOUT:?} — daemon may have crashed"
    );

    DaemonGuard { child, socket, kill_switch, _tmpdir: tmpdir }
}

/// Send one JSON request, return the raw response line.
fn send_request(socket: &str, json: &str) -> String {
    send_request_timeout(socket, json, Duration::from_secs(10))
}

fn send_request_timeout(socket: &str, json: &str, timeout: Duration) -> String {
    let mut stream = UnixStream::connect(socket).expect("connect to daemon socket");
    stream.set_read_timeout(Some(timeout)).unwrap();
    stream.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
    writeln!(stream, "{}", json).expect("write request");
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).expect("read response");
    line
}

/// Like send_request but returns None on any IO error (for adversarial tests).
fn try_send_request(socket: &str, json: &str) -> Option<String> {
    let Ok(mut stream) = UnixStream::connect(socket) else { return None };
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    stream.set_write_timeout(Some(Duration::from_secs(2))).ok()?;
    writeln!(stream, "{}", json).ok()?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    if line.is_empty() { None } else { Some(line) }
}

/// Subscribe to the daemon and collect up to `n` push messages.
fn subscribe_collect(socket: &str, n: usize, timeout: Duration) -> Vec<serde_json::Value> {
    let mut stream = UnixStream::connect(socket).expect("connect for subscribe");
    stream.set_read_timeout(Some(timeout)).unwrap();
    stream.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
    writeln!(stream, r#"{{"type":"Subscribe"}}"#).expect("write subscribe");

    let mut reader = BufReader::new(stream);
    // First line is the Ok ack.
    let mut ack = String::new();
    let _ = reader.read_line(&mut ack);

    let mut messages = Vec::new();
    for _ in 0..n {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    messages.push(v);
                }
            }
        }
    }
    messages
}

/// Parse response and return the Value.
fn parse(resp: &str) -> serde_json::Value {
    serde_json::from_str(resp).unwrap_or_else(|e| {
        panic!("failed to parse response as JSON: {e}\nResponse was: {resp:?}")
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// LAYER 1 — SMOKE TESTS
// Verify: daemon boots, protocol responds, cycles run.
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn e2e_boots_and_serves_status() {
    let guard = spawn_daemon_guard();
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Status", "expected Status response, got: {v}");
    assert_eq!(v["payload"]["running"], true, "daemon must report running=true");
}

#[test]
fn e2e_version_protocol() {
    let guard = spawn_daemon_guard();
    let resp = send_request(&guard.socket, r#"{"type":"GetVersion"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "VersionInfo", "expected VersionInfo, got: {v}");
    assert_eq!(v["payload"]["protocol"], 1, "protocol version must be 1");
    let build = v["payload"]["build"].as_str().unwrap_or("");
    assert!(!build.is_empty(), "build string must be non-empty");
}

#[test]
fn e2e_capabilities_coherent() {
    let guard = spawn_daemon_guard();
    let resp = send_request(&guard.socket, r#"{"type":"GetCapabilities"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Capabilities", "expected Capabilities, got: {v}");
    let caps = &v["payload"];
    // Non-root test runner: memorystatus requires root.
    assert_eq!(caps["can_memorystatus"], false, "non-root must have can_memorystatus=false");
    // macOS: taskpolicy always available.
    assert_eq!(caps["can_taskpolicy"], true, "macOS must have can_taskpolicy=true");
}

#[test]
fn e2e_cycles_increment() {
    let guard = spawn_daemon_guard();
    // Collect 3 push events (wait up to 30s — each cycle is ~2s dry-run).
    let pushes = subscribe_collect(&guard.socket, 3, CYCLE_TIMEOUT);
    assert!(
        pushes.len() >= 2,
        "expected ≥2 cycle pushes via Subscribe, got {}",
        pushes.len()
    );
    // cycles must be monotonically increasing.
    let c1 = pushes[0]["payload"]["metrics"]["cycles"].as_u64().unwrap_or(0);
    let c2 = pushes[1]["payload"]["metrics"]["cycles"].as_u64().unwrap_or(0);
    assert!(c2 > c1, "cycles must increase: {c1} → {c2}");
}

#[test]
fn e2e_doctor_returns_checks() {
    let guard = spawn_daemon_guard();
    let resp = send_request(&guard.socket, r#"{"type":"Doctor"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Doctor", "expected Doctor, got: {v}");
    let checks = v["payload"]["checks"].as_array().expect("checks must be array");
    assert!(!checks.is_empty(), "Doctor must return at least one check");
}

// ══════════════════════════════════════════════════════════════════════════════
// LAYER 2 — STATE MACHINE TESTS
// Verify: profile changes, kill-switch, TTL, rapid transitions.
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn e2e_set_profile_takes_effect() {
    let guard = spawn_daemon_guard();

    // Switch to AggressiveRoot.
    let resp = send_request(
        &guard.socket,
        r#"{"type":"SetProfile","payload":{"profile":"aggressive-root","ttl_minutes":null}}"#,
    );
    let v = parse(&resp);
    assert_eq!(v["type"], "Ok", "SetProfile must return Ok, got: {v}");

    // GetStatus should reflect the override in effective_profile.
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    // After SetProfile, the effective_profile (or override_active) must change.
    let effective = v["payload"]["effective_profile"].as_str().unwrap_or("");
    let override_active = v["payload"]["override_active"].as_bool().unwrap_or(false);
    assert!(
        effective.contains("aggressive") || override_active,
        "expected effective_profile 'aggressive-root' or override_active=true, got effective='{effective}' override_active={override_active}"
    );
}

#[test]
fn e2e_kill_switch_pauses_and_resumes() {
    let guard = spawn_daemon_guard();

    // Verify not paused initially.
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    assert_eq!(v["payload"]["kill_switch"], false, "kill_switch must start false");

    // Create kill switch file.
    std::fs::write(&guard.kill_switch, "").expect("create kill switch");

    // Wait for daemon to detect it (one cycle, ~2s).
    let start = Instant::now();
    loop {
        std::thread::sleep(CYCLE_WAIT);
        let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
        let v = parse(&resp);
        if v["payload"]["kill_switch"] == true {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(15),
            "daemon did not detect kill_switch within 15s"
        );
    }

    // Remove kill switch and verify daemon resumes.
    std::fs::remove_file(&guard.kill_switch).expect("remove kill switch");

    let start = Instant::now();
    loop {
        std::thread::sleep(CYCLE_WAIT);
        let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
        let v = parse(&resp);
        if v["payload"]["kill_switch"] == false {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(15),
            "daemon did not resume after kill_switch removed within 15s"
        );
    }
}

#[test]
fn e2e_rapid_profile_switching() {
    let guard = spawn_daemon_guard();

    // 20 rapid profile switches alternating Performance / SafeRoot.
    for i in 0..20 {
        let profile = if i % 2 == 0 { "Performance" } else { "SafeRoot" };
        let json = format!(
            r#"{{"type":"SetProfile","payload":{{"profile":"{profile}","ttl_minutes":null}}}}"#
        );
        // Daemon must still respond — not hang or panic.
        let resp = try_send_request(&guard.socket, &json);
        assert!(resp.is_some(), "daemon stopped responding at switch {i}");
    }

    // Daemon must still serve a coherent status after the barrage.
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Status", "daemon must serve Status after rapid switching");
}

#[test]
fn e2e_set_profile_with_ttl() {
    let guard = spawn_daemon_guard();

    // Set AggressiveRoot with a 2-minute TTL.
    let resp = send_request(
        &guard.socket,
        r#"{"type":"SetProfile","payload":{"profile":"aggressive-root","ttl_minutes":2}}"#,
    );
    let v = parse(&resp);
    assert_eq!(v["type"], "Ok", "SetProfile with TTL must return Ok");

    // Verify override_active is true.
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    assert_eq!(
        v["payload"]["override_active"], true,
        "override_active must be true when TTL is set"
    );
    assert!(
        v["payload"]["override_expires_at"].is_string(),
        "override_expires_at must be a string timestamp"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// LAYER 3 — ADVERSARIAL TESTS
// Try to break the daemon: floods, malformed input, races, disconnects.
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn e2e_socket_flood_32_concurrent() {
    let guard = spawn_daemon_guard();
    let socket = guard.socket.clone();

    let handles: Vec<_> = (0..32)
        .map(|_| {
            let s = socket.clone();
            std::thread::spawn(move || try_send_request(&s, r#"{"type":"GetStatus"}"#))
        })
        .collect();

    let ok = handles
        .into_iter()
        .filter_map(|h| h.join().ok().flatten())
        .filter(|resp| {
            serde_json::from_str::<serde_json::Value>(resp)
                .map(|v| v["type"] == "Status")
                .unwrap_or(false)
        })
        .count();

    // MAX_CONCURRENT_CLIENTS is 32 — expect most to succeed.
    assert!(
        ok >= 28,
        "at least 28/32 concurrent requests must succeed, got {ok}"
    );

    // Daemon must still be alive.
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Status", "daemon must survive 32-client flood");
}

#[test]
fn e2e_socket_flood_33_rejected_gracefully() {
    let guard = spawn_daemon_guard();
    let socket = guard.socket.clone();

    // Open 33 connections — one over the MAX_CONCURRENT_CLIENTS limit.
    let handles: Vec<_> = (0..33)
        .map(|_| {
            let s = socket.clone();
            std::thread::spawn(move || try_send_request(&s, r#"{"type":"GetStatus"}"#))
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().ok().flatten()).collect();
    let ok_count = results.iter().filter(|r| r.is_some()).count();
    let err_count = 33 - ok_count;

    // At least one must be rejected, but daemon must survive.
    assert!(err_count >= 1, "at least one request must be rejected when over the client limit");

    // After the flood: daemon must still serve requests normally.
    std::thread::sleep(Duration::from_millis(500));
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Status", "daemon must survive 33-client flood");
}

#[test]
fn e2e_malformed_json_doesnt_crash() {
    let guard = spawn_daemon_guard();

    let payloads: &[&str] = &[
        "}{{{not json",
        "",
        "\n\n\n",
        r#"{"type": 42}"#,
        r#"{"type": "GetStatus", "unexpected_extra_field": true}"#,
        r#"null"#,
        r#"[]"#,
    ];

    for payload in payloads {
        // Use a raw write — may or may not get a response, we don't care.
        if let Ok(mut stream) = UnixStream::connect(&guard.socket) {
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let _ = writeln!(stream, "{}", payload);
            // Drain any response (ignore result).
            let mut buf = String::new();
            let _ = BufReader::new(stream).read_line(&mut buf);
        }
    }

    // Daemon must still be alive and responding.
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Status", "daemon must survive malformed JSON payloads");
}

#[test]
fn e2e_oversized_request_rejected() {
    let guard = spawn_daemon_guard();

    // Send a 128 KB request — well above the 64 KB limit in the socket handler.
    let big_note = "x".repeat(128 * 1024);
    let oversized = format!(
        r#"{{"type":"Feedback","payload":{{"rating":"good","note":"{big_note}"}}}}"#
    );

    if let Ok(mut stream) = UnixStream::connect(&guard.socket) {
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
        let _ = writeln!(stream, "{}", oversized);
        let mut buf = String::new();
        let _ = BufReader::new(stream).read_line(&mut buf);
        // If we got a response, it should be an Error, not a crash.
        if !buf.is_empty() {
            let v = serde_json::from_str::<serde_json::Value>(&buf).unwrap_or_default();
            assert!(
                v["type"] == "Error" || v["type"] == "Ok",
                "oversized request must yield Error or Ok, not crash; got: {v}"
            );
        }
    }

    // Daemon must still serve normal requests.
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Status", "daemon must survive oversized request");
}

#[test]
fn e2e_rapid_subscribe_unsubscribe() {
    let guard = spawn_daemon_guard();

    // Open and immediately close 50 subscription connections.
    for _ in 0..50 {
        if let Ok(mut stream) = UnixStream::connect(&guard.socket) {
            stream.set_write_timeout(Some(Duration::from_millis(200))).ok();
            let _ = writeln!(stream, r#"{{"type":"Subscribe"}}"#);
            // Drop stream immediately — abrupt disconnect.
        }
    }

    // Brief pause to let the daemon clean up file descriptors.
    std::thread::sleep(Duration::from_millis(500));

    // Daemon must still serve requests without fd exhaustion.
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Status", "daemon must survive 50 rapid subscribe/unsubscribe");
}

#[test]
fn e2e_unknown_request_type_handled() {
    let guard = spawn_daemon_guard();

    // Send a request with an unknown type tag.
    if let Ok(mut stream) = UnixStream::connect(&guard.socket) {
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
        writeln!(stream, r#"{{"type":"FakeNonExistentCommand","payload":{{}}}}"#).ok();
        let mut buf = String::new();
        let _ = BufReader::new(stream).read_line(&mut buf);
        // Daemon may close the connection or return an error — both are fine.
        // What matters is it doesn't crash.
    }

    // Daemon must still serve valid requests.
    let resp = send_request(&guard.socket, r#"{"type":"GetVersion"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "VersionInfo", "daemon must survive unknown request type");
}

#[test]
fn e2e_concurrent_set_profile_race() {
    let guard = spawn_daemon_guard();
    let socket = guard.socket.clone();

    let profiles = ["aggressive-root", "safe-root", "balanced-root", "aggressive-root", "safe-root"];
    let handles: Vec<_> = profiles
        .iter()
        .enumerate()
        .map(|(i, profile)| {
            let s = socket.clone();
            let p = *profile;
            std::thread::spawn(move || {
                let json = format!(
                    r#"{{"type":"SetProfile","payload":{{"profile":"{p}","ttl_minutes":null}}}}"#
                );
                // Slight stagger to increase race probability.
                std::thread::sleep(Duration::from_millis(i as u64 * 5));
                try_send_request(&s, &json)
            })
        })
        .collect();

    for h in handles {
        h.join().ok(); // Ignore individual results — we care about post-state.
    }

    // After concurrent mutations, GetStatus must return a valid, parseable profile.
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Status", "daemon must survive concurrent SetProfile race");
    let profile = v["payload"]["profile"].as_str().unwrap_or("");
    assert!(!profile.is_empty(), "profile must not be empty after concurrent writes");
}

#[test]
fn e2e_client_disconnect_survives() {
    let guard = spawn_daemon_guard();

    // Abruptly disconnect 20 subscriptions without reading.
    for _ in 0..20 {
        if let Ok(mut stream) = UnixStream::connect(&guard.socket) {
            stream.set_write_timeout(Some(Duration::from_millis(100))).ok();
            let _ = writeln!(stream, r#"{{"type":"Subscribe"}}"#);
            // Drop → abrupt disconnect while daemon is waiting to push.
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    std::thread::sleep(Duration::from_millis(500));

    // Daemon must still run after 20 broken-pipe events.
    let resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Status", "daemon must survive 20 abrupt client disconnects");
}

#[test]
fn e2e_metrics_bounded_after_stress() {
    let guard = spawn_daemon_guard();

    // Wait for at least 3 cycles.
    let pushes = subscribe_collect(&guard.socket, 3, CYCLE_TIMEOUT);
    assert!(pushes.len() >= 2, "need ≥2 cycle pushes to verify metrics");

    let resp = send_request(&guard.socket, r#"{"type":"GetMetrics"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Metrics", "expected Metrics response, got: {v}");

    let m = &v["payload"];
    let cycle_count = m["cycles"].as_u64().unwrap_or(0);
    assert!(cycle_count >= 2, "cycles must be ≥2 after waiting, got {cycle_count}");

    // Counters must be non-negative (JSON numbers, present in the payload).
    for field in &["freezes_applied", "throttles_applied", "boosts_applied", "unfreezes_applied"] {
        let val = m[field].as_u64();
        assert!(
            val.is_some(),
            "RuntimeMetrics must contain '{field}' as a non-negative integer"
        );
    }

    // In dry-run non-root mode, no real freezes happen but bookkeeping may record them.
    // The key invariant: no counter should be astronomically wrong.
    let freezes = m["freezes_applied"].as_u64().unwrap_or(0);
    assert!(
        freezes < 10_000,
        "freezes_applied={freezes} is suspiciously large for a dry-run test"
    );
}

// ══════════════════════════════════════════════════════════════════════════════
// LAYER 4 — QUANTITATIVE SCORE TEST
// Composite fitness function for evolutionary improvement.
// Score = (cycles_in_10s × 20) + (socket_flood_ok × 3) + (profile_change_verified × 30)
//       + (health_clean × 20)
// Minimum passing threshold: 100 points.
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn e2e_score() {
    let guard = spawn_daemon_guard();

    // ── Component 1: Throughput — count cycles completed in 10 seconds ──────
    // Subscribe to push notifications: each push = one completed cycle.
    // We collect for exactly 10 seconds.
    let throughput_start = Instant::now();
    let throughput_window = Duration::from_secs(10);

    // We collect up to 40 pushes, bounded by the 10s window.
    let mut cycle_stream =
        UnixStream::connect(&guard.socket).expect("connect for throughput measurement");
    cycle_stream.set_read_timeout(Some(Duration::from_millis(500))).unwrap();
    cycle_stream.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
    writeln!(cycle_stream, r#"{{"type":"Subscribe"}}"#).expect("write subscribe");

    let mut reader = BufReader::new(cycle_stream);
    // Consume ack line.
    let mut ack = String::new();
    let _ = reader.read_line(&mut ack);

    let mut cycles_in_10s: u32 = 0;
    while throughput_start.elapsed() < throughput_window {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
            Ok(_) => {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if v["type"] == "Push" || v.get("payload").is_some() {
                        cycles_in_10s += 1;
                    }
                }
            }
        }
    }
    drop(reader);

    // ── Component 2: Concurrency — how many of 32 concurrent succeed ─────────
    let socket_c2 = guard.socket.clone();
    let handles: Vec<_> = (0..32)
        .map(|_| {
            let s = socket_c2.clone();
            std::thread::spawn(move || try_send_request(&s, r#"{"type":"GetStatus"}"#))
        })
        .collect();

    let socket_flood_ok: u32 = handles
        .into_iter()
        .filter_map(|h| h.join().ok().flatten())
        .filter(|resp| {
            serde_json::from_str::<serde_json::Value>(resp)
                .map(|v| v["type"] == "Status")
                .unwrap_or(false)
        })
        .count() as u32;

    // ── Component 3: Correctness — profile change visible after SetProfile ────
    let set_resp = send_request(
        &guard.socket,
        r#"{"type":"SetProfile","payload":{"profile":"aggressive-root","ttl_minutes":null}}"#,
    );
    let set_v = serde_json::from_str::<serde_json::Value>(&set_resp).unwrap_or_default();

    let profile_change_verified: u32 = if set_v["type"] == "Ok" {
        // Verify state is reflected immediately via GetStatus.
        let st_resp = send_request(&guard.socket, r#"{"type":"GetStatus"}"#);
        let st_v = serde_json::from_str::<serde_json::Value>(&st_resp).unwrap_or_default();
        let effective = st_v["payload"]["effective_profile"].as_str().unwrap_or("");
        let override_active = st_v["payload"]["override_active"].as_bool().unwrap_or(false);
        if effective.contains("aggressive") || override_active {
            1
        } else {
            0
        }
    } else {
        0
    };

    // ── Component 4: Stability — health=full after 3 cycles ──────────────────
    // We already spent 10s so cycles have run; check health directly.
    let health_resp = send_request(&guard.socket, r#"{"type":"GetHealth"}"#);
    let health_v = serde_json::from_str::<serde_json::Value>(&health_resp).unwrap_or_default();
    let cb_state = health_v["payload"]["circuit_breaker"].as_str().unwrap_or("");
    let op_mode = health_v["payload"]["operation_mode"].as_str().unwrap_or("");
    let health_clean: u32 = if cb_state == "closed" && op_mode == "full" { 1 } else { 0 };

    // ── Composite score ───────────────────────────────────────────────────────
    let score = (cycles_in_10s * 20)
        + (socket_flood_ok * 3)
        + (profile_change_verified * 30)
        + (health_clean * 20);

    // Write score to a temp file so it can be read by the evolve loop.
    let score_line = format!(
        "e2e_score: {} (cycles_in_10s={} socket_flood_ok={}/32 profile_verified={} health_clean={})\n",
        score, cycles_in_10s, socket_flood_ok, profile_change_verified, health_clean
    );
    eprintln!("{}", score_line.trim());
    let _ = std::fs::write("/tmp/apollo_e2e_score.txt", &score_line);

    assert!(
        score >= 100,
        "e2e_score={score} is below minimum threshold 100 \
         (cycles_in_10s={cycles_in_10s}, socket_flood_ok={socket_flood_ok}/32, \
         profile_change_verified={profile_change_verified}, health_clean={health_clean})"
    );
}

#[test]
fn e2e_health_report_no_degradation() {
    let guard = spawn_daemon_guard();

    // Let 2+ cycles complete cleanly.
    let _ = subscribe_collect(&guard.socket, 2, CYCLE_TIMEOUT);

    let resp = send_request(&guard.socket, r#"{"type":"GetHealth"}"#);
    let v = parse(&resp);
    assert_eq!(v["type"], "Health", "expected Health response, got: {v}");

    let h = &v["payload"];
    // Circuit breaker should be closed (no failures in dry-run).
    let cb_state = h["circuit_breaker"].as_str().unwrap_or("unknown");
    assert_eq!(
        cb_state, "closed",
        "circuit breaker must be 'closed' after clean dry-run cycles"
    );
    // Operation mode should be 'full' (no degradation triggered).
    let op_mode = h["operation_mode"].as_str().unwrap_or("unknown");
    assert_eq!(
        op_mode, "full",
        "operation mode must be 'full' after clean dry-run cycles"
    );
}
