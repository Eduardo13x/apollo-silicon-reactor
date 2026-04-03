//! Integration tests for daemon initialization sequence.
//!
//! Validates startup invariants WITHOUT requiring root privileges,
//! a live daemon, or real Unix sockets.

use std::path::Path;

use apollo_optimizer::engine::capabilities::detect_capabilities;
use apollo_optimizer::engine::daemon_helpers::{
    journal_path, kill_switch_path, learned_state_path, socket_path,
};
use apollo_optimizer::engine::learned_state::LearnedState;

// ── Test 1: detect_capabilities returns coherent report ─────────────────────

#[test]
fn detect_capabilities_returns_coherent_report() {
    let cap = detect_capabilities();

    // On macOS, taskpolicy and sysctl are always available.
    #[cfg(target_os = "macos")]
    {
        assert!(cap.can_taskpolicy, "taskpolicy must be available on macOS");
        assert!(cap.can_sysctl, "sysctl must be available on macOS");
        // mdutil and tmutil binaries exist on every macOS install.
        assert!(cap.can_mdutil, "/usr/bin/mdutil must exist on macOS");
        assert!(cap.can_tmutil, "/usr/bin/tmutil must exist on macOS");
    }

    // Non-root tests: memorystatus requires root.
    if !cap.is_root {
        assert!(
            !cap.can_memorystatus,
            "can_memorystatus must be false for non-root"
        );
    }

    // unavailable list must NOT contain capabilities that are reported as available.
    if cap.can_taskpolicy {
        assert!(
            !cap.unavailable.contains(&"taskpolicy".to_string()),
            "unavailable must not list available capabilities"
        );
    }
    if cap.can_sysctl {
        assert!(!cap.unavailable.contains(&"sysctl".to_string()));
    }
}

// ── Test 2: LearnedState loads gracefully from nonexistent path ─────────────

#[test]
fn learned_state_load_nonexistent_returns_none() {
    // First boot scenario: no persisted state file exists.
    let bogus = Path::new("/tmp/apollo-test-nonexistent-learned-state.json");
    // Ensure it truly does not exist.
    let _ = std::fs::remove_file(bogus);

    let result = LearnedState::load(bogus);
    assert!(
        result.is_none(),
        "LearnedState::load on missing file must return None (cold start)"
    );
}

// ── Test 3: Socket path is deterministic for non-root ───────────────────────

#[test]
fn socket_path_is_nonempty_and_deterministic() {
    let p1 = socket_path();
    let p2 = socket_path();

    assert!(!p1.is_empty(), "socket_path must not be empty");
    assert_eq!(p1, p2, "socket_path must be deterministic across calls");

    // Non-root always gets /tmp path.
    let is_root = unsafe { libc::geteuid() == 0 };
    if !is_root {
        assert_eq!(
            p1, "/tmp/apollo-optimizer.sock",
            "non-root socket path must be /tmp/apollo-optimizer.sock"
        );
    }
}

// ── Test 4: Journal path is non-empty and deterministic ─────────────────────

#[test]
fn journal_path_is_nonempty_and_deterministic() {
    let p1 = journal_path();
    let p2 = journal_path();

    assert!(!p1.is_empty(), "journal_path must not be empty");
    assert_eq!(p1, p2, "journal_path must be deterministic across calls");

    // Verify it ends with the expected filename.
    assert!(
        p1.ends_with("journal.jsonl"),
        "journal_path must end with journal.jsonl, got: {}",
        p1
    );

    // Other startup paths must also be non-empty and consistent.
    let kp = kill_switch_path();
    assert!(!kp.is_empty(), "kill_switch_path must not be empty");

    let lsp = learned_state_path();
    assert!(!lsp.is_empty(), "learned_state_path must not be empty");
    assert!(
        lsp.ends_with("learned_state.json"),
        "learned_state_path must end with learned_state.json, got: {}",
        lsp
    );
}

// ── Test 5: LearnedState roundtrip persist/load + validate ──────────────────

#[test]
fn learned_state_roundtrip_and_validate() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().join("learned_state.json");

    // Create a minimal LearnedState via JSON deserialization with all defaults.
    // Since all fields have #[serde(default)], an empty JSON object yields cold-start defaults.
    let mut state: LearnedState =
        serde_json::from_str("{}").expect("empty JSON must deserialize to LearnedState");

    assert_eq!(state.version, 1, "default version must be 1");
    assert!(
        state.signal_intelligence.is_none(),
        "cold-start signal_intelligence must be None"
    );
    assert!(
        state.outcome_tracker.is_none(),
        "cold-start outcome_tracker must be None"
    );
    assert_eq!(
        state.persist_generations, 0,
        "cold-start persist_generations must be 0"
    );

    // validate() must not panic on cold-start state.
    state.validate();

    // Persist and reload — roundtrip.
    state.persist(&path);
    let reloaded = LearnedState::load(&path).expect("must reload persisted state");

    assert_eq!(reloaded.version, state.version);
    assert_eq!(reloaded.persist_generations, state.persist_generations);
    assert!(reloaded.signal_intelligence.is_none());
    assert!(reloaded.outcome_tracker.is_none());
}
