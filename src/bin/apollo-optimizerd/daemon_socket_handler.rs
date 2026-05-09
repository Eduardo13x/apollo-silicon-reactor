//! # Daemon Socket Handler — startup glue
//!
//! Spawns the control socket server thread and synchronously waits for bind
//! confirmation before the daemon enters its hot loop.
//!
//! Without this guard, a second daemon instance whose `bind()` failed would
//! silently run headless — no socket, no control plane, but actively mutating
//! `frozen_state.json` concurrently with the first instance. On confirmed bind
//! failure this function terminates the process via `std::process::exit(1)`.
//!
//! A 5-second timeout is applied to the bind confirmation; if it elapses
//! without a result, the daemon continues under a warning (best-effort).
//!
//! Extracted from `main.rs` per the V110 Strangler Fig plan. [Fowler 2004]
//!
//! Note: the underlying `run_socket_server_with_notify` (which performs the
//! probe-bind, re-bind, and client-thread fan-out) already lives in
//! `socket_handler.rs`. This module is pure startup glue.
//!
//! Corresponding paper:
//! - [Fowler 2004] Strangler Fig Application pattern.

use std::thread;
use std::time::Duration;

use apollo_engine::engine::daemon_state::SharedState;

use super::socket_handler;

/// Spawn the control socket server thread and block until bind is confirmed
/// (or the 5 s timeout elapses).
///
/// # Behavior
/// - On successful bind → returns; caller proceeds into the main loop.
/// - On bind failure → logs, prints diagnostic hints, and `exit(1)`s.
/// - On timeout → logs a warning and returns (best-effort degrade).
///
/// The spawned thread continues running the full socket server after bind
/// confirmation is sent; this function does not join it.
pub fn spawn_control_socket(state: &SharedState) {
    // Spawn socket server and wait for bind confirmation before entering main loop.
    // If bind fails (e.g., another instance already running), exit(1) immediately.
    // Without this guard, a second daemon instance would silently run headless —
    // no socket, no control, but actively mutating frozen_state and frozen_state.json
    // concurrently with the first instance.
    let socket_state = state.clone();
    let (bind_tx, bind_rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();
    thread::spawn(move || {
        socket_handler::run_socket_server_with_notify(socket_state, bind_tx);
    });
    match bind_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(Ok(())) => { /* bind succeeded, continue */ }
        Ok(Err(e)) => {
            tracing::error!(err = ?e, "FATAL: socket bind failed — another instance may be running");
            eprintln!("apollo-optimizerd: socket bind failed: {e}");
            eprintln!("  Is another instance already running?");
            eprintln!("  Check: pgrep apollo-optimizerd");
            std::process::exit(1);
        }
        Err(_timeout) => {
            tracing::warn!("socket bind confirmation timed out — continuing anyway");
        }
    }
}
