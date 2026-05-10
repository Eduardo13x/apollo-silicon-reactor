//! Main-loop messages from socket-handler threads.
//!
//! The IPC socket runs in spawned threads; CLI Purge requests cannot spawn
//! `purge` directly without races against the main loop's tick. A one-shot
//! mpsc channel from socket thread → main loop, with a reply channel
//! piggybacking on each message, keeps purge spawning serial.

use std::sync::{mpsc, Mutex, OnceLock};

use apollo_engine::engine::protocol::DaemonResponse;

pub enum MainLoopMsg {
    CliPurge {
        response_tx: mpsc::Sender<DaemonResponse>,
    },
}

/// Initialized exactly once at daemon startup; socket threads use this
/// to forward CLI Purge requests to the main loop.
pub static MAIN_LOOP_TX: OnceLock<Mutex<mpsc::Sender<MainLoopMsg>>> = OnceLock::new();
