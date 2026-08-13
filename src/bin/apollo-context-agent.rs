//! User-session context publisher.
//!
//! The sampler emits only bounded numeric aggregates. It keeps no raw window,
//! process, audio, or frame data and does not write a state file.

use std::thread;
use std::time::Duration;

use apollo_engine::engine::context_agent::{send_once, ContextCollector};

const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

fn main() {
    let mut collector = ContextCollector::new();
    loop {
        let _ = send_once(&mut collector);
        thread::sleep(SAMPLE_INTERVAL);
    }
}
