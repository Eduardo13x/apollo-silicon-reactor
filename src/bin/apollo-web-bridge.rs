//! Unprivileged Chromium Native Messaging host for bounded WebFlow events.

use std::io::{self, BufReader, BufWriter};

use apollo_engine::engine::webflow_native::{
    process_bridge_payload, read_native_frame, send_event_to_context_agent, write_native_frame,
    BridgeAck, EventTokenBucket,
};
use apollo_engine::engine::webflow_types::webflow_monotonic_ms;

fn main() -> io::Result<()> {
    let mut input = BufReader::new(io::stdin().lock());
    let mut output = BufWriter::new(io::stdout().lock());
    let mut bucket = EventTokenBucket::new(webflow_monotonic_ms());

    while let Some(payload) = read_native_frame(&mut input)? {
        let ack = process_bridge_payload(&payload, &mut bucket, webflow_monotonic_ms(), |event| {
            match send_event_to_context_agent(&event)? {
                ack if ack == BridgeAck::ACCEPTED => Ok(()),
                _ => Err(io::Error::new(
                    io::ErrorKind::ConnectionRefused,
                    "context agent rejected WebFlow event",
                )),
            }
        });
        let response = serde_json::to_vec(&ack)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        write_native_frame(&mut output, &response)?;
    }
    Ok(())
}
