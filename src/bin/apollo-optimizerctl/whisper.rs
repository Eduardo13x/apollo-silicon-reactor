//! `apollo-optimizerctl whisper` — single-glyph status snippet for prompt
//! integration. No daemon socket round-trip; reads
//! `runtime_metrics.json` directly so the prompt overhead is one mmap +
//! one JSON parse (~1–3 ms on warm cache).
//!
//! Output contract:
//! - At most one short line; nothing on stale (>5 s) snapshot.
//! - Without `--always-on`, silent on a healthy idle system.
//! - With `--always-on`, prints `·` to mark "Apollo healthy".
//!
//! Sprint patch (2026-06-05). Glyph thresholds chosen to mirror the
//! existing TUI dashboard pressure bands:
//! - `▰` pressure ≥ 0.75 — critical
//! - `▱` pressure ≥ 0.55 — warning
//! - `▴` thermal == "Critical" (and pressure healthy)
//! - `·` always-on healthy
//!
//! Optional swap-pressure suffix: `swXG` when swap_mb > 1500.
//!
//! References:
//! - Dean & Barroso 2013 — keep latency-critical readers off the daemon
//!   hot path so RPC queuing cannot starve the prompt.

use std::fs;
use std::time::{Duration, SystemTime};

const RUNTIME_METRICS_PATHS: &[&str] = &[
    "/var/lib/apollo/runtime_metrics.json",
    "/tmp/apollo-runtime_metrics.json",
];
const STALE_THRESHOLD: Duration = Duration::from_secs(5);

/// Decide the glyph + optional suffix based on a parsed metrics JSON.
/// Returns `None` for "say nothing" cases (stale, unhealthy parse,
/// healthy + not always_on).
pub fn decide_line(
    json: &serde_json::Value,
    always_on: bool,
    age: Duration,
) -> Option<String> {
    if age > STALE_THRESHOLD {
        return None;
    }
    let pressure = json
        .get("memory_pressure")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let thermal = json
        .get("thermal_state")
        .and_then(|v| v.as_str())
        .unwrap_or("Nominal");
    let swap_mb = json.get("swap_mb").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let glyph = if pressure >= 0.75 {
        "▰"
    } else if pressure >= 0.55 {
        "▱"
    } else if thermal == "Critical" {
        "▴"
    } else if always_on {
        "·"
    } else {
        return None;
    };

    let suffix = if swap_mb > 1500.0 {
        format!(" sw{:.0}G", swap_mb / 1024.0)
    } else {
        String::new()
    };
    Some(format!("{glyph}{suffix}"))
}

pub fn run(always_on: bool) -> anyhow::Result<()> {
    for path in RUNTIME_METRICS_PATHS {
        let meta = match fs::metadata(path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let age = SystemTime::now()
            .duration_since(mtime)
            .unwrap_or(Duration::ZERO);
        let raw = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if raw.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(line) = decide_line(&v, always_on, age) {
            println!("{line}");
        }
        return Ok(());
    }
    // No runtime_metrics file reachable — silent. The whisper contract is
    // "say nothing rather than say something wrong."
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(pressure: f64, thermal: &str, swap_mb: f64) -> serde_json::Value {
        serde_json::json!({
            "memory_pressure": pressure,
            "thermal_state": thermal,
            "swap_mb": swap_mb,
        })
    }

    #[test]
    fn stale_returns_none() {
        let v = metrics(0.95, "Critical", 4000.0);
        assert!(decide_line(&v, true, Duration::from_secs(10)).is_none());
    }

    #[test]
    fn critical_pressure_emits_filled_glyph() {
        let v = metrics(0.80, "Nominal", 0.0);
        let line = decide_line(&v, false, Duration::from_secs(0)).expect("critical line");
        assert!(line.starts_with("▰"), "got {line}");
    }

    #[test]
    fn warning_pressure_emits_outlined_glyph() {
        let v = metrics(0.60, "Nominal", 0.0);
        let line = decide_line(&v, false, Duration::from_secs(0)).expect("warning line");
        assert!(line.starts_with("▱"), "got {line}");
    }

    #[test]
    fn healthy_without_always_on_returns_none() {
        let v = metrics(0.20, "Nominal", 100.0);
        assert!(decide_line(&v, false, Duration::from_secs(1)).is_none());
    }

    #[test]
    fn healthy_with_always_on_emits_dot() {
        let v = metrics(0.20, "Nominal", 100.0);
        let line = decide_line(&v, true, Duration::from_secs(1)).expect("always-on line");
        assert!(line.starts_with("·"), "got {line}");
    }

    #[test]
    fn swap_suffix_present_when_large() {
        let v = metrics(0.80, "Nominal", 4096.0);
        let line = decide_line(&v, false, Duration::from_secs(0)).expect("swap suffix line");
        assert!(line.contains("sw"), "expected swap suffix, got {line}");
    }
}
