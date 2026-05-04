//! BlockedActionJournal — dark-matter observability for actions gated out.
//!
//! Every safety/policy gate that prevents an action from executing emits a
//! `BlockedActionEvent` here. Downstream learning (OutcomeTracker,
//! RL reward) can then correlate blocked decisions with t+30s/t+120s outcomes
//! (e.g. OOM, thrashing spike) to discover gates that are too conservative.
//!
//! Candidate emitters (wired in a follow-up commit):
//!   • `user_context::UserContext::freeze_protected` — when it returns `true`,
//!     the caller should emit `BlockerKind::UserContextAssertion` (or
//!     `HardProtection` if `call_in_progress`). `freeze_protected` itself is a
//!     pure predicate with no journal handle, so emission stays at the call site.
//!   • `execute_actions` per-PID guards — PidInvalid, BudgetExhausted, thermal
//!     and interrupt phases.
//!   • `decide_actions` safety filters — ForegroundFamily, HardProtection.
//!
//! [Bengio 2013] Counterfactual reasoning requires observing the COUNTERFACTUAL,
//! not just the taken action. [Nygard 2018 §8.5] Adaptive capacity limits.

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::OnceLock;

/// Liveness counters for the shadow writer thread. If `writes_failed` grows
/// without `writes_succeeded` growing, the writer is dead or the disk is full
/// — caller can alert via RuntimeMetrics. [Nygard 2018 §9] observability must
/// observe itself.
static SHADOW_WRITES_OK: AtomicU64 = AtomicU64::new(0);
static SHADOW_WRITES_FAILED: AtomicU64 = AtomicU64::new(0);

pub fn shadow_writes_succeeded() -> u64 {
    SHADOW_WRITES_OK.load(Ordering::Relaxed)
}
pub fn shadow_writes_failed() -> u64 {
    SHADOW_WRITES_FAILED.load(Ordering::Relaxed)
}

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::engine::audit_types::PolicyDecisionTrace;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BlockerKind {
    /// Hard-protected name (classify_protection Unconditional).
    HardProtection,
    /// User-context sleep assertion, call, or recently-active.
    UserContextAssertion,
    /// Foreground family or foreground-app name match.
    ForegroundFamily,
    /// Thermal emergency or resource-interrupt phase.
    ThermalOrInterrupt,
    /// Per-cycle action budget exhausted.
    BudgetExhausted,
    /// PID validation failed (dead or recycled).
    PidInvalid,
    /// Epistemic uncertainty too high.
    EpistemicHigh,
    /// Other — free-form reason.
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockedActionEvent {
    pub t: DateTime<Utc>,
    pub action_kind: String, // "Freeze" / "Throttle" / "Boost"
    pub target_name: String,
    pub target_pid: Option<u32>,
    pub blocker: BlockerKind,
    /// Snapshot of relevant pressure indicators at block time.
    pub pressure: f64,
    pub swap_gb: f64,
    pub thrashing_score: f64,
    pub p_oom_30s: Option<f64>,
}

impl BlockedActionEvent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        action_kind: impl Into<String>,
        target_name: impl Into<String>,
        target_pid: Option<u32>,
        blocker: BlockerKind,
        pressure: f64,
        swap_gb: f64,
        thrashing_score: f64,
        p_oom_30s: Option<f64>,
    ) -> Self {
        Self {
            t: Utc::now(),
            action_kind: action_kind.into(),
            target_name: target_name.into(),
            target_pid,
            blocker,
            pressure,
            swap_gb,
            thrashing_score,
            p_oom_30s,
        }
    }
}

/// Append a single event as one JSON line to `path`. Creates the file if missing.
/// On failure returns io::Error — callers MUST decide whether to swallow (hot path)
/// or propagate. This is a best-effort observability primitive; a failed write must
/// never abort the daemon's main loop.
///
/// **Synchronous** — do NOT call from the daemon hot path. Use `emit_async` instead.
/// This function remains for tests and offline tools.
pub fn emit(path: &Path, event: &BlockedActionEvent) -> io::Result<()> {
    let line = serde_json::to_string(event)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{}", line)
}

/// Max shadow journal size before rotation (10 MB — same policy as journal.rs).
const MAX_SHADOW_BYTES: u64 = 10 * 1024 * 1024;

/// Rotate the shadow journal once per call if it exceeds the size cap.
/// Non-atomic by design: the writer thread owns it, so there's no concurrent
/// writer to race with. Old `.1` is clobbered; we keep only the most recent
/// rotation to bound disk usage at ~20 MB total. [Same policy as journal.rs
/// rotation_when_file_exceeds_10mb to prevent the 8.6GB TelemetryLogger-style
/// SSD saturation that froze the system 2026-04-09.]
fn rotate_if_needed(path: &Path) {
    if let Ok(meta) = std::fs::symlink_metadata(path) {
        if !meta.file_type().is_symlink() && meta.len() > MAX_SHADOW_BYTES {
            let rotated = path.with_extension("jsonl.1");
            let _ = std::fs::remove_file(&rotated);
            let _ = std::fs::rename(path, &rotated);
        }
    }
}

/// Background writer thread — mirrors the `apollo-frozen-writer` pattern
/// (daemon_helpers.rs:439) to keep filesystem I/O off the daemon hot path.
///
/// Per [Nygard 2018 §7] the daemon's 10ms per-cycle budget must not absorb
/// filesystem tail latency. Shadow events are serialized on the caller thread
/// (cheap — a few µs) then shipped via unbounded mpsc to a dedicated writer
/// thread. Send is non-blocking and never fails under normal operation; if the
/// writer has panicked, the send silently drops (best-effort by design).
fn writer_tx() -> &'static Sender<(PathBuf, String)> {
    static TX: OnceLock<Sender<(PathBuf, String)>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<(PathBuf, String)>();
        std::thread::Builder::new()
            .name("apollo-shadow-writer".to_string())
            .spawn(move || {
                while let Ok((path, line)) = rx.recv() {
                    // Rotate BEFORE opening — bounds disk usage at ~2 × 10 MB.
                    rotate_if_needed(&path);
                    // Best-effort: if open or write fails, bump fail counter
                    // and drop. We do NOT block or retry — observability must
                    // never stall. Callers monitor liveness via the exposed
                    // succeeded/failed counters.
                    match OpenOptions::new().create(true).append(true).open(&path) {
                        Ok(mut f) => match writeln!(f, "{}", line) {
                            Ok(()) => { SHADOW_WRITES_OK.fetch_add(1, Ordering::Relaxed); }
                            Err(_) => { SHADOW_WRITES_FAILED.fetch_add(1, Ordering::Relaxed); }
                        },
                        Err(_) => { SHADOW_WRITES_FAILED.fetch_add(1, Ordering::Relaxed); }
                    }
                }
            })
            .expect("failed to spawn apollo-shadow-writer");
        tx
    })
}

/// Async, non-blocking emit for hot-path use. Serializes the event on the
/// caller thread (~µs) and hands it to a background writer via mpsc. Returns
/// immediately. Errors during serialization are swallowed — observability must
/// never abort the daemon.
pub fn emit_async(path: PathBuf, event: &BlockedActionEvent) {
    let Ok(line) = serde_json::to_string(event) else {
        SHADOW_WRITES_FAILED.fetch_add(1, Ordering::Relaxed);
        return;
    };
    // Send fails iff writer thread panicked. Bump fail counter — callers
    // detect dead writer via SHADOW_WRITES_FAILED climbing without _OK.
    if writer_tx().send((path, line)).is_err() {
        SHADOW_WRITES_FAILED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Async, non-blocking emit for policy audit traces.
pub fn emit_audit_async(path: PathBuf, trace: &PolicyDecisionTrace) {
    let Ok(line) = serde_json::to_string(trace) else {
        SHADOW_WRITES_FAILED.fetch_add(1, Ordering::Relaxed);
        return;
    };
    if writer_tx().send((path, line)).is_err() {
        SHADOW_WRITES_FAILED.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_constructs_and_serializes() {
        let e = BlockedActionEvent::new(
            "Freeze",
            "firefox",
            Some(1234),
            BlockerKind::UserContextAssertion,
            0.66,
            1.07,
            29_599.0,
            Some(0.40),
        );
        let json = serde_json::to_string(&e).expect("serializes");
        assert!(json.contains("\"UserContextAssertion\""));
        assert!(json.contains("\"firefox\""));
        let back: BlockedActionEvent = serde_json::from_str(&json).expect("roundtrips");
        assert_eq!(back.blocker, BlockerKind::UserContextAssertion);
        assert_eq!(back.target_pid, Some(1234));
    }

    #[test]
    fn blocker_kind_other_holds_reason() {
        let b = BlockerKind::Other("custom-gate".to_string());
        let json = serde_json::to_string(&b).unwrap();
        assert!(json.contains("custom-gate"));
    }

    #[test]
    fn emit_async_is_nonblocking_and_writes_eventually() {
        // Unique path per-test to avoid cross-test pollution.
        let mut p = std::env::temp_dir();
        p.push(format!(
            "apollo-blocked-journal-emit-async-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        let _ = std::fs::remove_file(&p);

        let e = BlockedActionEvent::new(
            "Freeze",
            "bg-daemon",
            Some(9999),
            BlockerKind::UserContextAssertion,
            0.70,
            1.0,
            9_500.0,
            Some(0.35),
        );

        // Non-blocking: this must return in microseconds.
        let start = std::time::Instant::now();
        emit_async(p.clone(), &e);
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(5),
            "emit_async blocked hot path for {:?}",
            elapsed
        );

        // Poll up to 2s for the async writer to flush.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut contents = String::new();
        loop {
            if let Ok(s) = std::fs::read_to_string(&p) {
                if !s.trim().is_empty() {
                    contents = s;
                    break;
                }
            }
            if std::time::Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 1, "async writer should have flushed exactly one line");
        let back: BlockedActionEvent =
            serde_json::from_str(lines[0]).expect("parses as BlockedActionEvent");
        assert_eq!(back.target_pid, Some(9999));

        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rotate_if_needed_rotates_file_over_cap() {
        use std::io::Write as _;
        let mut p = std::env::temp_dir();
        p.push(format!(
            "apollo-shadow-rotate-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        let rotated = p.with_extension("jsonl.1");
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&rotated);

        // Write just over the cap.
        {
            let mut f = std::fs::File::create(&p).expect("create");
            let chunk = vec![b'x'; 1024 * 1024]; // 1 MB
            for _ in 0..11 {
                f.write_all(&chunk).expect("write chunk");
            }
        }
        let size_before = std::fs::metadata(&p).unwrap().len();
        assert!(size_before > MAX_SHADOW_BYTES);

        rotate_if_needed(&p);

        // Primary file is gone, rotated file exists.
        assert!(!p.exists(), "primary should have been renamed");
        assert!(rotated.exists(), "rotated file should exist");

        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&rotated);
    }

    #[test]
    fn rotate_if_needed_noop_below_cap() {
        use std::io::Write as _;
        let mut p = std::env::temp_dir();
        p.push(format!(
            "apollo-shadow-norotate-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        let _ = std::fs::remove_file(&p);

        std::fs::File::create(&p)
            .expect("create")
            .write_all(b"small")
            .expect("write");

        rotate_if_needed(&p);

        assert!(p.exists(), "small file must NOT rotate");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn emit_appends_one_jsonl_line() {
        use std::io::Read as _;

        // Per-test unique path under std::env::temp_dir() — no tempfile dep.
        let mut p = std::env::temp_dir();
        p.push(format!(
            "apollo-blocked-journal-emit-{}-{}.jsonl",
            std::process::id(),
            // nanos to avoid collision if multiple tests run in same PID
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        // Clean up any stale file from prior runs.
        let _ = std::fs::remove_file(&p);

        let e1 = BlockedActionEvent::new(
            "Freeze",
            "firefox",
            Some(1234),
            BlockerKind::UserContextAssertion,
            0.66,
            1.07,
            29_599.0,
            Some(0.40),
        );
        let e2 = BlockedActionEvent::new(
            "Throttle",
            "chrome",
            Some(2345),
            BlockerKind::HardProtection,
            0.80,
            2.10,
            15_000.0,
            None,
        );

        emit(&p, &e1).expect("first emit");
        emit(&p, &e2).expect("second emit");

        let mut contents = String::new();
        std::fs::File::open(&p)
            .expect("file exists")
            .read_to_string(&mut contents)
            .expect("read");
        let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2, "expected 2 non-empty lines, got {}", lines.len());
        for l in &lines {
            let _: BlockedActionEvent =
                serde_json::from_str(l).expect("each line parses as BlockedActionEvent");
        }

        // Clean up.
        let _ = std::fs::remove_file(&p);
    }
}
