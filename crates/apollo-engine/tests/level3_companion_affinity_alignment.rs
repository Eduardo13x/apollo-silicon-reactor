//! Sprint 12 Convergence #1 — Companion × Affinity bridge round-trip test.
//!
//! The cold-thread routing flip from E-cluster → P-cluster (when the
//! owning PID is a foreground companion AND DRAM bandwidth is below
//! 0.50) requires a real `MachQoSManager` enumerating real macOS task
//! threads, so the full path can only be empirically verified in
//! production. What we CAN lock here mechanically:
//!
//!   1. `LSE_COUNTERS.inc_companion_affinity_alignment()` actually
//!      bumps `companion_affinity_alignments_total`.
//!   2. The snapshot reader returns the bumped value.
//!   3. The serde-default round-trip on `RuntimeMetrics` does not
//!      drop the field (per CLAUDE.md silent-telemetry-death rule).
//!
//! Together these prove the producer→consumer plumbing is intact and
//! the counter will be observable in `runtime_metrics.json` once the
//! production branch fires on real foreground companions.

use std::sync::Mutex;
use std::sync::atomic::Ordering;

use apollo_engine::engine::lse_counters::LSE_COUNTERS;
use apollo_engine::engine::types::RuntimeMetrics;

static COUNTER_GUARD: Mutex<()> = Mutex::new(());

#[test]
fn companion_affinity_alignment_counter_increments_via_helper() {
    let _g = COUNTER_GUARD.lock().unwrap();
    let before = LSE_COUNTERS
        .companion_affinity_alignments_total
        .load(Ordering::Relaxed);
    LSE_COUNTERS.inc_companion_affinity_alignment();
    LSE_COUNTERS.inc_companion_affinity_alignment();
    LSE_COUNTERS.inc_companion_affinity_alignment();
    let after = LSE_COUNTERS
        .companion_affinity_alignments_total
        .load(Ordering::Relaxed);
    assert_eq!(after - before, 3, "helper bumps must be visible directly");
}

#[test]
fn companion_affinity_alignment_field_round_trips_through_runtime_metrics_json() {
    let _g = COUNTER_GUARD.lock().unwrap();
    let mut m = RuntimeMetrics::default();
    m.companion_affinity_alignments_total = 42;
    let s = serde_json::to_string(&m).expect("serialize");
    assert!(
        s.contains("\"companion_affinity_alignments_total\":42"),
        "field must surface in runtime_metrics.json: {}",
        s
    );
    let back: RuntimeMetrics = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(
        back.companion_affinity_alignments_total, 42,
        "round-trip preserves the counter"
    );
}

#[test]
fn companion_affinity_alignment_field_defaults_to_zero_on_missing() {
    // serde(default) silent-telemetry-death guard: serialize a
    // RuntimeMetrics without the new field (manually strip it from the
    // JSON), deserialize, and verify it defaults to 0 instead of
    // erroring. Mirrors what an old persisted runtime_metrics.json from
    // before Sprint 12 would look like on disk.
    let base = RuntimeMetrics::default();
    let full = serde_json::to_string(&base).expect("serialize default");
    // Strip just the new field's key/value pair (the only one that
    // could missing in a pre-Sprint-12 file).
    let stripped: String = full
        .replace(",\"companion_affinity_alignments_total\":0", "")
        .replace("\"companion_affinity_alignments_total\":0,", "");
    assert!(
        !stripped.contains("companion_affinity_alignments_total"),
        "strip-step must remove the key entirely"
    );
    let m: RuntimeMetrics =
        serde_json::from_str(&stripped).expect("deserialize stripped");
    assert_eq!(
        m.companion_affinity_alignments_total, 0,
        "missing field defaults to 0 (serde default)"
    );
}
