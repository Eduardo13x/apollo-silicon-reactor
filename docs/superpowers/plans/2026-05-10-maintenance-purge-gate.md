# Maintenance Purge Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a daemon_maintenance_tick that fires `purge` under sustained-but-non-crisis pressure (0.65 ≤ raw < 0.85) plus an `apollo-optimizerctl purge` CLI command, replacing the user need to run `sudo purge` manually.

**Architecture:** New `MaintenanceState` + `SwapDeltaWindow` types in `apollo-engine`. New `daemon_maintenance_tick` orchestrator in `apollo-optimizerd` bin, inserted between `survival_tick` and `dispatch_tick`. New `DaemonRequest::Purge` IPC variant routed via mpsc channel from socket thread to main loop. 7 new lockfree counters traversing the full Sprint 3 telemetry sync chain. Asymmetric cooldown semantics: survival writes `last_any_purge_at` but never reads it; maintenance reads + writes.

**Tech Stack:** Rust 2021 edition, Cargo workspace, sysinfo, libc, std::sync::mpsc, serde_json, serde::SystemTime/Instant. Existing patterns: `daemon_survival_tick.rs` for purge spawn pattern, `planner.rs::TrendWindow` for VecDeque ring buffer, `lse_counters.rs` for atomic counter pattern, `daemon_state.rs::sync_from_lockfree` for telemetry flush, `protocol.rs::is_privileged` for IPC privilege gating.

**Spec:** `docs/superpowers/specs/2026-05-10-maintenance-purge-design.md` (patched 2026-05-10 per NotebookLM round 2 review).

**NotebookLM checkpoints:** Phases marked 🧠 must end with a NotebookLM peer-review query against notebook `8344b94c-a014-4803-abea-076a55753cfd` before commit. Phases not marked do not require NotebookLM gating but may be queried opportunistically.

**Branch:** Create `sprint5-mes0-maintenance-gate` from master before Task 1.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `crates/apollo-engine/src/engine/maintenance_state.rs` | NEW | `MaintenanceState` + `SwapDeltaWindow` structs |
| `crates/apollo-engine/src/engine/mod.rs` | MODIFY | `pub mod maintenance_state;` re-export |
| `crates/apollo-engine/src/engine/lse_counters.rs` | MODIFY | 7 atomic counters + 7 `MetricsSnapshot` fields + 7 `.load(Relaxed)` lines |
| `crates/apollo-engine/src/engine/types.rs` | MODIFY | 7 `RuntimeMetrics` fields with `#[serde(default)]` |
| `crates/apollo-engine/src/engine/daemon_state.rs` | MODIFY | 7 lines in `sync_from_lockfree` |
| `crates/apollo-engine/src/engine/learned_state.rs` | MODIFY | Persist `last_any_purge_at` + `last_cli_purge_at` in `LearnedState::collect/apply` |
| `crates/apollo-engine/src/engine/protocol.rs` | MODIFY | `DaemonRequest::Purge` + `DaemonResponse::PurgeResult` + `is_privileged` |
| `src/bin/apollo-optimizerd/daemon_maintenance_tick.rs` | NEW | `run_maintenance_tick` + `should_fire` + `SkipReason` |
| `src/bin/apollo-optimizerd/daemon_init.rs` | MODIFY | `DaemonSubsystems` gets `maintenance_state: MaintenanceState` |
| `src/bin/apollo-optimizerd/daemon_survival_tick.rs` | MODIFY | Take `&mut MaintenanceState` instead of `&mut Option<Instant>`. Survival writes shared timestamp but reads its own local Instant cooldown. |
| `src/bin/apollo-optimizerd/socket_handler.rs` | MODIFY | Handle `DaemonRequest::Purge` via mpsc channel to main loop |
| `src/bin/apollo-optimizerd/main.rs` | MODIFY | Insert `run_maintenance_tick` call, drain `MainLoopMsg::CliPurge` post-tick |
| `src/bin/apollo-optimizerctl/main.rs` | MODIFY | New `Purge` subcommand |
| `crates/apollo-engine/tests/level3_maintenance_purge.rs` | NEW | 3 integration tests |

---

### Task 1: Create branch + scaffold MaintenanceState skeleton

**Files:**
- Create: `crates/apollo-engine/src/engine/maintenance_state.rs`
- Modify: `crates/apollo-engine/src/engine/mod.rs`

- [ ] **Step 1: Create branch**

```bash
git switch -c sprint5-mes0-maintenance-gate
```

Expected: `Switched to a new branch 'sprint5-mes0-maintenance-gate'`

- [ ] **Step 2: Create `maintenance_state.rs` skeleton**

```rust
// crates/apollo-engine/src/engine/maintenance_state.rs
//! Maintenance Purge Gate state — opportunistic non-crisis purge orchestration.
//!
//! See docs/superpowers/specs/2026-05-10-maintenance-purge-design.md
//!
//! Asymmetric cooldown: survival_tick writes last_any_purge_at but does not
//! read it (survival is physical-crisis sovereign). maintenance_tick reads
//! and writes (yields to anything recent).

use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaintenanceState {
    #[serde(skip)]
    pub swap_delta_window: SwapDeltaWindow,

    #[serde(default)]
    pub last_any_purge_at: Option<SystemTime>,

    #[serde(default)]
    pub last_cli_purge_at: Option<SystemTime>,

    #[serde(skip)]
    pub last_wake_at: Option<Instant>,
}

impl MaintenanceState {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SwapDeltaWindow {
    samples: VecDeque<(SystemTime, f64)>,
}

impl SwapDeltaWindow {
    pub const CAP: usize = 45;

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
```

- [ ] **Step 3: Add `pub mod maintenance_state;` to engine mod**

Open `crates/apollo-engine/src/engine/mod.rs`. Find the alphabetically appropriate spot among existing `pub mod` declarations (likely after `pub mod learned_state;` and before `pub mod mpc;` or similar). Add:

```rust
pub mod maintenance_state;
```

- [ ] **Step 4: Verify it compiles**

```bash
cargo check -p apollo-engine 2>&1 | tail -10
```

Expected: 0 errors. May see "unused" warnings for new fields — those are fine, will be consumed in later tasks.

- [ ] **Step 5: Commit**

```bash
git add crates/apollo-engine/src/engine/maintenance_state.rs crates/apollo-engine/src/engine/mod.rs
git commit -m "feat(maintenance): scaffold MaintenanceState + SwapDeltaWindow types"
```

---

### Task 2: SwapDeltaWindow push + cap behavior (TDD)

**Files:**
- Modify: `crates/apollo-engine/src/engine/maintenance_state.rs`

- [ ] **Step 1: Write failing test**

Append at the end of `maintenance_state.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_delta_window_drops_oldest_at_capacity() {
        let mut w = SwapDeltaWindow::default();
        let t = SystemTime::now();
        for i in 0..50 {
            w.push(t + Duration::from_secs(i as u64), i as f64);
        }
        assert_eq!(w.len(), SwapDeltaWindow::CAP);
        // First sample retained should be sample index 5 (50 - 45)
        assert_eq!(w.samples.front().unwrap().1, 5.0);
    }
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test -p apollo-engine --lib engine::maintenance_state::tests::swap_delta_window_drops_oldest_at_capacity 2>&1 | tail -10
```

Expected: FAIL with "no method named `push` found".

- [ ] **Step 3: Implement `push`**

Add to `impl SwapDeltaWindow`:

```rust
pub fn push(&mut self, t: SystemTime, delta_bps: f64) {
    if self.samples.len() >= Self::CAP {
        self.samples.pop_front();
    }
    self.samples.push_back((t, delta_bps));
}
```

- [ ] **Step 4: Verify pass**

```bash
cargo test -p apollo-engine --lib engine::maintenance_state::tests::swap_delta_window_drops_oldest_at_capacity 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/apollo-engine/src/engine/maintenance_state.rs
git commit -m "feat(maintenance): SwapDeltaWindow::push with capacity drop"
```

---

### Task 3: SwapDeltaWindow sustained_below — full window true case (TDD)

**Files:**
- Modify: `crates/apollo-engine/src/engine/maintenance_state.rs`

- [ ] **Step 1: Write failing test**

Append in `mod tests`:

```rust
#[test]
fn swap_delta_window_sustained_below_with_full_window_returns_true() {
    let mut w = SwapDeltaWindow::default();
    let now = SystemTime::now();
    // 45 samples spanning 90s, all below threshold 256_000.0
    for i in 0..45 {
        let t = now - Duration::from_secs(90) + Duration::from_secs(i * 2);
        w.push(t, 50_000.0);
    }
    assert!(w.sustained_below(256_000.0, 90));
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test -p apollo-engine --lib engine::maintenance_state::tests::swap_delta_window_sustained_below_with_full_window_returns_true 2>&1 | tail -5
```

Expected: FAIL with "no method named `sustained_below`".

- [ ] **Step 3: Implement `sustained_below`**

Add to `impl SwapDeltaWindow`:

```rust
pub fn sustained_below(&self, threshold_bps: f64, secs: u64) -> bool {
    let cutoff = match SystemTime::now().checked_sub(Duration::from_secs(secs)) {
        Some(t) => t,
        None => return false,
    };

    let recent: Vec<&(SystemTime, f64)> = self
        .samples
        .iter()
        .filter(|(t, _)| *t >= cutoff)
        .collect();

    let min_samples = (secs / 2).max(1) as usize;
    if recent.len() < min_samples {
        return false;
    }

    recent.iter().all(|(_, bps)| *bps < threshold_bps)
}
```

- [ ] **Step 4: Verify pass**

```bash
cargo test -p apollo-engine --lib engine::maintenance_state::tests::swap_delta_window_sustained_below_with_full_window_returns_true 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/apollo-engine/src/engine/maintenance_state.rs
git commit -m "feat(maintenance): SwapDeltaWindow::sustained_below baseline"
```

---

### Task 4: SwapDeltaWindow edge cases — spike, empty, partial (TDD)

**Files:**
- Modify: `crates/apollo-engine/src/engine/maintenance_state.rs`

- [ ] **Step 1: Write three failing tests**

Append in `mod tests`:

```rust
#[test]
fn swap_delta_window_sustained_below_with_one_spike_returns_false() {
    let mut w = SwapDeltaWindow::default();
    let now = SystemTime::now();
    for i in 0..30 {
        let t = now - Duration::from_secs(90) + Duration::from_secs(i * 2);
        w.push(t, 50_000.0);
    }
    // One spike inside the window
    w.push(now - Duration::from_secs(10), 500_000.0);
    assert!(!w.sustained_below(256_000.0, 90));
}

#[test]
fn swap_delta_window_sustained_below_empty_returns_false() {
    let w = SwapDeltaWindow::default();
    assert!(!w.sustained_below(256_000.0, 90));
}

#[test]
fn swap_delta_window_sustained_below_partial_window_returns_false() {
    let mut w = SwapDeltaWindow::default();
    let now = SystemTime::now();
    // Only 10 samples = ~20s of history
    for i in 0..10 {
        let t = now - Duration::from_secs(20) + Duration::from_secs(i * 2);
        w.push(t, 50_000.0);
    }
    assert!(!w.sustained_below(256_000.0, 90));
}
```

- [ ] **Step 2: Verify all three pass without changes**

(They should — `sustained_below` already handles each case.)

```bash
cargo test -p apollo-engine --lib engine::maintenance_state 2>&1 | tail -10
```

Expected: All 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/apollo-engine/src/engine/maintenance_state.rs
git commit -m "test(maintenance): SwapDeltaWindow edge cases (spike, empty, partial)"
```

---

### Task 5: MaintenanceState timestamp methods (TDD)

**Files:**
- Modify: `crates/apollo-engine/src/engine/maintenance_state.rs`

- [ ] **Step 1: Write failing tests**

Append in `mod tests`:

```rust
#[test]
fn secs_since_any_purge_none_returns_max() {
    let s = MaintenanceState::default();
    assert_eq!(s.secs_since_any_purge(), u64::MAX);
}

#[test]
fn secs_since_any_purge_clock_backwards_returns_zero() {
    let mut s = MaintenanceState::default();
    s.last_any_purge_at = Some(SystemTime::now() + Duration::from_secs(60));
    assert_eq!(s.secs_since_any_purge(), 0);
}

#[test]
fn mark_cli_purged_updates_both_timestamps() {
    let mut s = MaintenanceState::default();
    s.mark_cli_purged();
    assert!(s.last_cli_purge_at.is_some());
    assert!(s.last_any_purge_at.is_some());
}

#[test]
fn mark_purged_only_updates_any_not_cli() {
    let mut s = MaintenanceState::default();
    s.mark_purged();
    assert!(s.last_any_purge_at.is_some());
    assert!(s.last_cli_purge_at.is_none());
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test -p apollo-engine --lib engine::maintenance_state 2>&1 | tail -10
```

Expected: 4 tests FAIL — "no method named `secs_since_any_purge` / `mark_cli_purged` / `mark_purged`".

- [ ] **Step 3: Implement methods**

Add to `impl MaintenanceState`:

```rust
pub fn secs_since_any_purge(&self) -> u64 {
    match self.last_any_purge_at {
        None => u64::MAX,
        Some(t) => SystemTime::now()
            .duration_since(t)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    }
}

pub fn secs_since_cli_purge(&self) -> u64 {
    match self.last_cli_purge_at {
        None => u64::MAX,
        Some(t) => SystemTime::now()
            .duration_since(t)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    }
}

pub fn secs_since_wake(&self) -> u64 {
    match self.last_wake_at {
        None => u64::MAX,
        Some(t) => t.elapsed().as_secs(),
    }
}

pub fn push_swap_delta(&mut self, delta_bps: f64) {
    self.swap_delta_window.push(SystemTime::now(), delta_bps);
}

pub fn mark_purged(&mut self) {
    self.last_any_purge_at = Some(SystemTime::now());
}

pub fn mark_cli_purged(&mut self) {
    let now = SystemTime::now();
    self.last_cli_purge_at = Some(now);
    self.last_any_purge_at = Some(now);
}

pub fn observe_wake(&mut self) {
    self.last_wake_at = Some(Instant::now());
}
```

- [ ] **Step 4: Verify all tests pass**

```bash
cargo test -p apollo-engine --lib engine::maintenance_state 2>&1 | tail -10
```

Expected: 8 tests PASS (4 SwapDeltaWindow + 4 MaintenanceState).

- [ ] **Step 5: Commit**

```bash
git add crates/apollo-engine/src/engine/maintenance_state.rs
git commit -m "feat(maintenance): MaintenanceState timestamp + mutator methods"
```

---

### Task 6: 🧠 NotebookLM checkpoint — review MaintenanceState API

- [ ] **Step 1: Query NotebookLM**

Run via the MCP `mcp__notebooklm-mcp__notebook_query` tool against notebook `8344b94c-a014-4803-abea-076a55753cfd`:

Query body:
> Review crates/apollo-engine/src/engine/maintenance_state.rs (just committed). API surface: MaintenanceState{ swap_delta_window, last_any_purge_at, last_cli_purge_at, last_wake_at }. Methods: push_swap_delta, secs_since_any_purge/cli/wake, mark_purged, mark_cli_purged, observe_wake. SwapDeltaWindow{ CAP=45 } with push + sustained_below(threshold_bps, secs). Are there hidden race conditions or invariant breaks? Specifically: is the asymmetric cooldown (survival writes last_any_purge_at but never reads it) correctly modeled here, or is more API needed to enforce that asymmetry? Severity-rank findings.

- [ ] **Step 2: Read the response carefully**

If NotebookLM flags anything 🟠 High or above, address it inline. If only 🟡/🟢 surface, document in commit message and proceed. Discount one severity level mentally per CLAUDE.md NotebookLM rule.

- [ ] **Step 3: If fixes needed: apply, re-test, amend commit**

```bash
cargo test -p apollo-engine --lib engine::maintenance_state 2>&1 | tail -5
git add crates/apollo-engine/src/engine/maintenance_state.rs
git commit --amend --no-edit
```

- [ ] **Step 4: Note checkpoint outcome in plan progress**

Record in commit-message body: "NotebookLM checkpoint 1 (MaintenanceState API): <verdict + 1-line findings>".

---

### Task 7: Add 7 lockfree atomic counters

**Files:**
- Modify: `crates/apollo-engine/src/engine/lse_counters.rs`

- [ ] **Step 1: Read existing atomic structure**

```bash
grep -n "AtomicU64\|MetricsSnapshot\|pub fn snapshot" crates/apollo-engine/src/engine/lse_counters.rs | head -20
```

Note: existing pattern is `pub field_name: AtomicU64` in `LockFreeMetrics`, and `pub field_name: u64` in `MetricsSnapshot`, with `field_name: self.field_name.load(Ordering::Relaxed)` in `pub fn snapshot()`.

- [ ] **Step 2: Add 7 atomic fields to `LockFreeMetrics`**

Find the alphabetically appropriate location in `pub struct LockFreeMetrics { ... }` (likely after a `maintenance_*` block if one exists, else just after a related counter group). Add:

```rust
pub maintenance_purge_total: AtomicU64,
pub maintenance_purge_skipped_pressure_total: AtomicU64,
pub maintenance_purge_skipped_swap_floor_total: AtomicU64,
pub maintenance_purge_skipped_growing_total: AtomicU64,
pub maintenance_purge_skipped_idle_total: AtomicU64,
pub maintenance_purge_skipped_build_mode_total: AtomicU64,
pub maintenance_purge_skipped_rate_limit_total: AtomicU64,
```

- [ ] **Step 3: Add 7 fields to `MetricsSnapshot`**

In `pub struct MetricsSnapshot { ... }`, mirror:

```rust
pub maintenance_purge_total: u64,
pub maintenance_purge_skipped_pressure_total: u64,
pub maintenance_purge_skipped_swap_floor_total: u64,
pub maintenance_purge_skipped_growing_total: u64,
pub maintenance_purge_skipped_idle_total: u64,
pub maintenance_purge_skipped_build_mode_total: u64,
pub maintenance_purge_skipped_rate_limit_total: u64,
```

- [ ] **Step 4: Add 7 `.load()` lines to `pub fn snapshot()`**

In `impl LockFreeMetrics::snapshot`, add (matching the existing pattern):

```rust
maintenance_purge_total: self.maintenance_purge_total.load(Ordering::Relaxed),
maintenance_purge_skipped_pressure_total: self.maintenance_purge_skipped_pressure_total.load(Ordering::Relaxed),
maintenance_purge_skipped_swap_floor_total: self.maintenance_purge_skipped_swap_floor_total.load(Ordering::Relaxed),
maintenance_purge_skipped_growing_total: self.maintenance_purge_skipped_growing_total.load(Ordering::Relaxed),
maintenance_purge_skipped_idle_total: self.maintenance_purge_skipped_idle_total.load(Ordering::Relaxed),
maintenance_purge_skipped_build_mode_total: self.maintenance_purge_skipped_build_mode_total.load(Ordering::Relaxed),
maintenance_purge_skipped_rate_limit_total: self.maintenance_purge_skipped_rate_limit_total.load(Ordering::Relaxed),
```

- [ ] **Step 5: Verify compile**

```bash
cargo check -p apollo-engine 2>&1 | tail -10
```

Expected: 0 errors. Note: `Default` derive on `LockFreeMetrics` will auto-init new atomics to 0; if the struct uses manual `Default`, fix that too.

- [ ] **Step 6: Commit**

```bash
git add crates/apollo-engine/src/engine/lse_counters.rs
git commit -m "feat(maintenance): 7 lockfree atomic counters + MetricsSnapshot fields"
```

---

### Task 8: Add 7 RuntimeMetrics fields with #[serde(default)]

**Files:**
- Modify: `crates/apollo-engine/src/engine/types.rs`

- [ ] **Step 1: Locate `RuntimeMetrics` struct**

```bash
grep -n "pub struct RuntimeMetrics" crates/apollo-engine/src/engine/types.rs
```

- [ ] **Step 2: Add 7 fields**

In the `RuntimeMetrics` struct, near the other counter fields, add each line with `#[serde(default)]`:

```rust
#[serde(default)]
pub maintenance_purge_total: u64,
#[serde(default)]
pub maintenance_purge_skipped_pressure_total: u64,
#[serde(default)]
pub maintenance_purge_skipped_swap_floor_total: u64,
#[serde(default)]
pub maintenance_purge_skipped_growing_total: u64,
#[serde(default)]
pub maintenance_purge_skipped_idle_total: u64,
#[serde(default)]
pub maintenance_purge_skipped_build_mode_total: u64,
#[serde(default)]
pub maintenance_purge_skipped_rate_limit_total: u64,
```

- [ ] **Step 3: Verify compile**

```bash
cargo check -p apollo-engine 2>&1 | tail -10
```

Expected: 0 errors. If `Default` is derived, no manual init needed; if manual `Default` impl, add `0` for each field.

- [ ] **Step 4: Commit**

```bash
git add crates/apollo-engine/src/engine/types.rs
git commit -m "feat(maintenance): 7 RuntimeMetrics fields with serde(default)"
```

---

### Task 9: Wire 7 sync_from_lockfree flush lines (Sprint 3 lesson critical)

**Files:**
- Modify: `crates/apollo-engine/src/engine/daemon_state.rs`

- [ ] **Step 1: Locate sync_from_lockfree function**

```bash
grep -n "fn sync_from_lockfree" crates/apollo-engine/src/engine/daemon_state.rs
```

- [ ] **Step 2: Add 7 flush lines**

In `sync_from_lockfree`, add (matching existing pattern of `self.metrics.X = lf.X;`):

```rust
self.metrics.maintenance_purge_total = lf.maintenance_purge_total;
self.metrics.maintenance_purge_skipped_pressure_total = lf.maintenance_purge_skipped_pressure_total;
self.metrics.maintenance_purge_skipped_swap_floor_total = lf.maintenance_purge_skipped_swap_floor_total;
self.metrics.maintenance_purge_skipped_growing_total = lf.maintenance_purge_skipped_growing_total;
self.metrics.maintenance_purge_skipped_idle_total = lf.maintenance_purge_skipped_idle_total;
self.metrics.maintenance_purge_skipped_build_mode_total = lf.maintenance_purge_skipped_build_mode_total;
self.metrics.maintenance_purge_skipped_rate_limit_total = lf.maintenance_purge_skipped_rate_limit_total;
```

- [ ] **Step 3: Verify compile**

```bash
cargo check -p apollo-engine 2>&1 | tail -10
```

Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add crates/apollo-engine/src/engine/daemon_state.rs
git commit -m "feat(maintenance): wire 7 sync_from_lockfree flush lines (Sprint 3 chain)"
```

---

### Task 10: Telemetry round-trip integration test (Sprint 3 critical safeguard)

**Files:**
- Create: `crates/apollo-engine/tests/level3_maintenance_purge.rs`

- [ ] **Step 1: Write the integration test**

```rust
// crates/apollo-engine/tests/level3_maintenance_purge.rs
//! Integration tests for Maintenance Purge Gate.
//! Test #1 in this file is the Sprint 3 telemetry-death safeguard:
//! it round-trips all 7 maintenance counters through the full chain
//! (LockFreeMetrics → MetricsSnapshot → sync_from_lockfree → RuntimeMetrics → JSON)
//! and asserts literal substrings appear in serialized JSON.

use std::sync::atomic::Ordering;

use apollo_engine::engine::daemon_state::DaemonInternalState;
use apollo_engine::engine::lse_counters::LockFreeMetrics;

#[test]
fn maintenance_counters_round_trip_to_runtime_metrics_json() {
    let lf = LockFreeMetrics::default();
    lf.maintenance_purge_total.fetch_add(1, Ordering::Relaxed);
    lf.maintenance_purge_skipped_pressure_total
        .fetch_add(2, Ordering::Relaxed);
    lf.maintenance_purge_skipped_swap_floor_total
        .fetch_add(3, Ordering::Relaxed);
    lf.maintenance_purge_skipped_growing_total
        .fetch_add(5, Ordering::Relaxed);
    lf.maintenance_purge_skipped_idle_total
        .fetch_add(7, Ordering::Relaxed);
    lf.maintenance_purge_skipped_build_mode_total
        .fetch_add(11, Ordering::Relaxed);
    lf.maintenance_purge_skipped_rate_limit_total
        .fetch_add(13, Ordering::Relaxed);

    let snap = lf.snapshot();
    let mut state = DaemonInternalState::default();
    state.sync_from_lockfree(&snap);

    let json = serde_json::to_string(&state.metrics).expect("serialize RuntimeMetrics");

    assert!(
        json.contains(r#""maintenance_purge_total":1"#),
        "missing maintenance_purge_total in JSON: {json}"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_pressure_total":2"#),
        "missing pressure counter in JSON"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_swap_floor_total":3"#),
        "missing swap_floor counter"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_growing_total":5"#),
        "missing growing counter"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_idle_total":7"#),
        "missing idle counter"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_build_mode_total":11"#),
        "missing build_mode counter"
    );
    assert!(
        json.contains(r#""maintenance_purge_skipped_rate_limit_total":13"#),
        "missing rate_limit counter"
    );
}
```

> Note: If `DaemonInternalState::default()` does not exist, instead instantiate it directly with required fields. Adjust import path if struct is named differently — check `grep -n "pub struct DaemonInternalState" crates/apollo-engine/src/engine/daemon_state.rs`.

- [ ] **Step 2: Verify test passes**

```bash
cargo test -p apollo-engine --test level3_maintenance_purge 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/apollo-engine/tests/level3_maintenance_purge.rs
git commit -m "test(maintenance): Sprint 3 telemetry-death safeguard — counters round-trip to JSON"
```

---

### Task 11: Persist last_any_purge_at + last_cli_purge_at in LearnedState

**Files:**
- Modify: `crates/apollo-engine/src/engine/learned_state.rs`

- [ ] **Step 1: Locate `pub fn collect` and `pub fn apply` in `learned_state.rs`**

```bash
grep -n "pub fn collect\|pub fn apply\|pub struct LearnedState\b" crates/apollo-engine/src/engine/learned_state.rs | head
```

- [ ] **Step 2: Add 2 fields to `LearnedState` struct**

In the `pub struct LearnedState`, after an existing `#[serde(default)]` field block, add:

```rust
#[serde(default)]
pub last_any_purge_at: Option<std::time::SystemTime>,
#[serde(default)]
pub last_cli_purge_at: Option<std::time::SystemTime>,
```

> ⚠️ **CRITICAL**: both fields MUST have `#[serde(default)]`. Without it, the daemon will FAIL TO START on any host with a pre-existing `/var/lib/apollo/learned_state.json` (deserialize error → panic). NotebookLM r3 plan-review flagged this as a critical regression risk. Verify before commit:
>
> ```bash
> grep -B1 "pub last_any_purge_at\|pub last_cli_purge_at" crates/apollo-engine/src/engine/learned_state.rs
> ```
>
> Expected: each field preceded by `#[serde(default)]`.

- [ ] **Step 3: Update `collect()` signature + body**

`collect()` already takes references to many state structs. Add a `&MaintenanceState` parameter to the signature (find existing parameters with `grep -A20 "pub fn collect"`). In the body, populate:

```rust
last_any_purge_at: maintenance_state.last_any_purge_at,
last_cli_purge_at: maintenance_state.last_cli_purge_at,
```

- [ ] **Step 4: Update `apply()` to restore**

`apply()` writes restored state back into mutable references. Add `maintenance_state: &mut MaintenanceState` to the signature. In body:

```rust
maintenance_state.last_any_purge_at = self.last_any_purge_at;
maintenance_state.last_cli_purge_at = self.last_cli_purge_at;
// swap_delta_window NOT restored — let it warm up (~90s)
// last_wake_at NOT restored (Instant is process-relative)
```

- [ ] **Step 5: Add import**

At the top of `learned_state.rs`:

```rust
use crate::engine::maintenance_state::MaintenanceState;
```

- [ ] **Step 6: Update call sites**

```bash
grep -rn "LearnedState::collect\|\.collect(" src/ crates/ | grep -v "Vec::collect\|\.iter()" | head -20
```

Identify all callers of `collect()` and `apply()`. Add a `MaintenanceState` reference to each. The likely call sites are in `main.rs` daemon startup/shutdown and a `learning_pipeline` flush. For now, pass a placeholder `&MaintenanceState::default()` if `subsystems.maintenance_state` is not yet wired (it will be in Task 12).

- [ ] **Step 7: Verify compile**

```bash
cargo check 2>&1 | tail -15
```

Expected: 0 errors. Compile errors here likely indicate missed call sites — fix them.

- [ ] **Step 8: Commit**

```bash
git add -u crates/apollo-engine/src/engine/learned_state.rs src/ crates/
git commit -m "feat(maintenance): persist last_any_purge_at + last_cli_purge_at via LearnedState"
```

---

### Task 12: Wire MaintenanceState into DaemonSubsystems

**Files:**
- Modify: `src/bin/apollo-optimizerd/daemon_init.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs`

- [ ] **Step 1: Locate `DaemonSubsystems` struct**

```bash
grep -n "pub struct DaemonSubsystems" src/bin/apollo-optimizerd/daemon_init.rs
```

- [ ] **Step 2: Add field to struct**

```rust
pub maintenance_state: apollo_engine::engine::maintenance_state::MaintenanceState,
```

- [ ] **Step 3: Init in constructor**

In the `DaemonSubsystems::new()` (or equivalent init function), initialize:

```rust
maintenance_state: apollo_engine::engine::maintenance_state::MaintenanceState::new(),
```

- [ ] **Step 4: Replace placeholder `&MaintenanceState::default()` from Task 11**

Find the call sites for `LearnedState::collect/apply` modified in Task 11 step 6. Replace placeholders with `&subsystems.maintenance_state` / `&mut subsystems.maintenance_state`.

- [ ] **Step 5: Verify compile**

```bash
cargo check 2>&1 | tail -10
```

Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add -u src/bin/apollo-optimizerd/
git commit -m "feat(maintenance): wire MaintenanceState into DaemonSubsystems"
```

---

### Task 13: Refactor survival_tick — take &mut MaintenanceState (asymmetric semantics)

**Files:**
- Modify: `src/bin/apollo-optimizerd/daemon_survival_tick.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs`

- [ ] **Step 1: Read current survival_tick signature**

```bash
sed -n '40,60p' src/bin/apollo-optimizerd/daemon_survival_tick.rs
```

Current parameter: `last_purge_at: &mut Option<Instant>` (~line 52).

- [ ] **Step 2: Change parameter to MaintenanceState reference**

In the function signature, replace `last_purge_at: &mut Option<Instant>` with:

```rust
maintenance_state: &mut apollo_engine::engine::maintenance_state::MaintenanceState,
```

Also update the doc comment (~line 40): replace "rate-limit guard" doc with "asymmetric purge state — survival keeps its own local Instant cooldown but writes the shared timestamp on fire so maintenance backs off".

- [ ] **Step 3: Add a local survival-only Instant cooldown field**

Survival keeps local 10-min cooldown to remain physical-crisis sovereign. Since this is per-process and needn't persist, store a static `Mutex<Option<Instant>>` at module scope:

```rust
// Top of daemon_survival_tick.rs (inside the module, not inside a fn)
use std::sync::Mutex;
static SURVIVAL_LOCAL_COOLDOWN: Mutex<Option<std::time::Instant>> = Mutex::new(None);
```

- [ ] **Step 4: Update purge spawn body (~line 121-134)**

Replace the current block:

```rust
// Last-resort page reclaim: spawn `purge` when swap crosses 80% of
// exhaustion threshold. Rate-limited to once per 10 min.
let threshold = swap_exhaustion_threshold_bytes(snapshot.pressure.swap_total_bytes);
let swap_used = snapshot.pressure.swap_used_bytes;
if swap_used as f64 >= threshold as f64 * 0.80 {
    let can_purge = last_purge_at
        .map(|t| t.elapsed() >= Duration::from_secs(600))
        .unwrap_or(true);
    if can_purge {
        if std::process::Command::new("purge").spawn().is_ok() {
            *last_purge_at = Some(Instant::now());
        }
    }
}
```

with:

```rust
// Last-resort page reclaim: spawn `purge` when swap crosses 80% of
// exhaustion threshold. Survival reads its own local cooldown ONLY —
// never gated by the shared MaintenanceState.last_any_purge_at.
let threshold = swap_exhaustion_threshold_bytes(snapshot.pressure.swap_total_bytes);
let swap_used = snapshot.pressure.swap_used_bytes;
if swap_used as f64 >= threshold as f64 * 0.80 {
    let mut local = SURVIVAL_LOCAL_COOLDOWN.lock().unwrap_or_else(|e| e.into_inner());
    let can_purge = local
        .map(|t: Instant| t.elapsed() >= Duration::from_secs(600))
        .unwrap_or(true);
    if can_purge {
        if std::process::Command::new("purge").spawn().is_ok() {
            *local = Some(Instant::now());
            // Write shared timestamp so maintenance_tick yields.
            // (Survival itself does NOT read this field — asymmetric.)
            maintenance_state.mark_purged();
        }
    }
}
```

- [ ] **Step 5: Update caller in main.rs (~line 3300)**

Find where `run_survival_tick` is called. Replace `&mut last_purge_at` with `&mut subsystems.maintenance_state`. Also delete the now-unused `let mut last_purge_at: Option<Instant> = None;` declaration (~line 1057).

- [ ] **Step 6: Verify compile**

```bash
cargo check --bin apollo-optimizerd 2>&1 | tail -15
```

Expected: 0 errors. Imports may need adjustment; add `use std::time::{Duration, Instant};` if not already there.

- [ ] **Step 7: Verify existing survival tests still pass**

```bash
cargo test --bin apollo-optimizerd survival 2>&1 | tail -10
```

Expected: PASS (or tests don't exist for that path — neither breaks).

- [ ] **Step 8: Commit**

```bash
git add -u src/bin/apollo-optimizerd/
git commit -m "refactor(survival): take &mut MaintenanceState; asymmetric cooldown semantics

Survival keeps its own static SURVIVAL_LOCAL_COOLDOWN (10-min Instant) for
gating its own fire decision. After successful purge, it writes
maintenance_state.mark_purged() so maintenance_tick yields for 30 min.
Survival NEVER reads last_any_purge_at — physical-crisis sovereign.
"
```

---

### Task 14: 🧠 NotebookLM checkpoint — review survival refactor

- [ ] **Step 1: Query NotebookLM**

Tool: `mcp__notebooklm-mcp__notebook_query` against `8344b94c-a014-4803-abea-076a55753cfd`.

Query body:
> Review the survival_tick refactor (commit just made). The change: replaced `last_purge_at: &mut Option<Instant>` parameter with `&mut MaintenanceState`, added a static SURVIVAL_LOCAL_COOLDOWN: Mutex<Option<Instant>> for survival's own 10-min cooldown, and on successful purge calls `maintenance_state.mark_purged()` to write the shared timestamp WITHOUT reading it. Asymmetric semantics: survival writes-only, maintenance reads+writes. Are there hidden invariants this breaks? Specifically: can the static Mutex deadlock under concurrent ticks (the daemon main loop is single-threaded so theoretically no, but worth confirming)? Does write-only-no-read introduce a coordination gap with anything else in the codebase that might read last_any_purge_at?

- [ ] **Step 2: Apply any 🟠+ findings**

If NotebookLM flags issues at high severity (after one-level discount), fix inline + amend commit. Otherwise note in plan progress and proceed.

- [ ] **Step 3: Record outcome**

Add a follow-up commit annotation if material findings:

```bash
git commit --allow-empty -m "doc(maintenance): NotebookLM checkpoint 2 (survival refactor) — <verdict>"
```

---

### Task 15: Implement daemon_maintenance_tick — SkipReason enum + module skeleton

**Files:**
- Create: `src/bin/apollo-optimizerd/daemon_maintenance_tick.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs` (add `mod daemon_maintenance_tick;`)

- [ ] **Step 1: Create skeleton**

```rust
// src/bin/apollo-optimizerd/daemon_maintenance_tick.rs
//! Maintenance Purge tick — opportunistic non-crisis page reclaim.
//!
//! See docs/superpowers/specs/2026-05-10-maintenance-purge-design.md
//! Spec invariants:
//! - Pressure window: 0.65 ≤ raw < 0.85 (no overlap with survival ≥0.85)
//! - Swap floor: max(1.5 GB, 50% × swap_total)
//! - Swap delta sustained < 256 KB/s for 90s (via SwapDeltaWindow)
//! - User idle ≥120s + 10s post-wake quiet
//! - Build mode bypass (dev_runtime_active)
//! - Reads + writes shared last_any_purge_at (30 min)

use std::sync::atomic::Ordering;

use apollo_engine::collector::SystemSnapshot;
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::maintenance_state::MaintenanceState;
use apollo_engine::engine::user_context::UserContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    PressureLow,
    PressureSurvival,
    SwapFloor,
    Growing,
    Idle,
    PostWake,
    BuildMode,
    RateLimit,
}
```

- [ ] **Step 2: Add `mod daemon_maintenance_tick;` to main.rs**

In `src/bin/apollo-optimizerd/main.rs`, find the existing `mod daemon_*_tick;` declarations (alphabetically grouped). Insert:

```rust
mod daemon_maintenance_tick;
```

- [ ] **Step 3: Verify compile**

```bash
cargo check --bin apollo-optimizerd 2>&1 | tail -10
```

Expected: 0 errors. Some unused-import warnings are fine — they'll be consumed in next tasks.

- [ ] **Step 4: Commit**

```bash
git add src/bin/apollo-optimizerd/daemon_maintenance_tick.rs src/bin/apollo-optimizerd/main.rs
git commit -m "feat(maintenance): scaffold daemon_maintenance_tick + SkipReason enum"
```

---

### Task 16: should_fire — pressure window gates (TDD)

**Files:**
- Modify: `src/bin/apollo-optimizerd/daemon_maintenance_tick.rs`

- [ ] **Step 1: Write failing tests**

At the bottom of `daemon_maintenance_tick.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::collector::{PressureStats, SystemSnapshot};
    use apollo_engine::engine::user_context::UserContext;

    fn synth_snap(pressure: f64, swap_used: u64, swap_total: u64) -> SystemSnapshot {
        SystemSnapshot {
            timestamp: std::time::SystemTime::now(),
            pressure: PressureStats {
                memory_pressure: pressure,
                swap_used_bytes: swap_used,
                swap_total_bytes: swap_total,
                swap_delta_bytes_per_sec: 0.0,
                ..Default::default()
            },
            top_processes: vec![],
            ..Default::default()
        }
    }

    fn idle_ctx() -> UserContext {
        UserContext {
            idle_secs: 200,
            ..Default::default()
        }
    }

    #[test]
    fn should_fire_pressure_below_returns_pressure_low() {
        let snap = synth_snap(0.55, 3 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024);
        let ctx = idle_ctx();
        let state = MaintenanceState::default();
        assert_eq!(should_fire(&snap, &ctx, &state), Some(SkipReason::PressureLow));
    }

    #[test]
    fn should_fire_pressure_at_survival_returns_pressure_survival() {
        let snap = synth_snap(0.90, 3 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024);
        let ctx = idle_ctx();
        let state = MaintenanceState::default();
        assert_eq!(should_fire(&snap, &ctx, &state), Some(SkipReason::PressureSurvival));
    }
}
```

> If `SystemSnapshot` and `PressureStats` lack `Default` derive, the test helper instead uses field-by-field construction (check `crates/apollo-engine/src/collector.rs` for the actual fields). The test goal is just to exercise `should_fire` with synthetic inputs.

- [ ] **Step 2: Verify failure**

```bash
cargo test --bin apollo-optimizerd daemon_maintenance_tick 2>&1 | tail -10
```

Expected: FAIL with "no function `should_fire`".

- [ ] **Step 3: Implement `should_fire` skeleton with first 2 gates**

Add above `mod tests`:

```rust
pub(crate) fn should_fire(
    snap: &SystemSnapshot,
    _ctx: &UserContext,
    _state: &MaintenanceState,
) -> Option<SkipReason> {
    let p = snap.pressure.memory_pressure;
    if p < 0.65 {
        return Some(SkipReason::PressureLow);
    }
    if p >= 0.85 {
        return Some(SkipReason::PressureSurvival);
    }
    None
}
```

- [ ] **Step 4: Verify pass**

```bash
cargo test --bin apollo-optimizerd daemon_maintenance_tick 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bin/apollo-optimizerd/daemon_maintenance_tick.rs
git commit -m "feat(maintenance): should_fire pressure window gates"
```

---

### Task 17: should_fire — swap floor gate (TDD M1 cold-boot trap)

**Files:**
- Modify: `src/bin/apollo-optimizerd/daemon_maintenance_tick.rs`

- [ ] **Step 1: Write failing test**

In `mod tests`:

```rust
#[test]
fn should_fire_swap_floor_traps_m1_cold_boot() {
    // M1 cold boot: swap_total=800MB, swap_used=500MB (62.5% by ratio).
    // 1.5 GB absolute floor MUST kick in to skip.
    let snap = synth_snap(0.70, 500 * 1024 * 1024, 800 * 1024 * 1024);
    let ctx = idle_ctx();
    let state = MaintenanceState::default();
    assert_eq!(should_fire(&snap, &ctx, &state), Some(SkipReason::SwapFloor));
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test --bin apollo-optimizerd should_fire_swap_floor 2>&1 | tail -5
```

Expected: FAIL — currently returns `None` for this input.

- [ ] **Step 3: Add gate**

In `should_fire`, after the pressure block:

```rust
let swap_used = snap.pressure.swap_used_bytes;
let swap_total = snap.pressure.swap_total_bytes;
let swap_floor = std::cmp::max(1_536 * 1024 * 1024, swap_total / 2);
if swap_used < swap_floor {
    return Some(SkipReason::SwapFloor);
}
```

- [ ] **Step 4: Verify pass**

```bash
cargo test --bin apollo-optimizerd should_fire 2>&1 | tail -10
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bin/apollo-optimizerd/daemon_maintenance_tick.rs
git commit -m "feat(maintenance): should_fire swap floor gate (M1 cold-boot trap)"
```

---

### Task 18: should_fire — sustained-window growing gate (TDD)

**Files:**
- Modify: `src/bin/apollo-optimizerd/daemon_maintenance_tick.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn should_fire_growing_swap_returns_growing() {
    let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
    let ctx = idle_ctx();
    let mut state = MaintenanceState::default();
    // Empty window → sustained_below returns false → Growing
    assert_eq!(should_fire(&snap, &ctx, &state), Some(SkipReason::Growing));

    // Now feed sustained low samples and verify it advances
    let now = std::time::SystemTime::now();
    for i in 0..45 {
        let t = now - std::time::Duration::from_secs(90)
            + std::time::Duration::from_secs(i * 2);
        state.swap_delta_window.push(t, 50_000.0);
    }
    // Should advance past Growing now (next gate Idle still pending UserContext setup)
    assert_ne!(should_fire(&snap, &ctx, &state), Some(SkipReason::Growing));
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test --bin apollo-optimizerd should_fire_growing 2>&1 | tail -5
```

Expected: FAIL.

- [ ] **Step 3: Add gate**

In `should_fire`, after swap floor block:

```rust
if !_state.swap_delta_window.sustained_below(256_000.0, 90) {
    return Some(SkipReason::Growing);
}
```

(Drop the `_` underscore from `_state` parameter since now it's used.)

- [ ] **Step 4: Verify pass**

```bash
cargo test --bin apollo-optimizerd should_fire 2>&1 | tail -10
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bin/apollo-optimizerd/daemon_maintenance_tick.rs
git commit -m "feat(maintenance): should_fire growing gate via SwapDeltaWindow"
```

---

### Task 19: should_fire — idle, post-wake, build-mode, rate-limit gates (TDD batched)

**Files:**
- Modify: `src/bin/apollo-optimizerd/daemon_maintenance_tick.rs`

- [ ] **Step 1: Write failing tests**

```rust
fn make_ready_state() -> MaintenanceState {
    let mut state = MaintenanceState::default();
    let now = std::time::SystemTime::now();
    for i in 0..45 {
        let t = now - std::time::Duration::from_secs(90)
            + std::time::Duration::from_secs(i * 2);
        state.swap_delta_window.push(t, 50_000.0);
    }
    state
}

#[test]
fn should_fire_user_active_returns_idle() {
    let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
    let ctx = UserContext { idle_secs: 10, ..Default::default() };
    let state = make_ready_state();
    assert_eq!(should_fire(&snap, &ctx, &state), Some(SkipReason::Idle));
}

#[test]
fn should_fire_post_wake_quiet_returns_postwake() {
    let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
    let ctx = idle_ctx();
    let mut state = make_ready_state();
    state.observe_wake();  // last_wake_at = now, secs_since_wake = 0
    assert_eq!(should_fire(&snap, &ctx, &state), Some(SkipReason::PostWake));
}

#[test]
fn should_fire_build_mode_returns_build_mode() {
    let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
    let ctx = UserContext {
        idle_secs: 200,
        dev_runtime_active: true,
        ..Default::default()
    };
    let state = make_ready_state();
    assert_eq!(should_fire(&snap, &ctx, &state), Some(SkipReason::BuildMode));
}

#[test]
fn should_fire_rate_limit_returns_rate_limit() {
    let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
    let ctx = idle_ctx();
    let mut state = make_ready_state();
    state.last_any_purge_at = Some(std::time::SystemTime::now() - std::time::Duration::from_secs(100));
    assert_eq!(should_fire(&snap, &ctx, &state), Some(SkipReason::RateLimit));
}

#[test]
fn should_fire_all_gates_pass_returns_none() {
    let snap = synth_snap(0.70, 3_000_000_000, 4_000_000_000);
    let ctx = idle_ctx();
    let state = make_ready_state();
    assert_eq!(should_fire(&snap, &ctx, &state), None);
}
```

> Note: `UserContext { dev_runtime_active: bool, ... }` — verify the actual field exists. If it's a method `fn dev_runtime_active(&self) -> bool` instead, adjust the construction accordingly (likely `UserContext` exposes a method derived from process scan; check `crates/apollo-engine/src/engine/user_context.rs`). If it's a method, the test helper must construct `UserContext` such that the method returns true — may require a different field like `dev_runtimes_running: 1`.

- [ ] **Step 2: Verify failure**

```bash
cargo test --bin apollo-optimizerd should_fire 2>&1 | tail -15
```

Expected: 5 new tests FAIL.

- [ ] **Step 3: Implement remaining gates**

In `should_fire`, after Growing gate:

```rust
if !_ctx.is_idle_long() {
    return Some(SkipReason::Idle);
}
if _state.secs_since_wake() < 10 {
    return Some(SkipReason::PostWake);
}
if _ctx.dev_runtime_active() {
    return Some(SkipReason::BuildMode);
}
if _state.secs_since_any_purge() < 1800 {
    return Some(SkipReason::RateLimit);
}
None
```

(Drop underscore from `_ctx` and `_state` — both now used.)

Final form of `should_fire`:

```rust
pub(crate) fn should_fire(
    snap: &SystemSnapshot,
    ctx: &UserContext,
    state: &MaintenanceState,
) -> Option<SkipReason> {
    let p = snap.pressure.memory_pressure;
    if p < 0.65 { return Some(SkipReason::PressureLow); }
    if p >= 0.85 { return Some(SkipReason::PressureSurvival); }

    let swap_used = snap.pressure.swap_used_bytes;
    let swap_total = snap.pressure.swap_total_bytes;
    let swap_floor = std::cmp::max(1_536 * 1024 * 1024, swap_total / 2);
    if swap_used < swap_floor { return Some(SkipReason::SwapFloor); }

    if !state.swap_delta_window.sustained_below(256_000.0, 90) {
        return Some(SkipReason::Growing);
    }
    if !ctx.is_idle_long() { return Some(SkipReason::Idle); }
    if state.secs_since_wake() < 10 { return Some(SkipReason::PostWake); }
    if ctx.dev_runtime_active() { return Some(SkipReason::BuildMode); }
    if state.secs_since_any_purge() < 1800 { return Some(SkipReason::RateLimit); }

    None
}
```

- [ ] **Step 4: Verify pass**

```bash
cargo test --bin apollo-optimizerd should_fire 2>&1 | tail -15
```

Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bin/apollo-optimizerd/daemon_maintenance_tick.rs
git commit -m "feat(maintenance): should_fire idle/postwake/buildmode/ratelimit gates"
```

---

### Task 20: run_maintenance_tick — orchestrator with counter wiring

**Files:**
- Modify: `src/bin/apollo-optimizerd/daemon_maintenance_tick.rs`

- [ ] **Step 1: Implement orchestrator (returns whether fire happened so caller can record CausalGraph)**

Above `should_fire`:

```rust
/// Returns true if the maintenance tick fired a purge in this cycle.
/// Caller should record this into CausalGraph as "system_maintenance_purge"
/// for observational outcome tracking (≥30 samples before trusting).
pub fn run_maintenance_tick(
    snap: &SystemSnapshot,
    ctx: &UserContext,
    state: &mut MaintenanceState,
    lf_metrics: &LockFreeMetrics,
) -> bool {
    state.push_swap_delta(snap.pressure.swap_delta_bytes_per_sec);

    match should_fire(snap, ctx, state) {
        None => {
            if std::process::Command::new("purge").spawn().is_ok() {
                state.mark_purged();
                lf_metrics
                    .maintenance_purge_total
                    .fetch_add(1, Ordering::Relaxed);
                tracing::info!(
                    target: "apollo.maintenance",
                    swap_used_gb = snap.pressure.swap_used_bytes as f64 / 1_073_741_824.0,
                    pressure = snap.pressure.memory_pressure,
                    "maintenance purge fired"
                );
                return true;
            }
            false
        }
        Some(reason) => {
            let counter = match reason {
                SkipReason::PressureLow | SkipReason::PressureSurvival => {
                    &lf_metrics.maintenance_purge_skipped_pressure_total
                }
                SkipReason::SwapFloor => &lf_metrics.maintenance_purge_skipped_swap_floor_total,
                SkipReason::Growing => &lf_metrics.maintenance_purge_skipped_growing_total,
                SkipReason::Idle | SkipReason::PostWake => {
                    &lf_metrics.maintenance_purge_skipped_idle_total
                }
                SkipReason::BuildMode => &lf_metrics.maintenance_purge_skipped_build_mode_total,
                SkipReason::RateLimit => &lf_metrics.maintenance_purge_skipped_rate_limit_total,
            };
            counter.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
}
```

> NotebookLM r3 finding (plan review): the orchestrator must signal whether it fired so the caller can emit `system_maintenance_purge` into the CausalGraph. CausalGraph wiring happens in Task 24.

> Note: `tracing` may not be the project's logging framework. Check `grep -rn "tracing::info\|log::info" src/bin/apollo-optimizerd/ | head -3` and substitute the actual logging crate. If neither, omit the log line entirely.

- [ ] **Step 2: Verify compile**

```bash
cargo check --bin apollo-optimizerd 2>&1 | tail -10
```

Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add src/bin/apollo-optimizerd/daemon_maintenance_tick.rs
git commit -m "feat(maintenance): run_maintenance_tick orchestrator + counter wiring"
```

---

### Task 21: Integration test — swap_total_low cold-boot trap protection

**Files:**
- Modify: `crates/apollo-engine/tests/level3_maintenance_purge.rs`

- [ ] **Step 1: Add test**

> Note: this test cannot exercise the daemon binary's `run_maintenance_tick` directly from the engine tests directory. Instead, test the gate logic via the public `MaintenanceState` API. The skip-counter assertion will be exercised in the in-bin unit tests above.

```rust
use apollo_engine::engine::maintenance_state::{MaintenanceState, SwapDeltaWindow};

#[test]
fn maintenance_state_swap_floor_blocks_m1_cold_boot() {
    // Verify the swap_floor calculation matches spec.
    // M1 cold boot: swap_total = 800 MB, swap_used = 500 MB.
    let swap_total: u64 = 800 * 1024 * 1024;
    let swap_used: u64 = 500 * 1024 * 1024;
    let swap_floor = std::cmp::max(1_536u64 * 1024 * 1024, swap_total / 2);
    assert_eq!(swap_floor, 1_536 * 1024 * 1024);
    assert!(swap_used < swap_floor, "M1 cold boot should not trigger maintenance");
}

#[test]
fn maintenance_state_swap_floor_passes_for_typical_m1_8gb() {
    // Typical loaded M1 8GB: swap_total = 4 GB, swap_used = 2.5 GB.
    let swap_total: u64 = 4 * 1024 * 1024 * 1024;
    let swap_used: u64 = 2_560 * 1024 * 1024;  // 2.5 GB
    let swap_floor = std::cmp::max(1_536u64 * 1024 * 1024, swap_total / 2);
    assert_eq!(swap_floor, 2 * 1024 * 1024 * 1024);
    assert!(swap_used > swap_floor, "loaded M1 should pass swap_floor");
}
```

- [ ] **Step 2: Verify pass**

```bash
cargo test -p apollo-engine --test level3_maintenance_purge 2>&1 | tail -10
```

Expected: 3 tests pass (round-trip + 2 floor tests).

- [ ] **Step 3: Commit**

```bash
git add crates/apollo-engine/tests/level3_maintenance_purge.rs
git commit -m "test(maintenance): swap floor cold-boot trap + typical M1 8GB"
```

---

### Task 22: Integration test — 90s sustained window requires history

**Files:**
- Modify: `crates/apollo-engine/tests/level3_maintenance_purge.rs`

- [ ] **Step 1: Add test**

```rust
#[test]
fn maintenance_window_requires_90s_sustained() {
    let mut w = SwapDeltaWindow::default();
    let now = std::time::SystemTime::now();
    // 30 samples at 50_000 → 60s of history → insufficient
    for i in 0..30 {
        let t = now - std::time::Duration::from_secs(60)
            + std::time::Duration::from_secs(i * 2);
        w.push(t, 50_000.0);
    }
    assert!(!w.sustained_below(256_000.0, 90), "60s history should fail 90s requirement");

    // Add 15 more samples → 90s of history → sufficient
    for i in 0..15 {
        let t = now - std::time::Duration::from_secs(30)
            + std::time::Duration::from_secs(i * 2);
        w.push(t, 50_000.0);
    }
    assert!(w.sustained_below(256_000.0, 90), "90s history should pass");
}
```

- [ ] **Step 2: Verify pass**

```bash
cargo test -p apollo-engine --test level3_maintenance_purge 2>&1 | tail -5
```

Expected: 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/apollo-engine/tests/level3_maintenance_purge.rs
git commit -m "test(maintenance): 90s sustained-window history requirement"
```

---

### Task 23: 🧠 NotebookLM checkpoint — review tick logic + tests

- [ ] **Step 1: Query NotebookLM**

Tool: `mcp__notebooklm-mcp__notebook_query` against `8344b94c-a014-4803-abea-076a55753cfd`.

Query body:
> Review daemon_maintenance_tick.rs (just committed) including should_fire (8 gates: PressureLow, PressureSurvival, SwapFloor, Growing, Idle, PostWake, BuildMode, RateLimit) and run_maintenance_tick orchestrator + 9 unit tests + 4 integration tests. Are there any logic gaps in the gate ordering? E.g., is there a state where should_fire returns None but firing would actually be wrong? Are the test thresholds (1.5 GB swap floor, 256 KB/s bps, 90s window, 10s post-wake, 30 min rate-limit) aligned with M1 8GB realities? Any 🟠 High issues to address before wiring into main loop? Severity-rank.

- [ ] **Step 2: Apply findings inline**

If 🟠+ issues, fix and amend the most relevant prior commit. If only 🟡/🟢, document.

- [ ] **Step 3: Annotate progress**

```bash
git commit --allow-empty -m "doc(maintenance): NotebookLM checkpoint 3 (tick logic) — <verdict>"
```

---

### Task 24: Wire run_maintenance_tick into main loop

**Files:**
- Modify: `src/bin/apollo-optimizerd/main.rs`

- [ ] **Step 1: Find call to run_survival_tick**

```bash
grep -n "run_survival_tick\|run_dispatch_tick" src/bin/apollo-optimizerd/main.rs | head -5
```

- [ ] **Step 2: Insert maintenance_tick call between survival and dispatch + CausalGraph emit**

After `daemon_survival_tick::run_survival_tick(...)` and before `daemon_dispatch_tick::run_dispatch_tick(...)`, add:

```rust
let maintenance_fired = daemon_maintenance_tick::run_maintenance_tick(
    &snapshot,
    &policy_context.user_ctx,
    &mut subsystems.maintenance_state,
    &lf_metrics,
);
if maintenance_fired {
    // Record action cause for observational outcome tracking.
    // Validation requires ≥30 samples per CLAUDE.md supervision rule.
    causal_graph.record_action_with_resources(
        "system_maintenance_purge",
        snapshot.pressure.memory_pressure,
        snapshot.pressure.swap_used_bytes as f64 / 1_073_741_824.0,
        snapshot.pressure.compressor_pressure,
    );
}
```

> Note: identifiers `policy_context.user_ctx`, `subsystems`, `lf_metrics`, `snapshot`, `causal_graph` may differ. Match what's in scope at the survival call site (~line 3300). The exact `record_action_with_resources` signature must match what's in `crates/apollo-engine/src/engine/causal_graph.rs`; if it differs, find the canonical "record action cause" call site for survival's purge (`grep -rn "record_action.*purge" crates/`) and copy that signature.

- [ ] **Step 3: Verify compile**

```bash
cargo check --bin apollo-optimizerd 2>&1 | tail -10
```

Expected: 0 errors.

- [ ] **Step 4: Verify cargo test still green**

```bash
cargo test --bin apollo-optimizerd 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -u src/bin/apollo-optimizerd/main.rs
git commit -m "feat(maintenance): insert run_maintenance_tick between survival and dispatch"
```

---

### Task 25: Wake observer wires last_wake_at on wake events

**Files:**
- Modify: `src/bin/apollo-optimizerd/main.rs` (or wherever wake events are observed)

- [ ] **Step 1: Locate existing wake event observer**

```bash
grep -rn "WakeRuntimeState\|on_wake\|wake_event\|secs_since_wake" src/bin/apollo-optimizerd/ crates/apollo-engine/src/engine/ | head -10
```

- [ ] **Step 2: Add `subsystems.maintenance_state.observe_wake()` call**

The `WakeRuntimeState` exists in apollo-engine (per CLAUDE.md `wake_state.json` is persisted). Find the on-wake hook:

```bash
grep -rn "WakeRuntimeState\|wake_state\|woken_up\|on_wake" crates/apollo-engine/src/engine/ | head -20
```

In the existing wake-event handler (likely a method like `WakeRuntimeState::record_wake` or a main-loop branch detecting `last_wake_at` change), add:

```rust
subsystems.maintenance_state.observe_wake();
```

> If you genuinely cannot find a wake observer after grepping, fall back: in main.rs, detect a process-relative "woke up" by tracking `Instant`-since-last-tick > 30s (long jump indicates sleep). Wire this into a new helper that calls `subsystems.maintenance_state.observe_wake()`. Do NOT skip this step — the lid-open race is a documented spec risk (🟡 medium) and silently bypassing it weakens the PostWake gate.

- [ ] **Step 3: Verify compile + test**

```bash
cargo check --bin apollo-optimizerd 2>&1 | tail -5
```

Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add -u src/bin/apollo-optimizerd/
git commit -m "feat(maintenance): observe_wake on system wake events (or document no-op)"
```

---

### Task 26: Add DaemonRequest::Purge + DaemonResponse::PurgeResult

**Files:**
- Modify: `crates/apollo-engine/src/engine/protocol.rs`

- [ ] **Step 1: Add request variant**

In `pub enum DaemonRequest`, append:

```rust
/// Trigger an immediate maintenance purge through the daemon.
/// Subject to MaintenanceState rate-limits (5 min CLI + 1 min from any auto-purge).
Purge,
```

- [ ] **Step 2: Add response variant**

In `pub enum DaemonResponse`, append:

```rust
PurgeResult {
    fired: bool,
    reason: String,
},
```

- [ ] **Step 3: Wire is_privileged**

In `impl DaemonRequest::is_privileged`, add `Purge` to the `true` branch (alongside `Restore | PanicRestore`):

```rust
| Self::RevertSysctls
| Self::Purge => true,
```

- [ ] **Step 4: Add roundtrip test**

In `mod tests`:

```rust
#[test]
fn roundtrip_purge() {
    let rt = roundtrip(&DaemonRequest::Purge);
    assert!(matches!(rt, DaemonRequest::Purge));
}

#[test]
fn purge_is_privileged() {
    assert!(DaemonRequest::Purge.is_privileged());
}
```

- [ ] **Step 5: Verify compile + test**

```bash
cargo test -p apollo-engine engine::protocol::tests 2>&1 | tail -10
```

Expected: PASS, including 2 new tests.

- [ ] **Step 6: Commit**

```bash
git add crates/apollo-engine/src/engine/protocol.rs
git commit -m "feat(protocol): DaemonRequest::Purge + DaemonResponse::PurgeResult"
```

---

### Task 27: Socket handler routes Purge via mpsc to main loop

**Files:**
- Modify: `src/bin/apollo-optimizerd/socket_handler.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs` (or wherever main loop runs)

- [ ] **Step 1: Define MainLoopMsg enum**

In `socket_handler.rs` or a shared module imported by both:

```rust
pub enum MainLoopMsg {
    CliPurge {
        response_tx: std::sync::mpsc::Sender<apollo_engine::engine::protocol::DaemonResponse>,
    },
}
```

- [ ] **Step 2: Wire mpsc channel in main.rs at startup**

Find the place in `main.rs` where `socket_handler::serve` (or equivalent) is launched. Just before launch:

```rust
let (main_loop_tx, main_loop_rx) = std::sync::mpsc::channel::<MainLoopMsg>();
```

Pass `main_loop_tx` (a `Sender`) into the socket handler launch site. Keep `main_loop_rx` in main loop scope.

- [ ] **Step 3: Handle Purge variant in socket handler process_request**

Find the request match in `socket_handler.rs` (likely a `match req { ... }`). Add:

```rust
DaemonRequest::Purge => {
    let (response_tx, response_rx) = std::sync::mpsc::channel();
    if main_loop_tx
        .send(MainLoopMsg::CliPurge { response_tx })
        .is_err()
    {
        return DaemonResponse::PurgeResult {
            fired: false,
            reason: "main loop unreachable".into(),
        };
    }
    response_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .unwrap_or(DaemonResponse::PurgeResult {
            fired: false,
            reason: "timeout".into(),
        })
}
```

- [ ] **Step 4: Drain main_loop_rx in main loop epilogue**

After the dispatch tick (and any tick-end housekeeping), in main.rs:

```rust
while let Ok(msg) = main_loop_rx.try_recv() {
    match msg {
        MainLoopMsg::CliPurge { response_tx } => {
            let state = &mut subsystems.maintenance_state;
            let resp = if state.secs_since_cli_purge() < 300 {
                DaemonResponse::PurgeResult {
                    fired: false,
                    reason: format!("rate_limited — wait {}s", 300 - state.secs_since_cli_purge()),
                }
            } else if state.secs_since_any_purge() < 60 {
                DaemonResponse::PurgeResult {
                    fired: false,
                    reason: "rate_limited — auto-purge fired recently".into(),
                }
            } else if std::process::Command::new("purge").spawn().is_ok() {
                state.mark_cli_purged();
                lf_metrics.maintenance_purge_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                DaemonResponse::PurgeResult { fired: true, reason: "ok".into() }
            } else {
                DaemonResponse::PurgeResult {
                    fired: false,
                    reason: "purge spawn failed".into(),
                }
            };
            let _ = response_tx.send(resp);
        }
    }
}
```

- [ ] **Step 5: Verify compile**

```bash
cargo check --bin apollo-optimizerd 2>&1 | tail -10
```

Expected: 0 errors. May need to import `DaemonRequest`, `DaemonResponse` and adjust visibility of `MainLoopMsg`.

- [ ] **Step 6: Verify tests**

```bash
cargo test --bin apollo-optimizerd 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -u src/bin/apollo-optimizerd/
git commit -m "feat(maintenance): socket handler routes Purge via mpsc to main loop"
```

---

### Task 28: 🧠 NotebookLM checkpoint — review IPC channel handoff

- [ ] **Step 1: Query NotebookLM**

Query body:
> Review the IPC channel handoff for DaemonRequest::Purge (just committed). Socket thread (per-connection) creates a one-shot mpsc::channel, sends MainLoopMsg::CliPurge { response_tx } to the daemon main loop's mpsc channel, then blocks on response_rx.recv_timeout(2s). Main loop drains main_loop_rx.try_recv() each cycle epilogue, spawns `purge`, sends response back. Concerns to check: (1) what happens if main loop is mid-tick and takes >2s to drain (cycle p95 was reported at 96ms but worst-case can spike)? (2) what if multiple CLI clients hit Purge simultaneously — is the mpsc fan-in safe? (3) does the response_tx getting dropped cause a panic anywhere? Severity-rank.

- [ ] **Step 2: Apply findings**

If 🟠+, fix inline + amend commit.

- [ ] **Step 3: Annotate progress**

```bash
git commit --allow-empty -m "doc(maintenance): NotebookLM checkpoint 4 (IPC channel) — <verdict>"
```

---

### Task 29: Add `purge` subcommand to apollo-optimizerctl

**Files:**
- Modify: `src/bin/apollo-optimizerctl/main.rs`

- [ ] **Step 1: Locate clap subcommand definitions**

```bash
grep -n "Subcommand\|enum.*Cli\|enum.*Command" src/bin/apollo-optimizerctl/main.rs | head
```

- [ ] **Step 2: Add `Purge` variant**

In the clap subcommand enum, add (with whatever doc style the existing variants use):

```rust
/// Trigger an immediate maintenance purge through the daemon.
/// Rate-limited to 5 minutes between successive invocations.
Purge,
```

- [ ] **Step 3: Add handler arm**

In the `match cli.command { ... }` block, add:

```rust
Commands::Purge => {
    let req = DaemonRequest::Purge;
    let resp = send_request(&req)?;
    match resp {
        DaemonResponse::PurgeResult { fired: true, reason } => {
            println!("✅ purge fired ({reason})");
            Ok(())
        }
        DaemonResponse::PurgeResult { fired: false, reason } => {
            eprintln!("⚠️  purge skipped: {reason}");
            std::process::exit(1);
        }
        DaemonResponse::Error { message } => {
            eprintln!("❌ daemon error: {message}");
            std::process::exit(2);
        }
        other => {
            eprintln!("❌ unexpected response: {other:?}");
            std::process::exit(2);
        }
    }
}
```

> Note: identifiers `Commands`, `send_request` may differ. Match the existing pattern for e.g. `Commands::Restore` or `Commands::Doctor`.

- [ ] **Step 4: Verify compile**

```bash
cargo check --bin apollo-optimizerctl 2>&1 | tail -10
```

Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add src/bin/apollo-optimizerctl/main.rs
git commit -m "feat(ctl): apollo-optimizerctl purge subcommand"
```

---

### Task 30: Full workspace test run + clippy + format

**Files:** none (validation only)

- [ ] **Step 1: Run full test suite**

```bash
cargo test --workspace 2>&1 | tail -25
```

Expected: 0 failures. Total test count should be baseline + ~13 new (8 unit `maintenance_state` + 9 unit `daemon_maintenance_tick` + 3 integration `level3_maintenance_purge` + 2 protocol roundtrip).

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --all-targets --workspace 2>&1 | tail -20
```

Expected: clippy warnings ≤ baseline. Fix any new warnings before commit.

- [ ] **Step 3: Run format check**

```bash
cargo fmt --all -- --check 2>&1 | tail -10
```

Expected: no diff. If formatting needed:

```bash
cargo fmt --all
git add -u
git commit -m "style(maintenance): cargo fmt all"
```

- [ ] **Step 4: If anything failed, fix + amend appropriate prior commit**

Don't introduce a "fix everything" commit; amend the relevant prior commit so each task remains atomic.

- [ ] **Step 5: Build release**

```bash
cargo build --release --workspace 2>&1 | tail -5
```

Expected: `Compiling` … `Finished release` (binaries in `target/release/`). Watch for warnings — anything new should be fixed.

---

### Task 31: 🧠 NotebookLM final pre-deploy review

- [ ] **Step 1: Query NotebookLM**

Query body:
> Final pre-deploy review of the maintenance-purge-gate branch. Files touched: maintenance_state.rs (new), daemon_maintenance_tick.rs (new), lse_counters.rs (+7 atomics), types.rs (+7 RuntimeMetrics), daemon_state.rs (+7 sync lines), learned_state.rs (persist 2 SystemTime), protocol.rs (Purge variant + privileged), daemon_init.rs (DaemonSubsystems +1 field), daemon_survival_tick.rs (asymmetric refactor), socket_handler.rs (mpsc fan-out), main.rs (tick wiring + drain), apollo-optimizerctl main.rs (purge subcommand), level3_maintenance_purge.rs tests. Total ~17 files. Test counts grew baseline + ~15. Are there integration-level gaps not caught by per-file review? Specifically: any feedback loop with the existing 8 predictive subsystems (Kalman/CUSUM/Hazard/MPC/Markov etc.) where firing a maintenance purge could poison their training data because purge counts as an unmodeled exogenous shock? Should the maintenance purge cause string be recorded in CausalGraph? Severity-rank residual concerns.

- [ ] **Step 2: Apply 🟠+ findings**

For each material finding, either fix inline or document explicitly as "deferred to next sprint" in the commit log.

- [ ] **Step 3: Annotate progress**

```bash
git commit --allow-empty -m "doc(maintenance): NotebookLM checkpoint 5 (final review) — <verdict + residual gaps>"
```

---

### Task 32: Spec compliance final pass + dispatch code-quality review subagent

**Files:** none (review only)

- [ ] **Step 1: Re-read spec**

```bash
sed -n '1,50p' docs/superpowers/specs/2026-05-10-maintenance-purge-design.md
sed -n '50,150p' docs/superpowers/specs/2026-05-10-maintenance-purge-design.md
```

For each section ("Goal", "Non-goals", "Architecture", "Threshold logic", "Components", "Telemetry sync chain", "Persistence", "Testing strategy", "Risks ranked", "Backward compatibility", "Decisions log"), verify there is a corresponding implementation or test that covers it. List any gaps in your task tracker.

- [ ] **Step 2: Dispatch code-quality reviewer agent**

Spawn an agent (e.g., via the `Agent` tool with `subagent_type: "general-purpose"`) with prompt:

> You are reviewing the maintenance-purge-gate branch (commits since branching from master). Read the spec at docs/superpowers/specs/2026-05-10-maintenance-purge-design.md and check the diff: `git diff master...sprint5-mes0-maintenance-gate`. Independently identify: (1) any spec requirement not implemented, (2) any code added that's NOT in the spec (scope creep), (3) any code-quality issue the implementer self-review missed. Be specific with file paths and line numbers. Severity-rank.

- [ ] **Step 3: Apply review fixes inline**

Fix material findings in their original commits (via amend) so each task remains atomic and self-contained.

- [ ] **Step 4: Verify final test run still green**

```bash
cargo test --workspace 2>&1 | tail -10
```

Expected: 0 failures.

---

### Task 33: Deploy 3 binaries + smoke test

**Files:** none (deployment only)

> ⚠️ This task touches production state. The user has authorized continuous execution; verify each step before moving on, but do not stop for explicit consent.

- [ ] **Step 1: Confirm binaries built**

```bash
ls -la target/release/apollo-optimizerd target/release/apollo-optimizerctl target/release/apollo-menubar
```

Expected: 3 fresh binaries (modified within last 30 min).

- [ ] **Step 2: Backup current installed daemon**

```bash
sudo cp /usr/local/libexec/apollo-optimizerd /usr/local/libexec/apollo-optimizerd.bak.$(date -u +%Y%m%d-%H%M)
sudo cp /usr/local/bin/apollo-optimizerctl /usr/local/bin/apollo-optimizerctl.bak.$(date -u +%Y%m%d-%H%M)
```

- [ ] **Step 3: Install new binaries**

```bash
sudo cp target/release/apollo-optimizerd /usr/local/libexec/apollo-optimizerd
sudo cp target/release/apollo-optimizerctl /usr/local/bin/apollo-optimizerctl
sudo cp target/release/apollo-menubar /Applications/Apollo\ Menubar.app/Contents/MacOS/apollo-menubar 2>/dev/null || echo "menubar app bundle not at default path; skip if user uses different deploy"
```

> If the menubar app path differs, ask the user to identify the install location. The daemon and ctl are mandatory; menubar is optional for this feature.

- [ ] **Step 4: Restart daemon (bootout/bootstrap to refresh codesign)**

```bash
sudo launchctl bootout system/com.eduardocortez.systemoptimizerd
sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist
sleep 3
sudo launchctl print system/com.eduardocortez.systemoptimizerd | head -30
```

Expected: state = running, last exit reason normal.

- [ ] **Step 5: Smoke test — daemon health**

```bash
sudo apollo-optimizerctl status 2>&1 | head -20
```

Expected: cycles incrementing, failures=0, last_error=None.

- [ ] **Step 6: Smoke test — IPC purge command**

```bash
sudo apollo-optimizerctl purge
```

Expected: `✅ purge fired (ok)` OR `⚠️  purge skipped: rate_limited — ...` (latter if a recent auto-purge happened, also acceptable).

- [ ] **Step 7: Smoke test — runtime metrics include new counters**

```bash
sudo cat /var/lib/apollo/runtime_metrics.json | python3 -m json.tool | grep maintenance
```

Expected: 7 maintenance_purge* fields present (values may be 0 or N depending on whether tick fired).

- [ ] **Step 8: Smoke test — second purge in 2 min returns rate_limited**

```bash
sudo apollo-optimizerctl purge
```

Expected: `⚠️  purge skipped: rate_limited — wait NNNs`.

- [ ] **Step 9: Wait 5 minutes, third purge succeeds (manual confirmation)**

Document this step — user verifies later that after 5 min, `sudo apollo-optimizerctl purge` returns `ok` again.

- [ ] **Step 10: Commit deployment notes**

```bash
git commit --allow-empty -m "deploy(maintenance): bootout/bootstrap maintenance-purge-gate

Smoke tests:
- apollo-optimizerctl status: healthy, failures=0
- apollo-optimizerctl purge (first): <result>
- apollo-optimizerctl purge (rapid second): rate_limited as expected
- runtime_metrics.json: 7 maintenance_purge* counters present
"
```

---

### Task 34: Merge to master + push

**Files:** none

- [ ] **Step 1: Verify branch state**

```bash
git status
git log --oneline master..sprint5-mes0-maintenance-gate | head -40
```

Expected: clean working tree; ~30+ commits on this branch.

- [ ] **Step 2: Switch to master + merge**

```bash
git switch master
git merge --no-ff sprint5-mes0-maintenance-gate -m "Merge: Maintenance Purge Gate (sprint5-mes0-maintenance-gate)

Adds opportunistic non-crisis purge tick + apollo-optimizerctl purge CLI.
Asymmetric cooldown (survival writes-only, maintenance reads+writes).
Closes user pain: M1 8GB users no longer need manual sudo purge under
sustained moderate pressure.

Files: ~17 modified/created across apollo-engine + apollo-optimizerd + ctl.
Tests: +~15.
NotebookLM checkpoints: 5 passes (round 2 spec patch + 4 in-flight).
"
```

- [ ] **Step 3: Push (only if explicitly user-authorized)**

User said \"no soy tu guarderia\" earlier → continuous execution is authorized. Push:

```bash
git push origin master
```

Expected: master pushed to remote.

- [ ] **Step 4: Update MEMORY.md index**

Read `/Users/eduardocortez/.claude/projects/-Users-eduardocortez-proyectos-system-optimizer/memory/MEMORY.md`, append a one-line entry under "Estado actual" pointing to a new memory file (you write that file in Task 35).

- [ ] **Step 5: Final clean-up check**

```bash
sudo apollo-optimizerctl status 2>&1 | tail -20
```

Expected: still healthy 30+ minutes after deploy.

---

### Task 35: Write memory entry + close

**Files:**
- Create: `/Users/eduardocortez/.claude/projects/-Users-eduardocortez-proyectos-system-optimizer/memory/project_maintenance_purge_gate.md`
- Modify: `/Users/eduardocortez/.claude/projects/-Users-eduardocortez-proyectos-system-optimizer/memory/MEMORY.md`

- [ ] **Step 1: Write memory entry**

```markdown
---
name: Maintenance Purge Gate (2026-05-10)
description: Apollo opportunistic non-crisis purge tick + apollo-optimizerctl purge CLI; asymmetric survival/maintenance cooldown; addresses M1 8GB sudo-purge user pain
type: project
---

## Maintenance Purge Gate — 2026-05-10

**Branch merged:** `sprint5-mes0-maintenance-gate` → `master`
**Why:** M1 8GB users had to run `sudo purge` manually because Apollo's existing survival-mode purge only fires under crisis (pressure ≥ 0.85). Maintenance purge fires at 0.65 ≤ raw < 0.85 with strict guards.
**How to apply:** see spec at `docs/superpowers/specs/2026-05-10-maintenance-purge-design.md`. Asymmetric cooldown — survival writes `last_any_purge_at` but never reads it (physical-crisis sovereign); maintenance reads+writes (yields).

### Thresholds (post-NotebookLM r2 hardening)
- Pressure: 0.65 ≤ raw < 0.85 (uses raw memory_pressure, NOT effective)
- Swap floor: max(1.5 GB, 50% × swap_total) — relaxed from initial 2 GB after NotebookLM r2 flagged rigidity for M1 8GB typical swap_total range
- Swap delta sustained < 256 KB/s for 90s (via new SwapDeltaWindow VecDeque, CAP=45 samples × 2s)
- User idle ≥120s + 10s post-wake quiet
- Build mode bypass via dev_runtime_active
- Rate-limit 30 min asymmetric

### CLI
- `apollo-optimizerctl purge` — privileged IPC, mpsc channel handoff to main loop
- 5-min CLI-specific rate-limit + 1-min auto-purge spacing

### NotebookLM checkpoints (5 passes)
- r1 spec: GO with 3 mandatory fixes integrated
- r2 spec: surfaced shared-cooldown deadlock + 2 GB rigidity → spec patched
- r3 in-flight: MaintenanceState API, survival refactor, tick logic, IPC channel, final review

### Telemetry sync chain (Sprint 3 lesson explicit)
- 7 atomic counters: maintenance_purge_total + 6 skip-reason counters
- Round-trip integration test: literal JSON substring assertions for all 7
```

- [ ] **Step 2: Add MEMORY.md entry**

Append (or replace existing 2026-05-09 line) under "Estado actual":

```markdown
## Estado actual (2026-05-10) — Maintenance Purge Gate merged
- [Maintenance Purge Gate](project_maintenance_purge_gate.md) — closes manual `sudo purge` pain on M1 8GB
- Branch `sprint5-mes0-maintenance-gate` → `master`
- Asymmetric cooldown, 1.5 GB swap floor, 256 KB/s × 90s window
```

- [ ] **Step 3: Verify memory file structure**

```bash
ls -la /Users/eduardocortez/.claude/projects/-Users-eduardocortez-proyectos-system-optimizer/memory/project_maintenance_purge_gate.md
wc -l /Users/eduardocortez/.claude/projects/-Users-eduardocortez-proyectos-system-optimizer/memory/MEMORY.md
```

Expected: file exists, MEMORY.md still ≤ ~210 lines (truncation warning kicks in at 200; trim if needed).

---

## Verification end-to-end

After Task 35:

1. **Functional**:
   - `cargo test --workspace` — 0 failures, +~15 tests over baseline
   - `cargo clippy --all-targets` — warnings ≤ baseline
   - Build release — succeeds
2. **Production**:
   - `sudo apollo-optimizerctl status` — healthy, failures=0
   - `sudo apollo-optimizerctl purge` — fires or rate_limited as expected
   - `runtime_metrics.json` — 7 maintenance_purge* counters present
   - 30-min monitor post-deploy — no new last_error, p95_cycle stable
3. **Spec compliance**:
   - All 11 spec sections covered by code or test
   - 5 NotebookLM checkpoint annotations in commit log
4. **MEMORY**:
   - `project_maintenance_purge_gate.md` written
   - `MEMORY.md` index updated

## Rollback

If post-deploy issues:

```bash
sudo cp /usr/local/libexec/apollo-optimizerd.bak.* /usr/local/libexec/apollo-optimizerd
sudo cp /usr/local/bin/apollo-optimizerctl.bak.* /usr/local/bin/apollo-optimizerctl
sudo launchctl bootout system/com.eduardocortez.systemoptimizerd
sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist
sudo touch /var/run/apollo.disable  # also pauses optimization if needed
```

Branch already merged to master — for full code rollback, use `git revert -m 1 <merge-sha>`.
