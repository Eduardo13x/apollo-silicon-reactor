# Maintenance Purge Gate — Design Spec

**Date:** 2026-05-10
**Sprint:** post Sprint 5 Mes 0 — opportunistic feature commit
**Status:** Design synthesized from NotebookLM + Skeptic agent reviews, ready for implementation plan

## Goal

Apollo's existing auto-purge in `daemon_survival_tick.rs` only fires under crisis conditions (pressure ≥ 0.85 OR swap_delta > 1 MB/s OR p_oom_30s ≥ 0.80). Users on M1 8 GB report needing manual `sudo purge` because swap accumulates "stickily" at moderate pressures (0.55-0.70) without triggering survival mode. macOS keeps inactive pages resident rather than releasing them; Apollo currently has no maintenance-tier purge to reclaim them.

This spec adds:

1. **`daemon_maintenance_tick`** — a new per-cycle tick that fires `purge` under non-crisis but sustained-pressure conditions, with strict guards against disrupting active workloads.
2. **`apollo-optimizerctl purge`** — a CLI command for explicit user-triggered purge through the daemon (audited, rate-limited, no `sudo purge` workaround).
3. **Shared rate-limit infrastructure** — both survival and maintenance ticks consult a single `last_any_purge_at` timestamp on `MaintenanceState`, persisted in `LearnedState`, to prevent simultaneous double-purge.

## Non-goals

- ❌ Lower the survival_mode pressure threshold (0.85) — would cascade into Chromium demote / SIGSTOP cascades (rejected: "death loop" precedent in session 2026-04-30, NotebookLM concern).
- ❌ Replace existing `daemon_survival_tick` purge — the crisis-mode purge stays unchanged in semantics.
- ❌ Use effective_pressure (raw + boost factors) for the gate — purge fixes memory pressure only, not thermal/llm/hw pressure (Skeptic agent verdict, overrides NotebookLM's initial recommendation on this point).
- ❌ Make CLI command `--force` bypass rate-limit — user can already `sudo purge` if they want to override; daemon stays conservative.
- ❌ Implement causal A/B validation in this spec — `system_maintenance_purge` cause string emitted to CausalGraph, but outcome validation is observational only (≥30 samples per CLAUDE.md supervision rule before trusting).

## Architecture

```
src/bin/apollo-optimizerd/
└── daemon_maintenance_tick.rs       NEW (~120 LoC)
    └── pub fn run_maintenance_tick(...)

crates/apollo-engine/src/engine/
├── maintenance_state.rs              NEW (~80 LoC)
│   ├── struct MaintenanceState
│   └── struct SwapDeltaWindow
├── learned_state.rs (modify)         persist last_any_purge_at + last_cli_purge_at
├── lse_counters.rs (modify)          7 new atomic counters
├── types.rs (modify)                 7 new RuntimeMetrics fields
├── daemon_state.rs (modify)          7 new sync_from_lockfree flush lines
└── protocol.rs (modify)              DaemonRequest::Purge + DaemonResponse::PurgeResult

src/bin/apollo-optimizerd/
├── daemon_init.rs (modify)           DaemonSubsystems gets pub maintenance_state
├── daemon_survival_tick.rs (modify)  Update last_any_purge_at on survival fire (was local)
├── socket_handler.rs (modify)        process_request handler for Purge variant
└── main.rs (modify)                  Insert run_maintenance_tick call between ticks
```

### Crate boundaries

- `MaintenanceState` + `SwapDeltaWindow` live in `apollo-engine` (testable in isolation; survives crate split).
- `daemon_maintenance_tick` lives in `apollo-optimizerd` bin (orchestrates engine state with daemon-only concerns).
- `protocol.rs` change is engine-side (wire format).

### Tick ordering invariant

`survival_tick → maintenance_tick → dispatch_tick` (strict serial). Both purge ticks read/write the **same** `last_any_purge_at` field on `MaintenanceState`. If survival fires in cycle N, it updates the shared timestamp; maintenance_tick in the same cycle reads it and skips on rate-limit. **No flag-based "first wins"** — flags drift across forks/restarts; a single timestamp is the source of truth.

## Threshold logic (post-Skeptic hardening)

```rust
// daemon_maintenance_tick.rs — gate evaluation
fn should_fire(snap: &SystemSnapshot, ctx: &UserContext, state: &MaintenanceState) -> Option<SkipReason> {
    let raw_pressure = snap.pressure.memory_pressure;        // RAW kernel compressor ratio
    let swap_used    = snap.pressure.swap_used_bytes;
    let swap_total   = snap.pressure.swap_total_bytes;

    // 1. Pressure window: Elevated zone, NOT survival
    //    Lower bound 0.65 = avoid spurious purges in idle.
    //    Upper bound 0.85 = clean handoff to survival_tick (no overlap).
    if raw_pressure < 0.65 { return Some(SkipReason::PressureLow); }
    if raw_pressure >= 0.85 { return Some(SkipReason::PressureSurvival); }

    // 2. Absolute swap floor — closes M1 cold-boot calibration trap
    //    macOS dynamically grows swap_total; cold-boot 800MB allocation would
    //    fire at 50% with only 400MB swap, which is trivial. Floor at 2 GB.
    let swap_floor = std::cmp::max(2 * 1024 * 1024 * 1024, swap_total / 2);
    if swap_used < swap_floor { return Some(SkipReason::SwapFloor); }

    // 3. Swap delta sustained 90s via NEW window struct.
    //    Threshold 256 KB/s (100 KB/s is below collector noise on M1).
    //    90s window filters single-cycle anomalies that 60s would let through.
    if !state.swap_delta_window.sustained_below(256_000.0, 90) {
        return Some(SkipReason::Growing);
    }

    // 4. User idle ≥120s AND no wake event in last 10s.
    //    User opening lid at 119s → idle drops to 0; need wake quiet period
    //    to avoid purge firing on stale idle reading.
    if !ctx.is_idle_long() { return Some(SkipReason::Idle); }
    if state.secs_since_wake() < 10 { return Some(SkipReason::PostWake); }

    // 5. Build mode bypass — page cache invalidation during cargo/Xcode
    //    re-reads pages from SSD next access, hurting build wall time.
    if ctx.dev_runtime_active() { return Some(SkipReason::BuildMode); }

    // 6. Shared rate-limit (30 min since ANY purge: survival/maintenance/CLI)
    //    macOS Sequoia throttles repeat purges; 30 min prevents wasted spawn.
    if state.secs_since_any_purge() < 1800 { return Some(SkipReason::RateLimit); }

    None  // all gates passed → fire
}
```

### Threshold rationale (vs originally proposed)

| Param | Original | Final | Reason for change |
|---|---|---|---|
| Pressure source | effective | **raw** | Skeptic: purge addresses memory pressure only; effective pressure includes thermal/hw/llm/battery boosts that purge cannot fix. NotebookLM initially recommended effective for "consistency"; Skeptic argument wins on physical correctness. |
| Pressure window | ≥ 0.65 | **0.65 ≤ p < 0.85** | Clean handoff to survival_tick — no ambiguous overlap zone. |
| Swap pct | > 50% × total | **max(2 GB, 50% × total)** | Absolute 2 GB floor closes M1 cold-boot trap (small swap_total). |
| Swap delta | < 100 KB/s for 60s | **< 256 KB/s for 90s** | 100 KB/s is below collector noise floor on M1; 90s eliminates single-cycle anomalies that 60s could miss. |
| Idle gate | > 120s standalone | **UserContext::is_idle_long()** + 10s post-wake quiet | Reuse existing semantic + close lid-open race. |
| Build bypass | none | **skip if dev_runtime_active** | Prevent page cache invalidation during cargo/Xcode builds. |
| Rate-limit | 20 min isolated | **30 min shared** with survival | macOS purge throttling + SSD wear; single source of truth via `last_any_purge_at`. |

### CLI rate-limit (separate from auto-purge)

CLI `apollo-optimizerctl purge` has its own counter:

- `last_cli_purge_at` (separate from `last_any_purge_at`).
- Limit: 5 minutes between successive CLI invocations.
- Also enforces 1-minute spacing from any auto-purge (read `last_any_purge_at` too).
- No `--force` flag — explicit refusal communicated via `PurgeResult { fired: false, reason: "rate_limited" }`.
- User can `sudo purge` directly if they want to bypass; that's an explicit OS-level choice they own.

## Components

### `MaintenanceState` struct (in `apollo-engine`)

```rust
// crates/apollo-engine/src/engine/maintenance_state.rs

use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaintenanceState {
    /// 90s rolling window of (timestamp, swap_delta_bps) samples.
    /// Cap = 45 samples × 2s tick = 90s. Older samples drop on push.
    #[serde(skip)]                              // not persisted (rebuilds on warm-up)
    pub swap_delta_window: SwapDeltaWindow,

    /// Wall-clock timestamp of last successful purge from any tick or CLI.
    /// SystemTime (not Instant) survives sleep/wake.
    #[serde(default)]
    pub last_any_purge_at: Option<SystemTime>,

    /// Wall-clock timestamp of last CLI-triggered purge. Separate so CLI
    /// rate limit (5 min) is independent of auto-purge schedule (30 min).
    #[serde(default)]
    pub last_cli_purge_at: Option<SystemTime>,

    /// Instant of last observed system wake event (set by main loop's
    /// WakeRuntimeState observer). Used to enforce 10s post-wake quiet
    /// period before any maintenance decision.
    #[serde(skip)]
    pub last_wake_at: Option<Instant>,
}

impl MaintenanceState {
    pub fn new() -> Self { Self::default() }

    /// Push a new swap_delta_bps sample. Drops oldest when window full.
    pub fn push_swap_delta(&mut self, delta_bps: f64) {
        let now = SystemTime::now();
        self.swap_delta_window.push(now, delta_bps);
    }

    /// Seconds since last purge of any kind. Returns u64::MAX if never purged.
    /// Defensive: if SystemTime is backwards (clock change), returns 0.
    pub fn secs_since_any_purge(&self) -> u64 {
        match self.last_any_purge_at {
            None => u64::MAX,
            Some(t) => SystemTime::now()
                .duration_since(t)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    pub fn secs_since_cli_purge(&self) -> u64 { /* same pattern */ }

    /// Seconds since last wake event. Uses Instant (intentionally).
    /// Wake events update Instant; cooldown should not advance during sleep.
    pub fn secs_since_wake(&self) -> u64 {
        match self.last_wake_at {
            None => u64::MAX,
            Some(t) => t.elapsed().as_secs(),
        }
    }

    /// Mark a purge fired (any source). Updates shared timestamp.
    pub fn mark_purged(&mut self) {
        self.last_any_purge_at = Some(SystemTime::now());
    }

    /// Mark CLI-specific purge (also updates shared).
    pub fn mark_cli_purged(&mut self) {
        let now = SystemTime::now();
        self.last_cli_purge_at = Some(now);
        self.last_any_purge_at = Some(now);
    }

    pub fn observe_wake(&mut self) {
        self.last_wake_at = Some(Instant::now());
    }
}

#[derive(Debug, Clone, Default)]
pub struct SwapDeltaWindow {
    samples: VecDeque<(SystemTime, f64)>,
}

impl SwapDeltaWindow {
    const CAP: usize = 45;  // 90s at 2s tick

    pub fn push(&mut self, t: SystemTime, delta_bps: f64) {
        if self.samples.len() >= Self::CAP {
            self.samples.pop_front();
        }
        self.samples.push_back((t, delta_bps));
    }

    /// True iff ALL samples in the last `secs` are below threshold AND
    /// window has at least `secs` worth of history (no early-fire on warm-up).
    pub fn sustained_below(&self, threshold_bps: f64, secs: u64) -> bool {
        let cutoff = match SystemTime::now().checked_sub(Duration::from_secs(secs)) {
            Some(t) => t,
            None => return false,
        };

        // Filter samples within the last `secs` window
        let recent: Vec<&(SystemTime, f64)> = self.samples
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .collect();

        // Need enough samples to span the window
        // (at 2s tick, secs/2 samples expected)
        let min_samples = (secs / 2).max(1) as usize;
        if recent.len() < min_samples {
            return false;
        }

        recent.iter().all(|(_, bps)| *bps < threshold_bps)
    }

    pub fn len(&self) -> usize { self.samples.len() }
    pub fn is_empty(&self) -> bool { self.samples.is_empty() }
}
```

### `daemon_maintenance_tick` orchestration

```rust
// src/bin/apollo-optimizerd/daemon_maintenance_tick.rs

use apollo_engine::collector::SystemSnapshot;
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::maintenance_state::MaintenanceState;
use apollo_engine::engine::user_context::UserContext;
use std::sync::atomic::Ordering;

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

pub fn run_maintenance_tick(
    snap: &SystemSnapshot,
    ctx: &UserContext,
    state: &mut MaintenanceState,
    lf_metrics: &LockFreeMetrics,
) {
    // Update window with this cycle's delta (always — gate evaluation needs history)
    state.push_swap_delta(snap.pressure.swap_delta_bytes_per_sec);

    let decision = should_fire(snap, ctx, state);
    match decision {
        None => {
            // All gates passed — fire purge
            if std::process::Command::new("purge").spawn().is_ok() {
                state.mark_purged();
                lf_metrics.maintenance_purge_total
                    .fetch_add(1, Ordering::Relaxed);
                tracing::info!(
                    target: "apollo.maintenance",
                    swap_used_gb = snap.pressure.swap_used_bytes as f64 / 1_073_741_824.0,
                    pressure = snap.pressure.memory_pressure,
                    "maintenance purge fired"
                );
                // CausalGraph cause string handled by main loop post-tick drain
                // (consistent with existing purge_purgeable pattern at chromium_manager)
            }
            // If spawn fails: silent (best-effort), don't increment counter
        }
        Some(reason) => {
            let counter = match reason {
                SkipReason::PressureLow | SkipReason::PressureSurvival => &lf_metrics.maintenance_purge_skipped_pressure_total,
                SkipReason::SwapFloor => &lf_metrics.maintenance_purge_skipped_swap_floor_total,
                SkipReason::Growing => &lf_metrics.maintenance_purge_skipped_growing_total,
                SkipReason::Idle | SkipReason::PostWake => &lf_metrics.maintenance_purge_skipped_idle_total,
                SkipReason::BuildMode => &lf_metrics.maintenance_purge_skipped_build_mode_total,
                SkipReason::RateLimit => &lf_metrics.maintenance_purge_skipped_rate_limit_total,
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn should_fire(snap: &SystemSnapshot, ctx: &UserContext, state: &MaintenanceState) -> Option<SkipReason> {
    // (logic from "Threshold logic" section above)
}
```

### Tick wiring in main.rs

```rust
// src/bin/apollo-optimizerd/main.rs (excerpt)

// Existing call:
daemon_survival_tick::run_survival_tick(
    &snapshot, &signal_digest, cycle_count,
    /* ... */
    &mut state.maintenance_state,  // CHANGED: was last_purge_at: &mut Option<Instant>
    /* ... */
);

// NEW call inserted between survival and dispatch:
daemon_maintenance_tick::run_maintenance_tick(
    &snapshot,
    &policy_context.user_ctx,
    &mut state.maintenance_state,
    &lf_metrics,
);

// CausalGraph cause record (post-tick)
if last_action_was_maintenance_purge {
    causal_graph.record_action_with_resources(
        "system_maintenance_purge",
        snapshot.pressure.memory_pressure,
        /* ... */
    );
}

// dispatch_tick continues unchanged
```

### Survival tick refactor

`daemon_survival_tick::run_survival_tick` currently takes `last_purge_at: &mut Option<Instant>`. Replace with `&mut MaintenanceState` and update the survival purge spawn to call `state.mark_purged()` (which uses `SystemTime` shared with maintenance).

### CLI command — `apollo-optimizerctl purge`

```rust
// crates/apollo-engine/src/engine/protocol.rs (additions)

pub enum DaemonRequest {
    // ... existing variants ...
    Purge,
}

pub enum DaemonResponse {
    // ... existing variants ...
    PurgeResult { fired: bool, reason: String },
}

// is_privileged() must return true for Purge — same as Restore/PanicRestore
impl DaemonRequest {
    pub fn is_privileged(&self) -> bool {
        match self {
            // ... existing matches ...
            DaemonRequest::Purge => true,
        }
    }
}
```

```rust
// src/bin/apollo-optimizerd/socket_handler.rs (in process_request)

DaemonRequest::Purge => {
    if !is_peer_root(&stream)? {
        return DaemonResponse::PurgeResult {
            fired: false,
            reason: "unauthorized — root required".into(),
        };
    }

    // Send to main loop via channel (avoid spawning purge from socket thread)
    let (response_tx, response_rx) = std::sync::mpsc::channel();
    main_loop_tx.send(MainLoopMsg::CliPurge { response_tx })?;
    response_rx.recv_timeout(Duration::from_secs(2))
        .unwrap_or(DaemonResponse::PurgeResult {
            fired: false,
            reason: "timeout".into(),
        })
}
```

```rust
// src/bin/apollo-optimizerd/main.rs (in cycle epilogue)

// Drain main_loop messages
while let Ok(msg) = main_loop_rx.try_recv() {
    match msg {
        MainLoopMsg::CliPurge { response_tx } => {
            let state = &mut subsystems.maintenance_state;
            // Two rate-limits: CLI-specific (5 min) AND shared cooldown (1 min from any auto-purge)
            if state.secs_since_cli_purge() < 300 {
                let _ = response_tx.send(DaemonResponse::PurgeResult {
                    fired: false,
                    reason: format!("rate_limited — wait {}s", 300 - state.secs_since_cli_purge()),
                });
            } else if state.secs_since_any_purge() < 60 {
                let _ = response_tx.send(DaemonResponse::PurgeResult {
                    fired: false,
                    reason: "rate_limited — auto-purge fired recently".into(),
                });
            } else if std::process::Command::new("purge").spawn().is_ok() {
                state.mark_cli_purged();
                lf_metrics.maintenance_purge_total.fetch_add(1, Ordering::Relaxed);
                let _ = response_tx.send(DaemonResponse::PurgeResult {
                    fired: true,
                    reason: "ok".into(),
                });
            } else {
                let _ = response_tx.send(DaemonResponse::PurgeResult {
                    fired: false,
                    reason: "spawn failed".into(),
                });
            }
        }
    }
}
```

```rust
// src/bin/apollo-optimizerctl/main.rs (CLI handler)

Cli::Purge => {
    let req = DaemonRequest::Purge;
    let resp = send_request(req)?;
    match resp {
        DaemonResponse::PurgeResult { fired: true, .. } => {
            println!("✅ purge fired");
        }
        DaemonResponse::PurgeResult { fired: false, reason } => {
            eprintln!("⚠️  purge skipped: {}", reason);
            std::process::exit(1);
        }
        _ => {
            eprintln!("unexpected response from daemon");
            std::process::exit(2);
        }
    }
}
```

## Telemetry sync chain (Sprint 3 lesson — explicit)

Seven new counters, each must traverse the full 4-tier chain:

```
Tier 1 — atomics (lse_counters.rs):
  pub maintenance_purge_total: AtomicU64
  pub maintenance_purge_skipped_pressure_total: AtomicU64
  pub maintenance_purge_skipped_swap_floor_total: AtomicU64
  pub maintenance_purge_skipped_growing_total: AtomicU64
  pub maintenance_purge_skipped_idle_total: AtomicU64
  pub maintenance_purge_skipped_build_mode_total: AtomicU64
  pub maintenance_purge_skipped_rate_limit_total: AtomicU64

Tier 2 — MetricsSnapshot fields (snapshot() method, .load(Ordering::Relaxed) each):
  ↑ same 7 fields

Tier 3 — RuntimeMetrics fields (types.rs):
  ↑ same 7 fields with #[serde(default)] u64

Tier 4 — sync_from_lockfree (daemon_state.rs):
  self.metrics.maintenance_purge_total = lf.maintenance_purge_total;
  self.metrics.maintenance_purge_skipped_pressure_total = lf.maintenance_purge_skipped_pressure_total;
  ... 5 more lines (one per counter)

Verification:
  Integration test asserts JSON serialization contains literal counter values.
  See "Testing strategy" below.
```

## Persistence

Two fields persist via `LearnedState::collect()/apply()`:

- `last_any_purge_at: Option<SystemTime>`
- `last_cli_purge_at: Option<SystemTime>`

Why persist:
- Rate-limit must survive daemon crash + restart. Otherwise: crash → restart → maintenance fires within 30 min of previous.
- `SwapDeltaWindow` does NOT persist. It's a 90s rolling window — let it warm up after restart (~90s). Premature fire risk is bounded by other gates anyway.
- `last_wake_at: Option<Instant>` does NOT persist (Instant is process-relative; meaningless across restarts).

`LearnedState::self_improve()` does not need special handling for these fields — they're scalar timestamps, no growth to prune.

## Testing strategy

### Unit tests

In `crates/apollo-engine/src/engine/maintenance_state.rs`:

1. `swap_delta_window_sustained_below_with_full_window_returns_true` — feed 45 samples below threshold over 90s, assert true.
2. `swap_delta_window_sustained_below_with_one_spike_returns_false` — feed 30 below + 1 spike, assert false.
3. `swap_delta_window_sustained_below_empty_returns_false` — empty window, assert false.
4. `swap_delta_window_sustained_below_partial_window_returns_false` — feed 10 samples (only 20s), assert false (insufficient history).
5. `swap_delta_window_drops_oldest_at_capacity` — feed 50 samples, assert len == 45, oldest dropped.
6. `secs_since_any_purge_none_returns_max` — fresh state, assert u64::MAX.
7. `secs_since_any_purge_clock_backwards_returns_zero` — set timestamp to future, assert 0 (defensive).
8. `mark_cli_purged_updates_both_timestamps` — assert both `last_cli_purge_at` and `last_any_purge_at` updated.

In `src/bin/apollo-optimizerd/daemon_maintenance_tick.rs`:

9. `should_fire_all_gates_pass` — synthetic snapshot with all gates green, assert returns None (fire).
10. `should_fire_pressure_below_returns_pressure_low` — pressure 0.55, assert SkipReason::PressureLow.
11. `should_fire_pressure_at_survival_returns_pressure_survival` — pressure 0.90, assert SkipReason::PressureSurvival.
12. `should_fire_swap_floor_traps_m1_cold_boot` — swap_total=800MB, swap_used=500MB, pressure=0.70, assert SkipReason::SwapFloor.
13. `should_fire_growing_swap_returns_growing` — window with recent spike, assert SkipReason::Growing.
14. `should_fire_user_active_returns_idle` — idle_secs=10, assert SkipReason::Idle.
15. `should_fire_post_wake_quiet_returns_postwake` — last_wake_at = 5s ago, assert SkipReason::PostWake.
16. `should_fire_build_mode_returns_build_mode` — dev_runtime_active=true, assert SkipReason::BuildMode.
17. `should_fire_rate_limit_returns_rate_limit` — last_any_purge_at = 100s ago, assert SkipReason::RateLimit.

### Integration tests

In `crates/apollo-engine/tests/level3_maintenance_purge.rs` (NEW):

18. `maintenance_purge_skipped_when_swap_total_low` — synthetic SystemSnapshot with swap_total=800MB, swap_used=500MB, pressure=0.70, idle=200s, delta=0. Run maintenance_tick. Assert: zero purge spawn (mock or skip the spawn), `maintenance_purge_skipped_swap_floor_total == 1`.

19. `maintenance_purge_window_requires_90s_sustained` — feed window with 30 samples at delta=50_000 (under threshold). Assert `sustained_below(256000, 90) == false` (only 60s history). Feed 15 more samples. Assert `sustained_below(256000, 90) == true`.

20. `maintenance_counters_round_trip_to_runtime_metrics_json` — increment all 7 counters atomically with prime numbers (1, 2, 3, 5, 7, 11, 13). Trigger snapshot → sync_from_lockfree → serde_json::to_string. Assert literal substrings present in JSON: `"maintenance_purge_total":1`, `"maintenance_purge_skipped_pressure_total":2`, etc. **This test catches Sprint 3 telemetry-death pattern explicitly.**

### Smoke tests (post-deploy, manual)

21. `apollo-optimizerctl purge` from terminal returns `PurgeResult { fired: true, reason: "ok" }` and journal logs entry.
22. Run `apollo-optimizerctl purge` twice within 2 minutes — second returns `rate_limited`.
23. Wait 5+ minutes from second invocation — third `purge` returns `ok`.
24. Confirm runtime_metrics.json includes all 7 maintenance counters with non-stale values after some hours of operation.

## Risks ranked

| Risk | Severity | Mitigation |
|---|---|---|
| Sprint 3 telemetry-death pattern repeats (counters increment but never reach JSON) | 🔴 Critical | Integration test #20 explicitly verifies all 7 counters round-trip with literal JSON substrings. |
| `swap_used > 50% × swap_total` cold-boot trap on M1 | 🔴 Critical | Absolute floor `max(2 GB, 0.5 × total)`. Integration test #18 catches it. |
| "Sustained 60s" silently uses single-cycle data (no window infrastructure) | 🔴 Critical | NEW `SwapDeltaWindow` struct with explicit 90s window + min_samples requirement. Unit tests #1-5 cover. |
| Two purge code paths fire simultaneously | 🟠 High | Single shared `last_any_purge_at` on `MaintenanceState`. Both ticks read/write same field via `&mut MaintenanceState`. |
| CLI purge from socket thread races main-loop tick | 🟠 High | Channel-based handoff: socket thread sends `MainLoopMsg::CliPurge` with `response_tx`; main loop drains and replies. No direct spawn from socket thread. |
| Build mode page cache invalidation | 🟠 High | `dev_runtime_active` gate skips during cargo/Xcode/etc. Existing `safety::infrastructure_processes` pattern reused. |
| Wake event race (lid open at 119s idle) | 🟡 Medium | 10s post-wake quiet period via `MaintenanceState::secs_since_wake()`. `WakeRuntimeState` observer wires to `state.observe_wake()`. |
| Effective vs raw pressure ambiguity | 🟡 Medium | Spec mandates raw `snap.pressure.memory_pressure`. Skeptic agent argument over NotebookLM's effective recommendation: purge addresses memory only, not thermal/llm/hw. |
| Causal A/B undersampled for validation | 🟢 Low | `system_maintenance_purge` recorded in CausalGraph. Mark "Preliminary" until N≥30 samples per CLAUDE.md supervision rule. |
| User runs CLI purge 6× in 30 min (spam) | 🟢 Low | 5-min CLI rate limit. No `--force` flag. User can `sudo purge` directly if they want OS-level bypass. |

## Backward compatibility

- New fields in `RuntimeMetrics` use `#[serde(default)]` — old `runtime_metrics.json` files still deserialize.
- New `DaemonRequest::Purge` variant — old `apollo-optimizerctl` clients won't send it (they don't know about it). Daemon handles all known variants; unknown variants would already error today.
- `apollo-optimizerctl` binary gets new `purge` subcommand — old daemon would respond with "unknown request". For seamless rollout, deploy daemon first, then ctl client.

## Decisions log

| Q | Decision | Rationale |
|---|---|---|
| Pressure source | **Raw** (snap.pressure.memory_pressure) | Skeptic: purge fixes memory only. Effective pressure includes thermal/llm/hw boosts that purge cannot address. NotebookLM had recommended effective for "system consistency"; rejected on physical correctness grounds. |
| Window length | 90s, threshold 256 KB/s | 100 KB/s is below collector noise. 60s missed single-cycle anomalies. Skeptic recommendation. |
| Rate limit | 30 min shared via `last_any_purge_at` | macOS Sequoia throttles repeat purges; SSD wear; single source of truth prevents survival/maintenance double-fire. |
| Persistence | Yes for timestamps, No for window | Window warms up in 90s after restart; cooldown must survive crash to prevent fire-on-restart. |
| Build bypass | Yes (`dev_runtime_active`) | Page cache invalidation during cargo/Xcode regresses build wall time. |
| CLI rate limit | 5 min separate from auto | User intent is different concern from background scheduling. |
| `--force` flag on CLI | No | User can `sudo purge` if they really want; daemon stays conservative. |

## References

- NotebookLM peer review 2026-05-10 (notebook `8344b94c-a014-4803-abea-076a55753cfd`): GO with 3 mandatory fixes integrated.
- Skeptic adversarial agent review 2026-05-10: 🟡 yellow with 3 critical pre-implementation blockers integrated.
- Apollo invariants from CLAUDE.md (NotebookLM-not-gatekeeper section, supervision-mode rules).
- Sprint 3 lesson on telemetry sync chain (CLAUDE.md top section, integration test #20 directly addresses).
- `daemon_survival_tick.rs:121-134` — existing purge pattern (reference implementation).
- `safety.rs:540-583` — calibration trap precedent for swap thresholds.
- `user_context.rs::UserContext::idle_secs` — existing idle detection.
- `effective_pressure.rs` — boost factor compute (NOT used in this gate).
- `protocol.rs::is_privileged` — privileged request convention.
