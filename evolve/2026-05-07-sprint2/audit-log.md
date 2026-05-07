# Sprint 2 — Audit Findings

## Phase B2 — Retry+Jitter audit

**Result:** PASS — no production retry loops without backoff found.

Pre-grep located one finding only: `mach_qos.rs:1704` — a `for _ in 0..5` retry loop INSIDE the `with_all_tasks_no_leak` test (a port-leak detection test that calls `with_all_tasks` repeatedly). This is test-only code; no production hot-path implications.

Across `src/`, no production retry loops without exponential backoff found. The daemon's hot path uses idiomatic Rust patterns:
- `while let Ok(...) = rx.recv()` for worker thread teardown (sender-drop bounded)
- Cooldowns enforced by structured types (`FreezeCooldown`, `MemoryBudgetState`, `RecentlyApplied`) rather than naive retry loops

[Anti-pattern: Retry Storm — 1001 patterns slide 57 — N/A]


## Phase B3 — Compensating Tx audit

**Result:** PASS — all freeze paths have compensating unfreeze on shutdown.

Shutdown sequence at `src/bin/apollo-optimizerd/main.rs:4498-4549`:
1. ✓ `sysctl_governor.revert_persisted_changes` — sysctl Compensating Tx
2. ✓ `chromium_mgr.shutdown_cleanup()` — Chromium renderer thaw
3. ✓ NEW Phase B1.4: `recently_applied.save(&path)` — Inbox persistence
4. ✓ frozen_state main path unfreeze (BUG 19 fix)
5. ✓ resource_interrupt frozen_pids unfreeze
6. ✓ remove_crash_sentinel — graceful flag for next boot
7. ✓ remove socket_path — clean slate

No missing compensating transactions. The shutdown handler implements all
inverse operations for transactions Apollo applies during runtime.

[Compensating Transaction — 1001 patterns slide 49 — APPLIED]


## Phase B4 — ACL hygiene audit

**Result:** PASS — all 9 direct callers are orthogonal pre-skips, not bypasses.

`classify_protection()` at `src/engine/safety.rs:327` remains the SINGLE source of safety truth at the universal filter chokepoint and execute_actions safety layer. The 9 direct `is_protected_name()` callers serve a different purpose: per-site early-skip to avoid wasted work BEFORE candidate enters the action vector.

| Site | Purpose | Verdict |
|------|---------|---------|
| daemon_skill_tick.rs:87 | skill_registry pre-skip protected target | orthogonal early-skip |
| daemon_skill_tick.rs:160 | trial skill pre-skip | orthogonal |
| cognitive_tick.rs:269 | cognitive bus pre-skip | orthogonal |
| process_enrichment.rs:382 | governor decision pre-skip | orthogonal (Layer 1) |
| process_enrichment.rs:394 | governor convert pre-skip | orthogonal |
| main.rs:2239 | resource interrupt pre-skip | orthogonal |
| daemon_turbo_manager.rs:80 | turbo deactivation guard | orthogonal |
| daemon_thermal_freeze.rs:87,93 | thermal freeze guard | orthogonal |
| daemon_paging_hints.rs:83 | paging hint pre-filter | orthogonal |

NONE of these REPLACE classify_protection at the chokepoint. They are
defense-in-depth pre-skips that shed work early. No refactor needed.

[ACL Pattern — 1001 patterns slide 48 — VERIFIED]


## Phase B5 — Anti-pattern scan

**Result:** PASS — no No-Timeout, no Retry-Storm, no Ignoring-Idempotency.

### No-Timeout (recv()/lock()/wait() without timeout)

3 sites use `while let Ok(...) = rx.recv()`:
- `daemon_helpers.rs:458` — frozen_state writer thread
- `blocked_action_journal.rs:153` — async audit appender
- `execute_actions.rs:139` — execute_actions worker

All are `while let Ok(...) = rx.recv()` loops that exit when the sender
drops at shutdown — this is the IDIOMATIC Rust pattern for graceful
worker thread teardown, NOT an unbounded wait. Sender-side bounded
channels ensure backpressure.

Mutex use throughout daemon: `lock_recover()` pattern (handles poisoning)
or `try_lock()` for non-blocking probes. No raw `lock()` calls in hot
path that could deadlock.

condvar.wait_timeout used in main loop pacing — explicit timeout always
present.

### Retry-Storm (post-B2)

N/A — no production retries found. See Phase B2 audit above.

### Ignoring-Idempotency (post-A1+A2)

CLOSED by Phase A1 + A2 — `pid_identity_still_valid()` helper at
`main.rs:~140` (commit 984f565) mirrors `verify_pid_identity` exactly:
start_sec match + start_usec match (when both nonzero) + name match
(unconditional defense-in-depth). Used at both pre-emit (universal
filter) and post-drain (action_queue exit) chokepoints.

[Anti-patterns: No-Timeout / Retry-Storm / Ignoring-Idempotency —
1001 patterns slides 56, 57, 59 — ALL N/A or CLOSED]
