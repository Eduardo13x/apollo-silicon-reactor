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
