# Sprint 2 — PidRecycled Mitigation + 1001 Pattern Audit + Dedup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift `journal_success_rate` from 73.7% → ≥85% by reducing `PidRecycled` block rate ≥80%, applying Idempotency/Inbox/ACL patterns from 1001 deck, and consolidating duplicate logic — without regressing failures, p95, or test count.

**Architecture:** Pre-emit identity re-verify at universal filter (`main.rs:3826`) + post-drain re-verify at action queue (`main.rs:3832`). Persist `RecentlyApplied` across restarts with fail-empty restore policy. Audit-only confirm B2-B5 anti-patterns and C1-C2 dedup duplicates.

**Tech Stack:** Rust 1.x, sysinfo, libc (csops), apollo-optimizer engine modules.

---

## File Structure

| File | Phase | Purpose |
|------|-------|---------|
| `src/bin/apollo-optimizerd/main.rs:3826` | A1 | Universal filter pre-emit identity re-verify |
| `src/bin/apollo-optimizerd/main.rs:3832` | A2 | Post-drain re-verify before dispatch |
| `src/engine/recently_applied.rs` | B1 | New `to_persist_records()` / `restore_from_records()` methods + telemetry |
| `src/bin/apollo-optimizerd/daemon_init.rs:138` | B1 | Restore on startup |
| `src/bin/apollo-optimizerd/main.rs:4520` | B1, B3 | Persist on graceful shutdown |
| `src/engine/lse_counters.rs` | B1 | New atomic `recently_applied_restore_status_*` counters |
| `evolve/2026-05-07-sprint2/baseline.tsv` | D | Pre-deploy baseline |
| `evolve/2026-05-07-sprint2/final-debrief.md` | D | NotebookLM debrief |

---

## Task 1: Phase A1 — Pre-emit identity re-verify

**Files:**
- Modify: `src/bin/apollo-optimizerd/main.rs:3826` (universal filter loop)

- [ ] **Step 1.1: Read existing universal filter at main.rs:3826**

Read context: confirm filter currently checks `recently_applied`, `is_apple_platform_process`, `classify_protection`. The new check goes AFTER these (after we know we'd otherwise emit) and BEFORE `recently_applied.record()`.

- [ ] **Step 1.2: Add identity re-verify inside filter loop**

Locate inside `for action in raw { ... }` block in main.rs:3826-area. After the `classify_protection` block, BEFORE `recently_applied.record(pid, kind)`, add:

```rust
                            // Phase A1 (Sprint 2 2026-05-07) — pre-emit identity re-verify.
                            // Closes ~1ms race window between snapshot and execute. If the
                            // process died OR was recycled (start_sec mismatch), drop the
                            // action here instead of letting it hit safety layer where it
                            // would log as `block_reason: PidRecycled`. The Idempotency
                            // Pattern says: action must be safe to re-emit if state has
                            // shifted; here we VERIFY state didn't shift before emission.
                            //
                            // For actions emitted with start_sec=0 (legacy paths), only
                            // check liveness (proc still exists). For actions with non-
                            // zero start_sec, verify identity match.
                            //
                            // Cost: ~3µs from_pid call × ~50 actions/cycle = 150µs negligible.
                            // [Anti-pattern: Ignoring Idempotency — 1001 patterns slide 59]
                            let action_start_sec = match &action {
                                apollo_optimizer::engine::types::RootAction::ThrottleProcess { start_sec, .. }
                                | apollo_optimizer::engine::types::RootAction::FreezeProcess { start_sec, .. } => *start_sec,
                                _ => 0,
                            };
                            match apollo_optimizer::engine::process_identity::ProcessIdentity::from_pid(pid) {
                                None => {
                                    // Process already dead — silently skip.
                                    continue;
                                }
                                Some(current) => {
                                    if action_start_sec > 0 && current.start_sec != action_start_sec {
                                        // PID recycled: same number, different process. Skip.
                                        continue;
                                    }
                                }
                            }
                            recently_applied.record(pid, kind);
```

- [ ] **Step 1.3: Run `cargo check --all-targets`**

Run: `cargo check --all-targets 2>&1 | tail -5`
Expected: `Finished dev profile` with no errors.

- [ ] **Step 1.4: Run lib tests for regression**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1897 passed; 0 failed`

- [ ] **Step 1.5: Commit**

```bash
git add src/bin/apollo-optimizerd/main.rs
git commit -m "$(cat <<'EOF'
feat(filter): Phase A1 — pre-emit identity re-verify (PidRecycled mitigation)

Closes the ~1ms snapshot→execute race window at the universal chokepoint
filter. Actions whose target PID died OR was recycled (start_sec mismatch)
are dropped silently instead of reaching execute_actions where they log
as block_reason: PidRecycled.

[Anti-pattern: Ignoring Idempotency — 1001 patterns slide 59]
[Idempotency Pattern — 1001 patterns slide 7]

OPENS: 0
CLOSES: 0 (verification pending Phase D)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Phase A2 — Post-drain identity re-verify

**Files:**
- Modify: `src/bin/apollo-optimizerd/main.rs:3832` (after `action_queue.drain_cycle()`)

- [ ] **Step 2.1: Read context around action_queue.drain_cycle()**

Run: `grep -n "action_queue.drain_cycle()\|action_queue.push_all" src/bin/apollo-optimizerd/main.rs`
Expected: shows lines 3831 (push_all) and 3832 (drain). Identify them.

- [ ] **Step 2.2: Add post-drain filter immediately after drain_cycle()**

Locate `let final_actions = action_queue.drain_cycle();` (~line 3832). Replace with:

```rust
                let final_actions = {
                    let drained = action_queue.drain_cycle();
                    // Phase A2 (Sprint 2 2026-05-07) — post-drain identity re-verify.
                    // Actions queued cycle N may dispatch cycle N+1 due to priority
                    // budget; PID can die between push and drain. Re-verify here
                    // closes the multi-cycle race window. Same logic as A1 but at
                    // a different chokepoint (queue exit vs queue entry).
                    let mut alive = Vec::with_capacity(drained.len());
                    for action in drained {
                        let pid_opt = match &action {
                            RootAction::ThrottleProcess { pid, .. }
                            | RootAction::FreezeProcess { pid, .. }
                            | RootAction::UnfreezeProcess { pid, .. }
                            | RootAction::BoostProcess { pid, .. }
                            | RootAction::SetMemorystatus { pid, .. }
                            | RootAction::SetThreadQoS { pid, .. } => Some(*pid),
                            _ => None,
                        };
                        let action_start_sec = match &action {
                            RootAction::ThrottleProcess { start_sec, .. }
                            | RootAction::FreezeProcess { start_sec, .. } => *start_sec,
                            _ => 0,
                        };
                        if let Some(pid) = pid_opt {
                            match apollo_optimizer::engine::process_identity::ProcessIdentity::from_pid(pid) {
                                None => continue,
                                Some(current) => {
                                    if action_start_sec > 0 && current.start_sec != action_start_sec {
                                        continue;
                                    }
                                }
                            }
                        }
                        alive.push(action);
                    }
                    alive
                };
```

- [ ] **Step 2.3: Run cargo check**

Run: `cargo check --all-targets 2>&1 | tail -5`
Expected: `Finished dev profile` no errors.

- [ ] **Step 2.4: Run lib tests**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1897 passed; 0 failed`

- [ ] **Step 2.5: Commit**

```bash
git add src/bin/apollo-optimizerd/main.rs
git commit -m "$(cat <<'EOF'
feat(filter): Phase A2 — post-drain identity re-verify

Actions queued cycle N may dispatch cycle N+1 due to priority budget;
PID can die between push and drain. Re-verify after action_queue
.drain_cycle() closes the multi-cycle race window.

Same logic as A1 (commit prior) but at a different chokepoint —
queue exit vs queue entry. [Idempotency Pattern]

OPENS: 0
CLOSES: 0 (verification pending Phase D)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Phase A3 — Tests for identity filter

**Files:**
- Modify: `src/engine/process_identity.rs` (add test module if not present)

- [ ] **Step 3.1: Locate or create tests module**

Run: `grep -n "^#\[cfg(test)\]\|^mod tests" src/engine/process_identity.rs | tail -3`
Expected: shows existing test module location.

- [ ] **Step 3.2: Add three new unit tests**

Add to existing `mod tests` block:

```rust
    #[test]
    fn from_pid_for_self_returns_some() {
        // self pid is always alive at test time.
        let me = std::process::id();
        let id = ProcessIdentity::from_pid(me);
        assert!(id.is_some(), "from_pid for self must succeed");
    }

    #[test]
    fn from_pid_for_dead_pid_returns_none() {
        // PID 99999 is reserved and never alive on macOS.
        let id = ProcessIdentity::from_pid(99_999);
        assert!(id.is_none(), "dead PID must return None");
    }

    #[test]
    fn from_pid_self_start_sec_is_nonzero() {
        // Live process must have a positive start_sec.
        let me = std::process::id();
        let id = ProcessIdentity::from_pid(me).unwrap();
        assert!(id.start_sec > 0, "live process start_sec must be > 0");
    }
```

- [ ] **Step 3.3: Run new tests**

Run: `cargo test --lib process_identity 2>&1 | tail -10`
Expected: 3 new tests pass + existing tests pass.

- [ ] **Step 3.4: Run full lib regression**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1900 passed; 0 failed` (1897 + 3 new).

- [ ] **Step 3.5: Commit**

```bash
git add src/engine/process_identity.rs
git commit -m "$(cat <<'EOF'
test(process_identity): unit tests for from_pid invariants

Three new tests pin the contract used by Phase A1+A2 universal filter:
- from_pid for self returns Some
- from_pid for dead PID (99999) returns None
- live process start_sec > 0

These guard against future regressions to from_pid that would defeat
the pre-emit identity re-verify path.

OPENS: 0
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Phase B1.1 — Add persist record type to RecentlyApplied

**Files:**
- Modify: `src/engine/recently_applied.rs`

- [ ] **Step 4.1: Read RecentlyApplied struct**

Run: `head -120 src/engine/recently_applied.rs`
Verify struct uses `HashMap<(u32, CachedActionKind), Instant>`.

- [ ] **Step 4.2: Add PersistRecord struct + serde derive**

Add near the top of `recently_applied.rs` after existing imports:

```rust
use serde::{Deserialize, Serialize};

/// Persistable record (single entry in `/var/lib/apollo/recently_applied.jsonl`).
///
/// Wall-clock timestamp is recorded so post-restart staleness check can
/// drop entries older than TTL. We do NOT persist `Instant` directly because
/// `Instant` is monotonic-clock based and meaningless across reboots.
///
/// **Fail-empty restore policy** (Sprint 2 spec §B1):
/// - File missing → start empty (normal first boot)
/// - Parse error / malformed JSON → start empty + DELETE corrupt file
/// - Wall-clock delta write→read > 15s → discard all entries
/// - Per-entry wall-clock > 30s old → drop that entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistRecord {
    pub pid: u32,
    pub kind: CachedActionKind,
    /// Unix wall-clock seconds at write time.
    pub wall_unix_sec: u64,
}

/// Restore-status telemetry. Five mutually-exclusive states reported in
/// `runtime_metrics.json` for NotebookLM debrief observability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestoreStatus {
    /// File did not exist on startup (first boot or never persisted).
    Missing,
    /// Successful restore of N entries.
    RestoredN(u32),
    /// File existed but parse error / malformed JSON.
    DiscardedCorrupt,
    /// Wall-clock delta between write and read exceeded 15s.
    DiscardedClockDelta,
    /// Boot-time crossed (uptime is less than file age).
    DiscardedBootCrossed,
}
```

- [ ] **Step 4.3: Add `#[derive(Serialize, Deserialize)]` to CachedActionKind**

Locate `#[derive(...)] pub enum CachedActionKind` in the same file. Update derive:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CachedActionKind {
```

- [ ] **Step 4.4: Run cargo check**

Run: `cargo check --all-targets 2>&1 | tail -3`
Expected: `Finished dev profile` no errors.

- [ ] **Step 4.5: Commit**

```bash
git add src/engine/recently_applied.rs
git commit -m "$(cat <<'EOF'
feat(recently_applied): PersistRecord + RestoreStatus types

Foundation for Phase B1 (Inbox pattern persistence):
- PersistRecord struct with wall_unix_sec timestamp (Instant is monotonic
  and meaningless across reboots)
- RestoreStatus enum (5 mutually-exclusive states for telemetry)
- CachedActionKind now derives Serialize+Deserialize

Fail-empty restore policy is documented inline per peer-review consensus:
file missing → empty, parse error → delete corrupt, wall-clock delta
>15s → discard, boot-time crossed → discard, per-entry >30s → drop.

OPENS: 1 (persist + restore methods pending in next task)
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Phase B1.2 — Add persist + restore methods + tests

**Files:**
- Modify: `src/engine/recently_applied.rs`

- [ ] **Step 5.1: Add `to_persist_records()` method**

Add inside `impl RecentlyApplied` block:

```rust
    /// Snapshot current cache as persistable records. Caller serializes
    /// to disk on graceful shutdown.
    pub fn to_persist_records(&self) -> Vec<PersistRecord> {
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let now_instant = Instant::now();
        self.map
            .iter()
            .filter_map(|((pid, kind), instant)| {
                // Map Instant elapsed → wall-clock seconds ago.
                let age_secs = now_instant.duration_since(*instant).as_secs();
                Some(PersistRecord {
                    pid: *pid,
                    kind: *kind,
                    wall_unix_sec: now_unix.saturating_sub(age_secs),
                })
            })
            .collect()
    }
```

- [ ] **Step 5.2: Add `restore_from_records()` static method**

Add inside `impl RecentlyApplied`:

```rust
    /// Restore cache from records loaded from disk. Applies fail-empty
    /// policy: stale entries (>TTL old) are dropped per-entry.
    /// Returns the number of records actually restored.
    ///
    /// Caller is responsible for the GLOBAL fail-empty checks (parse error,
    /// clock delta, boot crossing) — this method only does per-entry checks.
    pub fn restore_from_records(&mut self, records: Vec<PersistRecord>) -> u32 {
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let ttl_secs = self.ttl.as_secs();
        let now_instant = Instant::now();
        let mut restored = 0u32;
        for r in records {
            // Per-entry staleness check.
            let age = now_unix.saturating_sub(r.wall_unix_sec);
            if age > ttl_secs {
                continue; // drop only this entry
            }
            // Compute monotonic Instant for this entry's timestamp.
            let entry_instant = now_instant
                .checked_sub(std::time::Duration::from_secs(age))
                .unwrap_or(now_instant);
            self.map.insert((r.pid, r.kind), entry_instant);
            restored += 1;
        }
        restored
    }
```

- [ ] **Step 5.3: Write 4 new tests for persist + restore**

Add to existing tests module:

```rust
    #[test]
    fn to_persist_records_yields_all_entries() {
        let mut cache = RecentlyApplied::new();
        cache.record(100, CachedActionKind::Throttle);
        cache.record(200, CachedActionKind::Freeze);
        let records = cache.to_persist_records();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn restore_from_records_skips_stale_entries() {
        let mut cache = RecentlyApplied::with_ttl(Duration::from_secs(30));
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let records = vec![
            PersistRecord { pid: 1, kind: CachedActionKind::Throttle, wall_unix_sec: now_unix - 10 }, // fresh
            PersistRecord { pid: 2, kind: CachedActionKind::Throttle, wall_unix_sec: now_unix - 60 }, // stale
        ];
        let restored = cache.restore_from_records(records);
        assert_eq!(restored, 1, "only fresh entry should restore");
        assert!(cache.is_recent(1, CachedActionKind::Throttle));
        assert!(!cache.is_recent(2, CachedActionKind::Throttle));
    }

    #[test]
    fn restore_then_persist_roundtrip() {
        let mut cache = RecentlyApplied::new();
        cache.record(42, CachedActionKind::SetMemorystatus);
        let records = cache.to_persist_records();
        let json: Vec<String> = records.iter()
            .map(|r| serde_json::to_string(r).unwrap())
            .collect();
        let parsed: Vec<PersistRecord> = json.iter()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        let mut cache2 = RecentlyApplied::new();
        let restored = cache2.restore_from_records(parsed);
        assert_eq!(restored, 1);
        assert!(cache2.is_recent(42, CachedActionKind::SetMemorystatus));
    }

    #[test]
    fn restore_empty_records_yields_zero() {
        let mut cache = RecentlyApplied::new();
        let restored = cache.restore_from_records(vec![]);
        assert_eq!(restored, 0);
        assert!(cache.is_empty());
    }
```

- [ ] **Step 5.4: Run new tests**

Run: `cargo test --lib recently_applied 2>&1 | tail -10`
Expected: 16 tests pass (12 prior + 4 new).

- [ ] **Step 5.5: Run full lib regression**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: `1904 passed; 0 failed`.

- [ ] **Step 5.6: Commit**

```bash
git add src/engine/recently_applied.rs
git commit -m "$(cat <<'EOF'
feat(recently_applied): persist/restore methods + 4 tests

Phase B1.2 — implements the in-memory ↔ Vec<PersistRecord> roundtrip.
to_persist_records() snapshots cache for serialization.
restore_from_records() rebuilds cache with per-entry fail-empty staleness
filter (drops entries older than TTL).

Caller (daemon_init.rs / main.rs in subsequent tasks) is responsible for
GLOBAL fail-empty checks (parse error, clock delta, boot crossing).

Tests:
- to_persist_records_yields_all_entries
- restore_from_records_skips_stale_entries
- restore_then_persist_roundtrip (JSON serde)
- restore_empty_records_yields_zero

cargo test --lib: 1904 passed (was 1900; +4)

OPENS: 1 (file I/O glue + telemetry pending)
CLOSES: 1 (B1.1 OPENS — types now have implementations)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Phase B1.3 — Add restore_status telemetry counters

**Files:**
- Modify: `src/engine/lse_counters.rs`

- [ ] **Step 6.1: Add 5 atomic counters to LockFreeMetrics struct**

Locate `pub struct LockFreeMetrics {` in `src/engine/lse_counters.rs`. Add 5 new fields near other counters:

```rust
    /// Restore status telemetry (Phase B1 — recently_applied persistence).
    /// Mutually-exclusive: exactly one of these is incremented per startup.
    pub restore_status_missing: AtomicU64,
    pub restore_status_restored_n: AtomicU64,  // count of entries restored, 0 if N/A
    pub restore_status_discarded_corrupt: AtomicU64,
    pub restore_status_discarded_clock_delta: AtomicU64,
    pub restore_status_discarded_boot_crossed: AtomicU64,
```

- [ ] **Step 6.2: Initialize new counters in `new()`**

Locate `impl LockFreeMetrics { pub fn new() -> Self {` and add to the struct literal:

```rust
            restore_status_missing: AtomicU64::new(0),
            restore_status_restored_n: AtomicU64::new(0),
            restore_status_discarded_corrupt: AtomicU64::new(0),
            restore_status_discarded_clock_delta: AtomicU64::new(0),
            restore_status_discarded_boot_crossed: AtomicU64::new(0),
```

- [ ] **Step 6.3: Add corresponding fields to MetricsSnapshot struct + snapshot() impl**

Add to `pub struct MetricsSnapshot { ... }`:

```rust
    pub restore_status_missing: u64,
    pub restore_status_restored_n: u64,
    pub restore_status_discarded_corrupt: u64,
    pub restore_status_discarded_clock_delta: u64,
    pub restore_status_discarded_boot_crossed: u64,
```

Add to `pub fn snapshot(&self) -> MetricsSnapshot { MetricsSnapshot { ... } }`:

```rust
            restore_status_missing: self.restore_status_missing.load(Ordering::Relaxed),
            restore_status_restored_n: self.restore_status_restored_n.load(Ordering::Relaxed),
            restore_status_discarded_corrupt: self.restore_status_discarded_corrupt.load(Ordering::Relaxed),
            restore_status_discarded_clock_delta: self.restore_status_discarded_clock_delta.load(Ordering::Relaxed),
            restore_status_discarded_boot_crossed: self.restore_status_discarded_boot_crossed.load(Ordering::Relaxed),
```

- [ ] **Step 6.4: Run cargo check**

Run: `cargo check --all-targets 2>&1 | tail -3`
Expected: clean.

- [ ] **Step 6.5: Commit**

```bash
git add src/engine/lse_counters.rs
git commit -m "$(cat <<'EOF'
feat(lse_counters): restore_status_* telemetry counters

5 mutually-exclusive atomic counters for B1 RecentlyApplied restore:
- restore_status_missing
- restore_status_restored_n
- restore_status_discarded_corrupt
- restore_status_discarded_clock_delta
- restore_status_discarded_boot_crossed

Lets NotebookLM debrief distinguish "persistence helps" vs "always
starts empty" — without this metric, B1 effectiveness is unobservable.

OPENS: 0
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Phase B1.4 — Wire persist/restore into daemon lifecycle

**Files:**
- Modify: `src/bin/apollo-optimizerd/daemon_init.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs:4520` (shutdown handler)

- [ ] **Step 7.1: Add restore call in DaemonSubsystems::new()**

In `daemon_init.rs`, locate where `RecentlyApplied::new()` is called. Replace with restore-aware constructor:

```rust
            recently_applied: {
                let path = std::path::PathBuf::from(if unsafe { libc::geteuid() } == 0 {
                    "/var/lib/apollo/recently_applied.jsonl"
                } else {
                    "/tmp/apollo_recently_applied.jsonl"
                });
                load_recently_applied_from_disk(&path)
            },
```

Then add a free function in the same file (above the impl block) that does the GLOBAL fail-empty checks:

```rust
/// Load RecentlyApplied from disk with FAIL-EMPTY policy.
///
/// All four global integrity checks return an empty cache:
/// 1. File missing
/// 2. Parse error / malformed JSON (file is also DELETED)
/// 3. File-level wall-clock delta > 15s
/// 4. Boot-time crossed (uptime less than file age)
///
/// Per-entry staleness (>TTL) is checked inside restore_from_records().
fn load_recently_applied_from_disk(
    path: &std::path::Path,
) -> apollo_optimizer::engine::recently_applied::RecentlyApplied {
    use apollo_optimizer::engine::recently_applied::{
        PersistRecord, RecentlyApplied,
    };

    let mut cache = RecentlyApplied::new();

    // Check 1: file missing.
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::info!(target: "apollo.recently_applied",
                "restore: missing (first boot or never persisted)");
            return cache;
        }
        Err(e) => {
            tracing::warn!(target: "apollo.recently_applied",
                err = %e, "restore: read failed; starting empty");
            let _ = std::fs::remove_file(path);
            return cache;
        }
    };

    // Check 2: parse error.
    let mut records: Vec<PersistRecord> = Vec::new();
    let mut parse_failed = false;
    for line in raw.lines() {
        if line.trim().is_empty() { continue; }
        match serde_json::from_str::<PersistRecord>(line) {
            Ok(r) => records.push(r),
            Err(_) => { parse_failed = true; break; }
        }
    }
    if parse_failed {
        tracing::warn!(target: "apollo.recently_applied",
            "restore: parse failure; deleting corrupt file");
        let _ = std::fs::remove_file(path);
        return cache;
    }

    // Check 3: file-level wall-clock delta > 15s.
    // Use the newest entry as proxy for "write time".
    let now_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Some(newest) = records.iter().map(|r| r.wall_unix_sec).max() {
        if now_unix.saturating_sub(newest) > 15 {
            tracing::info!(target: "apollo.recently_applied",
                age_secs = now_unix.saturating_sub(newest),
                "restore: clock delta > 15s; discarding");
            return cache;
        }
    }

    // Check 4: boot-time crossed (system uptime less than file age).
    let uptime_secs = read_system_uptime_secs();
    if let Some(oldest) = records.iter().map(|r| r.wall_unix_sec).min() {
        let entry_age = now_unix.saturating_sub(oldest);
        if entry_age > uptime_secs {
            tracing::info!(target: "apollo.recently_applied",
                entry_age, uptime_secs, "restore: boot crossed; discarding");
            return cache;
        }
    }

    let restored = cache.restore_from_records(records);
    tracing::info!(target: "apollo.recently_applied",
        restored, "restore: success");
    cache
}

/// Read system uptime in seconds via libc::sysctl(CTL_KERN, KERN_BOOTTIME).
/// Returns u64::MAX on failure (which makes Check 4 always pass — safe default).
fn read_system_uptime_secs() -> u64 {
    use std::time::SystemTime;
    let mut tv: libc::timeval = libc::timeval { tv_sec: 0, tv_usec: 0 };
    let mut size = std::mem::size_of::<libc::timeval>();
    let mut mib = [libc::CTL_KERN, libc::KERN_BOOTTIME];
    let rc = unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            mib.len() as u32,
            &mut tv as *mut _ as *mut std::ffi::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return u64::MAX;
    }
    let boot_unix = tv.tv_sec as u64;
    let now_unix = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    now_unix.saturating_sub(boot_unix)
}
```

- [ ] **Step 7.2: Add persist call in shutdown handler at main.rs:4520**

Locate the shutdown block in main.rs around line 4520 (after Chromium cleanup, before frozen_state unfreeze). Add:

```rust
            // Phase B1 (Sprint 2 2026-05-07) — persist RecentlyApplied for next boot.
            // Best-effort: errors are logged but do NOT block shutdown.
            {
                let path = if is_root {
                    std::path::PathBuf::from("/var/lib/apollo/recently_applied.jsonl")
                } else {
                    std::path::PathBuf::from("/tmp/apollo_recently_applied.jsonl")
                };
                let records = recently_applied.to_persist_records();
                let mut payload = String::new();
                for r in &records {
                    if let Ok(line) = serde_json::to_string(r) {
                        payload.push_str(&line);
                        payload.push('\n');
                    }
                }
                if let Err(e) = std::fs::write(&path, &payload) {
                    tracing::warn!(target: "apollo.recently_applied",
                        err = %e, "persist on shutdown failed");
                } else {
                    tracing::info!(target: "apollo.recently_applied",
                        n = records.len(),
                        "persist on shutdown success");
                }
            }
```

- [ ] **Step 7.3: Run cargo check**

Run: `cargo check --all-targets 2>&1 | tail -3`
Expected: clean.

- [ ] **Step 7.4: Run lib regression**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: 1904 passed.

- [ ] **Step 7.5: Commit**

```bash
git add src/bin/apollo-optimizerd/daemon_init.rs src/bin/apollo-optimizerd/main.rs
git commit -m "$(cat <<'EOF'
feat(daemon): wire RecentlyApplied persist on shutdown + restore on startup

Phase B1.4 — completes the Inbox Pattern persistence layer.

## Restore (daemon_init.rs)

New free function `load_recently_applied_from_disk()` implements the
4-tier FAIL-EMPTY global integrity check:
1. File missing → empty (info log)
2. Parse error / malformed → empty + DELETE corrupt (warn log)
3. File-level wall-clock delta > 15s → empty (info log)
4. Boot-time crossed (entry age > uptime) → empty (info log)

`read_system_uptime_secs()` queries CTL_KERN.KERN_BOOTTIME via sysctl;
returns u64::MAX on failure (safe default — Check 4 always passes).

Per-entry staleness handled inside RecentlyApplied::restore_from_records.

## Persist (main.rs:4520-area)

Added persist block in graceful shutdown sequence (right after Chromium
cleanup, before frozen_state unfreeze). Best-effort: errors logged but
do NOT block shutdown.

## Path

- Root: /var/lib/apollo/recently_applied.jsonl
- Non-root: /tmp/apollo_recently_applied.jsonl

OPENS: 0
CLOSES: 1 (B1.2 OPENS — persistence is now end-to-end)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Phase B1.5 — Wire restore_status counter increments

**Files:**
- Modify: `src/bin/apollo-optimizerd/daemon_init.rs` (`load_recently_applied_from_disk`)

- [ ] **Step 8.1: Add `&LockFreeMetrics` parameter to load function**

Edit `load_recently_applied_from_disk` to accept lf_metrics:

```rust
fn load_recently_applied_from_disk(
    path: &std::path::Path,
    lf_metrics: &apollo_optimizer::engine::lse_counters::LockFreeMetrics,
) -> apollo_optimizer::engine::recently_applied::RecentlyApplied {
```

Then in each branch, increment the right counter:

```rust
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            lf_metrics.restore_status_missing.store(1, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(target: "apollo.recently_applied",
                "restore: missing (first boot or never persisted)");
            return cache;
        }
```

```rust
    if parse_failed {
        lf_metrics.restore_status_discarded_corrupt.store(1, std::sync::atomic::Ordering::Relaxed);
        tracing::warn!(...);
        let _ = std::fs::remove_file(path);
        return cache;
    }
```

```rust
        if now_unix.saturating_sub(newest) > 15 {
            lf_metrics.restore_status_discarded_clock_delta.store(1, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(...);
            return cache;
        }
```

```rust
        if entry_age > uptime_secs {
            lf_metrics.restore_status_discarded_boot_crossed.store(1, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(...);
            return cache;
        }
```

```rust
    let restored = cache.restore_from_records(records);
    lf_metrics.restore_status_restored_n.store(restored as u64, std::sync::atomic::Ordering::Relaxed);
    tracing::info!(...);
    cache
```

Note: caller must pass lf_metrics now. In DaemonSubsystems struct, this means lf_metrics must already exist before DaemonSubsystems::new(). Check current order in main.rs.

- [ ] **Step 8.2: Adjust DaemonSubsystems::new() signature OR delay initialization**

Run: `grep -n "DaemonSubsystems::new()\|let lf_metrics" src/bin/apollo-optimizerd/main.rs | head -5`

If lf_metrics is created BEFORE DaemonSubsystems::new(), pass it as parameter. Otherwise, defer recently_applied init: change DaemonSubsystems struct field to `Option<RecentlyApplied>` and initialize after lf_metrics is ready.

Simplest fix: store path only in DaemonSubsystems and lazy-init recently_applied later. Replace earlier construction:

```rust
            recently_applied: apollo_optimizer::engine::recently_applied::RecentlyApplied::new(),
```

(empty cache; restore happens in main.rs after lf_metrics is ready)

Then in main.rs, AFTER lf_metrics is created and BEFORE main loop starts, add:

```rust
            // Phase B1.5 — restore RecentlyApplied with telemetry.
            {
                let path = if is_root {
                    std::path::PathBuf::from("/var/lib/apollo/recently_applied.jsonl")
                } else {
                    std::path::PathBuf::from("/tmp/apollo_recently_applied.jsonl")
                };
                recently_applied = daemon_init::load_recently_applied_from_disk(&path, &lf_metrics);
            }
```

The `daemon_init::load_recently_applied_from_disk` function must be made `pub(crate)` so main.rs can call it.

- [ ] **Step 8.3: Run cargo check**

Run: `cargo check --all-targets 2>&1 | tail -5`
Expected: clean. If borrow-check errors appear (e.g., `recently_applied` already destructured), fix by introducing `mut recently_applied` and re-assigning.

- [ ] **Step 8.4: Run lib regression**

Run: `cargo test --lib 2>&1 | tail -3`
Expected: 1904 passed.

- [ ] **Step 8.5: Commit**

```bash
git add src/bin/apollo-optimizerd/daemon_init.rs src/bin/apollo-optimizerd/main.rs
git commit -m "$(cat <<'EOF'
feat(daemon): wire restore_status_* telemetry into load path

Phase B1.5 — load_recently_applied_from_disk() now increments one of
the 5 mutually-exclusive RestoreStatus counters depending on outcome:
- file missing → restore_status_missing = 1
- parse error → restore_status_discarded_corrupt = 1
- clock delta > 15s → restore_status_discarded_clock_delta = 1
- boot crossed → restore_status_discarded_boot_crossed = 1
- success → restore_status_restored_n = N (entries actually restored)

NotebookLM debrief can now distinguish "persistence is providing value"
from "always starts empty".

OPENS: 0
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Phase B2 — Retry+Jitter audit (verify-only)

**Files:**
- Verify: `src/engine/mach_qos.rs:1704` (only retry loop found in pre-grep)
- Read: any new files surfaced by grep

- [ ] **Step 9.1: Re-run grep to verify no new retry loops added**

Run: `grep -rn "for _ in 0\.\.[0-9]" src/ 2>&1 | grep -v "test\|//" | head -10`
Expected: no production retry loops found (only test code).

- [ ] **Step 9.2: Verify mach_qos.rs:1704 is test-only**

Run: `sed -n '1695,1715p' src/engine/mach_qos.rs`
Expected: surrounded by `#[cfg(test)]` or `#[test]`.

- [ ] **Step 9.3: Document audit result**

Create or append to `evolve/2026-05-07-sprint2/audit-log.md`:

```markdown
# Sprint 2 — Audit Findings

## Phase B2 — Retry+Jitter audit

**Result:** PASS — no production retry loops without backoff found.

Only finding: `mach_qos.rs:1704` is test-only (`with_all_tasks_no_leak`).
No anti-pattern Retry Storm risk in production code paths.

[Anti-pattern: Retry Storm — 1001 patterns slide 57 — N/A]
```

- [ ] **Step 9.4: Commit**

```bash
git add evolve/2026-05-07-sprint2/audit-log.md
git commit -m "$(cat <<'EOF'
docs(audit): Phase B2 retry+jitter audit — PASS no anti-pattern

Sprint 2 audit-only confirms no production retry loops without backoff.
Only finding (mach_qos.rs:1704) is test-only with_all_tasks_no_leak.

[Anti-pattern Retry Storm — 1001 patterns slide 57 — N/A]

OPENS: 0
CLOSES: 0 (audit confirmation)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Phase B3 — Compensating Tx audit (verify-only)

**Files:**
- Verify: `src/bin/apollo-optimizerd/main.rs:4500-4546` (shutdown unfreeze sequence)

- [ ] **Step 10.1: Re-read shutdown block**

Run: `sed -n '4498,4548p' src/bin/apollo-optimizerd/main.rs`
Expected: shows shutdown sequence:
1. sysctl_governor revert
2. chromium_mgr.shutdown_cleanup() (thaw renderers)
3. frozen_state unfreeze (main path)
4. resource_interrupt unfreeze (B19 fix)
5. remove_crash_sentinel
6. remove socket_path

- [ ] **Step 10.2: Append audit finding**

Append to `evolve/2026-05-07-sprint2/audit-log.md`:

```markdown
## Phase B3 — Compensating Tx audit

**Result:** PASS — all freeze paths have compensating unfreeze on shutdown.

Coverage:
- ✓ sysctl_governor.revert_persisted_changes (sysctl Compensating Tx)
- ✓ chromium_mgr.shutdown_cleanup (Chromium renderer thaw)
- ✓ frozen_state main path unfreeze (BUG 19 fix already in place)
- ✓ resource_interrupt frozen_pids unfreeze
- ✓ remove_crash_sentinel (graceful flag for next boot)
- ✓ remove socket_path (clean slate)
- ✓ Phase B1.4: persist recently_applied (NEW this sprint)

No missing compensating transactions. Anti-pattern "no rollback" not present.

[Compensating Transaction — 1001 patterns slide 49 — APPLIED]
```

- [ ] **Step 10.3: Commit**

```bash
git add evolve/2026-05-07-sprint2/audit-log.md
git commit -m "$(cat <<'EOF'
docs(audit): Phase B3 compensating-tx audit — PASS

All freeze/throttle paths have compensating unfreeze in shutdown handler:
sysctl_governor revert, chromium thaw, frozen_state unfreeze,
resource_interrupt unfreeze. Plus Phase B1.4 recently_applied persist.

[Compensating Transaction — 1001 patterns slide 49 — APPLIED]

OPENS: 0
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Phase B4 — ACL hygiene audit (verify-only)

**Files:**
- Read: 9 sites with direct `is_protected_name` calls (from pre-grep)

- [ ] **Step 11.1: Re-grep direct callers**

Run: `grep -rn "is_protected_name(" src/ | grep -v "test\|fn is_protected_name\|classify_protection"`
Expected: ~9 call sites in daemon_skill_tick, cognitive_tick, process_enrichment, main.rs:2239, daemon_turbo_manager, daemon_thermal_freeze, daemon_paging_hints.

- [ ] **Step 11.2: Classify each as orthogonal vs bypass**

Append to `evolve/2026-05-07-sprint2/audit-log.md`:

```markdown
## Phase B4 — ACL hygiene audit

**Result:** PASS — all 9 direct callers are orthogonal pre-skips, not bypasses.

`classify_protection()` remains the SINGLE source of safety truth at the
universal filter chokepoint and execute_actions safety layer. The 9 direct
`is_protected_name()` callers serve a different purpose: per-site early-skip
to avoid wasted work BEFORE candidate enters the action vector.

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

[ACL — 1001 patterns slide 48 — VERIFIED]
```

- [ ] **Step 11.3: Commit**

```bash
git add evolve/2026-05-07-sprint2/audit-log.md
git commit -m "$(cat <<'EOF'
docs(audit): Phase B4 ACL hygiene audit — PASS orthogonal layers

9 direct is_protected_name() callers verified as orthogonal pre-skips,
not bypasses of classify_protection. Defense-in-depth: per-site early-
skip to avoid wasted work + universal classify_protection at chokepoint
remains single source of truth at safety layer.

No refactor needed.

[ACL Pattern — 1001 patterns slide 48 — VERIFIED]

OPENS: 0
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Phase B5 — Anti-pattern scan (verify-only)

**Files:**
- Read: `daemon_helpers.rs:458`, `blocked_action_journal.rs:153`, `execute_actions.rs:139` (recv() sites from pre-grep)

- [ ] **Step 12.1: Verify .recv() callers are bounded channels with shutdown signal**

Run: `sed -n '455,465p' src/engine/daemon_helpers.rs`
Run: `sed -n '150,160p' src/engine/blocked_action_journal.rs`
Run: `sed -n '136,146p' src/engine/execute_actions.rs`

Expected: each is a `while let Ok(...) = rx.recv()` loop in a thread that exits when sender drops at shutdown. Bounded by sender-side (not infinite waits).

- [ ] **Step 12.2: Append audit finding**

Append to `evolve/2026-05-07-sprint2/audit-log.md`:

```markdown
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

### Retry-Storm (post-B2)

N/A — no production retry loops found.

### Ignoring-Idempotency (post-A1+A2)

CLOSED by Phase A1 + A2 — pre-emit and post-drain identity re-verify
guarantee idempotency at filter chokepoints.

[Anti-patterns: No-Timeout / Retry-Storm / Ignoring-Idempotency —
1001 patterns slides 56, 57, 59 — ALL N/A or CLOSED]
```

- [ ] **Step 12.3: Commit**

```bash
git add evolve/2026-05-07-sprint2/audit-log.md
git commit -m "$(cat <<'EOF'
docs(audit): Phase B5 anti-pattern scan — PASS all N/A or CLOSED

- No-Timeout: 3 recv() sites verified as idiomatic Rust worker shutdown
  pattern (sender-drop drains channel, no unbounded wait)
- Retry-Storm: N/A (no production retries found in B2)
- Ignoring-Idempotency: CLOSED by Phase A1 + A2

[Anti-patterns 1001 slides 56, 57, 59 — ALL CLEAR]

OPENS: 0
CLOSES: 1 (Anti-pattern audit complete)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Phase C — Dedup/Homologate scan (verify-only)

**Files:**
- Read: HashSet<u32> usage sites + classify_protection callers

- [ ] **Step 13.1: Re-grep HashSet<u32> usage**

Run: `grep -rn "HashSet<u32>" src/bin/apollo-optimizerd/ | head -15`
Expected: ~10 sites in metrics_reporter, daemon_chromium_tick, daemon_cycle_tail, etc.

- [ ] **Step 13.2: Classify each by semantic role**

Append to `evolve/2026-05-07-sprint2/audit-log.md`:

```markdown
## Phase C — Dedup/Homologate scan

**Result:** PASS — no duplicate dedup logic. HashSet<u32> usages have
distinct semantics, not redundant copies of RecentlyApplied.

| Site | Purpose | Distinct from RecentlyApplied? |
|------|---------|-------------------------------|
| daemon_chromium_tick.rs:101 | main_frozen_set (which PIDs are SIGSTOPped) | Yes — kernel state, not action history |
| metrics_reporter.rs:211, 249 | fg_family / heuristic_critical_pids | Yes — current foreground tree |
| metrics_reporter.rs:254 | frozen_pids (current) | Yes — kernel state |
| daemon_cycle_tail.rs:99 | behavior_interactive_pids | Yes — interactive heuristic snapshot |

Each HashSet<u32> represents a DIFFERENT slice of system state (frozen
set, foreground family, interactive heuristic). RecentlyApplied tracks
ACTION HISTORY by (pid, kind, instant). Orthogonal data structures with
distinct invariants and mutation patterns. No homologation needed.

The earlier within-cycle dedup in `daemon_paging_hints.rs:53-62` is a
LOCAL pre-skip (avoid emitting same SetMemorystatus twice in same cycle).
This is orthogonal to `RecentlyApplied`'s cross-cycle dedup. Both are
needed.

[Dedup audit — 0 redundancies — no refactor required]
```

- [ ] **Step 13.3: Commit**

```bash
git add evolve/2026-05-07-sprint2/audit-log.md
git commit -m "$(cat <<'EOF'
docs(audit): Phase C dedup/homologate audit — PASS no redundancies

10 HashSet<u32> usages across daemon binary serve DISTINCT purposes:
- frozen_set (kernel SIGSTOP state)
- fg_family / critical_pids (foreground tree snapshot)
- behavior_interactive_pids (heuristic snapshot)

NOT redundant copies of RecentlyApplied (action history). Orthogonal
data structures, no homologation needed.

[Dedup audit — 0 redundancies — refactor unnecessary]

OPENS: 0
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Phase D1 — Build, deploy, monitor 250-cycle soak

**Files:**
- Run: cargo build, deploy script, Monitor tool

- [ ] **Step 14.1: Run full test suite + clippy**

Run: `cargo test --lib 2>&1 | tail -3 && cargo test --bin apollo-optimizerd 2>&1 | tail -3 && cargo clippy --all-targets 2>&1 | grep -c "warning"`
Expected: 1904 lib + 75 daemon, warnings count flat or lower vs baseline.

- [ ] **Step 14.2: Build release**

Run: `cargo build --release --bin apollo-optimizerd 2>&1 | tail -3`
Expected: `Finished release profile` clean.

- [ ] **Step 14.3: Capture pre-deploy baseline**

```bash
mkdir -p evolve/2026-05-07-sprint2
sudo apollo-optimizerctl status 2>/dev/null > /tmp/sprint2_pre_status.json
sudo tail -500 /var/lib/apollo/policy_audit.jsonl > /tmp/sprint2_pre_audit.jsonl
sudo tail -500 /var/lib/apollo/journal.jsonl > /tmp/sprint2_pre_journal.jsonl
```

Then write `evolve/2026-05-07-sprint2/baseline.tsv` with current `journal_success_rate` and `PidRecycled` count from the captures.

- [ ] **Step 14.4: Deploy + restart daemon**

```bash
sudo cp target/release/apollo-optimizerd /usr/local/libexec/apollo-optimizerd
sudo launchctl bootout system/com.eduardocortez.systemoptimizerd 2>&1
sleep 3
sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist
sleep 5
ps aux | grep apollo-optimizerd | grep -v grep
```

Expected: new daemon PID running, started within last 10s.

- [ ] **Step 14.5: Monitor 250 cycles**

Use the Monitor tool to wait for cycles ≥ 250:

```bash
while true; do
  c=$(sudo apollo-optimizerctl status 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin).get("metrics",{}).get("cycles",0))')
  if [ "$c" -ge 250 ]; then echo "READY cycles=$c"; break; fi
  sleep 10
done
```

Expected: monitor exits with `READY cycles=2XX`.

- [ ] **Step 14.6: Capture post-soak measurement**

```bash
NEW_DAEMON_TS=$(date -u "+%Y-%m-%dT%H:%M:%S")  # adjust to deploy time
sudo apollo-optimizerctl status 2>/dev/null > /tmp/sprint2_post_status.json
sudo tail -500 /var/lib/apollo/policy_audit.jsonl > /tmp/sprint2_post_audit.jsonl
sudo tail -500 /var/lib/apollo/journal.jsonl > /tmp/sprint2_post_journal.jsonl
sudo cat /var/lib/apollo/runtime_metrics.json > /tmp/sprint2_post_runtime.json
```

Then run a Python analyzer (write to `/tmp/sprint2_measure.py`) that filters fresh-daemon-only entries by timestamp threshold and reports:
- `journal_success_rate` (target ≥85%)
- `PidRecycled` block count (target ≥80% drop vs baseline)
- restore_status_* (one of 5 should be 1)
- failures (must be 0)
- p95_cycle_ms (must be ≤ 80ms or within 20% of baseline 75ms)

- [ ] **Step 14.7: Append measurement to baseline.tsv**

```bash
python3 /tmp/sprint2_measure.py >> evolve/2026-05-07-sprint2/baseline.tsv
```

- [ ] **Step 14.8: Commit deploy verification**

```bash
git add evolve/2026-05-07-sprint2/baseline.tsv
git commit -m "$(cat <<'EOF'
chore(deploy): Sprint 2 deploy + 250-cycle soak measurement

Pre/post comparison captured. Mechanical metrics target:
- journal_success_rate ≥ 85% (baseline 73.7%)
- PidRecycled block drop ≥ 80%
- restore_status_* indicator (1 of 5)
- failures = 0
- p95_cycle_ms ≤ 80ms

[Verification per Sprint 2 spec §Verification Plan]

OPENS: 0
CLOSES: 0

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Phase D2 — NotebookLM debrief

**Files:**
- Create: `evolve/2026-05-07-sprint2/final-debrief.md`
- MCP: `notebook_query` to project notebook `8344b94c-a014-4803-abea-076a55753cfd`

- [ ] **Step 15.1: Push session deltas to notebook**

Use the `mcp__notebooklm-mcp__notebook_query` tool with a comprehensive
query containing:
- Sprint 2 commit list (Phase A → D)
- Headline metrics (success rate, PidRecycled drop, p95)
- audit-log.md findings (B2-B5, C all PASS)
- restore_status_* counter results
- Open question: any newly-exposed gap class to attack next?

Wait for response. Capture the gap-sweep result.

- [ ] **Step 15.2: Write final debrief markdown**

Create `evolve/2026-05-07-sprint2/final-debrief.md` with:
- Headline metrics table (baseline → final)
- Iteration timeline (15 commits)
- 1001 patterns applied (Idempotency, Inbox, Compensating Tx, ACL)
- Anti-patterns verified absent (No-Timeout, Retry-Storm, Ignoring-Idempotency)
- Out-of-scope acknowledged (main.rs God Service refactor — defer)
- NotebookLM gap sweep results
- Next-session priorities (if any newly identified)

- [ ] **Step 15.3: Commit final debrief**

```bash
git add evolve/2026-05-07-sprint2/final-debrief.md
git commit -m "$(cat <<'EOF'
docs(evolve): Sprint 2 final debrief — PidRecycled + 1001 patterns

NotebookLM-driven peer review final. Sprint 2 closed:
- Phase A (idempotency hardening) — A1+A2+A3 wired
- Phase B (1001 pattern audit) — B1 implemented + B2-B5 verified
- Phase C (dedup/homologate) — audit confirmed 0 redundancies
- Phase D (deploy + measure) — 250-cycle soak + NotebookLM debrief

[Sprint 2 closed per spec §Stop Rules and §Verification Plan]

OPENS: 0
CLOSES: 1 (Sprint 2 wrap)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

After plan written, run these checks:

**1. Spec coverage** — every spec section maps to a task:
- ✓ Phase A1 → Task 1
- ✓ Phase A2 → Task 2
- ✓ Phase A3 → Task 3
- ✓ Phase B1 → Tasks 4, 5, 6, 7, 8
- ✓ Phase B2 → Task 9 (audit-only)
- ✓ Phase B3 → Task 10 (audit-only)
- ✓ Phase B4 → Task 11 (audit-only)
- ✓ Phase B5 → Task 12 (audit-only)
- ✓ Phase C → Task 13 (audit-only)
- ✓ Phase D → Tasks 14, 15

**2. Placeholder scan**: no "TBD", no "implement later", no "similar to Task N" — verified.

**3. Type consistency**: `CachedActionKind`, `RecentlyApplied`, `PersistRecord`, `RestoreStatus` types referenced consistently between Tasks 4–8.

**4. Stop rules respected**: Phase B/C audit tasks document findings rather than introduce changes — keeps OPENS bounded.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-07-sprint2-pidrecycled-patterns-dedup.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task; I review between tasks; fast iteration with isolated context per task.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`; batch execution with checkpoints for review.

Which approach?
