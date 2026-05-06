# Zero-Risk Production Merge Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring 3 isolated worktree commits (`gate_e`, `chromium adaptive`, `protection consolidation`) to production master with zero regression risk by adding explicit anti-flap to gate_e before merge, then sequential merge ordered by ascending blast radius, and gated deploy with mechanical rollback criteria.

**Architecture:** Single new module (`src/engine/freeze_cooldown.rs`) tracks PIDs thawed within last N cycles, exposed as a `HashSet<u32>` in `SharedState`. `decide_actions` filters gate_e candidates against the set. Cooldown TTL decremented in main loop tick. Merge order: smallest-risk first (Protection → Chromium → Gate_e+Cooldown). Each phase consults NotebookLM before mutation.

**Tech Stack:** Rust 2021, `Arc<Mutex<>>` for shared state, `apollo-optimizerd` daemon, `cargo test` for verification, NotebookLM MCP for peer review.

---

## Risk Analysis

| Commit | Worktree | Mutation surface | Pre-merge risk |
|--------|----------|------------------|----------------|
| `e768d6d` Protection | `agent-a4a1e514a0107dfa5` | `cognitive_tick.rs` +7/-3 | Trivial — 1 callsite swap |
| `7bdb931` Chromium | `agent-af42b7c1666fcf5b8` | `chromium_manager.rs` +173/-12 | Self-contained, RE_PURGE guard preserved |
| `80f16c2` Gate_e | `agent-a330670c592d46ae2` | `decide_actions.rs` +116/-6 | **HIGH** — resurrects freezer, oscillation possible |

**Critical risk identified by NotebookLM (conversation `379c81af`):**
- `MAX_FROZEN_CYCLES=150` in `planner.rs:173` is a **TTL** (when to thaw a frozen PID), NOT a post-thaw cooldown.
- Failure mode: PID frozen by gate_e → TTL fires → SIGCONT → process re-meets gate_e thresholds within 1 cycle → re-frozen.
- Without explicit anti-flap: oscillation under sustained swap pressure.

**Mitigation strategy:** introduce per-PID post-thaw cooldown set, populated on every gate_e thaw, decremented per cycle. Gate_e candidate filtering excludes PIDs in cooldown.

---

## File Structure

**New files:**
- `src/engine/freeze_cooldown.rs` — `FreezeCooldown` struct: `HashMap<u32, u8>` tracking remaining cooldown cycles per PID, with `tick()`, `mark_thawed(pid)`, `is_in_cooldown(pid) -> bool` methods.

**Modified files:**
- `src/engine/types.rs` — add `freeze_cooldown: Arc<Mutex<FreezeCooldown>>` to `SharedState` struct.
- `src/engine/decide_actions.rs` — filter gate_e freeze candidates against cooldown set.
- `src/bin/apollo-optimizerd/main.rs` — wire `FreezeCooldown` into `SharedState`, tick per cycle, mark thawed PIDs at unfreeze sites (3 sites at lines 1411, 1481, 1548).

**Constant:**
- `GATE_E_COOLDOWN_CYCLES: u8 = 60` — at ~2Hz tick rate ≈ 30s post-thaw window. Long enough for kernel to redistribute swap pressure, short enough that real new pressure events still trigger.

---

## Phase 0: Pre-Flight (no code changes)

### Task 0.1: Verify worktree state

- [ ] **Step 1: Confirm 3 worktrees exist with expected SHAs**

```bash
git worktree list
```

Expected output (paths abbreviated):
```
.../system-optimizer                                            59be645 [master]
.../worktrees/agent-a330670c592d46ae2  80f16c2 [worktree-agent-a330670c592d46ae2] locked
.../worktrees/agent-a4a1e514a0107dfa5  e768d6d [worktree-agent-a4a1e514a0107dfa5] locked
.../worktrees/agent-af42b7c1666fcf5b8  7bdb931 [worktree-agent-af42b7c1666fcf5b8] locked
```

- [ ] **Step 2: Capture baseline production metrics for rollback comparison**

```bash
sudo apollo-optimizerctl status > /tmp/baseline_pre_merge.json
sudo apollo-optimizerctl teach export | head -5 > /tmp/baseline_swap.txt
```

Expected: `failures: 0`, current AIS, current p95_cycle_ms recorded for later comparison.

---

## Phase 1: NotebookLM Consult — Cooldown Design

### Task 1.1: Validate cooldown design with NotebookLM

- [ ] **Step 1: Query NotebookLM with proposed design**

Use MCP tool `mcp__notebooklm-mcp__notebook_query`:
```
notebook_id: 8344b94c-a014-4803-abea-076a55753cfd
conversation_id: 379c81af-388d-483e-9d48-7ec30e4c6bde
query: "Voy a agregar anti-flap explícito para gate_e:
- nuevo módulo freeze_cooldown.rs con HashMap<u32, u8>
- al thaw por gate_e (TTL): mark_thawed(pid), set cooldown=60 cycles
- decide_actions filtra freeze_candidates: si pid in cooldown → skip
- main loop: cooldown.tick() per cycle (decrementa)
- 60 cycles ≈ 30s @ 2Hz

Preguntas:
1. ¿60 cycles es el threshold correcto? ¿O debería ser 150 (alineado con MAX_FROZEN_CYCLES)?
2. ¿Aplicar cooldown SOLO a thaws por gate_e, o a TODOS los thaws? Si solo gate_e, cómo distingo en el unfreeze path
3. ¿Persistir cooldown en learned_state.json o ephemeral en SharedState?
4. ¿Hay precedente de que un cooldown muy largo (>60s) cause undertreating de procesos zombi reales?"
```

- [ ] **Step 2: Document notebook response in plan**

Append to this plan section "NotebookLM Verdict (Phase 1)" with:
- Recommended cooldown duration
- Scope (gate_e-only vs all thaws)
- Persistence (yes/no)
- Any new constraints surfaced

- [ ] **Step 3: Adjust constants if notebook recommends different values**

If notebook recommends ≠60 cycles: update `GATE_E_COOLDOWN_CYCLES` in subsequent tasks.

---

## Phase 2: Implement FreezeCooldown Module

### Task 2.1: Create FreezeCooldown struct (TDD)

**Files:**
- Create: `src/engine/freeze_cooldown.rs`
- Modify: `src/engine/mod.rs` — add `pub mod freeze_cooldown;`

- [ ] **Step 1: Write the failing test**

Create `src/engine/freeze_cooldown.rs` with ONLY tests (no implementation):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newly_thawed_pid_is_in_cooldown() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        assert!(c.is_in_cooldown(1234));
    }

    #[test]
    fn untracked_pid_is_not_in_cooldown() {
        let c = FreezeCooldown::new();
        assert!(!c.is_in_cooldown(9999));
    }

    #[test]
    fn cooldown_expires_after_n_ticks() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        for _ in 0..GATE_E_COOLDOWN_CYCLES {
            c.tick();
        }
        assert!(!c.is_in_cooldown(1234));
    }

    #[test]
    fn cooldown_active_during_window() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        for _ in 0..(GATE_E_COOLDOWN_CYCLES - 1) {
            c.tick();
        }
        assert!(c.is_in_cooldown(1234));
    }

    #[test]
    fn re_thaw_resets_cooldown() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        for _ in 0..(GATE_E_COOLDOWN_CYCLES / 2) {
            c.tick();
        }
        c.mark_thawed(1234); // reset
        for _ in 0..(GATE_E_COOLDOWN_CYCLES / 2) {
            c.tick();
        }
        assert!(c.is_in_cooldown(1234)); // still in cooldown despite total ticks > N
    }
}
```

- [ ] **Step 2: Run tests, confirm they FAIL**

```bash
cargo test --lib freeze_cooldown 2>&1 | tail -10
```

Expected: compilation error — `FreezeCooldown not defined`.

- [ ] **Step 3: Write minimal implementation**

Replace test-only file with:

```rust
//! Per-PID post-thaw cooldown for gate_e anti-flap.
//!
//! Problem: gate_e (decide_actions.rs) resurrects the freezer for M1 8GB by
//! firing on swap_pct >= 0.85 + memory_pressure >= 0.70. The existing TTL
//! (`MAX_FROZEN_CYCLES=150` in planner.rs) thaws frozen PIDs after ~5 min.
//! Without an explicit cooldown, a thawed PID may still meet gate_e thresholds
//! and be re-frozen on the next cycle — oscillation under sustained pressure.
//!
//! This module tracks recently-thawed PIDs and prevents gate_e from re-freezing
//! them within the cooldown window. The kernel needs ~10-30s to redistribute
//! pressure after a SIGCONT; the cooldown gives swap reclaim a chance to take
//! effect before another freeze decision.
//!
//! Cooldown is **ephemeral** — does not persist across daemon restart.
//! On restart, the freezer treats the system as fresh; if real pressure
//! still exists, the gate_e logic will reapply naturally.
//!
//! [Nygard 2018] §8.5 — circuit breakers must include hold-down windows
//! after recovery to prevent thrashing on slow-decaying load conditions.

use std::collections::HashMap;

/// Cooldown duration in daemon cycles. At ~2 Hz tick rate this is ~30 s.
/// Tuned to be longer than typical kernel swap-redistribution latency
/// (10-15 s observed) but shorter than MAX_FROZEN_CYCLES (150 ≈ 5 min)
/// so a process held in real, sustained pressure will eventually re-freeze.
pub const GATE_E_COOLDOWN_CYCLES: u8 = 60;

#[derive(Debug, Default, Clone)]
pub struct FreezeCooldown {
    remaining: HashMap<u32, u8>,
}

impl FreezeCooldown {
    pub fn new() -> Self {
        Self {
            remaining: HashMap::new(),
        }
    }

    /// Mark a PID as recently thawed. Resets its cooldown counter.
    pub fn mark_thawed(&mut self, pid: u32) {
        self.remaining.insert(pid, GATE_E_COOLDOWN_CYCLES);
    }

    /// Returns true if the PID is in cooldown and must not be re-frozen by gate_e.
    pub fn is_in_cooldown(&self, pid: u32) -> bool {
        self.remaining.get(&pid).map(|&n| n > 0).unwrap_or(false)
    }

    /// Decrement all cooldown counters by 1; remove entries that reach 0.
    /// Call once per daemon cycle.
    pub fn tick(&mut self) {
        self.remaining.retain(|_pid, n| {
            *n = n.saturating_sub(1);
            *n > 0
        });
    }

    /// Number of PIDs currently in cooldown — for observability/metrics.
    pub fn active_count(&self) -> usize {
        self.remaining.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newly_thawed_pid_is_in_cooldown() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        assert!(c.is_in_cooldown(1234));
    }

    #[test]
    fn untracked_pid_is_not_in_cooldown() {
        let c = FreezeCooldown::new();
        assert!(!c.is_in_cooldown(9999));
    }

    #[test]
    fn cooldown_expires_after_n_ticks() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        for _ in 0..GATE_E_COOLDOWN_CYCLES {
            c.tick();
        }
        assert!(!c.is_in_cooldown(1234));
    }

    #[test]
    fn cooldown_active_during_window() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        for _ in 0..(GATE_E_COOLDOWN_CYCLES - 1) {
            c.tick();
        }
        assert!(c.is_in_cooldown(1234));
    }

    #[test]
    fn re_thaw_resets_cooldown() {
        let mut c = FreezeCooldown::new();
        c.mark_thawed(1234);
        for _ in 0..(GATE_E_COOLDOWN_CYCLES / 2) {
            c.tick();
        }
        c.mark_thawed(1234);
        for _ in 0..(GATE_E_COOLDOWN_CYCLES / 2) {
            c.tick();
        }
        assert!(c.is_in_cooldown(1234));
    }
}
```

Add to `src/engine/mod.rs` (alphabetically between existing pub mod declarations):
```rust
pub mod freeze_cooldown;
```

- [ ] **Step 4: Run tests to verify all pass**

```bash
cargo test --lib freeze_cooldown 2>&1 | tail -15
```

Expected: `5 passed; 0 failed`.

- [ ] **Step 5: Run clippy on new file**

```bash
cargo clippy --lib -- -D warnings 2>&1 | grep "freeze_cooldown" | head -5
```

Expected: zero output (no clippy warnings on new module).

- [ ] **Step 6: Commit**

```bash
git add src/engine/freeze_cooldown.rs src/engine/mod.rs
git commit -m "$(cat <<'EOF'
feat(freeze): add FreezeCooldown — per-PID post-thaw anti-flap for gate_e

Tracks recently-thawed PIDs to prevent oscillation when gate_e
(decide_actions.rs) re-fires immediately after a TTL-driven thaw.
GATE_E_COOLDOWN_CYCLES=60 at ~2Hz tick = 30s post-thaw window.

Standalone module — no callers yet. Wiring follows in subsequent commits.

[Nygard 2018] §8.5 circuit breaker hold-down semantics.

OPENS: 0
CLOSES: 0  # part of zero-risk-merge plan, closure on full wire

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Wire FreezeCooldown into SharedState

### Task 3.1: Add field to SharedState

**Files:**
- Modify: `src/engine/types.rs` (add `Arc<Mutex<FreezeCooldown>>` field)
- Modify: `src/bin/apollo-optimizerd/main.rs` (initialize field in constructor)

- [ ] **Step 1: Read current SharedState definition**

```bash
grep -n "pub struct SharedState\|frozen_state: Arc" src/engine/types.rs | head -5
```

Locate the struct definition and confirm the line range.

- [ ] **Step 2: Add field to SharedState**

In `src/engine/types.rs`, locate `pub struct SharedState` and add (alphabetically, near `frozen_state`):

```rust
    /// Per-PID post-thaw cooldown set. Prevents gate_e from re-freezing a PID
    /// that was just thawed by the TTL path. See `freeze_cooldown` module.
    pub freeze_cooldown: Arc<Mutex<crate::engine::freeze_cooldown::FreezeCooldown>>,
```

- [ ] **Step 3: Update constructor / Default impl**

If `SharedState` has a `Default` or `new()` impl in `types.rs`, add:
```rust
freeze_cooldown: Arc::new(Mutex::new(crate::engine::freeze_cooldown::FreezeCooldown::new())),
```

If `SharedState` is constructed inline in `main.rs` (around line 310 per existing grep), add the same field there.

- [ ] **Step 4: Verify compilation**

```bash
cargo check 2>&1 | tail -10
```

Expected: clean. If "missing field freeze_cooldown" errors appear, find the construction site and add the field.

- [ ] **Step 5: Commit**

```bash
git add src/engine/types.rs src/bin/apollo-optimizerd/main.rs
git commit -m "$(cat <<'EOF'
feat(state): wire FreezeCooldown into SharedState

Adds freeze_cooldown: Arc<Mutex<FreezeCooldown>> field. Not yet read by
decide_actions or written by unfreeze paths — those wires follow.

OPENS: 0
CLOSES: 0  # part of zero-risk-merge plan

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: Wire Cooldown into Unfreeze Paths

### Task 4.1: Mark thawed PIDs at unfreeze sites

**Files:**
- Modify: `src/bin/apollo-optimizerd/main.rs` (3 unfreeze sites at lines 1411, 1481, 1548)

- [ ] **Step 1: Locate the 3 unfreeze sites**

```bash
grep -n "frozen_state_path\|unfreeze_pids_verified" src/bin/apollo-optimizerd/main.rs | head -10
```

Confirm line numbers (may have drifted from 1411/1481/1548). For each site, read 20 lines of context to understand which PIDs are being thawed.

- [ ] **Step 2: After each successful SIGCONT, mark cooldown**

At each unfreeze site, after the SIGCONT command succeeds (look for `unfreeze_pids_verified` return value or `kill(pid, SIGCONT)` patterns), add:

```rust
// Mark thawed PIDs in cooldown to prevent gate_e re-freeze oscillation
{
    let mut cooldown = state.freeze_cooldown.lock_recover();
    for pid in &thawed_pids {
        cooldown.mark_thawed(*pid);
    }
}
```

The exact variable name (`thawed_pids`, `safe_pids`, `unfreeze_targets`) will depend on the site — use whatever Vec/Iter of `u32` PIDs is being thawed.

- [ ] **Step 3: Add `tick()` call once per main loop iteration**

In the main loop body (search for the start of the daemon's tick — likely a `loop { ... }` block in the run function around lines 700-900), add early in the iteration:

```rust
// Decrement freeze cooldown counters once per cycle
state.freeze_cooldown.lock_recover().tick();
```

- [ ] **Step 4: Verify compilation**

```bash
cargo check 2>&1 | tail -10
cargo test --bin apollo-optimizerd 2>&1 | tail -10
```

Expected: clean build, all existing daemon tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bin/apollo-optimizerd/main.rs
git commit -m "$(cat <<'EOF'
feat(daemon): tick freeze_cooldown + mark thawed PIDs

Per-cycle:
- freeze_cooldown.tick() decrements all cooldown counters
- After successful SIGCONT at any of the 3 unfreeze sites, mark_thawed(pid)
  populates the cooldown with GATE_E_COOLDOWN_CYCLES=60 entries

Without this wire, the cooldown HashMap stays empty and gate_e behaves
as if no anti-flap exists. Next commit (Phase 5) reads the cooldown
in decide_actions to filter freeze candidates.

OPENS: 0
CLOSES: 0  # part of zero-risk-merge plan

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5: Cherry-Pick Gate_e Worktree + Add Cooldown Filter

### Task 5.1: Cherry-pick gate_e commit

- [ ] **Step 1: Cherry-pick gate_e commit from worktree**

```bash
git cherry-pick 80f16c2 2>&1 | tail -10
```

Expected: clean cherry-pick (no conflicts — the worktree only modified `decide_actions.rs` and master has no parallel changes there since session start).

If conflicts appear: abort (`git cherry-pick --abort`) and re-evaluate.

- [ ] **Step 2: Verify build still clean**

```bash
cargo check 2>&1 | tail -5
cargo test --lib decide_actions 2>&1 | tail -10
```

Expected: gate_e tests (4) pass, full decide_actions suite green.

### Task 5.2: Filter gate_e candidates by cooldown

**Files:**
- Modify: `src/engine/decide_actions.rs` — at the gate_e candidate-filtering site

- [ ] **Step 1: Locate gate_e candidate selection**

```bash
grep -n "freeze_candidates\|gate_e\|swap-pct" src/engine/decide_actions.rs | head -15
```

Find the block that builds `freeze_candidates` after `extreme_freeze_ok` is true. Around line 1130-1200 based on prior session inspection.

- [ ] **Step 2: Add cooldown filter parameter**

The cleanest path is to pass the cooldown set as `&FreezeCooldown` parameter to `decide_actions`. Locate the function signature:

```bash
grep -n "pub fn decide_actions" src/engine/decide_actions.rs | head -3
```

Add to the existing parameter list (preserve backward compat by passing through a `&Option<FreezeCooldown>` if many callers exist; otherwise direct `&FreezeCooldown`):

```rust
pub fn decide_actions(
    // ... existing params ...
    freeze_cooldown: &crate::engine::freeze_cooldown::FreezeCooldown,
) -> Decision {
```

- [ ] **Step 3: Filter freeze_candidates**

In the freeze-candidate iteration block (where `freeze_candidates: Vec<...>` is consumed by `into_iter().take(3)`), add a filter:

```rust
let freeze_candidates: Vec<_> = freeze_candidates
    .into_iter()
    .filter(|(pid, _name, _rss, _cpu, _start_sec)| {
        // gate_e anti-flap: skip PIDs recently thawed
        if freeze_gate == "swap-pct" && freeze_cooldown.is_in_cooldown(*pid) {
            false
        } else {
            true
        }
    })
    .collect();
```

This filter is **only active when freeze_gate == "swap-pct"** (i.e., gate_e is the load-bearing gate). gate_a/b/c/d retain their original semantics — they fire on different physical conditions where post-thaw cooldown would be too conservative.

- [ ] **Step 4: Update all `decide_actions` callers**

```bash
grep -rn "decide_actions(" src/ --include="*.rs" | grep -v "fn decide_actions\|^[[:space:]]*//"
```

For each caller, pass the cooldown reference. In `main.rs` daemon loop:

```rust
let cooldown = state.freeze_cooldown.lock_recover();
let decision = decide_actions(/* ... existing args ..., */ &cooldown);
drop(cooldown); // release lock ASAP
```

In tests, pass `&FreezeCooldown::new()` (always-empty fallback).

- [ ] **Step 5: Add unit test for cooldown filter**

In `src/engine/decide_actions.rs::tests`, add:

```rust
#[test]
fn gate_e_skips_pid_in_cooldown() {
    use crate::engine::freeze_cooldown::FreezeCooldown;

    let mut cooldown = FreezeCooldown::new();
    cooldown.mark_thawed(1234);

    // Construct a synthetic snapshot that would trigger gate_e
    // (swap_pct=0.85, pressure=0.70, candidate pid=1234)
    // ... set up snapshot ...

    let decision = decide_actions(/* ... */, &cooldown);

    // Assert gate_e fired but pid 1234 was not selected as freeze target
    assert_eq!(decision.freeze_gate, "swap-pct");
    assert!(!decision.actions.iter().any(|a| matches!(a,
        RootAction::FreezeProcess { pid, .. } if *pid == 1234
    )));
}
```

If snapshot construction is too involved, add a more focused unit test on the filter logic itself rather than full pipeline.

- [ ] **Step 6: Verify all tests pass**

```bash
cargo build 2>&1 | tail -5
cargo test 2>&1 | tail -20
```

Expected: zero failures across the entire suite. If anything breaks, fix before commit.

- [ ] **Step 7: Commit**

```bash
git add src/engine/decide_actions.rs src/bin/apollo-optimizerd/main.rs
git commit -m "$(cat <<'EOF'
fix(freeze): gate_e respects FreezeCooldown — close oscillation gap

When gate_e is the load-bearing freeze gate ("swap-pct" label), filter
out PIDs in post-thaw cooldown. Other gates (a/b/c/d) retain unchanged
semantics because their physical triggers (delta, committed, thrashing,
swap-oom) reflect different failure modes where cooldown would be too
conservative.

Closes the NotebookLM-flagged oscillation risk: TTL-driven thaw of a
gate_e-frozen PID followed by immediate re-freeze on the next cycle.

OPENS: 0
CLOSES: 1  # zero-risk-merge: gate_e oscillation gap

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6: Cherry-Pick Remaining Worktrees

### Task 6.1: Merge Protection commit

- [ ] **Step 1: Cherry-pick smallest commit first (lowest risk)**

```bash
git cherry-pick e768d6d 2>&1 | tail -5
```

- [ ] **Step 2: Verify**

```bash
cargo build 2>&1 | tail -3
cargo test --bin apollo-optimizerd cognitive_tick 2>&1 | tail -5
```

Expected: clean build, cognitive_tick tests pass.

### Task 6.2: Merge Chromium commit

- [ ] **Step 1: Cherry-pick chromium adaptive**

```bash
git cherry-pick 7bdb931 2>&1 | tail -5
```

- [ ] **Step 2: Verify**

```bash
cargo build 2>&1 | tail -3
cargo test --lib chromium_manager 2>&1 | tail -5
```

Expected: 69+ chromium tests pass (5 new + existing).

### Task 6.3: Final full-suite validation

- [ ] **Step 1: Run full test suite**

```bash
cargo test --release 2>&1 | tail -20
```

Expected: zero failures. Note total count for post-deploy comparison.

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --all-targets 2>&1 | grep -c "^warning"
```

Compare to pre-merge baseline (~204 warnings). Expected: same or fewer.

- [ ] **Step 3: Capture merge state**

```bash
git log --oneline -8
echo "---"
git log master..HEAD --stat | tail -30
```

---

## Phase 7: Pre-Deploy Smoke Test

### Task 7.1: Build release binary

- [ ] **Step 1: Build release**

```bash
cargo build --release 2>&1 | tail -5
```

Expected: `Finished release profile`, no errors.

- [ ] **Step 2: Verify binary exists and is fresh**

```bash
ls -la target/release/apollo-optimizerd
file target/release/apollo-optimizerd
```

Expected: built within last 5 min, Mach-O 64-bit arm64.

- [ ] **Step 3: Run binary in dry-run mode (if supported)**

```bash
./target/release/apollo-optimizerd --help 2>&1 | head -10
```

Expected: help text prints — confirms binary is executable.

---

## Phase 8: Production Deploy + Monitor

### Task 8.1: Deploy with rollback safety

- [ ] **Step 1: Backup current production binary**

```bash
sudo cp /usr/local/libexec/apollo-optimizerd /usr/local/libexec/apollo-optimizerd.pre-merge-backup
```

- [ ] **Step 2: Replace production binary (preserving codesign)**

```bash
sudo cp target/release/apollo-optimizerd /usr/local/libexec/apollo-optimizerd
```

**Critical:** use `sudo cp`, NOT Python file write — Python strips linker-signed flag and breaks codesign.

- [ ] **Step 3: Restart daemon via launchctl**

```bash
sudo launchctl bootout system/com.eduardocortez.systemoptimizerd 2>&1
sleep 3
sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist
sleep 5
sudo apollo-optimizerctl status | head -5
```

Expected: `running: true`, profile reported, fresh `cycles` count.

### Task 8.2: 30-Minute Monitor Window

- [ ] **Step 1: Capture metrics every 5 min for 30 min**

Run this shell loop:
```bash
for i in 1 2 3 4 5 6; do
  echo "=== T+${i}5min ==="
  sudo apollo-optimizerctl status | python3 -c "
import json,sys
d=json.load(sys.stdin)
m=d['metrics']
print(f'cycles={m[\"cycles\"]} p95={m[\"p95_cycle_ms\"]}ms')
print(f'freezes={m[\"freezes_applied\"]} unfreezes={m[\"unfreezes_applied\"]} throttles={m[\"throttles_applied\"]}')
print(f'failures={m[\"failures\"]} thermal={d[\"thermal_state\"]}')
"
  sleep 300
done
```

- [ ] **Step 2: Apply rollback criteria**

| Metric | Pass | Rollback trigger |
|--------|------|------------------|
| `failures` delta | == 0 over 30 min | > 0 |
| `freezes_applied` rate | < 50/h | > 50/h (oscillation) |
| `unfreezes_applied / freezes_applied` ratio | < 1.5 | > 2.0 (thaw thrash) |
| `p95_cycle_ms` | < 250 ms | > 350 ms |
| `swap` GB | trending down OR stable | climbing > 1 GB in 30 min |
| `last_error` | None | non-None |

- [ ] **Step 3: Capture causal graph evidence for new edges**

```bash
sudo apollo-optimizerctl teach export | grep -A 12 "Causal Graph"
```

Expected: new edges appearing for `freeze:*` patterns now that gate_e is firing. Document for next session.

### Task 8.3: Rollback procedure (if any criterion fails)

- [ ] **Step 1: If rollback criterion hit, restore backup**

```bash
sudo cp /usr/local/libexec/apollo-optimizerd.pre-merge-backup /usr/local/libexec/apollo-optimizerd
sudo launchctl bootout system/com.eduardocortez.systemoptimizerd
sleep 3
sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist
```

- [ ] **Step 2: Confirm rollback**

```bash
sudo apollo-optimizerctl status | head -5
```

Expected: daemon back on pre-merge SHA.

- [ ] **Step 3: Revert master to pre-merge SHA**

```bash
git reset --hard 59be645
```

This is destructive — only run if production verifies rollback successful.

- [ ] **Step 4: Document failure mode**

Append to this plan, "Phase 8 — Rollback Postmortem", what metric tripped, when, and the snapshot data. Used to inform next iteration.

---

## Phase 9: Post-Deploy Notebook Sync

### Task 9.1: Push session result to NotebookLM

- [ ] **Step 1: Build session delta source text**

Compose summary including:
- All commits merged (SHAs + subjects)
- Pre/post deploy metric deltas
- Rollback (yes/no)
- New causal edges for gate_e

- [ ] **Step 2: Add as source via MCP**

```
mcp__notebooklm-mcp__source_add(
    notebook_id="8344b94c-a014-4803-abea-076a55753cfd",
    source_type="text",
    title="Session 2026-05-03 — zero-risk merge result",
    text="<the summary>",
    wait=true
)
```

- [ ] **Step 3: Add note**

```
mcp__notebooklm-mcp__note(
    notebook_id="8344b94c-a014-4803-abea-076a55753cfd",
    action="create",
    title="Zero-risk merge — outcome",
    content="3 commits merged. Anti-flap added (FreezeCooldown). Deploy result: <pass/fail>. Next session focus: 7 substring-scan latent bugs in protection."
)
```

---

## Phase 10: Cleanup

### Task 10.1: Remove worktrees

- [ ] **Step 1: Confirm commits are in master**

```bash
git log --oneline | head -10 | grep -E "(80f16c2|e768d6d|7bdb931)"
```

Expected: 3 SHAs (or their cherry-picked equivalents) in master log.

- [ ] **Step 2: Remove worktrees**

```bash
git worktree remove -f .claude/worktrees/agent-a330670c592d46ae2
git worktree remove -f .claude/worktrees/agent-a4a1e514a0107dfa5
git worktree remove -f .claude/worktrees/agent-af42b7c1666fcf5b8
git branch -D worktree-agent-a330670c592d46ae2 worktree-agent-a4a1e514a0107dfa5 worktree-agent-af42b7c1666fcf5b8
git worktree list
```

Expected: only main worktree remains.

---

## Rollback Tree (worst-case decision matrix)

```
Phase 8 monitor reports anomaly
├── failures > 0 OR last_error != None
│   └── IMMEDIATE rollback (Task 8.3)
├── freezes_applied > 50/h
│   └── IMMEDIATE rollback — gate_e oscillating despite cooldown
├── p95 > 350ms sustained
│   └── IMMEDIATE rollback — likely SharedState lock contention
├── swap climbing despite freezes
│   └── HOLD — system genuinely overloaded, not Apollo regression
└── all criteria pass at T+30min
    └── PROCEED to Phase 9 (notebook sync)
```

---

## Completion Criteria

This plan is **done** when:
1. All 3 worktree commits are in master (cherry-picked or merged)
2. `FreezeCooldown` module is integrated and tested
3. Production daemon runs ≥ 30 min without rollback trigger
4. NotebookLM has session result as new source
5. Worktrees cleaned up

**Total estimated cycles:** ~3 hours (25 min implementation + 30 min monitor + buffers).

**Rollback budget:** ~5 min if any monitor criterion trips.
