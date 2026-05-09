# Workspace Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert apollo-optimizer single-crate monolith (~110K LoC) into a Cargo workspace with `apollo-engine` library crate, reducing `cargo test` wall time from ~20 min to ≤8 min for engine-only runs without changing runtime behavior.

**Architecture:** Workspace root with three binaries (`apollo-optimizer`, `apollo-optimizerd`, `apollo-optimizerctl`, `apollo-menubar`) + `crates/apollo-engine/` library crate hosting the 148 engine modules and FFI native bridges. Root keeps a one-line facade `lib.rs` (`pub use apollo_engine::*`) so existing integration tests resolve `apollo_optimizer::engine::*` paths without rewrites.

**Tech Stack:** Rust 1.x edition 2021, Cargo workspaces, `cc` crate for C FFI compilation, `git mv` for path-preserving file moves.

**Spec:** `docs/superpowers/specs/2026-05-09-workspace-split-design.md`

**Time budget per task** (NotebookLM 2026-05-09 review — engineer should activate the abort threshold rather than overspend):

| Task | Est. wall-time | If exceeds | Action |
|---|---|---|---|
| Task 0 (baseline capture) | 30 min | 1h | Continue — baseline matters |
| Task 1 (workspace skeleton) | 30 min | 1h | Re-check Cargo.toml syntax |
| Task 2 (engine crate stub) | 30 min | 1h | Re-check member paths |
| Task 3 (physical move) | 30 min + ~10 min build | 1h total | Investigate path drift |
| Task 4 (sed migration) | 15 min | 30 min | sed pattern issue |
| Task 5 (facade replace) | 15 min | 30 min | Check Cargo.toml deps |
| Task 6 (compiler-driven repair) | 2-4 hours | 4 hours OR 30+ leaks | ABORT, rollback |
| Task 7 (visibility audit) | 1-2 hours | 3 hours | Defer to follow-up Sprint |
| Task 8 (test relocation) | 30 min | 1h | Skip ambiguous tests, leave in root |
| Task 9 (dashboard move) | 30 min | 1h | Investigate Cargo bin path |
| Task 10 (validation) | 30 min | 1h | Continue — validation is the gate |

Total expected: 6-9 hours. >12 hours signals scope creep — pause and reassess.

---

## Pre-flight: Baseline capture

### Task 0: Record current state for comparison

**Files:**
- Create: `evolve/2026-05-09-workspace-split/baseline.txt`

- [ ] **Step 1: Confirm git HEAD is clean and on main branch**

Run: `git status && git log -1 --oneline`
Expected: working tree clean, latest commit `808dc19 docs(spec): integrate NotebookLM adversarial review for Workspace Split` (or later)

- [ ] **Step 2: Create baseline directory**

Run: `mkdir -p evolve/2026-05-09-workspace-split`
Expected: directory exists, no error

- [ ] **Step 3: Capture cargo test wall-time baseline**

Run: `time cargo test --release --no-run 2>&1 | tail -3`

Wait until completion. Note the `real` time. Expected: ~20 min for full compile (tests aren't run yet, just compiled).

Then run: `time cargo test --release 2>&1 | tail -3`

Expected: ~20 min total (compile dominates), ~1967 tests pass, 1 known flake (`thread_selfcounts::selfcounts_increases_with_work`).

- [ ] **Step 4: Capture clippy warning count**

Run: `cargo clippy --release --all-targets 2>&1 | grep -c "^warning:"`
Expected: a number around 218 (current baseline). Save the number for Commit 4 comparison.

- [ ] **Step 5: Write baseline file**

Create `evolve/2026-05-09-workspace-split/baseline.txt` with these exact contents (replace `<value>` with measured values):

```
date_utc: 2026-05-09
commit: <git rev-parse HEAD>
cargo_test_wall_time: <real time from Step 3>
test_count_pass: <count of "passed" line>
test_count_fail: <count of "failed" line>
clippy_warning_count: <number from Step 4>
daemon_pid_pre_split: <ps -p $(sudo launchctl print system/com.eduardocortez.systemoptimizerd | grep -E "^	pid" | awk '{print $3}') -o etime= | tr -d ' '>
daemon_failures: <jq .metrics.failures /var/lib/apollo/runtime_metrics.json>
```

- [ ] **Step 6: Commit baseline**

```bash
git add evolve/2026-05-09-workspace-split/baseline.txt
git commit -m "chore(workspace-split): capture pre-split baseline metrics"
```

---

## Commit 1: Workspace skeleton

### Task 1: Convert root Cargo.toml to workspace

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Read current Cargo.toml**

Run: `cat Cargo.toml`

Confirm current structure: `[package]`, `[dependencies]`, `[dev-dependencies]`, `[build-dependencies]` sections present.

- [ ] **Step 2: Replace Cargo.toml with workspace + facade lib structure**

Overwrite `Cargo.toml` with:

```toml
[workspace]
members = [".", "crates/apollo-engine"]
resolver = "2"

# Workspace-level deps pin VERSIONS only. Per-crate Cargo.toml declares
# features explicitly to prevent feature unification (e.g. UI crate
# enabling Rayon-via-sysinfo would unintentionally activate it inside
# apollo-engine hot path — CLAUDE.md invariant: Rayon disabled).
[workspace.dependencies]
sysinfo = { version = "0.30", default-features = false }
serde = "1.0"
serde_json = "1.0"
clap = "4.4"
chrono = "0.4"
anyhow = "1.0"
libc = "0.2"
ctrlc = "3.4"
toml = "0.8"
ureq = "2.12"
tracing = "0.1"
tracing-subscriber = "0.3"
tempfile = "3"

[package]
name = "apollo-optimizer"
version = "1.0.0"
edition = "2021"

[lib]
path = "src/lib.rs"

[dependencies]
sysinfo = { workspace = true, features = ["serde"] }
serde = { workspace = true, features = ["derive"] }
serde_json.workspace = true
clap = { workspace = true, features = ["derive"] }
chrono = { workspace = true, features = ["serde"] }
anyhow.workspace = true
libc.workspace = true
ctrlc = { workspace = true, features = ["termination"] }
toml.workspace = true
ureq = { workspace = true, features = ["json"] }
tracing.workspace = true
tracing-subscriber = { workspace = true, features = ["json", "env-filter"] }

[dev-dependencies]
tempfile.workspace = true

[build-dependencies]
```

- [ ] **Step 3: Verify cargo recognizes workspace**

Run: `cargo metadata --no-deps --format-version 1 2>&1 | jq -r '.workspace_members[]' 2>/dev/null || cargo metadata --no-deps 2>&1 | head -3`

Expected: ERROR — workspace members include `apollo-engine` but the directory does not yet exist. This failure is expected, Task 2 creates it.

### Task 2: Create empty apollo-engine crate

**Files:**
- Create: `crates/apollo-engine/Cargo.toml`
- Create: `crates/apollo-engine/src/lib.rs`

- [ ] **Step 1: Create crate directory**

```bash
mkdir -p crates/apollo-engine/src
```

- [ ] **Step 2: Create apollo-engine Cargo.toml**

Write `crates/apollo-engine/Cargo.toml`:

```toml
[package]
name = "apollo-engine"
version = "1.0.0"
edition = "2021"

[lib]
path = "src/lib.rs"

[dependencies]
sysinfo = { workspace = true, features = ["serde"] }
serde = { workspace = true, features = ["derive"] }
serde_json.workspace = true
chrono = { workspace = true, features = ["serde"] }
anyhow.workspace = true
libc.workspace = true
toml.workspace = true
ureq = { workspace = true, features = ["json"] }
tracing.workspace = true
tracing-subscriber = { workspace = true, features = ["json", "env-filter"] }
```

- [ ] **Step 3: Create empty lib.rs stub**

Write `crates/apollo-engine/src/lib.rs`:

```rust
//! apollo-engine — core runtime library for apollo-optimizer.
//!
//! Sprint 5 Mes 0: empty stub. Engine modules migrate in Commit 2.
//! See docs/superpowers/specs/2026-05-09-workspace-split-design.md.
```

- [ ] **Step 4: Verify workspace metadata resolves**

Run: `cargo metadata --no-deps --format-version 1 2>&1 | jq -r '.workspace_members[]'`

Expected output (paths resolved):
```
apollo-optimizer 1.0.0 (path+file:///...)
apollo-engine 1.0.0 (path+file:///...)
```

If error: re-check Cargo.toml syntax in both root and crate.

- [ ] **Step 5: Verify root crate still compiles (engine still in src/engine/)**

Run: `cargo build --workspace 2>&1 | tail -5`

Expected: `Finished ... target(s) in <time>`. Both root crate and empty apollo-engine compile. Engine code is still physically in `src/engine/` referenced by `crate::engine::*` in bins — untouched. apollo-engine is empty, so it provides nothing yet.

- [ ] **Step 6: Run baseline tests to confirm nothing broke**

Run: `cargo test --workspace --release --no-run 2>&1 | tail -3`

Expected: compiles cleanly. Tests not yet executed (--no-run for speed).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates/apollo-engine/
git commit -m "feat(workspace): convert root to Cargo workspace + empty apollo-engine crate

First commit of Workspace Split (Sprint 5 Mes 0). Establishes the
workspace skeleton:
- Root Cargo.toml: [workspace] block listing 2 members + per-version
  workspace.dependencies (features per-crate, NOT shared, to prevent
  feature unification leaking Rayon into apollo-engine hot path).
- Root keeps [lib] (still pointing at src/lib.rs unchanged) and 4 bins.
- crates/apollo-engine/: empty stub crate. Engine modules migrate in
  Commit 2.

State: workspace compiles, all bins still work via crate::engine::*
(physically unchanged), apollo-engine provides no symbols yet.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Commit 2: Move engine module physically

### Task 3: Migrate engine source tree to apollo-engine

**Files:**
- Move: `src/engine/` → `crates/apollo-engine/src/engine/`
- Move: `src/engine_c/` → `crates/apollo-engine/native/`
- Modify: `crates/apollo-engine/src/lib.rs`
- Modify: `crates/apollo-engine/Cargo.toml` (add build script)
- Move/Modify: `build.rs` if exists at root, otherwise create at `crates/apollo-engine/build.rs`
- Modify: `src/lib.rs` (root) — temporary shim

- [ ] **Step 1: Verify build.rs location BEFORE moving anything**

Critical pre-check (NotebookLM 2026-05-09 review): the C FFI bridges (`ioreport_bridge.c`, `smc_bridge.c`) are compiled by a `build.rs` script. Determine where it currently lives:

```bash
ls -la build.rs 2>/dev/null && echo "BUILD.RS AT ROOT"
ls -la crates/apollo-engine/build.rs 2>/dev/null && echo "BUILD.RS IN APOLLO-ENGINE"
test ! -f build.rs && test ! -f crates/apollo-engine/build.rs && echo "BUILD.RS MISSING — will create in Task 3 Step 4"
```

Also verify `cc` crate is currently a build-dependency:
```bash
grep -E "^cc\s*=|cc\s*=\s*\".*\"" Cargo.toml
```

Expected: one of these prints. Note the location — Task 3 Step 4 branches on it.

- [ ] **Step 2: Move engine module physically**

```bash
git mv src/engine crates/apollo-engine/src/engine
```

Verify: `ls crates/apollo-engine/src/engine/ | head -5` shows familiar files (e.g. `types.rs`, `safety.rs`, `action_accumulator.rs`).

- [ ] **Step 3: Move engine_c FFI bridges**

```bash
git mv src/engine_c crates/apollo-engine/native
```

Verify: `ls crates/apollo-engine/native/` shows `ioreport_bridge.c`, `smc_bridge.c`.

- [ ] **Step 4: Move build.rs (if exists at root) and update paths**

If root had build.rs:
```bash
git mv build.rs crates/apollo-engine/build.rs
```

Open `crates/apollo-engine/build.rs` and replace any path references. Find lines like:
```rust
.file("src/engine_c/ioreport_bridge.c")
```
Replace with:
```rust
.file("native/ioreport_bridge.c")
```

If build.rs did not exist at root, create `crates/apollo-engine/build.rs`:
```rust
fn main() {
    cc::Build::new()
        .file("native/ioreport_bridge.c")
        .file("native/smc_bridge.c")
        .compile("apollo_native");
    println!("cargo:rerun-if-changed=native/ioreport_bridge.c");
    println!("cargo:rerun-if-changed=native/smc_bridge.c");
}
```

- [ ] **Step 5: Add cc as build-dependency to apollo-engine**

Edit `crates/apollo-engine/Cargo.toml` and append:

```toml
[build-dependencies]
cc = "1.0"
```

- [ ] **Step 6: Wire engine module in apollo-engine lib.rs**

Replace `crates/apollo-engine/src/lib.rs` with:

```rust
//! apollo-engine — core runtime library for apollo-optimizer.
//!
//! Hosts the engine module tree migrated from src/engine/ in Sprint 5
//! Mes 0 (Workspace Split). Public surface mirrors the pre-split
//! `apollo_optimizer::engine` namespace exactly.
//!
//! See docs/superpowers/specs/2026-05-09-workspace-split-design.md.

pub mod engine;
```

- [ ] **Step 7: Add temporary shim in root lib.rs**

This is the **one-commit shim** allowed by the boundary-leak rule. Bins still use `crate::engine::*` paths. Commit 3 removes the shim.

Edit `src/lib.rs` (root). Find any existing `pub mod engine;` line. Replace with:

```rust
//! apollo-optimizer — facade crate for binaries and integration tests.
//!
//! TEMPORARY (Commit 2 only): re-exports apollo_engine::engine so existing
//! `crate::engine::*` paths in src/bin/ keep compiling during the move.
//! Commit 3 removes this shim and bins migrate to `apollo_engine::engine::*`.

pub mod engine {
    pub use apollo_engine::engine::*;
}
```

If the file had other `pub mod` lines (e.g. `pub mod dashboard;`), preserve them.

- [ ] **Step 8: Verify workspace builds**

Run: `cargo build --workspace 2>&1 | tail -10`

Expected: `Finished ... target(s) in <time>`. apollo-engine compiles 91K LoC of engine + FFI; bins compile via shim. **First incremental build will be slow (~5-10 min) because nothing is cached.**

If errors:
- "file not found": check `git mv` operations — are paths correct?
- "build script failed": check build.rs paths point to `native/` not `src/engine_c/`
- "cannot find module `engine`": check `crates/apollo-engine/src/lib.rs` has `pub mod engine;`
- "duplicate module" in root: another `mod engine;` may still exist in `src/main.rs` or root lib — remove duplicates

- [ ] **Step 9: Verify cargo check passes per-crate**

Run: `cargo check -p apollo-engine 2>&1 | tail -5`
Expected: clean.

Run: `cargo check -p apollo-optimizer 2>&1 | tail -5`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat(workspace): move engine module + native FFI to apollo-engine crate

Physical migration only — no semantic changes.

- src/engine/ -> crates/apollo-engine/src/engine/ (148 .rs files, 91K LoC)
- src/engine_c/ -> crates/apollo-engine/native/ (C FFI bridges)
- build.rs relocated/created at apollo-engine crate root with cc paths
  pointing to native/ (was src/engine_c/)
- crates/apollo-engine/src/lib.rs declares pub mod engine
- Root src/lib.rs gets a TEMPORARY shim (one commit only):
  pub mod engine { pub use apollo_engine::engine::*; }
  This keeps bins compiling via crate::engine::* during the move.
  Commit 3 removes the shim and migrates bin imports.

State: workspace builds, bins still reference crate::engine::* via shim.
Bins themselves untouched.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Commit 3: Mechanical migration + compiler-driven repair

### Task 4: Sed bulk path migration in bins and tests

**Files:**
- Modify (sed): all `.rs` files under `src/bin/` and `tests/`

- [ ] **Step 1: Preview which files will change**

Run:
```bash
grep -rl "crate::engine::\|apollo_optimizer::engine::" src/bin tests
```

Expected: ~40-50 file paths. These are the files about to be sed'd.

- [ ] **Step 2: Run sed migration**

```bash
find src/bin tests -name "*.rs" -exec sed -i '' \
  -e 's|crate::engine::|apollo_engine::engine::|g' \
  -e 's|apollo_optimizer::engine::|apollo_engine::engine::|g' {} +
```

Note: macOS sed requires `-i ''` (empty backup arg). Linux sed uses `-i` alone.

- [ ] **Step 3: Spot-check the migration**

Run:
```bash
grep "crate::engine::\|apollo_optimizer::engine::" src/bin tests -r 2>/dev/null | head -10
```

Expected: empty output. If any matches remain, those files have unusual path forms (macros, raw strings) — investigate manually before continuing.

### Task 5: Replace shim with facade re-export

**Files:**
- Modify: `src/lib.rs` (root)

- [ ] **Step 1: Replace temporary shim with facade**

Overwrite `src/lib.rs`:

```rust
//! apollo-optimizer — facade crate for binaries and integration tests.
//!
//! Sprint 5 Mes 0: this lib is a thin facade that re-exports
//! `apollo-engine` so legacy paths `apollo_optimizer::engine::...` in
//! integration tests and any external consumer continue to resolve.
//!
//! Discipline:
//! - This file MUST stay this size (one re-export, no other modules).
//! - New logic goes in `apollo-engine` (lib) or `src/bin/<name>/` (bin).
//! - If you're tempted to add `pub mod foo` here, you're rebuilding the
//!   monolith. Don't.
//!
//! See docs/superpowers/specs/2026-05-09-workspace-split-design.md.

pub use apollo_engine::*;
```

- [ ] **Step 2: Verify root crate Cargo.toml depends on apollo-engine**

Run: `grep -A2 "\[dependencies\]" Cargo.toml | head -5`

If apollo-engine is not listed, edit `Cargo.toml` and add to `[dependencies]`:
```toml
apollo-engine = { path = "crates/apollo-engine" }
```

- [ ] **Step 3: First compile attempt**

Run: `cargo check --workspace 2>&1 | tail -30`

Expected: SOME compile errors. Don't panic — Task 6 fixes them iteratively.

### Task 6: Compiler-driven repair loop

**Files:**
- Modify: various files in `crates/apollo-engine/src/engine/` and `src/bin/`

This task is iterative: run `cargo check`, fix one error, repeat.

- [ ] **Step 1: First error category — `super::` paths inside engine modules**

Some engine files use `super::types::Foo` to reference siblings. After moving to apollo-engine, those still resolve correctly RELATIVE to the apollo-engine crate root (which is now `crates/apollo-engine/src/lib.rs` → `pub mod engine`).

Look for errors like:
```
error[E0432]: unresolved import `super::foo`
```

Diagnosis: most `super::` paths inside `crates/apollo-engine/src/engine/<module>.rs` are fine because `super::` still points to `engine::`. But if any `super::` was relying on root-crate scope (e.g. `super::dashboard::Foo` where dashboard was at root), it breaks.

Fix per occurrence: rewrite as `crate::engine::<correct_path>` (relative to apollo-engine crate, not the old root).

- [ ] **Step 2: Second error category — `pub(crate)` access from bins**

After the split, items declared `pub(crate)` in apollo-engine are no longer accessible from bins (different crate now). The compiler will flag:
```
error[E0603]: function `Foo::bar` is private
```

Fix per occurrence (case by case, NOT blanket sed):
1. Open the engine file declaring `pub(crate) fn bar`.
2. Confirm the bin actually NEEDS this function (not just convenience).
3. Promote with explicit doc comment:

```rust
/// Cross-crate access: required by apollo-optimizerd's daemon_dispatch_tick
/// to short-circuit the action queue under critical pressure. Promoted
/// from pub(crate) during Sprint 5 Mes 0 workspace split — see
/// docs/superpowers/specs/2026-05-09-workspace-split-design.md.
pub fn bar(...) { ... }
```

- [ ] **Step 3: Third error category — boundary leaks (apollo-engine importing from bin)**

If apollo-engine code references a type or function that lives in a bin (root `src/bin/<name>/`), the compiler will flag:
```
error[E0432]: unresolved import `crate::dashboard`
```
or
```
error[E0433]: failed to resolve: use of undeclared crate or module
```

This is a **boundary leak**. DO NOT add a shim. Action:

1. Add comment at the offending line:
```rust
// BOUNDARY-LEAK: apollo-engine references <symbol> defined in <bin>.
// TODO Sprint 5 Mes 1+: extract <symbol> to apollo-engine or the bin
// stops requiring engine to know about it.
```

2. Append to the spec file (`docs/superpowers/specs/2026-05-09-workspace-split-design.md` under "Discovered boundary leaks"):
```markdown
- `<symbol>` referenced by `<bin path>` from `<engine module path>` — context: <one line why>
```

3. Resolution path (NotebookLM 2026-05-09 review): **NEVER duplicate types** — duplicates create silent type mismatches at compile time (e.g. `engine::types::RootAction` ≠ `local::RootAction`, indistinguishable until the next sprint surfaces a serde or trait bound bug). Two acceptable resolutions:
   - **Promote visibility**: if the leaked symbol logically belongs in apollo-engine, make it `pub` there and delete the bin-side definition.
   - **Move symbol to engine**: if the symbol is currently in a bin but engine code references it, the symbol's home is wrong — relocate to apollo-engine.

   If neither resolution fits the symbol cleanly, the abort threshold (30 leaks) is your signal that this split is too deep for this Sprint.

If you reach >30 distinct boundary leaks, **STOP**. Per spec abort threshold, the split is too deep for this Sprint. Execute rollback:

```bash
git reset --hard HEAD~1   # back to Commit 2 state
# Or full rollback to pre-split tag:
git reset --hard <commit-before-task-0>
```

Document the leak inventory in the spec under "Discovered boundary leaks" before rolling back so the next Sprint has the data.

- [ ] **Step 4: Repeat compile-fix-recompile until clean**

Run repeatedly:
```bash
cargo check --workspace 2>&1 | grep -E "^error\[" | head -3
```

Each time, fix one error using the patterns from Steps 1-3. Should converge in <2 hours of work for a typical workspace split.

When `cargo check --workspace` is clean:

```bash
cargo check --workspace 2>&1 | tail -3
```
Expected: `Finished ... target(s) in <time>`. No errors.

- [ ] **Step 5: Verify apollo-engine compiles independently**

Run: `cargo check -p apollo-engine 2>&1 | tail -3`
Expected: clean.

This proves apollo-engine has no upward dependencies on bins (excluding any documented boundary-leak duplicates).

- [ ] **Step 6: Run engine tests for first time**

Run: `cargo test -p apollo-engine --release 2>&1 | tail -10`

Expected: most tests pass. Some `super::` test imports may need similar fixes. Apply the same patterns from Steps 1-3.

- [ ] **Step 7: Run full workspace tests**

Run: `cargo test --workspace --release 2>&1 | tail -10`

Expected: same pass/fail count as baseline (1967 pass, 1 known flake on `selfcounts_increases_with_work`).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(workspace): mechanical import migration + compiler-driven repair

Commit 3 of Workspace Split. Removes the temporary shim from Commit 2
and migrates all bin/test imports to the new crate path:
  crate::engine::Foo  ->  apollo_engine::engine::Foo
  apollo_optimizer::engine::Foo  ->  apollo_engine::engine::Foo

Root src/lib.rs replaced with one-line facade:
  pub use apollo_engine::*;

This keeps the legacy apollo_optimizer::engine::... paths used by
integration tests resolvable. The facade is intentionally minimal — no
modules, no logic, no growth allowed (see file doc comment).

Compiler-driven repair handled three categories beyond what sed could
mechanically address:
1. super:: paths inside engine modules
2. pub(crate) -> pub promotions for items bins legitimately need
3. Boundary leaks (apollo-engine -> bin references) surfaced and
   documented, NOT papered over with shims

Boundary leaks discovered: <count>. See spec for the running list.

State: cargo check --workspace clean, cargo test --workspace passes
the same suite as baseline (1967 pass, 1 known flake).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Commit 4: Explicit visibility audit

### Task 7: Audit types.rs / protocol.rs / SharedState

**Files:**
- Modify: `crates/apollo-engine/src/engine/types.rs`
- Modify: `crates/apollo-engine/src/engine/protocol.rs`
- Modify: `crates/apollo-engine/src/engine/daemon_state.rs` (SharedState)

This task is proactive (NOT minimal). NotebookLM peer-review flagged that blanket `pub(crate)` → `pub` violates Information Hiding. Each `pub` change needs justification.

- [ ] **Step 1: List `pub(crate)` items in types.rs that bins now access**

Run:
```bash
grep -n "pub(crate)" crates/apollo-engine/src/engine/types.rs | head -30
```

Note line numbers and item names. For each:
- Search bins for usage: `grep -rn "<item_name>" src/bin/`
- If bins use it directly via `apollo_engine::engine::types::<item>`, the item needs `pub` (it was promoted in Commit 3 already if compiler demanded it).
- If no bin uses it, KEEP `pub(crate)` — Information Hiding.

- [ ] **Step 2: Apply justifying doc comments to types.rs promotions**

For each item promoted from `pub(crate)` to `pub` in Commit 3, add a doc comment ABOVE it:

```rust
/// Cross-crate visibility: required by apollo-optimizerd's <module> for
/// <specific reason>. Was pub(crate) before Sprint 5 Mes 0 workspace split.
pub fn item_name(...) { ... }
```

If you cannot articulate a reason, the item should not be `pub`. Demote back to `pub(crate)` and refactor the bin to not need it.

- [ ] **Step 3: Repeat audit for protocol.rs**

Same pattern as Step 1-2 but for `crates/apollo-engine/src/engine/protocol.rs`. Wire-protocol types are typically all `pub` because they cross the IPC boundary — confirm but expect fewer changes.

- [ ] **Step 4: Audit SharedState in daemon_state.rs**

Open `crates/apollo-engine/src/engine/daemon_state.rs` and find the `SharedState` struct.

This is a god-node per the graph report (decide=55 edges, snap=45). Each `pub` field may have been promoted indiscriminately. Audit each:
- Does any bin access this field directly via `state.field` from outside apollo-engine?
- If yes: keep `pub`, add comment explaining what bin needs it.
- If no: demote to `pub(crate)`.

- [ ] **Step 5: Run cargo check after demotions**

Run: `cargo check --workspace 2>&1 | tail -10`

If any items you demoted ARE still referenced by bins, the compiler will flag them. Re-promote with the proper justification doc comment.

- [ ] **Step 6: Verify clippy delta vs baseline**

Run: `cargo clippy --workspace --all-targets --release 2>&1 | grep -c "^warning:"`

Compare to baseline number from Task 0 Step 4. Expected: same count or slightly fewer (visibility tightening can suppress some warnings, never adds them).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(visibility): audit pub(crate) -> pub promotions post split

Commit 4 of Workspace Split. NotebookLM adversarial review reframed
this commit as proactive visibility audit, NOT minimal compiler-demanded
promotion.

Audited modules:
- types.rs: <count> pub(crate) -> pub promotions, each with doc comment
  citing the bin module + reason.
- protocol.rs: <count> changes (wire types stay pub).
- daemon_state.rs SharedState: per-field audit. <count> demotions back
  to pub(crate) for fields no bin accesses cross-crate.

Discipline applied (Information Hiding, Parnas 1972):
- Default to most-restrictive that compiles.
- Each pub promotion gets a doc comment citing why a bin needs it.
- No blanket sed pub(crate) -> pub.

Verification:
- cargo check --workspace clean
- cargo clippy --workspace --all-targets: warnings <= baseline

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Commit 5: Test relocation + dashboard move

### Task 8: Move engine-specific integration tests

**Files:**
- Move: `tests/level3_golden_action_accumulator.rs` → `crates/apollo-engine/tests/level3_golden_action_accumulator.rs`
- Move: `tests/level3_telemetry_sync.rs` → `crates/apollo-engine/tests/level3_telemetry_sync.rs`

- [ ] **Step 1: Create apollo-engine tests directory**

```bash
mkdir -p crates/apollo-engine/tests
```

- [ ] **Step 2: Move the two engine-specific test files**

```bash
git mv tests/level3_golden_action_accumulator.rs crates/apollo-engine/tests/
git mv tests/level3_telemetry_sync.rs crates/apollo-engine/tests/
```

- [ ] **Step 3: Update test imports**

Open both moved files. Replace any `apollo_optimizer::engine::*` or `apollo_optimizer::*` imports with `apollo_engine::engine::*` (or `apollo_engine::*` if going through the engine root).

For each moved test file, run:
```bash
grep -n "apollo_optimizer\|apollo_engine" crates/apollo-engine/tests/level3_golden_action_accumulator.rs
grep -n "apollo_optimizer\|apollo_engine" crates/apollo-engine/tests/level3_telemetry_sync.rs
```

Sed if needed:
```bash
sed -i '' 's|apollo_optimizer::engine::|apollo_engine::engine::|g; s|apollo_optimizer::|apollo_engine::|g' \
  crates/apollo-engine/tests/level3_golden_action_accumulator.rs \
  crates/apollo-engine/tests/level3_telemetry_sync.rs
```

- [ ] **Step 4: Verify engine tests run**

Run: `cargo test -p apollo-engine --release 2>&1 | tail -10`

Expected: passes including the relocated tests. Test count for apollo-engine should include the 2 moved integration tests.

### Task 9: Move dashboard.rs to ctl binary

**Files:**
- Move: `src/dashboard.rs` → `src/bin/apollo-optimizerctl/dashboard.rs`
- Modify: `src/bin/apollo-optimizerctl.rs` (current ctl bin file) → `src/bin/apollo-optimizerctl/main.rs`

- [ ] **Step 1: Check current ctl binary structure**

Run: `ls -la src/bin/ | grep optimizerctl`

If you see `apollo-optimizerctl.rs` (single file), proceed to Step 2.
If you see `apollo-optimizerctl/` (directory), skip to Step 4.

- [ ] **Step 2: Create ctl binary directory and move the file**

```bash
mkdir -p src/bin/apollo-optimizerctl
git mv src/bin/apollo-optimizerctl.rs src/bin/apollo-optimizerctl/main.rs
```

Cargo recognizes `src/bin/<name>/main.rs` as a binary named `<name>`.

- [ ] **Step 3: Verify ctl binary still compiles**

Run: `cargo build --release --bin apollo-optimizerctl 2>&1 | tail -5`

Expected: `Finished ... target(s)`. Path change is detected, binary still produces `target/release/apollo-optimizerctl`.

- [ ] **Step 4: Move dashboard.rs to live with ctl**

```bash
git mv src/dashboard.rs src/bin/apollo-optimizerctl/dashboard.rs
```

- [ ] **Step 5: Update dashboard.rs to be a sibling module of ctl main**

Open `src/bin/apollo-optimizerctl/dashboard.rs`. Replace any `crate::dashboard` references with self-references if they appear. The file should not need substantive changes — it was a sibling module already, just at a different path.

- [ ] **Step 6: Update ctl main.rs imports**

Open `src/bin/apollo-optimizerctl/main.rs`. Find:
```rust
mod dashboard;  // or
use crate::dashboard;
```

If `mod dashboard;` declaration is missing (because dashboard was at crate root before), add it at the top:
```rust
mod dashboard;
```

- [ ] **Step 7: Remove dashboard re-export from root lib.rs (if present)**

Open `src/lib.rs` (root). It should have only:
```rust
pub use apollo_engine::*;
```

If it has `pub mod dashboard;`, remove that line. Dashboard is no longer in root crate.

- [ ] **Step 8: Verify all bins compile**

Run: `cargo build --workspace --release 2>&1 | tail -5`

Expected: clean.

- [ ] **Step 9: Verify ctl dashboard command works**

Run: `cargo run --release --bin apollo-optimizerctl -- dashboard 2>&1 | head -20`

Expected: dashboard ASCII renders without panic. (Daemon must be running; if not, dashboard shows "daemon not reachable" message — that's still a successful run of the binary.)

### Task 10: Final validation + benchmark

- [ ] **Step 1: Run full workspace test suite**

```bash
time cargo test --workspace --release 2>&1 | tail -10
```

Expected: 1967 pass, 1 flake (`selfcounts_increases_with_work`), wall time <12 min.

Record actual wall time. Compare to baseline from Task 0.

- [ ] **Step 2: Run engine-only test suite (the win we're after)**

```bash
time cargo test -p apollo-engine --release 2>&1 | tail -10
```

Expected: engine tests pass, wall time ≤8 min.

Record actual wall time.

- [ ] **Step 3: Run clippy across workspace**

```bash
cargo clippy --workspace --all-targets --release 2>&1 | grep -c "^warning:"
```

Expected: ≤ baseline count (218).

- [ ] **Step 4: Daemon smoke test**

Build + deploy + verify daemon startup:

```bash
sudo cp target/release/apollo-optimizerd /usr/local/libexec/apollo-optimizerd
sudo cp target/release/apollo-optimizerctl /usr/local/bin/apollo-optimizerctl
sudo cp target/release/apollo-menubar /usr/local/bin/apollo-menubar
sudo launchctl bootout system/com.eduardocortez.systemoptimizerd
sleep 2
sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist
sleep 10
sudo launchctl print system/com.eduardocortez.systemoptimizerd | grep -E "state = running|^	pid"
```

Expected: `state = running`, fresh PID, `last exit code = (never exited)`.

- [ ] **Step 5: Verify daemon emits a cycle**

```bash
sleep 30
sudo cat /var/lib/apollo/runtime_metrics.json | python3 -c "import json,sys; m=json.load(sys.stdin); print(f'cycles={m.get(\"cycles\")} p95={m.get(\"p95_cycle_ms\")}ms failures={m.get(\"failures\")} last_err={m.get(\"last_error\")}')"
```

Expected: cycles >0, failures=0, last_err=None.

- [ ] **Step 5b: Verify IPC socket protocol survived the split**

The wire protocol crossing Unix sockets between ctl and daemon depends on `engine/protocol.rs` types serialization. After visibility changes in Commit 4, confirm the protocol still round-trips correctly:

```bash
sudo apollo-optimizerctl status 2>&1 | head -20
```

Expected: JSON output starts with `{` and includes fields like `"effective_profile"`, `"metrics"`, `"failures"`. If you get `Error: failed to deserialize response` or `protocol mismatch`, a serde derivation got demoted incorrectly in Task 7 — investigate `protocol.rs` `pub` items.

```bash
sudo apollo-optimizerctl dashboard 2>&1 | grep -A2 "VEREDICTO" | head -3
```

Expected: VEREDICTO line renders (e.g. `🟢 Sistema optimizado` or `🟡 Presión moderada`). Confirms dashboard module relocation (Task 9) wired correctly.

- [ ] **Step 6: Update baseline file with post-split metrics**

Append to `evolve/2026-05-09-workspace-split/baseline.txt`:

```
---
post_split_date: <date>
post_split_cargo_test_workspace_wall_time: <time from Step 1>
post_split_cargo_test_engine_wall_time: <time from Step 2>
post_split_clippy_warning_count: <count from Step 3>
post_split_daemon_pid: <PID from Step 4>
post_split_daemon_failures: <failures from Step 5>
delta_workspace_test_wall_time: <baseline - post_split>
delta_engine_test_wall_time: <baseline - engine_only_post_split>
```

- [ ] **Step 7: Commit final validation + cleanup**

```bash
git add -A
git commit -m "chore(workspace-split): finalize Sprint 5 Mes 0

Commit 5 of Workspace Split. Test relocation + dashboard placement +
final validation:
- crates/apollo-engine/tests/ now hosts level3_golden_action_accumulator
  and level3_telemetry_sync (engine-specific integration tests).
- src/bin/apollo-optimizerctl/ converted to directory binary, dashboard.rs
  moved alongside main.rs (sibling module).
- Root /tests/ retains level1_unit and level2_integration (cross-cutting).

Validation captured in evolve/2026-05-09-workspace-split/baseline.txt:
- cargo test --workspace: <new wall time> (was ~20 min)
- cargo test -p apollo-engine: <new wall time> (was ~20 min, target ≤8)
- clippy delta: 0
- daemon smoke test: PID <pid>, failures=0, p95=<ms>

Sprint 5 Mes 0 closed. Architecture now ready for Mes 1 (constraints.rs).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-review checklist (run before declaring plan done)

Per writing-plans skill protocol:

- ✅ **Spec coverage**: Every section of the spec has a task. Architecture (Tasks 1-2), facade lib (Task 5), engine_c FFI (Task 3), dashboard relocation (Task 9), tests relocation (Task 8), workspace.dependencies features-explicit (Task 1), boundary-leak rule (Task 6), abort threshold (Task 6 Step 3), visibility audit (Task 7), validation (Task 10).

- ✅ **Placeholder scan**: All steps have concrete commands and code blocks. The `<placeholders>` in commit messages are intentional (filled by engineer at the moment based on actual measured values).

- ✅ **Type consistency**: Path names consistent across tasks (`crates/apollo-engine/src/engine/`, `apollo_engine::engine::*`). Crate name consistent (`apollo-engine`).

- ✅ **TDD discipline**: While Workspace Split is a refactor (not feature dev), each task has compile/test gates that fail-then-pass.

- ✅ **Atomic commits**: 5 commits + 1 baseline + 1 finalize = 7 commits, each independently reviewable.

- ✅ **Rollback path**: Spec defines `git reset --hard v0.6.1` + restore backup binaries.
