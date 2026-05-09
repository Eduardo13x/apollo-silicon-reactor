# Workspace Split — Design Spec

**Date**: 2026-05-09
**Sprint**: 5 — Mes 0 (pre-requisite for architect-grade upgrade)
**Status**: Design approved by user, ready for implementation plan

## Goal

Reduce `cargo test` wall time from ~20 min (single-crate monolith of ~110K LoC) to ≤8 min for `cargo test -p apollo-engine` and ≤12 min for `cargo test --workspace`, **without changing runtime behavior**.

This is a prerequisite for Sprint 5 Mes 1-3 (constraints.rs, NARS user-feedback wire, counterfactual loop close). Without faster compile, those sprints become unworkable on M1 8GB.

## Non-goals (out of scope)

- ❌ NO new subsystems (constraints.rs, NARS feedback wire, counterfactual — those are Mes 1-3)
- ❌ NO API rename / module restructure / public API cleanup
- ❌ NO version bumps for any dependency
- ❌ NO additional crate splits beyond apollo-engine (no apollo-protocol, apollo-ffi, apollo-signals — only apollo-engine)
- ❌ NO test rewrites — only relocation
- ❌ NO performance tuning of the runtime
- ❌ Apollo NOT published as third-party library — `apollo-engine` stays `version = "1.0.0"` unpublished

## Architecture

```
system-optimizer/                 (workspace root)
├── Cargo.toml                    [workspace + package for bins, no root lib]
├── Cargo.lock                    (single, shared across workspace)
├── crates/
│   └── apollo-engine/
│       ├── Cargo.toml
│       ├── build.rs              (compiles engine native FFI)
│       ├── native/               (was src/engine_c/)
│       │   ├── ioreport_bridge.c
│       │   └── smc_bridge.c
│       └── src/
│           ├── lib.rs            (pub mod engine;)
│           └── engine/           (was src/engine/, ~148 .rs files)
│               ├── action_accumulator.rs
│               ├── identity_cache_manager.rs
│               ├── safety.rs
│               ├── ... (rest unchanged)
│       └── tests/
│           ├── level3_golden_action_accumulator.rs
│           └── level3_telemetry_sync.rs
├── src/
│   └── bin/
│       ├── apollo-optimizer/     (cli, optional move from src/main.rs)
│       │   └── main.rs
│       ├── apollo-optimizerd/    (daemon)
│       │   ├── main.rs
│       │   └── ... (49 files)
│       ├── apollo-optimizerctl/
│       │   ├── main.rs
│       │   └── dashboard.rs      (moved from src/dashboard.rs)
│       └── apollo-menubar/
│           └── main.rs
└── tests/
    ├── level1_unit.rs            (cross-cutting, stays)
    └── level2_integration.rs     (cross-cutting, stays)
```

### Crate boundaries

**`apollo-engine` (lib)** — core runtime:
- types, runtime metrics, collector
- decision logic (decide_actions, action_accumulator)
- safety, sysctl_limits, identity_cache_manager
- planner (hierarchical Phase 0)
- MPC, NARS, RL, hazard model, Kalman, holt_winters, swap_predictor
- outcome tracker, causal graph, learning loops
- shadow_evaluator, adversarial_probe
- chromium_manager, mach_qos, jetsam_control
- FFI to engine native bridges (ioreport, smc)

**root crate (binaries only)**:
- `apollo-optimizer` (cli) — one-off commands
- `apollo-optimizerd` (daemon) — long-running service, hot loop
- `apollo-optimizerctl` (client) — CLI client + dashboard rendering
- `apollo-menubar` (UI) — menubar status

**Import direction enforced**:
- `apollo-engine` does NOT import from root crate
- Root bins import `apollo_engine::engine::*`
- ctl binary imports its own sibling `dashboard` module (NOT in apollo-engine)

### Why dashboard stays out of apollo-engine

`src/dashboard.rs` (1.2k LoC) is presentation/UI, not engine logic. It reads `RuntimeMetrics` but consuming engine types does not make it part of the engine domain. Keeping it near `apollo-optimizerctl`:

- Engine crate stays reusable / testable as core library
- UI/CLI/ASCII-formatting concerns stay with their binary consumer
- Future-proof: if engine ever published, no UI baggage

## Migration plan (5 commits + boundary-leak rule)

### Commit 1: Workspace skeleton

Cargo.toml root converted to workspace:
```toml
[workspace]
members = ["crates/apollo-engine", "."]
resolver = "2"

[workspace.dependencies]
sysinfo = { version = "0.30", default-features = false, features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
clap = { version = "4.4", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1.0"
libc = "0.2"
ctrlc = { version = "3.4", features = ["termination"] }
toml = "0.8"
ureq = { version = "2.12", features = ["json"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
tempfile = "3"

[package]
name = "apollo-optimizer"
version = "1.0.0"
edition = "2021"
# NO [lib] section — root is bins only
```

`crates/apollo-engine/Cargo.toml` — empty stub:
```toml
[package]
name = "apollo-engine"
version = "1.0.0"
edition = "2021"

[lib]
path = "src/lib.rs"

[dependencies]
sysinfo.workspace = true
serde.workspace = true
serde_json.workspace = true
chrono.workspace = true
anyhow.workspace = true
libc.workspace = true
toml.workspace = true
ureq.workspace = true
tracing.workspace = true
```

`crates/apollo-engine/src/lib.rs`:
```rust
// Empty stub — module migration happens in Commit 2.
```

**State after Commit 1**: workspace compiles, bins still work via `crate::engine` (untouched in root), apollo-engine empty.

### Commit 2: Move engine module physically

```bash
git mv src/engine crates/apollo-engine/src/engine
git mv src/engine_c crates/apollo-engine/native
# build.rs migration:
git mv build.rs crates/apollo-engine/build.rs  # if exists at root
# Update apollo-engine/build.rs paths from `src/engine_c/` to `native/`
# Update apollo-engine/src/lib.rs:
echo "pub mod engine;" > crates/apollo-engine/src/lib.rs
```

Root `lib.rs` may temporarily keep `pub mod engine` re-export shim for ONE COMMIT to keep root compiling. Bins still use `crate::engine::*`.

**State after Commit 2**: engine code physically in apollo-engine. Shim allowed to keep root buildable.

### Commit 3: Mechanical import migration (NO shim)

Mass sed across all bins + tests:
```bash
find src/bin tests -name "*.rs" -exec sed -i '' \
  -e 's|crate::engine::|apollo_engine::engine::|g' \
  -e 's|apollo_optimizer::engine::|apollo_engine::engine::|g' {} +
```

Remove `pub mod engine;` shim from root crate.

`cargo check --workspace` MUST pass with NO shim. Compiler-driven cleanup of `pub(crate)` → `pub` only where binaries explicitly demand access.

**Boundary-leak rule** (CRITICAL):

If circular dependency surfaces (root → apollo-engine → root):
- DO NOT add shim
- DO NOT re-export to mask it
- Add `// BOUNDARY-LEAK: <symbol> referenced by <bin> from apollo-engine` comment
- Add TODO with file:line citation
- Document in this spec under "Discovered boundary leaks"
- Continue split — circular deps are valuable signal for future architectural cleanup

**State after Commit 3**: `apollo_engine::engine::types::RootAction` is the canonical path. NO shim. Boundary leaks if any are surfaced.

### Commit 4: Visibility cleanup minimum

Only changes the compiler demanded in Commit 3. Specifically:
- `pub(crate)` → `pub` for items that bins now access via `apollo_engine::*`
- NO renames
- NO module restructure
- NO API redesign

`cargo clippy --workspace --all-targets` MUST show no new warnings beyond baseline.

### Commit 5: Test relocation + final cleanup

```bash
git mv tests/level3_golden_action_accumulator.rs crates/apollo-engine/tests/
git mv tests/level3_telemetry_sync.rs crates/apollo-engine/tests/
# Update test imports

git mv src/dashboard.rs src/bin/apollo-optimizerctl/dashboard.rs
# If apollo-optimizerctl is currently single-file (src/bin/apollo-optimizerctl.rs):
git mv src/bin/apollo-optimizerctl.rs src/bin/apollo-optimizerctl/main.rs
# (Cargo recognizes src/bin/<name>/main.rs)

# Optional homogeneity: src/main.rs → src/bin/apollo-optimizer/main.rs
# (defer if it adds churn)
```

Root `/tests/` keeps `level1_unit.rs` and `level2_integration.rs` (cross-cutting integration).

**Migration rule** (anti-desorden): if a test cannot be cleanly classified as engine-internal vs cross-cutting, leave in root `/tests/`. Don't block the split on taxonomy perfection.

## Validation gates

| Commit | Gate |
|---|---|
| 1 | `cargo build --workspace` clean |
| 2 | `cargo build --workspace` clean (shim allowed) |
| 3 | `cargo build --workspace` clean (NO shim) + `cargo test -p apollo-engine` passes |
| 4 | `cargo clippy --workspace --all-targets` no new warnings vs baseline |
| 5 | `cargo test --workspace` full suite + benchmark `cargo test -p apollo-engine` wall time vs 20 min baseline |

## Success metrics (Definition of Done)

- ✅ `cargo test -p apollo-engine --release`: target ≤8 min (vs ~20 min baseline)
- ✅ `cargo test --workspace --release`: target ≤12 min
- ✅ All existing tests pass (1967 baseline)
- ✅ Daemon smoke test: deploy + 100 cycles healthy + 0 failures
- ✅ Zero shims remaining post-Commit 3
- ✅ Cargo.lock has only additive entries (apollo-engine), no version drift
- ✅ ctl `dashboard` command renders identically pre/post split
- ✅ Codesigning works for daemon binary post-split (launchctl bootstrap succeeds)

## Risks ranked

| Risk | Severity | Mitigation |
|---|---|---|
| Circular dep root↔engine surfaced in Commit 3 | 🟠 High | Per boundary-leak rule: surface, don't paper. May add 1-2 cleanup commits before merge. |
| Cargo.lock drift / dep version mismatch post-split | 🟡 Medium | `[workspace.dependencies]` forces single version. Diff Cargo.lock pre/post — should only add apollo-engine entries. |
| Build script (engine native FFI) breaks under new path | 🟡 Medium | build.rs `OUT_DIR` semantics differ in workspace member. Test compile in Commit 2 BEFORE moving bins. |
| Test discovery breaks (integration tests dir per crate) | 🟢 Low | `cargo test --workspace` walks all members. Verified in Commit 5 gate. |
| Daemon binary codesigning under new build path | 🟢 Low | `target/release/apollo-optimizerd` location unchanged for workspace members. `sudo cp` + launchctl unchanged. |
| Deploy break: ctl binary moved location | 🟢 Low | Build path stays `target/release/apollo-optimizerctl`. Deploy script unchanged. |

## Discovered boundary leaks

(Empty until implementation surfaces any.)

## Rollback plan

If split breaks at any commit:

```bash
# Hard rollback to v0.6.1 stable tag
git reset --hard v0.6.1
sudo cp ~/backups/apollo-v0.6.1/apollo-optimizerd /usr/local/libexec/
sudo cp ~/backups/apollo-v0.6.1/apollo-optimizerctl /usr/local/bin/
sudo cp ~/backups/apollo-v0.6.1/apollo-menubar /usr/local/bin/
sudo launchctl bootout system/com.eduardocortez.systemoptimizerd
sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist
```

Total rollback time: <2 min. Backup binaries already exist at v0.6.1.

## Success measurement protocol

```bash
# Baseline (pre-split, current master, captured pre-Commit 1)
time cargo test --release 2>&1 | tail -5    # baseline ~20 min

# Post-split (after Commit 5 merged)
time cargo test -p apollo-engine --release   # target: ≤8 min
time cargo test --workspace --release        # target: ≤12 min
```

If post-split wall time ≥ baseline, rollback. If ≤ baseline, success.

## Decisions log (from brainstorming Q1-Q6)

| Q | Decision | Rationale |
|---|---|---|
| Q1: Aggressiveness | A — minimal split (apollo-engine only) | Resolves real pain (cargo test 20min) without coordination overhead. Maximal splits invite circular deps. |
| Q2: Imports | A puro — mechanical path migration | No shim. Boundary leaks surfaced, not papered. Compiler is the test. |
| Q3: engine_c (FFI) | A — inside apollo-engine as `native/` | Wrappers (iokit_sensors.rs, smc_direct.rs) live in engine; separating bridges creates artificial split. |
| Q4: dashboard.rs | B+ — near ctl binary, NOT in engine | Dashboard is UI/presentation. Engine stays reusable. ctl bin owns its render layer. |
| Q5: Tests location | A — engine tests in `crates/apollo-engine/tests/`, cross-cutting in root | Tests live where the responsibility they validate lives. Anti-desorden fallback: ambiguous tests stay in root. |
| Q6: Workspace deps | A — `[workspace.dependencies]` centralized | Single version source. Drift-free. No version bumps in this PR. |

## References

- NotebookLM peer-review session 2026-05-08 (notebook `8344b94c-a014-4803-abea-076a55753cfd`): GO-with-limited-scope verdict, workspace split as critical prerequisite
- CLAUDE.md supervision-mode rules (top section)
- Sprint 4 backlog: `.plan/SPRINT4_BACKLOG.md`
- Tag stable: `v0.6.1`
