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
├── Cargo.toml                    [workspace + package for bins + facade lib]
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
│   ├── lib.rs                    (facade ONLY: pub use apollo_engine::*)
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

### Why root keeps a facade `lib.rs`

NotebookLM peer-review (2026-05-09) flagged a CRITICAL issue: Cargo integration tests in root `tests/` cannot import types from a binary-only crate. Without `[lib]` in root, `level1_unit.rs` and `level2_integration.rs` would fail to compile because they currently do `use apollo_optimizer::engine::...`.

Solution: keep a **facade lib only** in root, no logic:

```rust
// src/lib.rs (root, post-split)
//! apollo-optimizer facade — re-exports apollo-engine for binary
//! consumers and integration tests. NO original logic lives here.
//! Engine internals are owned by `apollo-engine`.
pub use apollo_engine::*;
```

Discipline:
- Facade NEVER hosts new modules. If you're tempted to add code here, it goes in `apollo-engine` or a bin.
- Old `apollo_optimizer::engine::...` paths in tests resolve via this re-export. Fewer tests need rewriting.
- Bin crates use `apollo_engine::*` directly (cleaner, no facade hop) — facade is for tests + backward compat only.

This keeps the workspace honest (no monolith disguised) while letting the existing test suite compile without a `tests/` rewrite Sprint of its own.

## Migration plan (5 commits + boundary-leak rule)

### Commit 1: Workspace skeleton

Cargo.toml root converted to workspace:
```toml
[workspace]
members = ["crates/apollo-engine", "."]
resolver = "2"

[workspace.dependencies]
# Pin VERSIONS only here. Crates declare their own features explicitly to
# prevent feature unification leak (e.g. UI crate enabling Rayon-via-sysinfo
# unintentionally activating it in apollo-engine hot path).
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
path = "src/lib.rs"   # facade ONLY (re-exports apollo-engine)
```

Per-crate `[dependencies]` declares features explicitly:
```toml
# crates/apollo-engine/Cargo.toml
[dependencies]
sysinfo = { workspace = true, features = ["serde"] }
serde = { workspace = true, features = ["derive"] }
chrono = { workspace = true, features = ["serde"] }
ctrlc = { workspace = true, features = ["termination"] }
ureq = { workspace = true, features = ["json"] }
tracing-subscriber = { workspace = true, features = ["json", "env-filter"] }
# Other deps inherit version from workspace, no extra features
```

This pattern prevents feature unification across workspace members. The CLAUDE.md hot-path invariant ("Rayon disabled — coordination overhead exceeds benefit on M1, measured: 110/4049 samples") stays enforced.

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

### Commit 3: Mechanical import migration + compiler-driven cleanup (NO shim)

NotebookLM peer-review flagged sed-alone is insufficient because:
1. `super::` paths inside engine modules break when files move to a new crate
2. `pub(crate)` declarations re-scope to apollo-engine; bins lose access
3. Some imports are nested in macros / `cfg!` blocks that sed misses

Approach: **sed primary + compiler-driven repair**.

Step 3a — bulk sed:
```bash
find src/bin tests -name "*.rs" -exec sed -i '' \
  -e 's|crate::engine::|apollo_engine::engine::|g' \
  -e 's|apollo_optimizer::engine::|apollo_engine::engine::|g' {} +
```

Step 3b — replace the root `pub mod engine` shim added in Commit 2 with the facade re-export `pub use apollo_engine::*;`. Tests using the legacy `apollo_optimizer::engine::...` path resolve through the facade.

Step 3c — compiler-driven repair loop:
```
while ! cargo check --workspace; do
  # fix one error, commit progress as needed
done
```
Common fixes:
- `super::foo` paths inside engine modules: rewrite as `crate::engine::foo` (relative to apollo-engine crate root, not the old root crate)
- `pub(crate)` items used by bins: promote to `pub` ONLY with explicit comment justifying why bin needs cross-crate access (no blanket promotion)
- Macro-internal paths: handle case-by-case, never silently `use ... as ...`

**Boundary-leak rule** (CRITICAL):

If circular dependency surfaces (root → apollo-engine → root):
- DO NOT add shim
- DO NOT re-export to mask it
- Add `// BOUNDARY-LEAK: <symbol> referenced by <bin> from apollo-engine` comment
- Add TODO with file:line citation
- Document in this spec under "Discovered boundary leaks"
- Continue split — circular deps are valuable signal for future architectural cleanup

**Abort threshold**: if Commit 3 surfaces >30 distinct boundary leaks (not just simple `pub` promotions, real circular deps), STOP. The graph report shows god-nodes `decide()` (55 edges), `snap()` (45 edges), `find_decision()` (44 edges) suggesting `SharedState` may be too coupled to extract cleanly in this Sprint. Reassess scope before Commit 4.

**State after Commit 3**: `apollo_engine::engine::types::RootAction` is the canonical path. NO `pub mod engine` shim. Facade re-export only. Boundary leaks (if any) surfaced and counted.

### Commit 4: Explicit visibility audit (proactive, not minimal)

NotebookLM peer-review reframed this commit: NOT "minimal where the compiler demanded" but proactive audit of the visibility surface that was implicitly exposed by the split.

Audit targets:
- `types.rs`: every `pub(crate)` item — does an external bin actually need it? Promote to `pub` only with justification comment.
- `protocol.rs`: same audit. Wire protocol must be intentionally public.
- `daemon_state::SharedState`: god-node per graph report. Audit each field's pub level. Document any `pub` promotion as ARCH-DECISION with rationale.
- Other widely-used types (`RootAction`, `RuntimeMetrics`, `MetricsState`): confirm pub semantics match actual cross-crate usage.

Discipline (Information Hiding [Parnas 1972]):
- Default to most-restrictive that compiles: `pub(crate)` > `pub(super)` > `pub`
- Each `pub` promotion in this commit gets a doc comment explaining WHY a bin needs it
- NO blanket sed `pub(crate)` → `pub` — every change is intentional

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
| Circular dep root↔engine surfaced in Commit 3 | 🟠 High | Per boundary-leak rule: surface, don't paper. >30 leaks = abort threshold. Graph report's god-nodes (decide=55, snap=45 edges) make this realistic. |
| Tests in root `tests/` fail without root `[lib]` | 🔴 Critical → Mitigated | Facade lib `pub use apollo_engine::*` keeps `apollo_optimizer::engine::...` paths resolvable. Caught in NotebookLM review. |
| Feature unification leak (e.g. Rayon enabled via UI crate) | 🟡 Medium | Workspace deps pin VERSIONS only; per-crate `[dependencies]` declares features explicitly. CLAUDE.md hot-path invariant ("Rayon disabled") preserved. |
| `super::` paths inside engine modules break when files relocate | 🟠 High | Commit 3 reformulated as sed primary + compiler-driven repair loop. NOT sed-only. |
| `pub(crate)` mass promotion violates Information Hiding | 🟠 High | Commit 4 reframed as proactive audit, not blanket promotion. Each `pub` change gets doc comment justifying it. |
| Cargo.lock drift / dep version mismatch post-split | 🟡 Medium | `[workspace.dependencies]` single version source. Diff Cargo.lock pre/post — should only add apollo-engine entries. |
| Build script (engine native FFI) breaks under new path | 🟡 Medium | build.rs uses `CARGO_MANIFEST_DIR` for `cc` source paths. Test compile in Commit 2 BEFORE moving bins. |
| Wall-time targets ≤8 / ≤12 min optimistic for M1 8GB | 🟡 Medium | Keep `.cargo/config.toml` `codegen-units=256` + `debug=false` in profile.test (already set). Linker swap is the bottleneck on M1 RAM. |
| Test discovery breaks (integration tests per crate) | 🟢 Low | `cargo test --workspace` walks all members. Verified in Commit 5 gate. |
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
- NotebookLM adversarial spec review 2026-05-09: 3 critical fixes integrated (facade lib, sed-vs-compiler-driven, proactive visibility audit) + feature unification + abort threshold
- CLAUDE.md supervision-mode rules (top section)
- Sprint 4 backlog: `.plan/SPRINT4_BACKLOG.md`
- Tag stable: `v0.6.1`
- Parnas D.L. 1972 — "On the Criteria To Be Used in Decomposing Systems into Modules" (Information Hiding principle, cited by NotebookLM for visibility audit rationale)
