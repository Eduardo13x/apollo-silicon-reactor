# Workspace Split Plan — `crates/apollo-engine`

**Goal:** Reduce `cargo test` from ~20 min to ~5 min for engine-only test runs by splitting
the monolith into a Cargo workspace. The engine crate (~48k LOC, 104 modules) is the hot path
for both compilation and testing.

---

## Current State (baseline)

| Location | Files | LOC |
|----------|-------|-----|
| `src/engine/` | 104 | ~48,458 |
| `src/*.rs` (lib root + collector etc.) | 4 | ~1,381 |
| `src/bin/` (daemon, ctl, main) | ~12 | ~8,660 |
| **Total** | **~120** | **~58,499** |

Single crate → full recompile on any change → 20 min test cycle.

---

## Target Structure

```
system-optimizer/            ← workspace root
├── Cargo.toml               ← workspace manifest (NEW — replaces package manifest)
├── .cargo/config.toml       ← unchanged (applies workspace-wide)
├── crates/
│   └── apollo-engine/       ← NEW crate (pure logic, no bins, no macOS-only deps)
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           └── (move all of src/engine/ here verbatim)
└── apollo-optimizer/        ← renamed from root, or keep root as the "shell" crate
    ├── Cargo.toml           ← depends on apollo-engine
    └── src/
        ├── lib.rs           ← (thin, re-exports from apollo-engine as needed)
        ├── collector.rs
        ├── sysctl_tuner.rs
        ├── main.rs
        └── bin/
            ├── apollo-optimizerd/
            └── apollo-optimizerctl.rs
```

### Option B (less disruptive): keep root as workspace member

Keep `apollo-optimizer` at the repo root as a workspace member — avoids moving `src/bin/`
and `src/main.rs`. Only the engine is extracted:

```
Cargo.toml              ← workspace [workspace] + keep [package] apollo-optimizer here
crates/apollo-engine/
    Cargo.toml
    src/lib.rs          ← pub mod declarations mirroring src/engine/mod.rs
    src/…               ← moved from src/engine/
```

**Recommendation:** Option B. Smallest diff, least breakage.

---

## Step-by-Step Migration

### Step 1 — Convert root `Cargo.toml` to workspace

Replace the root `Cargo.toml` `[package]` section with a `[workspace]` header,
then keep `[package]` for the shell crate:

```toml
[workspace]
members = [
    ".",                  # apollo-optimizer (shell crate)
    "crates/apollo-engine",
]
resolver = "2"

[package]
name = "apollo-optimizer"
version = "1.0.0"
edition = "2021"
# ... rest of current [package] unchanged
```

### Step 2 — Create `crates/apollo-engine/Cargo.toml`

```toml
[package]
name = "apollo-engine"
version = "1.0.0"
edition = "2021"

[dependencies]
sysinfo  = { version = "0.30", default-features = false, features = ["serde"] }
serde    = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono   = { version = "0.4", features = ["serde"] }
anyhow   = "1.0"
libc     = "0.2"
tracing  = "0.1"
toml     = "0.8"
ureq     = { version = "2.12", features = ["json"] }
# NOTE: no clap, no ctrlc, no tray-icon, no winit, no cocoa, no objc
# Those belong to the shell crate only.

[dev-dependencies]
tempfile = "3"

[build-dependencies]
cc = "1.0"
```

### Step 3 — Create `crates/apollo-engine/src/lib.rs`

Copy the contents of the current `src/engine/mod.rs` verbatim:

```rust
// crates/apollo-engine/src/lib.rs
pub mod capabilities;
pub mod decide_actions;
// ... all 104 pub mod declarations
```

### Step 4 — Move source files

```bash
mkdir -p crates/apollo-engine/src
cp -r src/engine/* crates/apollo-engine/src/
# Do NOT delete src/engine/ yet — keep until imports are fixed.
```

### Step 5 — Update imports in shell crate

In `src/lib.rs` (and `src/bin/**`), change:

```rust
// OLD
use crate::engine::X;
use apollo_optimizer::engine::X;

// NEW
use apollo_engine::X;
```

Scope of changes:
- `src/lib.rs` — re-export or forward-use `apollo_engine`
- `src/bin/apollo-optimizerd/main.rs` (~5,454 LOC) — bulk find/replace
- `src/bin/apollo-optimizerctl.rs`
- `src/main.rs`
- `src/collector.rs`, `src/sysctl_tuner.rs`

Estimated occurrences: ~400–600 `use crate::engine::` / `use apollo_optimizer::engine::` patterns.
Use `cargo fix` or a sed pass:

```bash
# Dry-run estimate
grep -r 'crate::engine\|apollo_optimizer::engine' src/ | wc -l
```

### Step 6 — Add engine as dependency in shell crate

```toml
# root Cargo.toml [dependencies]
apollo-engine = { path = "crates/apollo-engine" }
```

### Step 7 — Remove `src/engine/` after validation

```bash
cargo check --workspace   # must pass first
rm -rf src/engine/
```

---

## Import Pattern Reference

| Old pattern | New pattern |
|-------------|-------------|
| `use crate::engine::protocol::DaemonResponse;` | `use apollo_engine::protocol::DaemonResponse;` |
| `use crate::engine::safety::SafetyEngine;` | `use apollo_engine::safety::SafetyEngine;` |
| `apollo_optimizer::engine::types::Profile` | `apollo_engine::types::Profile` |
| `mod engine;` in `src/lib.rs` | remove; add `use apollo_engine;` |

Internal cross-module refs inside `src/engine/` (e.g. `use crate::engine::types::X` from
within `src/engine/signal_intelligence.rs`) become `use crate::types::X` — these are already
correct once `engine/` is the crate root. **No change needed inside the engine crate itself.**

---

## Estimated Impact

### Compilation savings

| Scenario | Before | After (estimate) |
|----------|--------|-----------------|
| `cargo test` (full, cold) | ~20 min | ~20 min (same — cold build compiles everything) |
| `cargo test -p apollo-engine` (warm, engine change) | ~20 min | **~7–9 min** |
| `cargo test -p apollo-engine learning_pipeline` (warm) | ~20 min | **~3–5 min** |
| `cargo check -p apollo-engine` (incremental) | ~4 min | **~45–90 sec** |
| Shell crate test after engine-only change | ~20 min | **~1–2 min** (re-links only) |

Basis: `apollo-engine` is ~48k/58k = 82% of total LOC. However, the macOS-only UI deps
(tray-icon, winit, cocoa, objc) add significant compile time to the shell crate that would
now be decoupled. Engine crate has no UI deps → faster dep graph.

### Specific command from the task

```bash
cargo test --package apollo-engine learning_pipeline
```

Expected time: **3–5 minutes** (warm), **8–11 minutes** (cold).
Current equivalent: not possible (no package boundary) — full 20 min always.

---

## Risk Assessment

### What can go wrong

| Risk | Likelihood | Severity | Mitigation |
|------|-----------|----------|------------|
| `build.rs` / `cc` crate generates FFI bindings that reference both halves | Medium | High | Move `build.rs` to engine crate; expose generated symbols via engine's pub API |
| `cfg(target_os = "macos")` blocks split across crate boundary | Low | Medium | Keep macOS-only UI deps in shell; pure-logic macOS deps (libc, mach) stay in engine |
| Circular dependency: shell needs engine, engine needs collector | Low | High | `collector.rs` and `sysctl_tuner.rs` stay in shell; engine only exposes traits |
| `#[cfg(test)]` integration tests that import from both halves | Medium | Low | Create a `tests/` directory at workspace root for cross-crate integration tests |
| Incremental cache invalidation on first post-split build | Certain | Low | Expected: one full rebuild, then fast incremental |
| `target-cpu=native` still applies to engine crate via `.cargo/config.toml` | Certain | Low | Acceptable; `[profile.test]` mitigates with `codegen-units=256` |

### What to test after migration

1. `cargo check --workspace` — must be clean (0 errors, 0 warnings from new imports)
2. `cargo test --workspace` — full suite must stay at 2179 passed / 10 ignored
3. `cargo test -p apollo-engine` — engine tests pass in isolation
4. `cargo build --release` — release binary is identical (check `sha256sum`)
5. Run `./scripts/install-root-daemon.sh` in a VM — launchd integration intact
6. `cargo run --bin apollo-optimizerctl -- status` against running daemon — socket protocol intact

### Recommended test order

```bash
cargo check --workspace 2>&1 | grep -c '^error'   # must be 0
cargo test --workspace 2>&1 | tail -10
cargo test -p apollo-engine 2>&1 | tail -5
cargo build --release && sha256sum target/release/apollo-optimizer{,d,ctl}
```

---

## Effort Estimate

| Phase | Time |
|-------|------|
| Step 1-3: manifests + lib.rs | 30 min |
| Step 4: file move | 5 min |
| Step 5: import rewrites (sed + manual) | 2–4 hours |
| Step 6-7: validation + cleanup | 1 hour |
| **Total** | **4–6 hours** |

The import rewrite (Step 5) is the long pole. A sed one-liner covers 90%:

```bash
find src -name '*.rs' -exec sed -i '' \
    's/use crate::engine::/use apollo_engine::/g;
     s/crate::engine::/apollo_engine::/g' {} \;
```

The remaining 10% are pattern variants (`super::engine`, inline paths in macros) that need
manual review.

---

## Decision Gate

Only proceed with the workspace split if:
- `cargo test` warm-cycle time is confirmed > 10 min after applying `[profile.test]` quick wins
- The team has 4+ hours for a focused migration session
- There is a clean git state (tag before starting: `git tag pre-workspace-split`)
