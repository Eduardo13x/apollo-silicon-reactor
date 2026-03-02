# Agent Notes (apollo-optimizer)

This repo is a macOS-focused system optimizer written in Rust (edition 2021).
Binaries:
- `apollo-optimizer` (CLI): `src/main.rs`
- `apollo-optimizerd` (daemon, Unix socket API): `src/bin/apollo-optimizerd.rs`
- `apollo-optimizerctl` (daemon client): `src/bin/apollo-optimizerctl.rs`

Safety: code here can freeze/throttle processes, toggle Spotlight, and write to `/var/*`
when running as root. Keep behavior conservative and never introduce interactive prompts
in daemon paths (no hanging on password requests).

## Commands

Common build/run/lint/test:

```bash
# Build
cargo build
cargo build --release

# Run main CLI
cargo run -- --help
cargo run -- snapshot --output system_snapshot.json
cargo run -- optimize
cargo run -- daemon

# Run daemon + client from source
cargo run --bin apollo-optimizerd -- daemon --profile balanced-root
cargo run --bin apollo-optimizerctl -- status

# Format
cargo fmt --all
cargo fmt --all -- --check

# Lint
cargo clippy --all-targets
# Strict (currently fails repo-wide due to existing lints)
cargo clippy --all-targets -- -D warnings

# Tests
cargo test
cargo test --doc

# Single test (substring) / module-qualified patterns
cargo test my_test_name
cargo test engine::safety::tests::enforce_limits

# Target-specific
cargo test --lib
cargo test --bin apollo-optimizerd
cargo test --bin apollo-optimizerctl
```

Notes:
- `.cargo/config.toml` enables `-C target-cpu=native`; release uses LTO and `panic=abort`.
- Avoid running multiple `cargo` commands concurrently (shared `target/`).

### Root daemon (launchd)

Installs binaries into `/usr/local/*` and loads a LaunchDaemon:

```bash
./scripts/install-root-daemon.sh
```

Uninstalls and triggers a safety restore:

```bash
./scripts/uninstall-root-daemon.sh
```

Operational tip: prefer building as your user, then running the produced binaries as root
(via launchd or `sudo`) rather than compiling as root.

### Socket locations

- Root daemon socket: `/var/run/apollo-optimizer.sock`
- Non-root socket: `/tmp/apollo-optimizer.sock`

### Operational files (daemon)

When running as root, the daemon persists state under `/var/lib/apollo` and uses `/var/run`:
- Kill switch: `/var/run/apollo.disable` (when present, daemon pauses optimization)
- Journal: `/var/lib/apollo/journal.jsonl`
- Runtime metrics: `/var/lib/apollo/runtime_metrics.json`
- Governor state: `/var/lib/apollo/governor_state.json`
- Profile timeline: `/var/lib/apollo/profile_timeline.jsonl`
- Wake state: `/var/lib/apollo/wake_state.json`
- Frozen PID state: `/var/lib/apollo/frozen_state.json`

When non-root, equivalents are written under `/tmp`.

Config created by the install script (if missing): `/etc/apollo-optimizer/config.toml`.

## Rust Code Style

Formatting:
- Use `rustfmt` defaults; do not fight the formatter.
- Prefer small, testable helpers over deeply nested control flow.

Imports:
- Stable grouping order: `std` -> external crates -> local crate (`crate::...` / `apollo_optimizer::...`).
- Prefer grouped imports (`use std::{...}`) over many single-line imports.

Naming:
- Types/traits/enums: `PascalCase`; functions/vars/modules: `snake_case`; constants: `UPPER_SNAKE_CASE`.
- Serialized protocol strings: explicit `kebab-case` (see `#[serde(rename_all = "kebab-case")]`).

Types/ownership:
- Borrow (`&str`, `&[T]`) in hot paths; allocate at boundaries (I/O, protocol, logs).
- Keep mutex-guarded sections short; drop guards early; avoid holding locks across I/O.

Errors:
- Binaries should typically use `fn main() -> anyhow::Result<()>` and propagate with `?`.
- Add context at boundaries with `anyhow::Context` (files, sockets, subprocesses).
- In long-running loops (daemon), prefer best-effort handling: record/log error, keep system safe, continue.
- Avoid `unwrap()`/`expect()` in production; exceptions only for impossible invariants or explicit recovery.

Mutex poisoning convention used in this repo:
- Recover via `lock().unwrap_or_else(|e| e.into_inner())` (or a helper like `safe_lock()`).

External commands / privilege:
- Prefer `std::process::Command` (no shell).
- Never introduce interactive prompts in daemon paths; if you must use `sudo`, use `sudo -n`.
- Be conservative with global state changes: Spotlight (`mdutil`), sysctls, signals (`SIGSTOP`/`SIGCONT`).

Unsafe / FFI:
- Keep `unsafe` blocks small and localized; wrap platform-specific FFI behind safe helpers where possible.
- Document non-obvious invariants (pointer ownership, sizes, lifetimes).

Protocol & persistence:
- `src/engine/protocol.rs` uses JSON-tagged enums (`type`/`payload`); keep tags stable.
- Files under `/var/lib/apollo` or `/tmp` are operational state; consider backward compatibility if changing schema.

Daemon behavior constraints:
- Avoid adding blocking I/O on the hot path; keep per-cycle work bounded.
- Prefer defensive cleanup: if Apollo froze processes and crashes/restarts, it should unfreeze on startup.

Logging:
- Output may end up in launchd logs; avoid high-frequency spam in tight loops.
- Prefer structured JSON for machine-readable data when appropriate.

## Repo Notes

- No `rustfmt.toml`/`clippy.toml` currently; use defaults.
- No Cursor rules found (`.cursor/rules/`, `.cursorrules`).
- No Copilot instructions found (`.github/copilot-instructions.md`).
