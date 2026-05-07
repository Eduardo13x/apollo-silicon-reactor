# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical lesson — NotebookLM is NOT a final gatekeeper (2026-05-07)

NotebookLM peer-review (notebook `8344b94c-a014-4803-abea-076a55753cfd`) is a research librarian, not a senior engineer. It paraphrases and elaborates with confidence; it does NOT catch logic gaps in your diff, calibrate severity honestly, or push back when you're wrong.

**Observed failure (Sprint 3)**: Sprint declared "closed" with NotebookLM verdict citing Hellerstein 130ms and 3 severity-ranked gaps. Empirical re-verification 7h later found **5 real bugs** that NotebookLM missed despite having full source access:
1. Cache only cached 2/6 RootAction variants (start_sec=0 path skipped)
2. ABA window introduced by my own lookup_by_pid fix without invalidate_pid wire
3. SetSysctl no-op writes (clamp emitted, write succeeded but `before == after`)
4. network-optimizer raw emit at main.rs:3577 bypassing Phase C clamp
5. Dashboard verdict using sticky `survival_mode_activations > 0` as live state flag

**Mandatory adversarial pass before declaring any sprint "closed":**
1. Mechanical re-verification in production — read live `runtime_metrics.json` + `journal.jsonl` post-deploy with fresh-only timestamp filter. Classify entries by actual outcome. Compare against pre-deploy baseline.
2. Re-read your own diff with adversarial lens — for each modified file ask "what variants does this miss?", "what lifecycle event invalidates this state?", "what call site bypasses this guard?".
3. Sample size sanity — anything <500 events post-deploy is preliminary, not closed. NotebookLM treats N=8 as conclusive; do not.

If any of those 3 passes find an issue, sprint is **not** closed regardless of NotebookLM verdict.

**Inflation pattern**: NotebookLM defaults to severity-ranked tables with ≥1 🔴 Critical entry per debrief. Discount one severity level mentally — if everything is Critical, nothing is.

**What NotebookLM is good for**: surfacing forgotten cross-session context, structured debrief templates, paper citation lookup, ideation for Phase 1 brainstorming. Use as research aid, never as validation gate.

See `~/.claude/skills/apollo-evolve/SKILL.md` for full discipline.

## Project Overview

**apollo-optimizer** is a macOS system optimization tool written in Rust (edition 2021). It intelligently manages system resources by freezing/throttling processes, tuning sysctls, and optimizing for specific workloads (e.g., LLM/AI workloads).

### Three Binaries

1. **apollo-optimizer** (CLI) - `src/main.rs`
   - Entry point for one-off commands: `snapshot`, `optimize`, `clean`, `turbo`, `daemon`, `startup`, `llm`, `restore`
   - Can also run the daemon directly

2. **apollo-optimizerd** (Daemon) - `src/bin/apollo-optimizerd.rs`
   - Long-running process providing continuous optimization
   - Listens on Unix socket (`/var/run/apollo-optimizer.sock` as root, `/tmp/apollo-optimizer.sock` otherwise)
   - Maintains state in `/var/lib/apollo/` (root) or `/tmp/` (non-root): journal, metrics, governor state, etc.

3. **apollo-optimizerctl** (Client) - `src/bin/apollo-optimizerctl.rs`
   - CLI client to communicate with the daemon
   - Commands like `status`, `profile set`, etc.

## Build & Development Commands

```bash
# Build
cargo build
cargo build --release

# Run CLI
cargo run -- --help
cargo run -- snapshot --output system_snapshot.json
cargo run -- optimize
cargo run -- daemon

# Run daemon + client from source
cargo run --bin apollo-optimizerd -- daemon --profile balanced-root
cargo run --bin apollo-optimizerctl -- status

# Code style
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --all-targets

# Tests
cargo test                                    # All tests
cargo test --doc                             # Doctests only
cargo test my_test_name                      # Filter by substring
cargo test engine::safety::tests::enforce_limits  # Full module path
cargo test --lib                             # Library only
cargo test --bin apollo-optimizerd           # Specific binary
```

**Notes:**
- `.cargo/config.toml` enables `-C target-cpu=native` and LTO for release builds
- Avoid running multiple `cargo` commands concurrently (shared `target/` directory)
- Release builds use `panic=abort`

## Installation & Deployment

The project includes scripts for launchd integration:

```bash
# Install as root daemon (compiles, installs binaries, loads launchd service)
./scripts/install-root-daemon.sh

# Uninstall and restore system state
./scripts/uninstall-root-daemon.sh
```

**Best practice:** Build as your user, then run the produced binaries as root (via launchd or `sudo`) rather than compiling as root.

## Architecture

### Core Modules (in `src/engine/`)

- **`protocol.rs`** – Wire protocol for daemon ↔ client communication (JSON with `type`/`payload` tags; keep tags stable)
- **`types.rs`** – Core data structures (profiles, actions, capabilities)
- **`safety.rs`** – Validates and enforces optimization constraints (CPU/memory limits, frozen process tracking)
- **`capabilities.rs`** – Detects system capabilities (CPU core count, memory, OS version)
- **`decide_actions.rs`** – Decision logic for what optimizations to apply (can integrate LLM)
- **`execute_actions.rs`** – Applies optimizations (SIGSTOP, sysctl tuning, etc.)
- **`profile_governor.rs`** – Manages optimization profiles (balance, performance, efficiency)
- **`journal.rs`** – Audit trail of actions and state changes (written to `/var/lib/apollo/journal.jsonl`)
- **`usage_model.rs`** – Models system usage patterns
- **`llm.rs`** – Optional LLM integration for adaptive optimization

### Library Modules (in `src/`)

- **`collector.rs`** – Gathers system metrics (CPU, memory, process info via `sysinfo` crate)
- **`optimizer.rs`** – Main optimization engine orchestration
- **`reactor.rs`** – Event loop for daemon mode
- **`sysctl_tuner.rs`** – System parameter tuning

### Data & State Files

When running as root, the daemon persists state under `/var/lib/apollo/`:
- `journal.jsonl` – Audit log of optimizations
- `runtime_metrics.json` – Current system metrics
- `governor_state.json` – Profile configuration state
- `profile_timeline.jsonl` – History of profile switches
- `wake_state.json` – Wake/sleep event tracking
- `frozen_state.json` – List of currently frozen processes

Non-root instances use `/tmp` equivalents. Kill switch: `/var/run/apollo.disable` (presence pauses optimization).

## Code Style & Patterns

### Imports & Naming
- **Import grouping:** `std` → external crates → local crate (`crate::...` / `apollo_optimizer::...`)
- **Naming:** Types/traits/enums → `PascalCase`; functions/vars/modules → `snake_case`; constants → `UPPER_SNAKE_CASE`
- **Serialized strings:** explicit `kebab-case` (use `#[serde(rename_all = "kebab-case")]`)

### Error Handling
- Binaries: use `fn main() -> anyhow::Result<()>` with `?` propagation
- Add context at boundaries: `anyhow::Context` (files, sockets, commands)
- In long-running loops (daemon): prefer best-effort handling—record/log error, keep system safe, continue
- Avoid `unwrap()`/`expect()` except for impossible invariants
- Mutex poisoning: recover via `lock().unwrap_or_else(|e| e.into_inner())`

### Types & Ownership
- Use borrowing (`&str`, `&[T]`) in hot paths; allocate at boundaries (I/O, protocol, logs)
- Keep mutex-guarded sections short; drop guards early; avoid holding locks across I/O

### External Commands & Privilege
- Use `std::process::Command` (no shell)
- Never introduce interactive prompts in daemon code (if `sudo` is needed, use `sudo -n`)
- Be conservative with global state changes (Spotlight, sysctls, process signals)

### Unsafe & FFI
- Keep `unsafe` blocks small and localized; wrap behind safe helpers where possible
- Document non-obvious invariants (pointer ownership, sizes, lifetimes)

### Daemon Specifics
- Avoid blocking I/O on the hot path; keep per-cycle work bounded
- Prefer defensive cleanup: if Apollo froze processes and crashes/restarts, it should unfreeze on startup
- Avoid high-frequency logging in tight loops (output may end up in launchd logs)
- Use structured JSON for machine-readable data where appropriate

## Key Safety Constraints

This code can:
- Freeze/throttle processes (via SIGSTOP/SIGCONT)
- Toggle Spotlight (mdutil)
- Write to `/var/*` when running as root
- Tune system parameters (sysctls)

**Behavior must remain conservative.** Always validate actions against the safety module before execution. Ensure optimization constraints are enforced (e.g., don't freeze system-critical processes, respect memory/CPU limits).

## Dependencies

Key crates:
- `sysinfo` – System metrics collection
- `serde`/`serde_json` – Serialization
- `clap` – CLI argument parsing
- `chrono` – Time handling
- `anyhow` – Error handling
- `libc` – Low-level system calls
- `ctrlc` – SIGINT handling
- `toml` – Config file parsing
- `ureq` – HTTP requests (for LLM integration)

## Current Development: v0.6.0 "Self-Evolving"

**Base:** v0.5.0 (tag `v0.5.0`, commit `a3f2216`). Backup binarios en `~/backups/apollo-v0.5.0/`.

### Key Problem
Three independent learning loops (RL, OutcomeTracker, PredictiveAgent) never cross-feed. `mach_qos.rs` is purely reactive — ignores Markov predictions.

### Plan (Nivel 1 — ~70 lines, 0 new modules, 0 new deps)
1. **Router adaptativo** in `signal_intelligence.rs` — skip heavy subsystems when pressure < 0.40
2. **EMA Q-learning** in `rl_threshold.rs` — decaying alpha replaces fixed 0.10
3. **Cable A**: OutcomeTracker → RL reward signal (in daemon main loop)
4. **Cable B**: OutcomeTracker low_value → PredictiveAgent context
5. **Cable C**: Markov prediction → `mach_qos.set_tier()` proactive QoS
6. **Budget cognitivo**: Router uses per-predictor outcome scores

### Dead Code (confirmed, safe to remove)
- `optimizer.rs:optimize()` — never called in modern daemon
- `TransformerPredictor` — disabled
- `TelemetryLogger` — disabled

Full plan: see memory file `project_v060_evolution.md`.

## Unified Persistence Layer

All learned state is persisted in a single file (`learned_state.json`) via `LearnedState` in `src/engine/learned_state.rs`. This replaces the pattern of each subsystem persisting independently.

**What's persisted:**
- Signal intelligence: hazard model, MPC, Kalman filters, learned zones, utility EMAs
- Outcome tracker: Bayesian weights, experience memory, co-occurrence graph, HRPO groups
- Specialist accuracy tracker: per-specialist EMA weights

**Self-improvement:** Before each persist, `self_improve()` prunes stale co-occurrence entries, noisy weights, and caps experience memory at 300 records. After each restore, `validate()` clamps out-of-range values.

**Restore quality monitoring:** `RestoreQualityMonitor` tracks effectiveness for 50 cycles post-restore. If restored state is stale (quality < 0.35), zones are reset to defaults.

**Adding a new component:** Add a `#[serde(default)]` field to `LearnedState`, populate in `collect()`, restore in `apply()`.

## Quick Reference

| Task | Command |
|------|---------|
| Build release | `cargo build --release` |
| Run all tests | `cargo test` |
| Run single test | `cargo test test_name` |
| Format code | `cargo fmt --all` |
| Lint | `cargo clippy --all-targets` |
| Run CLI | `cargo run -- <subcommand>` |
| Install daemon | `./scripts/install-root-daemon.sh` |
| Uninstall daemon | `./scripts/uninstall-root-daemon.sh` |

