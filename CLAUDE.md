# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Ponytail — lazy senior dev mode (2026-06-14)

Lazy means efficient, not careless. Best code is code never written. Before writing code, stop at the first rung that holds: (1) does this need to exist? — no: skip it (YAGNI); (2) stdlib does it? — use it; (3) native platform feature? — use it; (4) installed dependency? — use it; (5) one line? — one line; (6) only then: the minimum that works. Deletion over addition, boring over clever, fewest files. No abstraction/dependency/boilerplate nobody asked for. Mark intentional simplifications with a `ponytail:` comment naming the ceiling + upgrade path.

**Never lazy about** (these WIN over minimalism): the Key Safety Constraints below, complete mediation through `safety.rs`, conservative daemon behavior, trust-boundary validation, error handling that prevents data loss, security. Apollo's defensive guards are the carve-out, never the chopping block. On any conflict, this project's safety/supervision rules override ponytail.

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

## Supervision mode (active 2026-05-07 onward)

User explicitly opted into supervision over autopilot. Rules:

1. **NEVER declare work complete without mechanical verification AND raw output shown to user.** "Tests pass" is not proof — paste the test runner output. "Deploy succeeded" is not proof — paste `launchctl print` state. "Bug closed" is not proof — paste the journal/runtime_metrics diff that demonstrates the fix.
2. **User is the gatekeeper, not the AI, not NotebookLM.** Present evidence, let the user draw conclusions. Do not pre-conclude on the user's behalf.
3. **Show commands before running them on irreversible/shared state** (deploy, restart daemon, modify production config). Wait for explicit go-ahead.
4. **Prefer "preliminary" over "closed"** when sample size is small or evidence indirect. Reserve "closed" for outcomes verified in production with N≥500 events or equivalent stress signal.
5. **Surface discrepancies, don't paper over them.** If NotebookLM says X and prod metrics say Y, report both — don't pick the prettier one.
6. **No celebratory summaries.** End-of-task message lists what was changed and what remains uncertain. Skip the victory lap.

## Project Overview

**apollo-silicon-reactor** (formerly `apollo-optimizer`) is a macOS system optimization daemon written in Rust (edition 2021) for Apple Silicon M1 8GB baseline. See `README.md` for full description, qualities, and academic foundation.

### Three Binaries

1. **apollo-optimizer** (CLI) — `src/main.rs`. One-off commands: `snapshot`, `optimize`, `clean`, `turbo`, `daemon`, `startup`, `llm`, `restore`.
2. **apollo-optimizerd** (Daemon) — `src/bin/apollo-optimizerd/main.rs`. Long-running. Unix socket: `/var/run/apollo-optimizer.sock` (root) or `/tmp/apollo-optimizer.sock`. State under `/var/lib/apollo/` (root) or `/tmp/`.
3. **apollo-optimizerctl** (Client) — `src/bin/apollo-optimizerctl/main.rs`. CLI + TUI dashboard.

## Build & Development Commands

```bash
# Build
cargo build --release

# Run from source
cargo run -- snapshot --output system_snapshot.json
cargo run --bin apollo-optimizerd -- daemon --profile balanced-root
cargo run --bin apollo-optimizerctl -- status

# Code style
cargo fmt --all
cargo clippy --all-targets

# Tests
cargo test                                    # All tests
cargo test --doc                              # Doctests only
cargo test my_test_name                       # Filter by substring
cargo test --lib                              # Library only
cargo test --bin apollo-optimizerd            # Specific binary
```

**Notes:**
- `.cargo/config.toml` enables `-C target-cpu=native` and LTO for release builds
- Avoid running multiple `cargo` commands concurrently (shared `target/` directory)
- Release builds use `panic=abort`

## Installation & Deployment

```bash
# Install as root daemon
sudo ./scripts/install-root-daemon.sh

# Guarded deploy (3-gate: test-diff, pre-snap, post-90s sanity ≥87 AIS)
./scripts/apollo-deploy-gate.sh

# Uninstall + restore
sudo ./scripts/uninstall-root-daemon.sh
```

**Best practice:** Build as your user, then run produced binaries as root (via launchd or `sudo`) rather than compiling as root.

**Deployment gotcha:** `sudo cp` preserves the linker-signed codesign. Do NOT use `python3 open().write()` — it strips the linker-signed flag and triggers Launch Constraint Violation. After replacing the binary, do `sudo launchctl bootout system/com.eduardocortez.systemoptimizerd && sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist`. The deploy gate script does this correctly.

**launchctl label**: `com.eduardocortez.systemoptimizerd`. **Binary path**: `/usr/local/libexec/apollo-optimizerd`.

## Architecture

### Workspace layout

```
src/                              # Top-level CLI binary (apollo-optimizer)
src/bin/apollo-optimizerd/        # Root daemon — ~30 independent `tick` modules (Strangler Fig)
src/bin/apollo-optimizerctl/      # CLI client + TUI dashboard (4-10Hz differential rendering)
crates/apollo-engine/             # Cognitive engine library
  src/engine/                     # Decision logic, NARS, causal graph, scorer, ~150 modules
  tests/                          # Integration tests (level3_*)
scripts/                          # install/uninstall/deploy-gate
```

### Key engine modules (in `crates/apollo-engine/src/engine/`)

- **`safety.rs`** — Validates and enforces optimization constraints. Single source of truth for protected processes.
- **`decide_actions.rs`** — Decision logic. Wires PolicyScorer + Gate Tower + asymmetric override.
- **`execute_actions.rs`** — Applies optimizations (SIGSTOP, sysctl, jetsam tier). Attaches Rationale to journal.
- **`action_policy.rs`** — `PolicyScorer`, `PolicyFeature` trait, `Contribution { benefit, cost, uncertainty, hard_veto }`.
- **`shadow_evaluator.rs`** — Asymmetric scorer/gate cutover (±0.30 threshold, REJECT-only override).
- **`nars_belief.rs`** — Non-Axiomatic Reasoning (Pei Wang 2013) with adaptive drift threshold.
- **`causal_graph.rs`** — Causal honesty: external_blame discounts impact score by 0.30 (Pearl 2009).
- **`learned_state.rs`** — `LearnedState` unified persistence + `PolicyRollbackGuard` (Sutton 2018 §11.7).
- **`user_presence.rs`** — 3-tier idle/HID/sleep modulator with `CRITICAL_PRESSURE_BYPASS = 0.65`.
- **`lse_counters.rs`** — `pub static LSE_COUNTERS: LockFreeMetrics` (ARMv8.1 atomics, single instance — never construct local copies).
- **`audit_types.rs`** — `Rationale { action_class, trigger, evidence, expected_outcome }`.
- **`shadow_signals.rs`** — Lock-free static atomics for cross-module signal handoff.

### Data & State Files (root install)

Under `/var/lib/apollo/`:
- `runtime_metrics.json` — Current system + cognitive metrics (11 LSE counters end-to-end)
- `journal.jsonl` — Audit log of optimizations (every action carries a Rationale)
- `learned_state.json` — Unified persistence (hazard model, Kalman, NARS, OutcomeTracker, RollbackGuard)
- `governor_state.json` — Profile configuration state
- `profile_timeline.jsonl` — History of profile switches
- `wake_state.json` — Wake/sleep event tracking
- `frozen_state.json` — Currently frozen processes

Non-root instances use `/tmp` equivalents. Kill switch: `/var/run/apollo.disable` (presence pauses optimization).

## Code Style & Patterns

### Imports & Naming
- **Import grouping:** `std` → external crates → local crate (`crate::...` / `apollo_engine::...`)
- **Naming:** Types/traits/enums → `PascalCase`; functions/vars/modules → `snake_case`; constants → `UPPER_SNAKE_CASE`
- **Serialized strings:** explicit `kebab-case` (use `#[serde(rename_all = "kebab-case")]`)

### Error Handling
- Binaries: `fn main() -> anyhow::Result<()>` with `?` propagation
- Add context at boundaries: `anyhow::Context` (files, sockets, commands)
- In the daemon loop: best-effort handling — record/log error, keep system safe, continue
- Avoid `unwrap()`/`expect()` except for impossible invariants
- Mutex poisoning: recover via `lock().unwrap_or_else(|e| e.into_inner())`

### Types & Ownership
- Borrow (`&str`, `&[T]`) in hot paths; allocate only at boundaries (I/O, protocol, logs)
- Mutex-guarded sections must be short; drop guards before any syscall

### External Commands & Privilege
- Use `std::process::Command` (no shell)
- Never introduce interactive prompts in daemon code (if `sudo` is needed, use `sudo -n`)
- Be conservative with global state changes (Spotlight, sysctls, process signals)

### Unsafe & FFI
- Keep `unsafe` blocks small and localized; wrap behind safe helpers where possible
- Document non-obvious invariants (pointer ownership, sizes, lifetimes)

### Daemon Specifics
- No blocking I/O on the hot path; keep per-cycle work bounded
- Defensive cleanup on startup: if Apollo froze processes and crashed/restarted, unfreeze them
- Avoid high-frequency logging in tight loops (output ends up in launchd logs)
- Use structured JSON for machine-readable data where appropriate

## Key Safety Constraints

This code can: freeze/throttle processes (SIGSTOP/SIGCONT), toggle Spotlight (mdutil), write to `/var/*` as root, tune sysctls.

**Behavior must remain conservative.** Always validate actions against `safety.rs` before execution.

**Hard rules:**
- **Never freeze**: `kernel_task`, `launchd`, `WindowServer`, `Spotlight (mds)`, `configd`, `Antigravity`, `Claude`, `Brave/Chromium*` (breaks Brave IPC contract — 3 regression cycles), `rustc` / `cargo` during active builds
- **Build mode**: ≥2 active compilers → thresholds shift -8pp (more conservative)
- **Cascade bypass**: `user_presence_modulator` returns `1.0×` when `memory_pressure ≥ 0.65` — survival beats UX politeness
- **Asymmetric scorer**: PolicyScorer may BLOCK a gate-accept but NEVER override a gate-reject

## Unified Persistence Layer

All learned state persists in `learned_state.json` via `LearnedState` in `crates/apollo-engine/src/engine/learned_state.rs`.

**What's persisted:**
- Signal intelligence: hazard model, MPC, Kalman filters, learned zones, utility EMAs
- Outcome tracker: Bayesian weights, experience memory, co-occurrence graph, HRPO groups
- Specialist accuracy tracker: per-specialist EMA weights
- Policy rollback guard: shift records for auto-revert on quality drop

**Self-improvement:** Before each persist, `self_improve()` prunes stale co-occurrence entries, noisy weights, and caps experience memory at 300 records. Evaluates `PolicyRollbackGuard` and auto-restores `pre_value` for ZoneAlpha + RlBandUpper when quality < 0.35. After each restore, `validate()` clamps out-of-range values.

**Restore quality monitoring:** `RestoreQualityMonitor` tracks effectiveness for 50 cycles post-restore. If restored state is stale (quality < 0.55, warmup 60 cycles), zones reset to defaults.

**Adding a new component:** Add a `#[serde(default)]` field to `LearnedState`, populate in `collect()`, restore in `apply()`.

## Agent Skills

These skills are available when the task matches their description:

- **apollo-evolve** (`~/.claude/skills/apollo-evolve/SKILL.md`) - Darwinian systems evolution for Rust. Paper-backed architecture, measured improvements, commit-per-phase discipline. Trigger: `/apollo-evolve`
- **diagnose** (`~/.claude/skills/diagnose/SKILL.md`) - Disciplined diagnosis loop for hard bugs and performance regressions. Trigger: `/diagnose`
- **tdd** (`~/.claude/skills/tdd/SKILL.md`) - Test-driven development with red-green-refactor loop. Trigger: `/tdd`
- **grill-me** (`~/.claude/skills/grill-me/SKILL.md`) - Stress-test plans and designs. Trigger: `/grill-me`
- **improve-codebase-architecture** (`~/.claude/skills/improve-codebase-architecture/SKILL.md`) - Find refactoring opportunities. Trigger: `/improve`
- **zoom-out** (`~/.claude/skills/zoom-out/SKILL.md`) - Get broader context or higher-level perspective. Trigger: `/zoom-out`

## LSE Counter Discipline

11 lock-free counters live in `pub static LSE_COUNTERS: LockFreeMetrics`. Producer = engine module bumps via `LSE_COUNTERS.inc_*()`. Consumer = `daemon_state.rs::sync_from_lockfree` copies into `RuntimeMetrics` for `runtime_metrics.json`.

**Silent telemetry-death pattern (Sprint 9 fix `4b13a39`):** Never construct local `LockFreeMetrics` instances. Always reference the global static: `let lf_metrics: &'static LockFreeMetrics = &LSE_COUNTERS;`. Local Arc copies silently increment forever but never reach `runtime_metrics.json`.

**Adding a new counter:** Add `AtomicU64` field to `LockFreeMetrics` + `inc_*` helper, mirror as `#[serde(default)] pub u64` field on `RuntimeMetrics`, wire copy in `sync_from_lockfree`, then bump from producer.
