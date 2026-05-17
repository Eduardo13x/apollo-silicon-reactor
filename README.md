# apollo-silicon-reactor

![Rust](https://img.shields.io/badge/rust-2021-orange.svg)
![macOS](https://img.shields.io/badge/macOS-Tahoe-blue.svg)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2B-success.svg)
![AIS](https://img.shields.io/badge/AIS-S--tier-brightgreen.svg)

**apollo-silicon-reactor** (formerly `apollo-optimizer`) is an autonomous system-optimization daemon engineered for **macOS Apple Silicon, M1 8 GB baseline**. Pure Rust, no Python, no shell scripting in the hot loop. It is not a generic process manager — it is an adaptive agent with 9 wired learning phases that redistributes CPU, RAM, and thermal headroom against actual user intent instead of XNU's "fair" scheduler defaults.

The reactor learns. Every cycle it observes ~400-600 live processes, scores candidate actions through a neuro-cognitive stack (NARS reasoning, causal graph, asymmetric policy scorer, RL agent, user-presence modulator), commits its rationale to an audit journal, and either reverts or reinforces its own learned parameters depending on the AIS quality score that follows.

## Why this exists

macOS Apple Silicon laptops with 8 GB RAM live in a constant memory-pressure ceiling. The XNU scheduler is workload-agnostic and optimizes for fairness, not intent — when you are mid-build, Spotlight reindexing wins CPU cycles you needed for `rustc`. apollo-silicon-reactor closes that loop with an agent that knows which process you actually want responsive right now, learns from its own mistakes, and reverses any optimization it picked that turned out to hurt AIS.

## Qualities

- **Adaptive, not reactive** — 22+ learned parameters auto-tuned per workload; no static thresholds in the hot path
- **Causally honest** — discounts its own contribution by 30% when an exogenous confounder (thermal throttle, network spike) was the real cause of the observed effect (Pearl 2009)
- **Self-reverting** — if a learned parameter shift drops AIS quality below 0.35, the policy-rollback guard restores `pre_value` automatically (Sutton & Barto 2018 §11.7)
- **Asymmetric scorer** — the policy scorer can OVERRIDE a gate-accept (safe direction), but is never allowed to override a gate-reject (unsafe direction)
- **User-presence aware** — three-tier modulator (idle / semi-active / active) reads HID idle, HID event rate, sleep assertions, and audio activity to decide how aggressive to be; bypasses politeness at `pressure ≥ 0.65` for survival
- **Battery-aware** — penalizes high-wakeup actions when on battery (proxy for energy cost); silent when plugged in
- **Lock-free hot path** — 11 ARMv8.1 atomic counters publish telemetry without mutex contention (Hellerstein 2012)
- **Auditable** — every executed action attaches a `Rationale { action_class, trigger, evidence, expected_outcome }` to the journal for post-hoc analysis
- **Safety-first** — 13 invariants enforced mechanically in `safety.rs`; protected processes (Antigravity, Claude, Brave, rustc, system services) can never be frozen
- **Disciplined deploys** — 3-gate guard requires test-diff evidence + pre-snap baseline + post-90s sanity check (AIS ≥ 87) before allowing a binary swap

## Architecture

Cargo workspace, three binaries, JSON-tagged Unix-socket IPC:

| Binary | Role | Lifecycle |
|---|---|---|
| `apollo-optimizerd` | Long-running root daemon | `launchd` service, `/var/lib/apollo/` state |
| `apollo-optimizerctl` | CLI client + live TUI dashboard | Connects to `/var/run/apollo-optimizer.sock` |
| `apollo-optimizer` | One-shot CLI (`snapshot`, `optimize`, `restore`, `llm`) | Process-per-invocation |

```mermaid
graph LR
    A[apollo-optimizerctl] -- JSON IPC --> B[apollo-optimizerd]
    B -- SENSE --> C[sysinfo + IOKit + Mach FFI]
    B -- THINK --> D[Neuro-Cognitive Stack]
    D -- DECIDE --> E[PolicyScorer + Gate Tower]
    E -- ACT --> F[SIGSTOP / sysctl / launchctl]
    D -- persist --> G[learned_state.json]
    F -- journal --> H[journal.jsonl + audit]
```

The daemon hot loop is decomposed into ~30 independent `tick` modules (Strangler Fig pattern). Per-cycle work is bounded; no blocking I/O on the hot path; lock guards are dropped before any syscall.

## Cognitive System

11 lock-free LSE counters publish end-to-end through `runtime_metrics.json`. Each counter corresponds to one wired learning phase. Counters that read 0 are not bugs — they are wired-dormant by design until their trigger fires (Crisis arousal, thermal transition, scorer/gate disagreement, etc).

| Phase | Counter | Purpose |
|---|---|---|
| Skill-Aware Prediction | `skill_aware_modulations_total` | Weights trial skills by historical success per workload |
| Arousal-Based Decay | `arousal_decay_accelerations_total` | Crisis flushes NARS beliefs faster (McGaugh 2004) |
| Companion Graph | `companion_cross_group_inferences_total` | Directional `P(proc \| fg_app)` via Lift normalization |
| Adaptive Drift Threshold | `adaptive_drift_threshold_raises_total` | Welford online variance, self-calibrating drift sensitivity |
| Causal External Blame | `causal_external_thermal_blames_total` | Discounts impact score by 0.30 when thermal confounder present |
| Policy Rollback | `policy_rollback_evaluations_total` | Reverts learned params when AIS quality < 0.35 |
| User Presence | `user_presence_suppressions_total` | Idle/HID/sleep-assertion 3-tier modulator with pressure≥0.65 bypass |
| Battery-Aware Cost | `battery_aware_penalty_emissions_total` | Penalizes wakeup/ctx-switch noise on battery |
| Journal Rationale | `journal_rationales_attached_total` | Attaches evidence-based reasoning metadata to every action |
| Scorer Override Reject | `scorer_override_rejects_total` | Asymmetric ±0.30 cutover — scorer can BLOCK gate-accepts |
| Scorer Disagreement | `scorer_disagreement_strong_accepts_total` | Logs gate-rejects the scorer wanted to accept (never overrides) |

### Academic foundation

- **Pei Wang (2013)** — Non-Axiomatic Reasoning, TruthValue revision, Bayesian forgetting
- **Pearl (2009 §3)** — Confounder adjustment, external-blame discount geometry
- **Sutton & Barto (2018 §11.7)** — Model-free policy correction, auto-revert on quality drop
- **Hellerstein et al. (2012)** — Feedback control of computing systems, lock-free counter hot paths
- **Nygard (2018)** — Release It! resilience patterns, circuit breakers, bulkhead pattern
- **Lakshminarayanan (2017)** — Simple scalable predictive uncertainty, RSS composition
- **McGaugh (2004)** — Emotional arousal accelerates memory consolidation/decay
- **Welford (1962)** — Online variance for adaptive thresholding

## Safety invariants

`safety.rs` enforces these mechanically; bypassing them is impossible from outside the module.

- **Never freeze**: `kernel_task`, `launchd`, `WindowServer`, `Spotlight (mds)`, `configd`, `Antigravity`, `Claude`, `Brave/Chromium*` (Brave IPC contract), `rustc` / `cargo` during active builds
- **Cascade bypass**: `user_presence_modulator` returns `1.0×` (full optimization) when `memory_pressure ≥ 0.65`, regardless of HID activity or sleep assertion — survival beats UX politeness
- **Asymmetric scorer**: PolicyScorer may BLOCK a gate-accept (safe direction) but is NEVER allowed to override a gate-reject (unsafe direction)
- **Supervision mode** (`CLAUDE.md`): no work declared "complete" without mechanical re-verification of `runtime_metrics.json` + adversarial diff re-read + N≥500 sample size

## Quick start

```bash
# Build (release: target-cpu=native, LTO, panic=abort)
cargo build --release

# Install as root daemon (codesign-preserving cp to /usr/local/libexec, launchd bootstrap)
sudo ./scripts/install-root-daemon.sh

# Status + cognitive health
apollo-optimizerctl status

# Live TUI dashboard (4-10Hz differential rendering, zero flicker)
apollo-optimizerctl dashboard

# One-shot snapshot
apollo-optimizer snapshot --output system_snapshot.json

# Uninstall + restore
sudo ./scripts/uninstall-root-daemon.sh
```

## Deploy discipline

`scripts/apollo-deploy-gate.sh` enforces three gates before any binary swap:

1. **Gate 1 — Test evidence**: HEAD (or merged branches via `git log -3 --no-merges`) must add/modify at least one `#[test]`. The Disobedience Rule from `CLAUDE.md`: write the failing test first.
2. **Gate 2 — Pre-snapshot**: capture `runtime_metrics.json` + cycle count + AIS before swap.
3. **Gate 3 — Post-snapshot (90s)**: AIS ≥ 87.0 floor, failures = 0, `last_error = None`, cycles progressing. Otherwise the script alerts loudly. Rollback is suggested, never executed — the human decides (supervision rule).

```bash
./scripts/apollo-deploy-gate.sh --dry-run   # gates 1+2 only, no deploy
./scripts/apollo-deploy-gate.sh             # full guarded deploy
```

Binary swap uses `sudo cp` to preserve the linker-signed flag (do NOT use `python3 open().write()` — it strips the codesign and triggers Launch Constraint Violation).

## Repository layout

```
src/                         # CLI binary (apollo-optimizer)
src/bin/apollo-optimizerd/   # Root daemon (long-running)
src/bin/apollo-optimizerctl/ # CLI client + TUI dashboard
crates/apollo-engine/        # Cognitive engine library
  src/engine/                # Decision logic, NARS, causal graph, scorer
  tests/                     # Integration tests (level3_*)
scripts/                     # install/uninstall/deploy-gate
.cargo/config.toml           # target-cpu=native, LTO
CLAUDE.md                    # Project doctrine (supervision mode, anti-patterns)
```

## Development

```bash
cargo test                              # Full suite (~2100 lib tests)
cargo test --doc                        # Doctests
cargo test engine::nars                 # Module filter
cargo clippy --all-targets              # Lint
cargo fmt --all                         # Format
```

Avoid running multiple `cargo` commands concurrently — they contend on the shared `target/` directory.

## License

See `LICENSE` (TBD).
