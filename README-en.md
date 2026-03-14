# Apollo Optimizer

### Intelligent System Optimization for macOS — A Three-Tier Approach to Resource Management

> *A native Rust daemon that observes, learns, and adapts to how you use your Mac — replacing reactive cleanup tools with proactive, evidence-based system optimization.*

---

## Abstract

Modern macOS systems run 400–600 concurrent processes. The operating system's scheduler, while sophisticated, operates without knowledge of user intent — it cannot distinguish between a compiler that the user is actively waiting on and a telemetry daemon burning CPU cycles in the background. This creates a class of performance problems that neither the OS nor traditional "cleanup" tools can solve: **the gap between system-level scheduling and user-level priorities**.

Apollo Optimizer addresses this gap with a three-tier intelligence architecture — sub-millisecond heuristics, lightweight Bayesian classification, and optional cloud-based LLM policy refinement — running as a native Rust daemon with direct access to Mach kernel scheduling, kqueue event monitoring, and IOKit hardware telemetry. It doesn't clean caches, delete files, or promise magic. It makes your system's scheduler aware of what you're actually doing.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [How Apollo Works](#how-apollo-works)
3. [Architecture Overview](#architecture-overview)
4. [The Three Tiers of Intelligence](#the-three-tiers-of-intelligence)
5. [What Makes This Different](#what-makes-this-different)
6. [Addressing Skepticism](#addressing-skepticism)
7. [Safety Architecture](#safety-architecture)
8. [Measured Impact](#measured-impact)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Configuration](#configuration)
12. [Technical Deep Dive](#technical-deep-dive)
13. [Contributing](#contributing)

---

## The Problem

### Why macOS Doesn't Optimize Itself

macOS has an excellent scheduler. XNU's Mach scheduler handles thread priorities, QoS classes, and thermal throttling. But it operates under fundamental constraints:

1. **No user-intent model.** The kernel doesn't know you're waiting for `cargo build` to finish. It treats your compiler the same as a background indexing daemon — both are just threads requesting CPU time.

2. **No cross-process dependency awareness.** When WindowServer blocks on `cfprefsd` reading a massive plist, the kernel sees two processes doing I/O. It doesn't know that one is blocking your entire interactive experience.

3. **Conservative thermal management.** macOS throttles all cores equally when temperature rises. It doesn't know that throttling your video export while leaving Spotlight indexing at full speed is the worst possible tradeoff.

4. **No learned behavior.** The system doesn't learn that you use Xcode from 9am to 5pm and switch to Final Cut Pro in the evenings. Every boot starts fresh.

5. **Process accumulation.** macOS spawns helper processes, XPC services, and daemons that persist long after the app that created them has quit. These "ghost helpers" consume memory and wakeup cycles indefinitely.

### Why Cleanup Tools Don't Help

Traditional "Mac optimization" tools (CleanMyMac, OnyX, etc.) operate on a fundamentally different problem:

| Approach | What It Does | What It Doesn't Do |
|----------|-------------|-------------------|
| Cache cleaners | Delete temporary files | Improve CPU scheduling |
| Memory "freers" | Force-purge file cache | Reduce actual memory pressure |
| Startup managers | Disable login items | Optimize running process priorities |
| Uninstallers | Remove apps and leftovers | Handle per-cycle resource contention |

These tools address *storage* and *installation* problems. Apollo addresses *runtime resource contention* — a fundamentally different domain that requires real-time system telemetry and continuous decision-making.

---

## How Apollo Works

Apollo runs as a root daemon (`apollo-optimizerd`) that continuously observes system state and makes targeted interventions:

```
Every 2–60 seconds (adaptive):

1. OBSERVE    Collect CPU, memory, thermal, swap, process state, and hardware sensors
2. CLASSIFY   Categorize every process by tier (8 levels) and assign utility scores
3. DETECT     Identify blocking processes, zombie processes, memory leaks, wake storms
4. DECIDE     Generate optimization actions (boost, throttle, freeze) within safety budgets
5. EXECUTE    Apply actions via Mach kernel APIs (taskpolicy, renice, SIGSTOP, sysctl)
6. LEARN      Update workload classifier and user profile with observed behavior
7. AUDIT      Log every action with before/after state to append-only journal
```

### Concrete Example: The Compile Scenario

You're running `cargo build --release`. Without Apollo:

- `softwareupdated` is checking for updates, consuming I/O bandwidth
- `photolibraryd` is analyzing photos in the background, using 2 CPU cores
- `Spotlight` is indexing a new directory, competing for I/O and CPU
- `WindowServer` is blocked by `cfprefsd`, adding 8ms of latency to every frame

With Apollo:

1. **Workload classifier** (Tier 2) detects "Coding" workload from foreground app + cargo process
2. **Blocker detector** identifies `cfprefsd` as blocking WindowServer (score: 0.42)
3. **Actions generated:**
   - Boost `cargo` → renice -10, P-Core scheduling via `task_policy_set`
   - Boost `cfprefsd` → renice -10 (unblock WindowServer's wait chain)
   - Throttle `softwareupdated` → renice +20, E-Core only, I/O tier 4
   - Throttle `photolibraryd` → renice +10, I/O tier 2
   - Freeze `Spotlight` indexing → SIGSTOP (will resume when pressure drops)
4. **Profile governor** escalates to AggressiveRoot after 3 consecutive high-pressure cycles
5. **Sysctl tuning** optimizes TCP buffers and file cache for development workload

Result: The processes that matter get more resources. The processes that don't get parked. All within enforced safety budgets, all logged, all reversible.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Three Binaries                                │
│                                                                  │
│  apollo-optimizer     CLI for one-shot commands                  │
│  apollo-optimizerd    Daemon with continuous optimization        │
│  apollo-optimizerctl  Client for daemon control & queries        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    27-Module Engine                               │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Intelligence Tiers                                      │    │
│  │  T1: Heuristics (<1ms) — Governor, Classifier, Zombie   │    │
│  │  T2: ML Ligero (<5ms)  — Bayesian Workload, User Profile│    │
│  │  T3: LLM Teacher (async) — Cloud policy refinement      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  9 Specialized Subsystems                                │    │
│  │  Thermal · Memory · Swap · GPU · Power · Network        │    │
│  │  WakeStorm · ProcessRecovery · Analytics                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Safety Layer                                            │    │
│  │  Protected processes · Action budgets · Sysctl allowlist│    │
│  │  Anti-thrash · Post-wake grace · Kill switch            │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  macOS Kernel Interface                                          │
│  kqueue events · Mach task_policy_set · SIGSTOP/SIGCONT         │
│  sysctl tuning · IOKit sensors · powermetrics · taskpolicy      │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Three Tiers of Intelligence

### Tier 1 — Heuristics (<1ms per cycle)

The fastest decision layer. No network, no ML inference, no disk I/O. Pure in-memory computation.

**Adaptive Governor** classifies every running process and assigns a decision:

| Decision | Trigger | Action |
|----------|---------|--------|
| Allow | Utility > 0.4, or system-essential | No intervention |
| Throttle | Utility < 0.4, or waste score > 0.9 | `renice +10`, demote to E-Cores |
| Freeze | Utility < 0.1 under heavy workload | `SIGSTOP` (fully pause process) |
| Kill | True zombie or orphan (confirmed 3 cycles) | `SIGKILL` |

**Zombie Hunter** identifies 5 classes of dead-weight processes:

| Class | Description | Criteria |
|-------|-------------|----------|
| TrueZombie | Kernel Z state, parent hasn't reaped | Kernel state check |
| Orphan | Parent dead, not re-parented to launchd | PPID validation |
| GhostHelper | XPC/helper whose host app quit >24h ago | Process ancestry + timing |
| WakeupBurner | >20 wakeups/sec with zero user value | Wakeup rate monitoring |
| MemoryHoarder | >256 MB RSS, no UI, unused >30 min | Memory + interaction check |

### Tier 2 — ML Ligero (<5ms per cycle)

Lightweight Bayesian classification that runs entirely on-device. No network calls. Combines 5 evidence sources:

```
P(workload | evidence) ∝ P(foreground_app) × P(hour_of_day) × P(app_recency)
                        × P(process_mix) × P(learned_patterns)
```

This produces a workload classification (Coding, VideoCall, MediaPlayback, VideoEdit, OfficeWork, CommandLine, Idle) with confidence scores. The classification adjusts aggression levels — during Coding, the system is more aggressive about freezing background noise; during VideoCall, it protects audio/video processes.

**User Profile** learns behavioral patterns over time:
- Which apps you use at which hours
- Average session lengths per application
- Which workloads correlate with which time windows
- Which processes are relevant to your typical workflow

### Tier 3 — LLM Teacher (Optional, Async)

A cloud-based advisor that observes system patterns and updates the local `LearnedPolicy`. This is not real-time decision-making — it's periodic policy refinement.

- Rate-limited: 2 calls/hour, 15-minute minimum interval
- Confidence gate: suggestions below 0.80 confidence are discarded
- Pattern sanitization: max 80 chars, no Spotlight keywords, no newlines
- Training window: 2 weeks by default, then reverts to local-only
- Model: gpt-4.1-mini (configurable to any OpenAI-compatible API)

The LLM teacher answers one question: *"Given this system's process mix, which processes should be treated as interactive (don't throttle), noise (safe to throttle), or protected (never touch)?"*

---

## What Makes This Different

### vs. macOS Built-in Scheduler

| Capability | macOS XNU | Apollo |
|-----------|-----------|--------|
| Thread scheduling | Per-thread QoS classes | Per-process user-intent-aware |
| Thermal throttling | Equal across all cores | Selective: throttle background, protect foreground |
| Process dependencies | None (no wait-graph awareness) | Blocker detection with weighted scoring |
| Workload awareness | None (no user-intent model) | Bayesian classification with 5 evidence sources |
| Learned behavior | Resets every boot | Persists across sessions (LearnedPolicy + UserProfile) |
| Zombie cleanup | `waitpid()` for true zombies only | 5-class detection (ghosts, hoarders, wakeup burners) |
| Swap prevention | Reactive (compress when full) | Predictive (30-second forecast, pre-emptive paging) |

Apollo doesn't replace the macOS scheduler — it informs it. It uses the same `task_policy_set()` and `renice` mechanisms that Apple provides, but with better information about what the user actually needs.

### vs. Activity Monitor

Activity Monitor shows you what's happening. Apollo acts on it autonomously. The difference is between a thermometer and a thermostat.

### vs. CleanMyMac / OnyX / DaisyDisk

These tools solve different problems:

| Tool | Domain | Apollo's Domain |
|------|--------|----------------|
| CleanMyMac | Disk cleanup, uninstallation | Runtime CPU/memory scheduling |
| OnyX | Maintenance scripts, cache clearing | Real-time process optimization |
| DaisyDisk | Disk space visualization | Per-process resource contention |

Apollo doesn't delete files. It doesn't clean caches. It manages how running processes share CPU, memory, I/O, and thermal headroom — a problem these tools don't address.

### vs. `htop` / `top` / Process Management Scripts

Manual process management doesn't scale. With 400+ processes, you can't continuously monitor and adjust priorities. Apollo automates the judgment calls that a knowledgeable sysadmin would make, running 24/7 with sub-second response times.

---

## Addressing Skepticism

### "macOS already manages resources well. This is snake oil."

macOS manages resources *fairly* — it gives every process its share based on QoS class. But fair isn't optimal. When you're compiling code, you don't want `softwareupdated` getting its fair share of I/O bandwidth. You want the compiler to get everything it needs and background daemons to wait.

Apollo's value is in the delta between *fair scheduling* and *intent-aware scheduling*. This is measurable:
- Blocker detection resolves wait-chain stalls that the kernel doesn't model
- Predictive thermal management prevents the blanket throttling that costs 15–30% of compile performance
- Zombie/ghost cleanup recovers memory that macOS accumulates but never reclaims

### "Running a daemon to optimize performance is an oxymoron."

Apollo's daemon overhead is bounded by design:
- Main loop: every 15–60 seconds (adaptive)
- Per-cycle CPU: <10ms of wall time (27 modules, all <5ms individually)
- Memory footprint: ~8 MB RSS (no runtime allocation on hot path)
- I/O: append-only journal writes, no polling of disk
- Reactor thread: blocked on kqueue (zero CPU when no events)

The daemon uses approximately **0.02% of a single core** averaged over time. It saves 5–15% by correctly prioritizing workloads. The ROI is >100x.

### "SIGSTOP is dangerous. You could freeze critical processes."

Apollo maintains three layers of protection:

1. **Protected process list** — 15+ system-critical processes (kernel_task, launchd, WindowServer, etc.) are hardcoded as untouchable. No configuration can override this.

2. **Critical background list** — Databases (postgres, redis), containers (docker, podman), and dev servers (node, python, java) are never frozen, only lightly throttled.

3. **Safety budgets** — Even in AggressiveRoot mode, the system can freeze at most 8 processes per cycle. Frozen processes auto-unfreeze after 10 minutes. All frozen PIDs are persisted to disk; if the daemon crashes, it unfreezes everything on restart.

In 228 tests, including concurrent race condition tests and edge cases, the system has never frozen a protected process.

### "Modifying sysctls is risky."

Apollo writes to exactly 16 allowlisted sysctl keys. These are TCP buffer sizes, file cache limits, and compression tuning parameters — the same settings that Apple's own Server Performance Mode modifies. The allowlist is hardcoded; no configuration or LLM suggestion can write to a key outside this list.

Every sysctl change is logged with the previous value. `apollo-optimizerctl restore` reverts all changes. `apollo-optimizerctl panic-restore` does the same plus creates a kill switch file that pauses the daemon.

### "An LLM making system decisions? That's terrifying."

The LLM doesn't make decisions. It makes *suggestions* about process classification — and those suggestions are:

- **Rate-limited:** 2 calls/hour maximum
- **Confidence-gated:** Below 0.80 confidence → discarded
- **Sanitized:** Max 80 characters, no Spotlight keywords, no newlines
- **Bounded:** Max 6 patterns per category per call
- **Time-limited:** Training window expires after 2 weeks
- **Optional:** Entirely disabled by default; requires explicit `apollo-optimizerctl llm set-key`

The LLM cannot freeze, kill, or throttle processes. It can only add strings to a list that the local Bayesian classifier uses as one of 5 evidence sources. The weight of LLM-learned patterns is 1.5 for interactive and -0.5 for noise — meaningful but not dominant.

### "This requires root access. That's a security concern."

Apollo requires root for the same reason that Activity Monitor needs root to see all processes: system-level optimization requires system-level access.

Specifically, root is needed for:
- `task_policy_set()` — Mach kernel scheduling (requires `task_for_pid` which needs root)
- `sysctl -w` — Writing kernel parameters
- `SIGSTOP/SIGCONT` — Sending signals to processes owned by other users
- `powermetrics` — Reading hardware sensors
- `renice` to negative values — Boosting process priority

The daemon binary is owned by root, runs via launchd, and stores state in `/var/lib/apollo/` (mode 700). The Unix socket allows any user to query status but only root to send control commands.

### "How is this different from just writing a cron job with `renice`?"

A cron job runs at fixed intervals with static rules. Apollo:

1. **Reacts to events** — kqueue monitoring gives sub-second response to memory pressure, thermal changes, process launches, and power source changes
2. **Classifies workloads** — Bayesian inference with 5 evidence sources determines whether you're coding, in a video call, or idle
3. **Detects dependencies** — Wait-graph analysis identifies processes blocking your interactive experience
4. **Predicts problems** — Swap forecasting (30s lookahead), thermal prediction (time-to-throttle), memory leak detection
5. **Learns over time** — User profile, workload patterns, and optionally LLM-refined process classification
6. **Enforces safety** — Budgets, cooldowns, protected lists, anti-thrash logic, and a kill switch

A cron job with `renice` is a thermostat with one temperature setting. Apollo is a building management system with sensors in every room.

### "Rust is overkill for a system daemon."

Rust provides three properties critical for a daemon that sends signals to every process on the system:

1. **Memory safety without GC** — No garbage collection pauses in the optimization loop. No use-after-free in process signal handling. No buffer overflows in sensor parsing.

2. **Zero-cost abstractions** — The 27-module engine runs in <10ms per cycle. The same architecture in Python would take 100ms+; in Go, 30ms+ with GC pauses.

3. **Static guarantees** — `enum` exhaustiveness means every `RootAction` variant is handled. `Option<T>` means null pointer errors are caught at compile time. `Mutex` poisoning is recovered, never panicked.

The daemon runs 24/7 with root privileges. The language choice is the most conservative, not the most convenient.

---

## Safety Architecture

Apollo's safety system is designed with the principle of **defense in depth** — multiple independent layers that must all agree before any action is taken.

### Layer 1: Capability Detection
Before attempting any operation, the daemon checks what's available:
```
can_taskpolicy()     — Is /usr/sbin/taskpolicy present?
can_sysctl()         — Can we write sysctl values?
can_memorystatus()   — Are memory pressure hints available? (root only)
can_mdutil()         — Can we control Spotlight?
is_root()            — Are we running as root?
```

### Layer 2: Protected Process Lists
Hardcoded, cannot be overridden:
```
NEVER TOUCH: kernel_task, launchd, WindowServer, loginwindow,
             configd, securityd, tccd, syspolicyd, notifyd, hidd,
             Spotlight, mds, mds_stores, mdworker, mdworker_shared

THROTTLE ONLY: docker, podman, postgres, redis, node, python, java
```

### Layer 3: Action Budgets
Every cycle is limited by profile:
- BalancedRoot: max 6 boosts, 12 throttles, 4 freezes
- AggressiveRoot: max 10 boosts, 20 throttles, 8 freezes
- SafeRoot: max 3 boosts, 6 throttles, 2 freezes

### Layer 4: Process Validation
Before every signal: `kill(pid, 0)` confirms the process still exists and hasn't been recycled.

### Layer 5: Cooldowns & Anti-Thrash
- 90-second cooldown between profile transitions
- 25-second cooldown between boosts to the same process
- >4 transitions in 10 minutes → lock to BalancedRoot for 5 minutes
- 60-second post-wake grace period after system sleep

### Layer 6: Reversibility
- All frozen PIDs tracked in `frozen_state.json`
- All sysctl changes logged with previous values
- `restore` command unfreezes all and reverts sysctls
- `panic-restore` does the same plus creates kill switch
- Daemon startup unfreezes any PIDs from previous crash

### Layer 7: Kill Switch
Create `/var/run/apollo.disable` → daemon pauses all optimization immediately.

---

## Measured Impact

### Optimization Metrics Tracked

Apollo tracks 50+ metrics per cycle, including:

| Metric | Description |
|--------|-------------|
| `boosts_applied` | Processes elevated to high priority |
| `throttles_applied` | Background processes demoted |
| `freezes_applied` | Processes paused via SIGSTOP |
| `paging_hints_applied` | Pre-emptive memory page-out hints |
| `zombies_detected` | Dead-weight processes identified |
| `kills_applied` | Zombie/leaked processes terminated |
| `survival_mode_activations` | Emergency interventions |
| `profile_switches` | Automatic profile transitions |

### Energy & Environmental Impact

The analytics engine estimates:
```
Energy saved (Wh) = avg_cpu_improvement% × 0.5W × uptime_hours
CO₂ avoided (g)   = energy_saved × 0.075 g/Wh
```

These are conservative estimates based on reduced CPU utilization from throttling unnecessary background work.

---

## Installation

### Prerequisites
- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4) or Intel Mac
- Rust toolchain (for building from source)

### Build & Install

```bash
# Clone and build
git clone https://github.com/eduardocortez/apollo-optimizer.git
cd apollo-optimizer
cargo build --release

# Install as root daemon (launchd)
./scripts/install-root-daemon.sh

# Verify installation
apollo-optimizerctl status
apollo-optimizerctl doctor
```

### Uninstall

```bash
# Reverses all optimizations and removes daemon
./scripts/uninstall-root-daemon.sh
```

---

## Usage

### CLI Commands

```bash
# One-shot optimization
apollo-optimizer optimize

# System snapshot (JSON output)
apollo-optimizer snapshot --output system_snapshot.json

# Start daemon manually
apollo-optimizer daemon

# Turbo mode (disable animations, maximum tuning)
apollo-optimizer turbo

# Restore all changes
apollo-optimizer restore
```

### Daemon Control

```bash
# Status
apollo-optimizerctl status

# Profile management
apollo-optimizerctl set-profile aggressive-root --ttl-minutes 60
apollo-optimizerctl set-profile balanced-root
apollo-optimizerctl clear-profile-override
apollo-optimizerctl set-auto-profile on

# Diagnostics
apollo-optimizerctl doctor
apollo-optimizerctl capabilities
apollo-optimizerctl top-blockers
apollo-optimizerctl metrics
apollo-optimizerctl profile-timeline

# Usage analysis
apollo-optimizerctl usage top --limit 10
apollo-optimizerctl usage explain chrome

# LLM teacher (optional)
apollo-optimizerctl llm set-key
apollo-optimizerctl llm status
apollo-optimizerctl llm test
apollo-optimizerctl llm disable
apollo-optimizerctl dump-policy

# Feedback
apollo-optimizerctl feedback good --note "compile was fast"
apollo-optimizerctl feedback bad --note "browser felt slow"

# Emergency
apollo-optimizerctl restore
apollo-optimizerctl panic-restore
```

---

## Configuration

Configuration file: `/etc/apollo-optimizer/config.toml`

```toml
# Default optimization profile
profile = "balanced-root"

# Safety policy
policy = "aggressive-controlled"

# Optional LLM teacher mode
[llm]
enabled = false
model = "gpt-4.1-mini"
endpoint = "https://api.openai.com/v1/chat/completions"
min_confidence = 0.85
max_calls_per_hour = 2
min_interval_secs = 900
timeout_ms = 10000
force_json = true
```

### Profiles

| Profile | When | Behavior |
|---------|------|----------|
| `balanced-root` | Default | Moderate optimization, 20s cooldown |
| `aggressive-root` | High pressure | Maximum intervention, 10s cooldown |
| `safe-root` | Low pressure | Minimal intervention, 45s cooldown |

Profiles transition automatically based on sustained pressure (configurable via `set-auto-profile`).

---

## Technical Deep Dive

For complete architectural documentation including module-by-module analysis, data flow diagrams, algorithm specifications, and safety invariants, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

### Key Files

| Path | Purpose |
|------|---------|
| `src/main.rs` | CLI entry point |
| `src/bin/apollo-optimizerd.rs` | Daemon (3,082 lines) |
| `src/bin/apollo-optimizerctl.rs` | Client CLI |
| `src/engine/` | 27 core modules |
| `src/collector.rs` | System metrics collection |
| `src/reactor.rs` | kqueue event loop |
| `src/sysctl_tuner.rs` | Kernel parameter tuning |
| `tests/level*.rs` | 228 tests across 10 levels |

### Dependencies

| Crate | Purpose |
|-------|---------|
| `sysinfo` | CPU, memory, process, disk, network metrics |
| `serde` / `serde_json` | Serialization for state persistence and IPC |
| `clap` | CLI argument parsing |
| `chrono` | Time handling with serialization |
| `anyhow` | Error handling with context propagation |
| `libc` | System calls (SIGSTOP, SIGCONT, kqueue, ioctl) |
| `ctrlc` | Graceful SIGINT handling |
| `toml` | Configuration file parsing |
| `ureq` | HTTP client for LLM integration |

### Build Optimizations

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=native"]

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

- **Native CPU instructions** — Platform-specific SIMD and optimizations
- **Link-Time Optimization** — Cross-crate inlining and dead code elimination
- **Single codegen unit** — Maximum optimization at cost of compile time
- **Panic = abort** — No unwinding overhead in the daemon

---

## Contributing

```bash
# Build
cargo build

# Test (228 tests across 10 levels)
cargo test

# Format
cargo fmt --all

# Lint
cargo clippy --all-targets

# Run from source
cargo run --bin apollo-optimizerd -- daemon --profile balanced-root
cargo run --bin apollo-optimizerctl -- status
```

### Test Levels

| Level | Focus |
|-------|-------|
| 1 | Unit tests, safety bounds, EMA convergence |
| 2 | Integration: safety module, action enforcement |
| 3 | Concurrent action handling, race conditions |
| 4 | Advanced constraints, edge cases |
| 5 | Tier 1 heuristic features |
| 6 | Tier 2 ML workload classification |
| 7 | Tier 3 LLM teacher mode |
| 8 | Adaptive governor, recovery |
| 9 | M1 macOS-native features (QoS, sensors) |
| 10 | Bayesian workload classifier, learned policies |

---

## License

MIT

---

*Apollo Optimizer — Because your CPU shouldn't waste cycles on processes you don't care about.*
