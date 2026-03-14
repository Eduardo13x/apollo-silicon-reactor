# Apollo Optimizer — System Architecture

> **Version:** 0.1.0 · **Language:** Rust 2021 · **Platform:** macOS (Apple Silicon native)
> **Last updated:** 2026-03-03

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [System Overview](#2-system-overview)
3. [Three-Tier Intelligence Model](#3-three-tier-intelligence-model)
4. [Binary Architecture](#4-binary-architecture)
5. [Core Engine Modules (27)](#5-core-engine-modules)
6. [Decision Pipeline](#6-decision-pipeline)
7. [Safety & Constraint System](#7-safety--constraint-system)
8. [Reactive Nervous System](#8-reactive-nervous-system)
9. [Profile Governor State Machine](#9-profile-governor-state-machine)
10. [LLM Teacher Integration](#10-llm-teacher-integration)
11. [Data Flow & Persistence](#11-data-flow--persistence)
12. [Hardware Telemetry Layer](#12-hardware-telemetry-layer)
13. [Module Reference Table](#13-module-reference-table)

---

## 1. Design Philosophy

Apollo Optimizer is built on four architectural principles:

1. **Observe, don't guess.** Every decision is backed by measured system state — CPU pressure, memory trends, thermal sensors, swap velocity, process wakeup rates, and user interaction patterns.

2. **Tiered intelligence with bounded latency.** The system uses three tiers of decision-making, each with strict latency contracts: heuristics (<1ms), lightweight Bayesian ML (<5ms), and optional cloud LLM (async, rate-limited). The hot path never waits for ML or network.

3. **Conservative by default, aggressive by evidence.** The daemon starts in `BalancedRoot` profile. It escalates to `AggressiveRoot` only after 3 consecutive cycles of sustained pressure above 0.72. It de-escalates after 6 consecutive cycles below 0.55. Anti-thrash logic locks the profile if oscillation is detected.

4. **Reversibility as a first-class property.** Every optimization action (SIGSTOP, renice, sysctl write) is logged in an append-only journal (`journal.jsonl`) with before/after state. The system can restore to pre-optimization state at any point. If the daemon crashes, it unfreezes all tracked processes on restart.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Apollo Optimizer System                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐      │
│   │ apollo-      │   │ apollo-          │   │ apollo-           │      │
│   │ optimizer    │   │ optimizerd       │   │ optimizerctl      │      │
│   │ (CLI)        │   │ (Daemon)         │   │ (Client)          │      │
│   │              │   │                  │   │                   │      │
│   │ One-shot     │   │ Continuous       │   │ IPC queries &     │      │
│   │ commands     │   │ optimization     │   │ profile control   │      │
│   └──────┬───────┘   └────────┬─────────┘   └─────────┬─────────┘      │
│          │                    │                        │                │
│          │                    │    Unix Socket IPC     │                │
│          │                    │◄──────────────────────►│                │
│          │                    │                        │                │
│          ▼                    ▼                                         │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                     Core Engine (27 modules)                 │      │
│   │                                                              │      │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │      │
│   │  │ Tier 1     │  │ Tier 2     │  │ Tier 3                 │  │      │
│   │  │ Heuristics │  │ ML Ligero  │  │ LLM Teacher (optional) │  │      │
│   │  │ <1ms       │  │ <5ms       │  │ async, rate-limited    │  │      │
│   │  └────────────┘  └────────────┘  └────────────────────────┘  │      │
│   │                                                              │      │
│   │  ┌────────────────────────────────────────────────────────┐  │      │
│   │  │              9 Specialized Subsystems                  │  │      │
│   │  │  Thermal · Memory · Swap · GPU · Power · Network      │  │      │
│   │  │  WakeStorm · ProcessRecovery · Analytics              │  │      │
│   │  └────────────────────────────────────────────────────────┘  │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                    macOS Kernel Interface                     │      │
│   │  kqueue · Mach task_policy_set · SIGSTOP/SIGCONT · sysctl   │      │
│   │  IOKit sensors · powermetrics · mdutil · taskpolicy          │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Three-Tier Intelligence Model

### Tier 1 — Heuristics (<1ms)

| Module | Responsibility |
|--------|---------------|
| `adaptive_governor.rs` | Central per-process decision engine (Allow / Throttle / Freeze / Kill) |
| `process_classifier.rs` | Categorizes processes into 8 tiers by behavior |
| `zombie_hunter.rs` | Identifies 5 classes of dead-weight processes |

**Process Classification Tiers:**

```
┌─────────────────────┬──────────────────────────────────────────────┐
│ Tier                │ Criteria                                      │
├─────────────────────┼──────────────────────────────────────────────┤
│ ActiveForeground    │ User interaction within last 30 seconds       │
│ BackgroundVisible   │ App opened, not focused, used within 5 min    │
│ AppHelper           │ XPC/helper subprocess of a running app        │
│ SystemEssential     │ Kernel services, audio, display server        │
│ SilentDaemon        │ Background service, no observable user value  │
│ Stale               │ Not used in >24 hours                         │
│ ZombieOrphan        │ Dead or orphaned process                      │
│ Telemetry           │ Analytics, crash reporters, diagnostics       │
└─────────────────────┴──────────────────────────────────────────────┘
```

**Utility Score (0.0–1.0):**
```
utility = base_interaction_score      // 1.0 if <30s, 0.7 if <5min, 0.0 if >1h
        + gui_window_bonus    (0.3)
        + network_active      (0.2)
        - high_wakeup_penalty (0.5)   // >50 wakeups/sec
        + user_profile_boost  (var)   // from learned user behavior
```

**Decision Thresholds:**
- `utility < 0.1` AND heavy workload → **Freeze** (SIGSTOP)
- `utility < 0.4` → **Throttle** (renice +10)
- `waste_score > 0.9` → **Throttle** (override)
- Zombie/Orphan → **Kill** (SIGKILL)

### Tier 2 — ML Ligero (<5ms)

| Module | Responsibility |
|--------|---------------|
| `workload_classifier.rs` | Bayesian workload classification with 5 evidence sources |
| `user_profile.rs` | Behavioral learning: per-app stats, hour-of-day model |

**Bayesian Evidence Sources:**

| Source | Weight | Description |
|--------|--------|-------------|
| ForegroundApp | 2.0 | Direct match against known app categories |
| HourPrior | 0.3 | 24-hour probability distribution from user history |
| AppRecency | 0.1–0.8 | How recently the user interacted with matching apps |
| ProcessMix | variable | Count of matching background processes |
| LlmLearned | 1.5 / -0.5 | Patterns from LearnedPolicy (interactive/noise) |

**Workload Types:** Coding, VideoCall, MediaPlayback, VideoEdit, OfficeWork, CommandLine, Idle, General

### Tier 3 — LLM Teacher (Optional, Async)

| Module | Responsibility |
|--------|---------------|
| `llm.rs` | Cloud-based policy refinement via OpenAI-compatible API |

The LLM teacher operates on a fundamentally different time scale — it doesn't make real-time decisions. Instead, it observes system patterns and updates the `LearnedPolicy` (lists of interactive, noise, and protected process patterns) that Tier 2 consumes. Rate-limited to 2 calls/hour with a 15-minute minimum interval.

---

## 4. Binary Architecture

### `apollo-optimizer` (CLI) — `src/main.rs`

One-shot entry point. Subcommands:

| Command | Action |
|---------|--------|
| `snapshot` | Collect system metrics → save JSON |
| `optimize` | Run optimization engine once |
| `clean` | Disk cleanup |
| `turbo` | Maximum performance mode (disable animations, extreme tuning) |
| `daemon` | Start continuous optimization loop |
| `startup` | Configure smart startup (prevent app reopening) |
| `llm` | Aggressive optimization for AI/LLM workloads |
| `restore` | Reverse all optimizations, unfreeze all |

### `apollo-optimizerd` (Daemon) — `src/bin/apollo-optimizerd.rs`

Long-running background service (3,082 lines). Core responsibilities:

- **Main optimization loop** with adaptive tick rate (2s–60s)
- **Unix socket server** for IPC with `apollo-optimizerctl`
- **Reactor thread** for event-driven response (kqueue)
- **State persistence** across restarts (7 state files)
- **Profile governor** with automatic transitions
- **Post-wake grace period** (60s after sleep/wake)
- **Kill switch** (`/var/run/apollo.disable`)

**SharedState** (Arc<Mutex<...>>):
```rust
SharedState {
    // Profile & policy
    profile: OptimizationProfile,
    latency_target: LatencyTarget,
    governor: ProfileGovernor,

    // Process tracking
    frozen: HashSet<u32>,           // Currently frozen PIDs
    frozen_since: HashMap<u32, DateTime>,
    last_blockers: Vec<BlockerScore>,

    // System state
    thermal_state: String,
    throttle_level: String,
    wake_state: WakeRuntimeState,

    // Intelligence modules
    adaptive_governor: AdaptiveGovernor,
    workload_classifier: WorkloadClassifier,
    user_profile: UserProfile,
    llm_state: LlmState,
    learned_policy: LearnedPolicy,
    usage_model: UsageModel,

    // Hardware interfaces
    mach_qos: MachQoSManager,
    iokit_reader: IOKitSensorReader,

    // Metrics & reactor
    metrics: RuntimeMetrics,        // 50+ counters
    reactor_event_weight: f64,
    reactor_mode: String,
    reactor_health: String,
}
```

### `apollo-optimizerctl` (Client) — `src/bin/apollo-optimizerctl.rs`

Lightweight CLI client (218 lines). Connects to daemon socket, sends JSON requests, displays responses.

**Wire Protocol** (tagged JSON):
```json
// Request
{"type": "SetProfile", "payload": {"profile": "aggressive-root", "ttl_minutes": 60}}

// Response
{"type": "Ok"}
```

---

## 5. Core Engine Modules

The engine comprises 27 modules in `src/engine/`. Organized by function:

### Types & Protocol (3 modules)

| Module | Lines | Key Types |
|--------|-------|-----------|
| `types.rs` | 356 | `OptimizationProfile`, `RootAction`, `SafetyPolicy`, `RuntimeMetrics`, `BlockerScore`, `DaemonStatus` |
| `protocol.rs` | 80 | `DaemonRequest` (23 variants), `DaemonResponse` (12 variants) |
| `journal.rs` | ~60 | `JournalEntry` — append-only JSONL audit trail |

### Safety & Capabilities (2 modules)

| Module | Key Functions |
|--------|--------------|
| `safety.rs` | `protected_processes()`, `critical_background_processes()`, `allowlisted_sysctls()`, `enforce_limits()` |
| `capabilities.rs` | `can_taskpolicy()`, `can_sysctl()`, `can_memorystatus()`, `can_mdutil()`, `is_root()` |

### Decision & Execution (3 modules)

| Module | Lines | Purpose |
|--------|-------|---------|
| `decide_actions.rs` | 271 | Context classification → blocker detection → action generation |
| `execute_actions.rs` | 239 | Process existence validation → signal/renice/sysctl execution |
| `profile_governor.rs` | 349 | Pressure scoring → profile transitions → anti-thrash → overrides |

### Intelligence (6 modules)

| Module | Lines | Purpose |
|--------|-------|---------|
| `adaptive_governor.rs` | 352 | Central heuristic decision engine (Tier 1) |
| `process_classifier.rs` | 270 | 8-tier process categorization |
| `zombie_hunter.rs` | 274 | 5-class dead-weight detection |
| `workload_classifier.rs` | ~400 | Bayesian workload classification (Tier 2) |
| `user_profile.rs` | 435 | Behavioral learning: app stats, hour model, session history |
| `llm.rs` | 564 | Cloud LLM advisor (Tier 3) |

### Hardware & System (4 modules)

| Module | Lines | Purpose |
|--------|-------|---------|
| `iokit_sensors.rs` | 305 | Hardware telemetry via `powermetrics` (temps, power, utilization) |
| `mach_qos.rs` | 279 | Mach kernel QoS classes: P-Core vs E-Core scheduling |
| `thermal_manager.rs` | 229 | Predictive thermal management with 60-sample history |
| `power_management.rs` | 331 | Battery modes, power estimation, critical actions |

### Specialized Subsystems (9 modules)

| Module | Purpose |
|--------|---------|
| `memory_analyzer.rs` | RSS/VMS/WSS profiling, leak detection (>70% growth → leak) |
| `swap_predictor.rs` | Linear swap forecasting (30s ahead), time-to-critical |
| `gpu_manager.rs` | GPU power states, workload-specific optimization |
| `network_optimizer.rs` | TCP tuning profiles (HighThroughput, LowLatency, Balanced, Battery) |
| `wake_storm_detector.rs` | Wakeup rate anomaly detection (>10/sec = storm) |
| `process_recovery.rs` | Automatic kill/restart of memory-leaking processes |
| `analytics.rs` | Cumulative impact metrics, energy/CO₂ estimates |
| `usage_model.rs` | Process usage tracking for `usage top` / `usage explain` |

---

## 6. Decision Pipeline

```
                              ┌──────────────────────┐
                              │   System Snapshot     │
                              │   (sysinfo + IOKit)   │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Context Classification│
                              │                        │
                              │  CPU>88% OR Mem>90%    │
                              │    → ThermalConstrained │
                              │  CPU>72% OR Mem>78%    │
                              │    → BackgroundPressure │
                              │  Otherwise             │
                              │    → InteractiveFocus   │
                              └──────────┬───────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
         ┌──────────▼──────────┐        │         ┌──────────▼──────────┐
         │  Blocker Detection  │        │         │  Process Classifier  │
         │                     │        │         │                      │
         │  Wait-graph scoring:│        │         │  8 tiers × utility   │
         │  interactive_wait   │        │         │  score → per-process │
         │  × 0.45 +           │        │         │  decisions           │
         │  cpu_spike × 0.35 + │        │         │                      │
         │  seen_recent × 0.10 │        │         │  Zombie Hunter runs  │
         │  + reactor × 0.10   │        │         │  in parallel         │
         │                     │        │         │                      │
         │  score > 0.30       │        │         └──────────┬──────────┘
         │  → Boost            │        │                    │
         └──────────┬──────────┘        │                    │
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Workload Classifier  │
                              │  (Bayesian, Tier 2)   │
                              │                        │
                              │  Confirms/adjusts      │
                              │  aggression level      │
                              │  based on workload     │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Safety Enforcement   │
                              │                        │
                              │  • Protected processes │
                              │  • Action budgets      │
                              │  • Sysctl allowlist    │
                              │  • Critical bg procs   │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Execute Actions      │
                              │                        │
                              │  PID validation →      │
                              │  taskpolicy / renice / │
                              │  SIGSTOP / SIGCONT /   │
                              │  sysctl -w / mdutil    │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Journal + Metrics    │
                              │                        │
                              │  Append JSONL audit →  │
                              │  Merge lock-free       │
                              │  counters              │
                              └──────────────────────┘
```

### Action Types (RootAction enum)

| Action | Mechanism | Reversible |
|--------|-----------|------------|
| `BoostProcess` | `taskpolicy -l 0 -t 0` + `renice -10` | Yes (renice 0) |
| `ThrottleProcess` | `taskpolicy -l {2\|4} -d 4` + `renice {+10\|+20}` | Yes (renice 0) |
| `FreezeProcess` | `taskpolicy -d 4` + `SIGSTOP` | Yes (SIGCONT) |
| `UnfreezeProcess` | `SIGCONT` | N/A |
| `SetSysctl` | `sysctl -w key=value` (allowlist only) | Yes (stored before) |
| `SetMemorystatus` | `sysctl kern.memorystatus_vm_pressure_send=PID` | Yes |
| `ToggleSpotlight` | `mdutil -i {on\|off} /` | Yes |
| `QuarantineDaemon` | I/O demotion + CPU throttle | Yes |

---

## 7. Safety & Constraint System

### Protected Processes (never touched)

```
kernel_task   launchd       WindowServer   loginwindow
configd       securityd     tccd           syspolicyd
notifyd       hidd          UserEventAgent
Spotlight     mds           mds_stores     mdworker     mdworker_shared
```

### Critical Background Processes (throttle lightly, never freeze)

```
podman   docker   colima   qemu-system          // Containers
postgres mysqld   redis-server   mongod          // Databases
node     python   java     nginx                 // Dev servers
go       ruby     php                            // Language runtimes
```

### Allowlisted Sysctls (16 tunable parameters)

```
net.inet.tcp.sendspace          net.inet.tcp.recvspace
net.inet.tcp.delayed_ack        net.inet.tcp.win_scale_factor
net.inet.tcp.autorcvbufmax      net.inet.tcp.autosndbufmax
vm.compressor_poll_interval     vm.compressor_sample_min
kern.maxvnodes                  kern.maxfiles
kern.ipc.somaxconn              kern.ipc.maxsockbuf
iogpu.wired_limit_mb            debug.iogpu.wired_limit
debug.lowpri_throttle_enabled   kern.memorystatus_vm_pressure_send
```

### Action Budgets Per Cycle

| Profile | Boosts | Throttles | Hints | Freezes | Cooldown |
|---------|--------|-----------|-------|---------|----------|
| AggressiveRoot | 10 | 20 | 12 | 8 | 10s |
| BalancedRoot | 6 | 12 | 6 | 4 | 20s |
| SafeRoot | 3 | 6 | 3 | 2 | 45s |

### Safety Invariants

1. Never freeze system-critical processes — checked against `protected_processes()`
2. Never freeze critical background work — checked against `critical_background_processes()`
3. All external commands use `std::process::Command` — no shell injection
4. Sysctl writes strictly allowlisted — only 16 keys writable
5. Profile transitions cooldown 90 seconds — prevents rapid oscillation
6. Anti-thrash: >4 transitions in 10 min → lock to BalancedRoot for 5 min
7. Developer floor: never drop to SafeRoot during active dev/interactive sessions
8. Post-wake grace: 60s of suppressed aggression after system wake
9. LLM patterns sanitized: max 80 chars, no newlines, no Spotlight keywords, confidence ≥ 0.80
10. All frozen PIDs tracked in `frozen_state.json` — unfrozen on daemon restart

---

## 8. Reactive Nervous System

The reactor thread provides event-driven response via macOS kernel facilities:

```
┌─────────────────────────────────────────────────────────┐
│                   kqueue Event Loop                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Nerve 1: EVFILT_VM (Memory Pressure)                   │
│    └─ NOTE_VM_PRESSURE → immediate re-optimization      │
│                                                          │
│  Nerve 2: Darwin Notification (Thermal)                  │
│    └─ com.apple.system.thermalpressurelevel              │
│    └─ Temperature change → thermal cascade               │
│                                                          │
│  Nerve 3: Darwin Notification (Lifecycle)                │
│    └─ com.apple.launchd.spawn                            │
│    └─ New process launch → classify & decide             │
│                                                          │
│  Nerve 4: Darwin Notification (Power)                    │
│    └─ com.apple.system.powersources.source               │
│    └─ AC plug/unplug → power mode switch                 │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  On any event:                                           │
│    1. Increment event counters                           │
│    2. Set fast_tick_until (accelerate main loop to 2s)   │
│    3. Collect fresh snapshot                              │
│    4. Run optimization cycle immediately                 │
└─────────────────────────────────────────────────────────┘
```

**Adaptive Tick Rate:**
- Normal: 60s between cycles
- Pro workload detected: 15s
- Reactor event fired: 2s (for fast_tick_duration)

---

## 9. Profile Governor State Machine

```
                    ┌──────────────────────────┐
                    │      SafeRoot            │
                    │  (conservative: 3/6/2)   │
                    └──────────┬───────────────┘
                               │
                    pressure ≥ 0.40 │ 3 consecutive
                    ───────────────►│
                               │◄──────────────
                    pressure ≤ 0.28 │ 6 consecutive
                               │
                    ┌──────────▼───────────────┐
                    │     BalancedRoot          │
                    │  (default: 6/12/4)       │
                    └──────────┬───────────────┘
                               │
                    pressure ≥ 0.72 │ 3 consecutive
                    ───────────────►│
                               │◄──────────────
                    pressure ≤ 0.55 │ 6 consecutive
                               │
                    ┌──────────▼───────────────┐
                    │    AggressiveRoot         │
                    │  (max: 10/20/8)          │
                    └──────────────────────────┘

    Anti-thrash: >4 transitions in 10 min → lock BalancedRoot for 5 min
    Developer floor: dev session active → never drop below BalancedRoot
    Manual override: user-set profile with TTL (expires automatically)
```

**Pressure Score Formula:**
```
score = 0.35 × cpu_pressure
      + 0.35 × ram_pressure
      + 0.20 × interactive_wait_ratio
      + 0.10 × reactor_event_weight
```

---

## 10. LLM Teacher Integration

```
┌──────────────────────────────────────────────────────────────┐
│                    LLM Teacher Mode                           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Configuration:                                               │
│    model: gpt-4.1-mini (OpenAI-compatible)                   │
│    min_confidence: 0.80                                       │
│    max_calls_per_hour: 2                                      │
│    min_interval: 15 minutes                                   │
│    timeout: 5 seconds                                         │
│    training_window: 2 weeks (configurable)                    │
│                                                               │
│  Input (system summary):                                      │
│    CPU pressure, memory state, thermal level,                │
│    top 10 processes, current profile, current patterns        │
│                                                               │
│  Output (structured JSON):                                    │
│    suggest_profile: OptimizationProfile                       │
│    suggest_latency_target: LatencyTarget                      │
│    add_interactive_patterns: Vec<String>  (max 6)            │
│    add_noise_patterns: Vec<String>  (max 6)                  │
│    add_protected_patterns: Vec<String>  (max 6)              │
│    confidence: f64  (must be ≥ 0.80)                         │
│    rationale: String                                          │
│                                                               │
│  Safeguards:                                                  │
│    • Spotlight patterns NEVER accepted                        │
│    • Patterns sanitized: max 80 chars, no newlines            │
│    • Max 6 patterns per category per call                     │
│    • Confidence gate: ≥ 0.80 required                         │
│    • Rate limiting: 2 calls/hour, 15 min minimum interval     │
│    • Training window expires → reverts to heuristics only     │
│                                                               │
│  Storage:                                                     │
│    /var/lib/apollo/learned_policy.json  (600, root:root)     │
│    /var/lib/apollo/suggestions.jsonl    (LLM response log)   │
│    /var/lib/apollo/feedback.jsonl       (user rating log)    │
│    /var/lib/apollo/llm_key_secret      (600, root:root)     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 11. Data Flow & Persistence

### State Files (root: `/var/lib/apollo/`, non-root: `/tmp/`)

| File | Format | Purpose | Update Frequency |
|------|--------|---------|------------------|
| `journal.jsonl` | JSONL (append) | Audit trail of every action (before/after) | Every action |
| `runtime_metrics.json` | JSON | 50+ counters (boosts, freezes, thermal, etc.) | Every cycle |
| `governor_state.json` | JSON | Active profile, cooldowns, transition counts | On transition |
| `profile_timeline.jsonl` | JSONL (append) | History of profile switches | On transition |
| `frozen_state.json` | JSON | Currently frozen PIDs | Every freeze/unfreeze |
| `wake_state.json` | JSON | Wake/sleep event tracking | On wake/sleep |
| `learned_policy.json` | JSON | ML-learned process patterns (37 patterns) | On LLM update |
| `usage_model.json` | JSON | Per-process usage statistics | Periodic |
| `suggestions.jsonl` | JSONL (append) | LLM suggestion history | On LLM call |
| `feedback.jsonl` | JSONL (append) | User feedback ratings | On user feedback |

### Metrics Tracked (RuntimeMetrics — 50+ fields)

**Optimization counters:** cycles, boosts_applied, throttles_applied, freezes_applied, unfreezes_applied, paging_hints_applied, sysctl_applied

**Safety counters:** failures, invalid_sysctl_denied, critical_background_skips, heuristic_kills_downgraded

**System state:** effective_profile, thermal_state, throttle_level, current_workload, ml_confidence

**Thermal/power:** iokit_p_cluster_temp, iokit_e_cluster_temp, iokit_package_watts

**Reactor:** reactor_pulses, reactor_mode, reactor_health, reactor_events_total

**Survival:** survival_mode_activations, kills_applied, zombies_detected

---

## 12. Hardware Telemetry Layer

### IOKit Sensor Reader (`iokit_sensors.rs`)

Data gathered via `powermetrics` (requires root):

| Sensor | Source |
|--------|--------|
| P-Cluster temperature | Firestorm cores (performance) |
| E-Cluster temperature | Icestorm cores (efficiency) |
| GPU temperature | Apple GPU |
| NAND temperature | Storage controller |
| Package power (W) | Total SoC power draw |
| CPU power (W) | CPU subsystem |
| GPU power (W) | GPU subsystem |
| DRAM power (W) | Memory subsystem |
| P-Core utilization (%) | Performance core load |
| E-Core utilization (%) | Efficiency core load |
| Battery charge (%) | Current battery level |
| Discharge rate (W) | Battery drain rate |

### Mach QoS Manager (`mach_qos.rs`)

Direct kernel scheduling control via `task_policy_set()`:

| QoS Class | Target | Effect |
|-----------|--------|--------|
| USER_INTERACTIVE | P-Cores (Firestorm) | Maximum throughput, lowest latency |
| USER_INITIATED | P-Cores, lower priority | High throughput |
| DEFAULT | Scheduler decides | Balanced |
| UTILITY | Mixed cores | Reduced energy impact |
| BACKGROUND | E-Cores (Icestorm) only | Throttled I/O, minimal energy |

---

## 13. Module Reference Table

| Module | Type | Latency | Output | Used By |
|--------|------|---------|--------|---------|
| `adaptive_governor` | Decision | <1ms | Allow/Throttle/Freeze/Kill per process | Daemon loop |
| `process_classifier` | Classification | <2ms | 8-tier categorization + utility score | AdaptiveGovernor |
| `zombie_hunter` | Detection | <5ms | Dead-weight classification + 3-cycle confirm | AdaptiveGovernor |
| `workload_classifier` | ML Classification | <1ms | Workload type + confidence (Bayesian) | AdaptiveGovernor |
| `user_profile` | Learning | <1ms | App stats, hour model, session history | WorkloadClassifier |
| `profile_governor` | State Machine | <1ms | Effective profile + transition decisions | Daemon loop |
| `decide_actions` | Pipeline | <5ms | Vec<RootAction> for execution | Daemon loop |
| `execute_actions` | Execution | <10ms | Outcomes (counters, errors, skips) | Daemon loop |
| `safety` | Validation | <1ms | Budget enforcement, protected lists | DecideActions, ExecuteActions |
| `capabilities` | Detection | <1ms | System capability report | Daemon startup |
| `thermal_manager` | Monitoring | <1ms | ThermalState + prediction + trend | AdaptiveGovernor, PowerManager |
| `memory_analyzer` | Profiling | <5ms | Per-process leak probability, WSS | ProcessRecoveryManager |
| `swap_predictor` | Forecasting | <1ms | Swap trend + time-to-critical | Daemon loop |
| `gpu_manager` | Monitoring | <1ms | GPU power state + recommendations | ThermalManager |
| `power_management` | Control | <1ms | Battery mode + power estimation | Daemon loop |
| `network_optimizer` | Tuning | <1ms | TCP sysctl recommendations | SysctlTuner |
| `wake_storm_detector` | Detection | <1ms | Storm severity + mitigation actions | ZombieHunter |
| `process_recovery` | Recovery | <1ms | Kill targets with backoff schedule | Daemon loop |
| `analytics` | Reporting | <5ms | Energy saved (Wh), CO₂ avoided (g) | Status queries |
| `iokit_sensors` | Telemetry | ~500ms | Hardware temps, power, battery | ThermalManager, PowerManager |
| `mach_qos` | Scheduling | <1ms | Mach task_policy_set (P/E core routing) | Daemon loop |
| `llm` | Cloud Advisory | 1–10s | Learned patterns, profile suggestions | Async (rate-limited) |
| `journal` | Audit | <1ms | Append-only action log | ExecuteActions |
| `usage_model` | Analytics | <1ms | Per-process usage statistics | Status queries |
| `protocol` | IPC | <1ms | JSON wire format (23 requests, 12 responses) | Daemon ↔ Client |
| `types` | Data | N/A | Core structs, enums, constants | Everything |

---

## Build & Deployment

### Compilation

```bash
cargo build --release     # LTO + native CPU + panic=abort
```

Produces three binaries:
- `target/release/apollo-optimizer` (CLI)
- `target/release/apollo-optimizerd` (Daemon)
- `target/release/apollo-optimizerctl` (Client)

### Installation (launchd)

```bash
./scripts/install-root-daemon.sh
```

Installs:
- `/usr/local/libexec/apollo-optimizerd` (daemon binary)
- `/usr/local/bin/apollo-optimizerctl` (client binary)
- `/Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist` (launchd service)
- `/var/lib/apollo/` (state directory, mode 700)
- `/etc/apollo-optimizer/config.toml` (configuration, mode 600)

### Test Suite

228 tests across 10 levels:

| Level | Focus | Tests |
|-------|-------|-------|
| 1 | Unit tests, safety bounds, EMA convergence | ~20 |
| 2 | Integration: safety module, action enforcement | ~25 |
| 3 | Concurrent action handling, race conditions | ~20 |
| 4 | Advanced constraints, edge cases | ~15 |
| 5 | Tier 1 heuristic features | ~25 |
| 6 | Tier 2 ML workload classification | ~20 |
| 7 | Tier 3 LLM teacher mode | ~25 |
| 8 | Adaptive governor, recovery | ~30 |
| 9 | M1 macOS-native features (QoS, sensors) | ~20 |
| 10 | Bayesian workload classifier, learned policies | ~28 |

```bash
cargo test                    # Run all 228 tests
cargo test level1             # Run specific level
```

---

*This document describes the architecture as of commit `09b0924` (2026-03-03).*
