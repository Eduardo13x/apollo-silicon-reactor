# Apollo Optimizer: Technical Deep Dive
## The "Shadow Architecture" & Intent-Aware Scheduling

This document details the advanced internal mechanisms of Apollo Optimizer that go beyond traditional system maintenance. It describes how the engine interacts with the macOS XNU kernel and Apple Silicon hardware.

---

### 1. The Wait-Graph Deadlock Resolver (`wait_graph.rs`)
Unlike simple process managers, Apollo performs **Wait-Graph Introspection**. Before freezing a process with `SIGSTOP`, the engine must ensure it doesn't cause a system-wide deadlock.
- **Mach Thread Analysis**: It uses `proc_pidinfo` with `PROC_PIDLISTTHREADS` and `PROC_PIDTHREADINFO` to inspect the `pth_run_state` of every thread.
- **Deadlock Veto**: If a frozen process is in `TH_STATE_WAITING` and the candidate for freezing is a likely lock-holder (running threads), Apollo **vetos the freeze** or proactively **unfreezes the waiter** to prevent indefinite stalls.
- **Stuck-Detection**: Periodically identifies "stuck-frozen" PIDs that were caught mid-IPC and recovers them.

### 2. Compressor-Aware Memory Management (`compressor_aware.rs`)
Apollo optimizes how it handles memory based on the **content** of the RAM, using `task_info(TASK_VM_INFO)`.
- **Compression Ratio Heuristic**: It calculates `phys_footprint / (phys_footprint + compressed)`. 
- **Freeze vs. Hint**:
    - **High Ratio (Text/Data)**: Process is frozen via `SIGSTOP`. The kernel keeps it compressed in RAM, making recovery nearly instant.
    - **Low Ratio (Media/Encrypted)**: Instead of freezing (which would force expensive Swap I/O), Apollo sends a `PressureHint` via `memorystatus_control`, asking the app to release its internal caches safely.

### 3. Multi-Phase Thermal Bail-out (`thermal_bailout.rs`)
Traditional macOS thermal management throttles everything. Apollo implements a **4-Phase Graduated Cooling Strategy** with hysteresis:
- **Phase 1 (80-85°C) - Gentle**: Reduces background I/O and hints purgeable memory.
- **Phase 2 (85-90°C) - Moderate**: Forces all background tasks to E-Cores (Icestorm) and throttles GPU.
- **Phase 3 (90-95°C) - Aggressive**: Freezes all non-essential daemons and caps P-Core (Performance) utilization target to 40%.
- **Phase 4 (>95°C) - Emergency**: Freezes everything except protected system services and the active foreground app; P-Core cap drops to 10%.

### 4. Granular I/O Tiering (`io_tiering.rs`)
Apollo escapes the binary "boost/throttle" limitation by utilizing all 5 levels of Darwin's disk I/O priority via `taskpolicy -d`:
1. **Tier 0 (Interactive)**: Swap paging and active compilation.
2. **Tier 1 (Standard)**: Visible background apps.
3. **Tier 2 (Utility)**: Spotlight and Time Machine indexing.
4. **Tier 3 (Throttle)**: Silent daemons.
5. **Tier 4 (Passive)**: Completely deferrable telemetry; only executes if the SSD is idle.

### 5. UMA Scavenging & GPU Eco-Mode
Specifically for Apple Silicon, the engine monitors the **Unified Memory Architecture (UMA)**.
- **Texture Scavenging**: When UMA pressure > 72%, it identifies background processes with heavy GPU weighted scores and moves them to "Passive" policies to save bandwidth for the active workspace.
- **Holographic Optimization**: Simultaneously coordinates Wait-Graph, UMA Scavenging, and Preemptive Paging to maintain a "holographic" (multi-dimensional) view of system health.

### 6. OS Mutation (Turbo & LLM Modes)
When a professional workload is detected, Apollo "mutates" the operating system environment to reduce overhead:
- **UI Decapitation**: Disables `NSAutomaticWindowAnimationsEnabled` and `DisableAllAnimations` via `defaults write` to save GPU cycles.
- **Shadow Removal**: Strips window shadows to reduce `WindowServer` compositing load.
- **RAM Guardian**: Proactively restarts `Finder` and `Dock` if they breach specific memory ceilings (>500MB and >300MB respectively).

---

### Core Kernel Interfaces Used
- `task_policy_set()`: Fine-grained QoS and Core routing.
- `memorystatus_control()`: Direct Jetsam/Memory pressure injection.
- `proc_pidinfo()`: Low-level thread and wait-state introspection.
- `powermetrics`: Hardware-level wattage and temperature telemetry.
