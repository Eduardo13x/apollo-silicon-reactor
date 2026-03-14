# Apollo Optimizer — Agent Coordination Context

## Active Sprint: Hardening & Stability (2026-03-05)

### TOP 10 ISSUES TO FIX (Priority Order)

| # | Issue | Owner | Status | File(s) |
|---|-------|-------|--------|---------|
| 1 | frozen/frozen_since desync (2 sets, 1 state) | Agent-Daemon | PENDING | apollo-optimizerd.rs |
| 2 | Socket symlink TOCTOU → privilege escalation | Agent-Persistence | PENDING | llm.rs, apollo-optimizerd.rs |
| 3 | read_line() sin limite + policy string sin limite | Agent-Daemon | PENDING | apollo-optimizerd.rs |
| 4 | Sysctl values sin validacion (kern.maxfiles=1) | Agent-Safety | PENDING | execute_actions.rs, safety.rs |
| 5 | Policy substring bypass ("kernel_tas" evade) | Agent-Safety | PENDING | apollo-optimizerd.rs (validation) |
| 6 | kqueue sin timeout en reactor | Agent-Daemon | PENDING | apollo-optimizerd.rs |
| 7 | Thermal thresholds desalineados (manager vs interrupt) | Agent-Thermal | PENDING | thermal_manager.rs, thermal_interrupt.rs |
| 8 | Journal sin rotation → disk exhaustion | Agent-Persistence | PENDING | journal.rs |
| 9 | Atomic ordering (Relaxed→Release en sequence) | Agent-Thermal | PENDING | thermal_interrupt.rs |
| 10 | Background collectors sin watchdog | Agent-Persistence | PENDING | background_collectors.rs, smc_reader.rs |

### FILE OWNERSHIP (prevents conflicts)

- **Agent-Daemon** (apollo-optimizerd.rs): Issues #1, #3, #6, plus integration of #2/#5
- **Agent-Safety** (safety.rs, execute_actions.rs, process_identity.rs): Issues #4, #5
- **Agent-Thermal** (thermal_interrupt.rs, thermal_manager.rs): Issues #7, #9
- **Agent-Persistence** (llm.rs, journal.rs, background_collectors.rs, smc_reader.rs): Issues #2, #8, #10
- **Agent-Types** (types.rs): Shared struct changes — coordinated

### RULES
1. Each agent ONLY edits files in their ownership
2. If a change requires another file, document it in AGENT_HANDOFF.md
3. NO cargo build/test — the orchestrator does that
4. Use `lock_recover()` for all mutex access (trait in lock_ext.rs)
5. Read files before editing
6. Keep changes minimal and focused

### ARCHITECTURE REMINDERS
- Daemon loop: ~3500 lines in apollo-optimizerd.rs
- SharedState: ~60 Arc<Mutex<T>> fields (line 157-217)
- Reactor: kqueue thread (line 370-610)
- Socket server: line 1819-1860
- Main loop: line 2362-3500+
- LockRecover trait: src/engine/lock_ext.rs
- Protected processes: src/engine/safety.rs
