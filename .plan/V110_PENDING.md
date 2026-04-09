# v1.1.0 — Pendiente post v1.0.0

Estado al 2026-04-03. Dos targets duros que quedaron sin cerrar en v1.0.0.

---

## Target 1 — main.rs ≤4100L (hoy: 4962L, gap: −862L)

Estrategia: extraer 5 secciones identificadas (~1321L disponibles → llevaría main.rs a ~3641L).

| Módulo nuevo | LOC aprox | Líneas fuente | Qué mueve |
|---|---|---|---|
| `daemon_process_collector.rs` | ~641 | 1500–2140 | Árbol de procesos, enriquecimiento GUI/net/CPU, memory scan top-50 |
| `daemon_freeze_executor.rs` | ~291 | 4109–4400 | TTL unfreeze, confirmación candidatos 2+ ciclos, budget enforcement |
| `daemon_action_safety.rs` | ~183 | 3278–3400, 4046–4107 | Sysctl governor, filtrado de acciones seguras |
| `daemon_wake_handler.rs` | ~127 | 1124–1250 | Post-wake grace, wake state management |
| `daemon_turbo_manager.rs` | ~79 | 1252–1330 | Display-off turbo freeze/unfreeze |

**Riesgo**: `daemon_process_collector.rs` (641L) tiene muchas dependencias — probablemente necesita pase de fix post-extracción.

**Lo que NO se mueve**: control flow del hot loop (cycle_count, condvar wait, last_cycle_instant), guardas de lock ordering sobre SharedState, reactor pulse monitoring.

---

## Target 2 — Tests ≥2500 (hoy: 2263, gap: +237)

Los 4 módulos extraídos en v1.0.0 tienen 0 tests sobre 1763L de código:

| Módulo | LOC | Tests hoy | Dificultad |
|---|---|---|---|
| `socket_handler.rs` | 878 | 0 | Alta — depende de SharedState completo |
| `metrics_reporter.rs` | 385 | 0 | Media |
| `learning_tick.rs` | 373 | 0 | Alta — depende de SharedState + LearningContext |
| `daemon_init.rs` | 127 | 0 | Baja |

**Riesgo**: tests de módulos bin requieren instanciar `SharedState` (muchos `Arc<Mutex<>>`). Puede que el resultado real sea 150–180 tests nuevos en vez de 237 si el setup es muy costoso.

---

## Target 3 — Workspace Split (opcional, v1.2.0)

Mueve `src/engine/` → `crates/apollo-engine/` como crate separado.
- **Beneficio principal**: `cargo test -p apollo-engine learning_pipeline` → 3–5 min vs 20 min
- **NO reduce main.rs directamente** — es fix de compilación, no de LOC
- Plan detallado: `.plan/WORKSPACE_SPLIT.md`
- Bloqueante: ninguno. Se puede hacer después de cerrar targets 1 y 2.

---

## Estrategia de ejecución

Atacar Target 1 y Target 2 en paralelo con agentes simultáneos:
- Wave 1: 3 agentes para extracciones pequeñas (daemon_action_safety, daemon_wake_handler, daemon_turbo_manager)
- Wave 2: 2 agentes para las grandes (daemon_process_collector, daemon_freeze_executor) + agente de tests
- Fix pass: un agente para resolver errores de compilación post-extracción

Guard: `cargo check --tests` (NO `cargo test` — tarda 20 min)

---

## Sensor-Consumer Debts — 2026-04-08 god-sensor session

The 2026-04-08 session added five new sensor axes (`VmRate`,
`thrashing_score`, `CpuSaturation`, per-process `cpu_contention`,
system-wide `stall_fraction`). All five are produced, persisted and
exposed via `RuntimeMetrics` + `StabilityOracle` → RL reward. Two
proposed decision-path consumers were DELIBERATELY deferred after
analysis — the sensors are in place but the behavioural wiring
requires empirical validation before it can be committed without
regressing existing decisions.

### DEBT-SENSOR-01 — Boost protected pid on high `cpu_contention`  ✅ CLOSED 2026-04-09

**Originally proposed:** if a foreground-family pid has
`cpu_contention > 0.6` sustained for N cycles, promote it to
`SchedulingTier::Foreground` via `mach_qos.set_tier()`.

**Why originally deferred:** the foreground family is ALREADY
promoted to Foreground by the existing `boost_foreground_family`
path. A second contention-triggered boost is a no-op for the
foreground set, and there was no "protected but not foreground"
concept to give it meaning.

**How actually closed (Phase 4 of the 2026-04-08 plan):** the
deferral note mis-framed where the value of `stall_fraction`
lives. The right consumer is NOT a boost on the protected pid
(no-op) but an aggressive-throttle override on the
NON-interactive pids. When the system is CPU-stalled at the
system level, the only lever to free CPU for the protected set
is to push the non-interactive tail to E-cores harder. Wired in
`decide_actions::decide_actions`:

```
let system_cpu_stalled = contention_tracker::global()
    .lock().stall_fraction(0.85) >= 0.5;
...
let aggressive = aggressive || system_cpu_stalled;
```

When the gate fires, every background-noise pid this cycle is
treated as "aggressive throttle" (Background QoS / E-cores)
regardless of context, freeing P-core headroom for the
protected set on the next quantum. Behaviour-preserving when
the tracker is empty (stall_fraction = 0 → no-op).

### DEBT-SENSOR-02 — Skip freeze of a CPU-starved candidate

**Proposed:** in `decide_actions` extreme-pressure freeze path, skip
candidates whose `cpu_contention > 0.7` — the scheduler is already
refusing them CPU so freezing adds no throughput benefit.

**Why deferred:** freezing reclaims MEMORY, not CPU. A starved pid
still holds its RSS, and memory pressure is the reason the freeze
path triggered. Skipping a starved-but-RSS-heavy pid would leave its
memory in residency and defeat the whole purpose of the branch. The
proposed consumer was structurally wrong on reflection — the sensor
does not add decision value in this specific branch.

**Unblocks when:** a different decision path (e.g. QoS tiering or
boost eligibility) needs contention-awareness. The
`ContentionTracker::stall_fraction()` global aggregate is the more
likely real consumer.

### Closure rationale

Both debts are converted from "silent promise in commit message
footer" to "tracked decision to defer with reasoning", which under
the apollo-evolve divergence stop rule counts as CLOSES because the
structural debt (unstructured open-ended promise) is replaced by
a bounded decision record.
