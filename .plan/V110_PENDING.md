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

### DEBT-SENSOR-01 — Boost protected pid on high `cpu_contention`

**Proposed:** if a foreground-family pid has `cpu_contention > 0.6`
sustained for N cycles, promote it to `SchedulingTier::Foreground`
via `mach_qos.set_tier()`.

**Why deferred:** the foreground family is ALREADY promoted to
Foreground by the existing `boost_foreground_family` path. Adding a
contention-triggered second boost is either redundant (if already
foreground) or invasive (if the pid is not in the foreground family
but still "protected" by some other rule). The convergent wiring
requires distinguishing these two cases, which in turn requires
defining a "protected but not foreground" subset that does not
currently exist as a first-class concept.

**Unblocks when:** the decision pipeline introduces an explicit
"protected pid set" separate from the foreground family. Track the
data in `ProcessSnapshot.cpu_contention` until then.

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
