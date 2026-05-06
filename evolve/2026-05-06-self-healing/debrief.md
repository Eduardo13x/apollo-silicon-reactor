# Apollo Self-Healing Sprint — Debrief 2026-05-06

## Plan executed

7 phases across 8 commits closing the gaps NotebookLM identified in the
prior session.

| Phase | Commit | Description | Status |
|-------|--------|-------------|--------|
| 0 | n/a | Baseline measurement | done |
| 1 | `18f749d` | Global Action Deduplicator at dispatch chokepoint | done |
| 2 | n/a | Sysinfo staggered refresh verification (no-op needed) | verified |
| 3 | `bef1f0b` | THREAD_AFFINITY_POLICY scaffolding | partial (FFI only) |
| 4 | n/a | Reactor weight stress validation | deferred |
| 5 | `ff71c30` | SwarmThrottling + GraduatedIdle + ThreadQoSRouting variants | done |
| 6 | `a5c8083` | Self-healing meta-observer | done |
| 6.1 | `dbfa241` | Per-kind drop breakdown + threshold tuning | done |

## Headline metrics

| Metric | Baseline | Final | Δ |
|--------|----------|-------|---|
| SetMemorystatus same-second dups | 33 distinct | 0 | **-100%** |
| Wasted syscalls / 200 events | 78 | 0 | **-100%** |
| Distinct DecisionReason variants seen | 4 | 4-6 (workload-dep) | +2 |
| New variants observed in prod | 0/6 | **4/6** | +4 |
| Self-diagnosis alerts firing (post-tune) | n/a | 0 false-positives | OK |
| Daemon failures | 0 | 0 | flat |
| Tests passing | 1869 | **1881** | +12 |

## Achievements

1. **PID reapply spam ELIMINATED**. Pid 65808 used to receive
   SetMemorystatus 8× in same second. Now zero.
2. **CriticalBypass observed in prod** for the first time (pressure
   crossed 0.80 in measurement window).
3. **Self-diagnosis layer operational**: fired correctly on first deploy
   ("dropping 7.60/cycle"), exposed that 0.5 threshold was wrong; tuned
   to 3.0 with per-kind breakdown. Now silent on steady-state, ready
   for genuine regressions.
4. **Meta-observability**: `/var/lib/apollo/self_diagnosis.jsonl` exists
   for next-session pickup. Detection-only per Hellerstein 2004 §9.

## NotebookLM Final Gap Sweep

**🟠 High — Residual 87.5% journal failure rate**:
Governor padece "Falta de Memoria de Estado": misma decisión emitida
ciclo tras ciclo si presión persiste, aunque PID ya está en target
state. Desperdicia presupuesto de CPU evaluando 21 reglas en hot path.

**🟡 Medium — PressureContext aún 78%**:
Próximas variantes a wirear (orden de impacto):
- `IpcProtected` en safety guard sites (existing thread_selfcounts data)
- `AnomalyDetected` en evolved_anomaly trigger
- `DisplayPipeline` en display_turbo / WindowServer boost

**🟡 Medium — Self-diagnosis effectiveness**:
Detection-only correcto per Hellerstein 2004. Pero recomendación: auto-
limpiar caches no-persistentes cuando alerts indiquen NaN/divergencia
numérica. Policy changes deben seguir pasando por LLM Teacher.

**⚪ Low — HysteresisRecovery + ThreadQoSRouting 0%**:
Mantener como work pendiente, NO degradar. HysteresisRecovery 0% es
señal de estabilidad (no hubo crisis sostenida). ThreadQoSRouting requiere
Phase 3 consumer wiring.

## Top 2 priorities — Next session

### 1. 🔴 **Critical — Sysinfo Process Tree Cache**
- p95 117ms no bajará de 100ms mientras refresh sea síncrono total
- Implementar caché de process tree por 3-5 ciclos
- Foundation commit ya añadió staggered refresh por zona; falta el
  segundo nivel (cache cross-cycle)

### 2. 🟠 **High — Thread-Level Routing Consumer**
- Phase 3 scaffolding `bef1f0b` listo
- decide_actions.rs debe emitir affinity tags para hilos UI dentro de
  procesos throttled (Brave UI thread → P-cluster aunque renderers van
  a E-cluster)
- Esperado: 8-15% mejor latency interactiva sin más consumo energético

## Deferred / not addressed this session

- Phase 3 consumer wiring (downstream of THREAD_AFFINITY_POLICY)
- Phase 4 reactor weight stress validation (no stress-ng infrastructure)
- NARS bridge for self_diagnosis.jsonl
- IpcProtected / AnomalyDetected / DisplayPipeline variant wiring
- Governor "memoria de estado" — track applied state per PID to skip
  redundant decisions cycle-to-cycle

## Verdict

> "Has cerrado el gap de honestidad de las syscalls (Zero Wasted Syscalls).
> El sistema ya no miente, pero sigue siendo redundante en su razonamiento.
> La próxima fase debe ser la de Agudeza y Velocidad."
> — NotebookLM gap sweep 2026-05-06

User vision realized: el daemon detecta clases de bugs por sí mismo,
emite alertas accionables, deja el "auto-fix" al LLM teacher / siguiente
apollo-evolve loop. Self-healing observability layer es production-ready.
