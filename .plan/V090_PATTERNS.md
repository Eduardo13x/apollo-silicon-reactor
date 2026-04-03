# Patrones Aplicados — Apollo v0.9.0

Fecha de análisis: 2026-04-02
Base: v0.8.0 commit `9932d0a`

---

## Patrones YA implementados (reconocerlos)

| Patrón | Archivo | Estado | Notas |
|--------|---------|--------|-------|
| Circuit Breaker | `src/engine/circuit_breaker.rs` | Completo | Closed→Open→HalfOpen, sliding window, 8 tests |
| Bulkhead | `src/bin/apollo-optimizerd/main.rs` + `daemon_state.rs` | Parcial | 6 dominios en daemon_state.rs; main.rs aún usa ~40 Mutex planos (DEBT-004) |
| Health Check | `src/engine/daemon_state.rs` (ReactorStatus), `src/engine/protocol.rs` (GetHealth, Doctor) | Completo | "ok"/"stalled"/"degraded" health strings; GetHealth expone CB + degradación |
| Event Sourcing (lite) | `src/engine/journal.rs` | Completo | Append-only JSONL, rotación a 10MB, simlink protection; `read_journal()` reconstruye estado |
| Backpressure | `src/engine/action_queue.rs` | Completo | 3 tiers (Urgent/Normal/Background), `drain_cycle(max_per_cycle)`, `backpressure_ratio()` |
| Degradación graceful | `src/engine/degradation.rs` | Completo | 4 modos (Full/Conservative/Observe/Emergency); CB + kernel_task CPU gates |
| Rate Limiting (implícito) | `action_queue.rs` `max_per_cycle` | Completo | Limita acciones ejecutadas por ciclo; Urgent no tiene tope |
| Request/Response | `src/engine/protocol.rs` (DaemonRequest/DaemonResponse) | Completo | JSON con `type`/`payload` tags; versión de protocolo explícita |
| Fire & Forget (push) | `protocol.rs` Subscribe / `socket_handler.rs` `broadcast_current_status()` | Completo | El daemon hace push de StatusPush a suscriptores en cada ciclo |
| Feature Toggle (runtime) | Kill switch `/var/run/apollo.disable` | Completo | Presencia del archivo pausa la optimización |
| Materialized View | `src/engine/signal_intelligence.rs` SignalDigest | Completo | Vista precalculada de Kalman+CUSUM+Entropy+Hazard+MPC; el daemon no recalcula por separado |
| CQRS (lite) | `socket_handler.rs` (reads) vs `main.rs` hot loop (writes) | Parcial | Separación de comandos de lectura (GetStatus, GetMetrics) del path de escritura (freeze/throttle); sin bus de eventos formal |
| Idempotencia | `journal.rs` `append_journal()` | Parcial | Escrituras son append-only (idempotentes en caso de retry). Las acciones SIGSTOP/SIGCONT son idempotentes por diseño del kernel. No hay `deduplication key` explícita en el protocolo. |
| Timeout | `socket_handler.rs` (read timeout implícito por Unix socket) + `action_queue.rs` ciclo de drain | Parcial | No hay timeout explícito en DaemonRequest; el socket se cierra si el cliente desconecta. |
| Strangler Fig (iniciado) | `src/engine/pipeline/` (LearningContext, DecisionStage, PeriodicStage) | En progreso | 3 de N stages extraídos del monolito main.rs. main.rs sigue siendo 5524L. |
| Anti-Corruption Layer (lite) | `src/engine/pipeline/decision_stage.rs` PolicyContext | Parcial | `PolicyContext` traduce ~8 parámetros Bayesianos sueltos en un struct tipado; sin ACL formal entre capas |

---

## Patrones a implementar en v0.9.0 (High priority)

| Patrón | Problema que resuelve | PR sugerida | Archivo objetivo |
|--------|-----------------------|-------------|-----------------|
| **Bulkhead completo** | DEBT-004: SharedState plano con ~40 Arc<Mutex<>> independientes causa contención y hace imposible aislar fallos por dominio | PR #1 (DEBT-004) | `src/bin/apollo-optimizerd/main.rs` → migrar a `daemon_state.rs` |
| **Strangler Fig (continuar)** | DEBT-010: main.rs 5524L es un God Service. Los stages de pipeline existen pero quedan inline | PRs sucesivos post-DEBT-004 | `src/engine/pipeline/` + extracciones de main.rs |
| **Anti-Corruption Layer** | Durante la migración Strangler Fig, las referencias a SharedState plano y al nuevo agrupado coexisten. Sin ACL explícita, es fácil introducir regresiones silenciosas | Como parte de PR DEBT-004 | Wrapper `domain_access.rs` o métodos de acceso en `daemon_state.rs` |
| **Feature Toggle (compile-time)** | La migración SharedState necesita poder activar/desactivar el nuevo SharedState agrupado sin romper el daemon. Un `cfg(feature = "grouped-state")` permite comparación en staging antes de hacer el corte | Como parte de PR DEBT-004 | `Cargo.toml` + `#[cfg(feature)]` en main.rs |
| **Idempotencia explícita** | El protocolo DaemonRequest no tiene request-id ni dedup key. Si el cliente reintenta `SetProfile` o `LlmSetKey`, el efecto se aplica dos veces sin detección | PR independiente | `src/engine/protocol.rs` — agregar `request_id: Option<String>` + cache en socket_handler |
| **Timeout explícito** | No hay timeout configurable por request en el socket handler. Un cliente malicioso o colgado puede bloquear un thread del pool indefinidamente | PR independiente | `src/bin/apollo-optimizerd/socket_handler.rs` |
| **Dead Letter Queue** | Las acciones que fallan en `execute_actions` son registradas en el journal como `success: false`, pero no hay cola de reintentos separada. Acciones críticas (Unfreeze) que fallan se pierden | PR independiente | `src/engine/action_queue.rs` — añadir tier DLQ con reintentos bounded |

---

## Patrones para considerar en v1.0+ (Medium/Low)

| Patrón | Aplicabilidad a Apollo | Prioridad | Notas |
|--------|------------------------|-----------|-------|
| **Event Sourcing completo** | El journal.jsonl ES event sourcing, pero el estado en memoria no se reconstruye desde el journal — se carga desde archivos de snapshot independientes | Medium | Unificar con `learned_state.rs` como fuente única de verdad |
| **CQRS formal** | socket_handler hace reads, main loop hace writes, pero comparten el mismo SharedState. Un bus de eventos (mpsc) desacoplaría las lecturas | Low | Overkill para un daemon single-node; útil si se agrega API HTTP o múltiples clientes |
| **Saga** | Las optimizaciones compuestas (freeze + sysctl + QoS) son transacciones distribuidas. Si el paso 2 falla, el estado queda inconsistente | Medium | Implementar como `CompensatingTransaction` en execute_actions.rs: si el Sysctl falla, revertir el freeze previo |
| **Compensating Transaction** | Subpatrón de Saga. Actualmente el daemon desencola frozen_state en startup como cleanup, pero no como compensación inline | Medium | En execute_actions: en error de acción intermedia, emitir RootAction::Unfreeze automático |
| **Outbox** | Las escrituras al journal.jsonl pueden fallar (disco lleno, permisos). Si journal falla y la acción ya se ejecutó, hay inconsistencia | Low | Patrón Outbox: escribe al journal ANTES de ejecutar; marca como committed después |
| **Materialized View (expandir)** | SignalDigest ya es una MV. `DaemonStatus` y `HealthReport` se reconstruyen por request — podrían ser MVs mantenidas incrementalmente | Low | Reduce latencia de GetStatus en el socket handler |
| **Sidecar** | El `apollo-optimizerctl` ya actúa como sidecar: se comunica con el daemon principal sin compartir proceso. El patrón está bien aplicado. | Already done | Documentarlo como decisión arquitectónica intencional |
| **Backpressure → adaptativo** | ActionQueue tiene `backpressure_ratio()` calculado pero no hay consumidor que cambie el comportamiento basado en él. El ratio debería alimentar al Router adaptativo de SignalIntelligence | Medium | Wire: `action_queue.backpressure_ratio()` → `lctx.signal_intel` → router zone |
| **Leader Election** | No aplica — Apollo es un daemon single-instance por diseño. El kill switch (`apollo.disable`) cumple la función de "ceder liderazgo". | N/A | |
| **Service Discovery** | No aplica — socket path es fijo (`/var/run/apollo-optimizer.sock`). Una abstracción de discovery sería sobreingeniería. | N/A | |
| **Canary / Blue-Green** | El `LaunchD` permite cargar dos versiones, pero Apollo no tiene mecanismo de rollback automático en caso de regresión de AIS score | Low | Script de install podría verificar AIS antes de hacer el cutover |
| **Retry (explícito)** | execute_actions ya tiene el Circuit Breaker para fallos en cascada, pero no tiene retry con backoff para acciones individuales que fallan transientemente | Low | Agregar retry con exponential backoff para `SetSysctl` y `SetMemorystatus`; no para SIGSTOP (sería peligroso reintentarlo) |

---

## Anti-patrones actuales a resolver

| Anti-patrón | Dónde existe en Apollo | Severidad | Solución en v0.9.0 |
|-------------|----------------------|-----------|-------------------|
| **God Service** | `src/bin/apollo-optimizerd/main.rs` (5524 líneas): contiene SharedState, run_reactor(), main(), init (~400L), init del hot loop (~400L), hot loop (~3200L) | Critical (DEBT-010) | Strangler Fig: extraer stages a `src/engine/pipeline/`. Target: <3000L post-migración SharedState |
| **Shared Database (equivalente)** | SharedState plano con ~40 Arc<Mutex<>> independientes: todos los componentes (socket_handler, main loop, reactor thread) leen/escriben los mismos campos sin contrato de ownership | High (DEBT-004) | Bulkhead: migrar a los 6 grupos de dominio de `daemon_state.rs`. Cada grupo tiene un contrato claro de quién escribe |
| **Chatty Microservices (lite)** | socket_handler adquiere múltiples locks independientes para construir DaemonStatus: `metrics.lock()`, `thermal_state.lock()`, `throttle_level.lock()`, `reactor_status.lock()` — cada GetStatus son 4-8 lock acquisitions | Medium | Bulkhead completo: `MetricsState` agrupa todos estos campos en un solo Mutex |
| **No Timeout** | Socket handler no tiene read timeout configurable en DaemonRequest. Un cliente que abre conexión y no envía nada bloquea un thread | High | Agregar `set_read_timeout(Some(Duration::from_secs(30)))` en `handle_client()` |
| **Retry Storm (riesgo latente)** | Si el daemon reinicia y encuentra procesos congelados, los descongela correctamente. Pero si el Circuit Breaker abre durante un pico y el caller (e.g., el systemd equivalent) reinicia el daemon repetidamente, puede haber cascada | Low | El Circuit Breaker actual mitiga esto, pero agregar rate limit en reconnect del launchd con `ThrottleInterval` en el .plist |
| **Ignoring Idempotency** | `LlmSetKey` y `SetProfile` no tienen request-id. Un cliente que retransmite por timeout aplica el efecto dos veces | Medium | Agregar `request_id: Option<String>` a DaemonRequest, cache en socket_handler |

---

## Mapping patrón → componente (detalle)

### Circuit Breaker
**Archivo**: `src/engine/circuit_breaker.rs`

Implementación completa con máquina de estados Closed→Open→HalfOpen. Parámetros:
- `failure_threshold`: 5 fallos en 60s para trip a Open
- `timeout`: 30s en Open antes de pasar a HalfOpen
- `success_threshold`: 2 éxitos consecutivos para volver a Closed

El `SharedState` en main.rs tiene `circuit_breaker: Arc<Mutex<CircuitBreaker>>`. El `DegradationController` lo consume: si CB está Open >5min, escala a Emergency. El endpoint `GetHealth` expone el estado.

**Gap**: `circuit_breaker` y `degradation` NO están en `daemon_state.rs` (ver DEBT-004 comentario: "FALTA en daemon_state.rs <- agregar en v0.9.0").

---

### Bulkhead
**Archivo**: `src/engine/daemon_state.rs` (diseño) vs `src/bin/apollo-optimizerd/main.rs` (implementación actual)

`daemon_state.rs` define los 6 bulkheads de dominio:
- `MetricsState`: RuntimeMetrics, thermal, reactor counters (~32 accesos, más contendido)
- `PolicyState`: profile, governor, learned_policy, adaptive_governor
- `ProcessState`: frozen_state, last_blockers, wake_state
- `HardwareState`: hw_snapshot, mach_qos, sysctl_governor
- `LlmDomainState`: llm_cfg, llm_state, paths
- `UsageDomainState`: usage_model, usage_tracker, paths

La implementación real en main.rs usa ~40 Arc<Mutex<>> independientes. Esto viola el Bulkhead: un fallo en MetricsState lock no aísla PolicyState, pero tampoco otorga las garantías de throughput que el patrón promete.

**Acción v0.9.0**: PR DEBT-004 — migrar main.rs a usar `SharedState` de `daemon_state.rs`.

---

### Event Sourcing (journal.rs)
**Archivo**: `src/engine/journal.rs`

El `journal.jsonl` es append-only: cada `JournalEntry` tiene timestamp, `RootAction`, before/after metrics, success flag, reason. Cumple las propiedades de Event Sourcing:
- Inmutabilidad: append-only
- Auditabilidad: `read_journal()` devuelve toda la historia
- Rotación: a 10MB → `.jsonl.1`
- Protección: rechaza symlinks

**Limitación**: No es Event Sourcing completo porque el estado en memoria (`frozen_state`, `learned_policy`) NO se reconstruye a partir del journal. Se carga desde snapshots independientes. Para ES completo, los snapshots deberían ser el estado _proyectado_ del journal.

**Para v1.0**: Unificar LearnedState y journal como fuente única; el startup haría replay del journal desde el último snapshot.

---

### Backpressure
**Archivo**: `src/engine/action_queue.rs`

`ActionQueue` implementa backpressure con 3 tiers de prioridad. El método `drain_cycle(max_per_cycle)` actúa como la válvula: el productor (decide_actions) puede generar 50 acciones, pero el consumidor (execute_actions) solo procesa `max_per_cycle` por ciclo. Las Urgent nunca se retienen.

`backpressure_ratio()` = `(normal.len() + background.len()) / capacity`. Este ratio se calcula pero **no está wired al Router adaptativo** de SignalIntelligence. Oportunidad de v0.9.0.

---

### CQRS
**Archivos**: `src/bin/apollo-optimizerd/socket_handler.rs` (read side) + `src/bin/apollo-optimizerd/main.rs` hot loop (write side)

El daemon implementa CQRS de facto:
- **Query side**: socket_handler.rs despacha `GetStatus`, `GetMetrics`, `GetTopBlockers`, etc. Usa `try_lock` en MetricsState para no bloquear el hot loop.
- **Command side**: main.rs ejecuta el ciclo de optimización (freeze, throttle, QoS, sysctl) y escribe SharedState.

**Limitación**: No hay separación formal de modelos (read model vs write model). El socket handler lee directamente del mismo SharedState que el loop escribe, con la misma estructura. CQRS formal requeriría un read model separado (p.ej., un `RuntimeMetricsView` inmutable proyectado desde el loop).

---

### Strangler Fig
**Archivos**: `src/engine/pipeline/learning_context.rs`, `decision_stage.rs`, `periodic_stage.rs`

El pattern está en ejecución: main.rs es el "legacy monolith" y `src/engine/pipeline/` es la estructura destino. Progreso actual:

| Stage | Estado |
|-------|--------|
| LearningContext (9 subsistemas) | Completo — wired en v0.8.0 |
| DecisionStage + PolicyContext | Completo — wired en v0.8.0 |
| PeriodicStage con run_periodic() | Completo — wired en v0.8.0 |
| ObservationStage | Pendiente (v0.9.0) |
| SharedState migration | Pendiente (v0.9.0, DEBT-004) |
| Init stage extraction | Pendiente (v0.9.0+) |

La migración se ejecuta de forma incremental: cada PR extrae un bloque funcional, el monolito sigue corriendo con el bloque nuevo inyectado. Este es el Strangler Fig textbook.

---

### Anti-Corruption Layer
**Archivo actual**: `src/engine/pipeline/decision_stage.rs` (PolicyContext)

`PolicyContext` es un ACL embrionario: traduce 8 parámetros sueltos de OutcomeTracker/LearnedPolicy en un struct tipado que `decide_actions` puede consumir sin conocer la estructura interna del daemon.

**Gap**: Durante la migración SharedState (DEBT-004), el acceso a `state.metrics.lock()` vs `state.metrics_state.lock().metrics` necesitará un adaptador explícito para que el código que no se ha migrado aún pueda coexistir. Este adaptador IS el ACL.

**Acción v0.9.0**: Crear `domain_access.rs` o métodos de acceso en `daemon_state.rs` que provean la interfaz del SharedState plano, delegando al agrupado internamente. Esto permite migrar main.rs call-site por call-site sin un big-bang rewrite.

---

### Health Check
**Archivos**: `src/engine/protocol.rs` (GetHealth, Doctor), `src/engine/daemon_state.rs` (ReactorStatus)

Tres niveles de health check:
1. `ReactorStatus.health`: `"ok"` | `"stalled"` | `"collector-stalled"` — verificado cada ciclo
2. `DaemonRequest::Doctor` — diagnóstico profundo vía socket
3. `DaemonRequest::GetHealth` — expone estado de CircuitBreaker y DegradationController

**Sugerencia v0.9.0**: Exponer `backpressure_ratio` y `action_queue.pending_count()` en GetHealth para observabilidad completa de pipeline health.

---

### Idempotencia
**Estado**: Parcial

- **Journal**: `append_journal` es idempotente en sentido de que múltiples llamadas con el mismo entry generan múltiples líneas (no hay dedup). Correcto para un audit log.
- **Acciones del kernel**: SIGSTOP en un proceso ya congelado es idempotente (macOS lo acepta). SIGCONT en un proceso corriendo es idempotente.
- **Protocolo**: `DaemonRequest` no tiene `request_id`. `SetProfile` aplicado dos veces tiene el mismo efecto final (idempotente por naturaleza del SET). Pero `LlmSetKey` y `Feedback` NO son idempotentes — cada llamada añade una nueva entrada.

**Acción v0.9.0**: Agregar `request_id: Option<String>` a DaemonRequest. socket_handler mantiene un `HashSet<String>` de los últimos N request_ids procesados (TTL 60s) y retorna el resultado cacheado en caso de retry.

---

### Dead Letter Queue
**Estado**: Ausente como patrón formal

Actualmente, cuando `execute_actions` falla:
1. El error se loguea
2. El circuit_breaker registra `record_failure()`
3. La entrada en journal tiene `success: false`
4. La acción se descarta

No hay cola de reintentos. Una acción `UnfreezeProcess` fallida (p.ej., por EPERM transitorio) se pierde. El proceso queda congelado indefinidamente hasta que el siguiente ciclo de decisión lo vuelva a seleccionar.

**Acción v0.9.0**: En `action_queue.rs`, agregar una cola `dlq: VecDeque<(RootAction, u8)>` con contador de intentos. Las Urgent que fallan se re-encolan hasta 3 intentos antes de escalar. Normal y Background se descartan después de 1 fallo (comportamiento actual).

---

## Resumen de prioridades v0.9.0

```
Alta prioridad (bloquean escalabilidad o tienen riesgo de producción):
  1. Bulkhead completo → PR DEBT-004 (SharedState migration)
  2. No Timeout → agregar read_timeout en socket_handler
  3. DLQ para Urgent actions → action_queue.rs
  4. ACL durante migración → domain_access.rs

Media prioridad (mejoran robustez y observabilidad):
  5. Idempotencia explícita → request_id en protocol.rs
  6. Backpressure → SignalIntelligence wire (backpressure_ratio al router)
  7. circuit_breaker + degradation → agregar a daemon_state.rs

Baja prioridad / v1.0+:
  8. Event Sourcing completo (replay desde journal)
  9. Compensating Transaction en execute_actions
 10. Materialized View para DaemonStatus
```
