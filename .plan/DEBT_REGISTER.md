# Registro de Deuda Tecnica — Apollo v0.8.0

Ultima actualizacion: 2026-04-02
Base de analisis: commit `1257633` (v0.7.0)

---

## Leyenda

| Severidad | Definicion |
|-----------|------------|
| Critical | Bloquea escalabilidad del codigo o puede causar bugs en produccion |
| High | Degrada mantenibilidad significativamente |
| Medium | Mejora robustez o cobertura |
| Low | Cosmetico / calidad de API |

| Estado | Significado |
|--------|-------------|
| `open` | No trabajado |
| `in_progress` | En una PR activa |
| `resolved` | Cerrado y mergeado |
| `deferred` | Pospuesto con justificacion |

---

## Deuda Activa

### DEBT-001 — OptimizerEngine es codigo muerto en el daemon
- **Severidad**: High
- **Estado**: `resolved`
- **Archivos**: `src/optimizer.rs` (1418 lineas)
- **Diagnostico**: `OptimizerEngine` y `HeuristicEngine` son el motor del CLI legacy (`src/main.rs`). El daemon moderno (`apollo-optimizerd`) nunca los importa ni llama. Los subcomandos del CLI que los usan: `Turbo`, `Llm`, `Restore`, `Optimize`, `Clean`, `Startup`, `Daemon`.
- **Riesgo de dejar esto**: Confunde a nuevos contribuidores que ven dos "motores". Aumenta tiempo de compilacion. Cualquier refactor en tipos compartidos (e.g. `OptimizationProfile`) requiere actualizar optimizer.rs aunque sea muerto.
- **Resuelto por**: PR #1 (M1 Commit B)
- **Resolucion**: af8c15d — optimizer.rs deleted as dead code

### DEBT-002 — SystemReactor es codigo muerto en el daemon
- **Severidad**: High
- **Estado**: `resolved`
- **Archivos**: `src/reactor.rs` (225 lineas)
- **Diagnostico**: `SystemReactor` implementa un event loop basado en kqueue para el CLI daemon mode. El daemon moderno tiene su propio `run_reactor()` en main.rs (lineas 271+) que no depende de este modulo. Call sites: solo `src/main.rs` lineas 148-149.
- **Riesgo**: Igual a DEBT-001. Adicionalmente, `SystemReactor::start()` llama `optimizer.optimize()` 5 veces internamente — si alguien lo activa por error en el daemon, ejecuta logica obsoleta.
- **Resuelto por**: PR #1 (M1 Commit B)
- **Resolucion**: af8c15d — reactor.rs deleted as dead code

### DEBT-003 — lib.rs exporta modulos muertos como API publica
- **Severidad**: Low
- **Estado**: `resolved`
- **Archivos**: `src/lib.rs`
- **Diagnostico**: `pub mod optimizer` y `pub mod reactor` aparecen en la API publica de la crate. Ningun binario en `src/bin/` los importa via `apollo_optimizer::optimizer`. Esto da la impresion equivocada de que son parte de la API estable.
- **Resuelto por**: PR #1 (M1 Commit B)
- **Resolucion**: 1690a43 — lib.rs cleaned, prelude added

### DEBT-004 — SharedState plano (40 campos) en lugar del agrupado de daemon_state.rs
- **Severidad**: High
- **Estado**: `deferred` -> **v0.9.0**
- **Archivos**: `src/bin/apollo-optimizerd/main.rs` (struct lines 162-219), `src/engine/daemon_state.rs`
- **Diagnostico**: Existe una version bien disenada del SharedState en `daemon_state.rs` con 6 grupos de dominio (MetricsState, PolicyState, ProcessState, HardwareState, LlmDomainState, UsageDomainState). El daemon usa una version plana con ~40 campos Arc<Mutex<>> independientes. Cada acceso a cualquier campo requiere su propio lock. Con el patron agrupado, toda la informacion de "metricas" estaria en un solo Mutex.
- **Por que diferido?**: La migracion requiere cambiar ~300 sitios de acceso en main.rs (5487 lineas), socket_handler.rs, y llm_daemon.rs simultaneamente. El riesgo de introducir un deadlock o borrow error es alto. Se hace en v0.9.0 con una PR dedicada y checklist de migracion campo-por-campo.
- **Campos del SharedState actual** (en main.rs) vs **daemon_state.rs**:
  ```
  main.rs plano:              daemon_state.rs agrupado:
  profile                 ->  PolicyState.profile
  latency_target          ->  PolicyState.latency_target
  metrics                 ->  MetricsState.metrics
  frozen_state            ->  ProcessState.frozen_state
  last_blockers           ->  ProcessState.last_blockers
  thermal_state           ->  MetricsState.thermal_state
  throttle_level          ->  MetricsState.throttle_level
  reactor_event_weight    ->  MetricsState.reactor_event_weight
  fast_tick_until         ->  MetricsState.fast_tick_until
  thermal_level_real      ->  MetricsState.thermal_level_real
  reactor_status          ->  MetricsState.reactor_status
  governor                ->  PolicyState.governor
  timeline                ->  PolicyState.timeline
  wake_state              ->  ProcessState.wake_state
  llm_cfg                 ->  LlmDomainState.llm_cfg
  llm_state               ->  LlmDomainState.llm_state
  learned_policy          ->  PolicyState.learned_policy (diferencia!)
  usage_model             ->  UsageDomainState.usage_model
  adaptive_governor       ->  PolicyState.adaptive_governor
  mach_qos                ->  HardwareState.mach_qos
  last_hw_snapshot        ->  HardwareState.last_hw_snapshot
  sysctl_governor_status  ->  HardwareState.sysctl_governor_status
  circuit_breaker         ->  FALTA en daemon_state.rs <- agregar en v0.9.0
  degradation             ->  FALTA en daemon_state.rs <- agregar en v0.9.0
  ```

### DEBT-005 — run_periodic() es un stub vacio
- **Severidad**: High
- **Estado**: `resolved`
- **Archivos**: `src/engine/pipeline/periodic_stage.rs` (lineas 122-140)
- **Diagnostico**: `run_periodic()` tiene la interfaz correcta (acepta `PeriodicContext`, devuelve `PeriodicResult`) pero el cuerpo solo setea flags booleanos sin ejecutar ninguna logica real. La logica de persistencia/GC esta inline en main.rs como bloques `if cycle_count % N == 0`.
- **Resuelto por**: PR #4 (M2 Periodic Stage, Commit A + B)
- **Resolucion**: 8233b63, 9932d0a — run_periodic() has real GC logic, wired into daemon loop

### DEBT-006 — DecisionStage::run() existe pero no esta en el hot path
- **Severidad**: High
- **Estado**: `resolved`
- **Archivos**: `src/engine/pipeline/decision_stage.rs`, `src/bin/apollo-optimizerd/main.rs`
- **Diagnostico**: `DecisionStage::run()` esta implementado (no es stub). Acepta `PolicyContext<'a>` y hace el wrapping de `decide_actions()`. El daemon llama `decide_actions()` directamente con ~16 parametros posicionales. Mientras tanto DecisionStage y PolicyContext estan definidos pero no instanciados en el loop.
- **Resuelto por**: PR #3 (M2 Decision Stage)
- **Resolucion**: 5e09d69 — DecisionStage::run() + PolicyContext wired into hot path

### DEBT-007 — LearningContext definido, testeado, pero nunca instanciado en el daemon
- **Severidad**: Critical (es el bloqueador de DEBT-005 y DEBT-006)
- **Estado**: `resolved`
- **Archivos**: `src/engine/pipeline/learning_context.rs`, `src/bin/apollo-optimizerd/main.rs`
- **Diagnostico**: `LearningContext<'a>` agrupa 9 `&'a mut` borrows en una sola estructura. Esta definido, tiene 2 tests propios que pasan. El daemon pasa estos 9 como variables separadas a cada funcion. Mientras no este wired, `PeriodicStage` no puede recibir las referencias que necesita y `DecisionStage` no puede simplificar su firma.
- **Dependencias**: DEBT-005 y DEBT-006 estan BLOQUEADOS por este item.
- **Resuelto por**: PR #2 (M2 Learning Context)
- **Resolucion**: 01260cd — LearningContext instantiated in daemon hot loop

### DEBT-008 — 19 modulos engine sin bloque #[cfg(test)]
- **Severidad**: Medium
- **Estado**: `resolved`
- **Archivos**: protocol.rs, types.rs, journal.rs, capabilities.rs, lock_ext.rs, daemon_state.rs, decide_actions.rs, user_profile.rs, wake_storm_detector.rs (+ 10 hardware-dependientes)
- **Diagnostico**: Los modulos con logica pura (sin hardware) carecen de tests inline. Hay 2004 tests en el suite de integracion pero son tests de decision/calidad, no de correctitud de estructuras de datos ni serializacion.
- **Riesgo principal**: Un cambio en la serializacion de `DaemonRequest` romperia compatibilidad cliente-daemon silenciosamente sin que ningun test lo detecte.
- **Resuelto por**: PRs #5, #6, #7
- **Resolucion**: 74d41f0, bb45950, 0d117fa, c7ceeed, 6bd14e0 — 8/9 modules covered (decide_actions.rs still needs tests)

### DEBT-009 — unwrap_or silente en learning_pipeline.rs
- **Severidad**: Low
- **Estado**: `open`
- **Archivos**: `src/engine/learning_pipeline.rs` (lineas 273, 292, 293, 340)
- **Diagnostico**: 4 llamadas a `.unwrap_or(default)` en codigo de produccion que silencian errores de deserializacion o acceso a archivos. No causan panic, pero ocultan problemas de estado corrupto.
- **Resuelto por**: PR #6 (como parte de error hardening)

### DEBT-010 — main.rs es un monolito de 5487 lineas
- **Severidad**: Critical
- **Estado**: `in_progress`
- **Archivos**: `src/bin/apollo-optimizerd/main.rs`
- **Diagnostico**: Un solo archivo contiene: definicion de SharedState, run_reactor(), main(), toda la inicializacion del daemon (~400 lineas), toda la inicializacion del hot loop (~400 lineas mas), y el hot loop mismo (~3200 lineas). Los stages de pipeline existen en `src/engine/pipeline/` pero no estan conectados.
- **Reduccion esperada post-M2**: ~600-800 lineas (el loop se compacta cuando los 3 stages estan wired).
- **Reduccion a largo plazo (v0.9.0)**: Migracion SharedState reduciria ~200 lineas adicionales.
- **Resuelto parcialmente por**: PRs #2, #3, #4
- **Nota**: main.rs es 5524 lineas (crecio ligeramente — pipeline wired pero logica inline permanece)

### DEBT-011 — No hay prelude de API publica en lib.rs
- **Severidad**: Low
- **Estado**: `resolved`
- **Archivos**: `src/lib.rs`
- **Diagnostico**: Los consumidores de la crate (ctl, menubar, binarios futuros) navegan paths como `apollo_optimizer::engine::protocol::DaemonRequest`. No hay un modulo `prelude` con los tipos mas usados.
- **Resuelto por**: PR #1 (M1 Commit C)
- **Resolucion**: 1690a43 — prelude module added to lib.rs

### DEBT-012 — Sin integration test para startup del daemon
- **Severidad**: Medium
- **Estado**: `open`
- **Archivos**: `tests/`
- **Diagnostico**: Los 2004 tests existentes cubren decisiones de optimizacion y funcionalidad de subsistemas. Ningun test verifica que la secuencia de inicializacion del daemon (construccion de SharedState, load de LearnedState, deteccion de capacidades) funciona correctamente en conjunto.
- **Resuelto por**: PR #7 (M3 Integration Test)

---

## Deuda Descartada

### ~~TransformerPredictor~~ — Solo referencia historica en un comentario
- `main.rs:876` contiene `"Darwin-Boltzmann Anomaly Detector: replaces disabled TransformerPredictor"`. No hay codigo muerto — es documentacion inline. No requiere accion.

### ~~TelemetryLogger~~ — Activo, no muerto
- `src/engine/telemetry_logger.rs` (487 lineas) esta completamente implementado con 5 tests propios. Inactivo en produccion por diseno (pipeline de ML training no esta conectado aun). No es deuda — es una feature incompleta, no codigo muerto.
