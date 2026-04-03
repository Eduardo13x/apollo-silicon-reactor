# Apollo v0.8.0 "Production Ready" — Plan Maestro

## Estado de ejecucion (2026-04-02)

| PR | Spec | Estado | Commits |
|----|------|--------|---------|
| #1 | M1_DEAD_CODE.md | COMPLETO | a2c1ccc, af8c15d, 1690a43 |
| #2 | M2_LEARNING_CONTEXT.md | COMPLETO | 01260cd |
| #3 | M2_DECISION_STAGE.md | COMPLETO | 5e09d69 |
| #4 | M2_PERIODIC_STAGE.md | COMPLETO | 8233b63, 9932d0a |
| #5 | M3_TEST_PROTOCOL.md | COMPLETO | 74d41f0, bb45950 |
| #6 | M3_TEST_JOURNAL.md | COMPLETO | 0d117fa, c7ceeed |
| #7 | M3_TEST_STRUCTURES.md | PARCIAL — falta Commit C (daemon_startup.rs) | 6bd14e0 |

**Tests**: 2130 passed, 10 ignored (24 suites)
**main.rs**: 5524 lineas (vs 5487 baseline)

**Objetivo**: Subir madurez de 62% -> 85%+ sin romper apollo
**Base**: v0.7.0 commit `1257633`
**Restriccion critica**: El daemon (`apollo-optimizerd`) NUNCA debe romperse. El CLI (`apollo-optimizer`) es secundario.

---

## Estructura de este folder

| Archivo | Contenido |
|---------|-----------|
| `README.md` | Este archivo -- indice y decisiones de alto nivel |
| `DEBT_REGISTER.md` | Registro de deuda tecnica con severidad y estado |
| `M1_DEAD_CODE.md` | Spec completa: eliminacion de codigo muerto + API prelude |
| `M2_LEARNING_CONTEXT.md` | Spec completa: wire LearningContext (fundacion de M2) |
| `M2_DECISION_STAGE.md` | Spec completa: wire DecisionStage en hot path |
| `M2_PERIODIC_STAGE.md` | Spec completa: implementar + wire run_periodic() |
| `M3_TEST_PROTOCOL.md` | Spec completa: tests protocol.rs + types.rs |
| `M3_TEST_JOURNAL.md` | Spec completa: tests journal.rs + lock_ext.rs + capabilities.rs |
| `M3_TEST_STRUCTURES.md` | Spec completa: tests daemon_state + user_profile + wake_storm |
| `M3_TEST_INTEGRATION.md` | Spec completa: test de integracion startup |

---

## Orden de ejecucion (IMPORTANTE — no cambiar sin leer riesgos)

```
Rama: refactor/m1-dead-code-api
  └── Commit A: CLI thin-replacements (quitar dependencia de OptimizerEngine)
  └── Commit B: DELETE optimizer.rs + reactor.rs + lib.rs cleanup
  └── Commit C: pub mod prelude en lib.rs
      └── PR #1 -> merge a main

Rama: refactor/m2-learning-context  (base: main post-PR1)
  └── Commit A: Instanciar LearningContext en el hot loop
      └── PR #2 -> merge a main  <- FUNDACION, el resto depende de esto

Rama: refactor/m2-decision-stage  (base: main post-PR2)
  └── Commit A: Wire DecisionStage::run() + PolicyContext
      └── PR #3 -> merge a main

Rama: refactor/m2-periodic-stage  (base: main post-PR3)
  └── Commit A: Implementar run_periodic() (cuerpo real)
  └── Commit B: Wire run_periodic() en daemon loop
      └── PR #4 -> merge a main

Rama: test/m3-protocol-types  (base: main, INDEPENDIENTE)
  └── Commit A: Tests protocol.rs (25 variants serde + is_privileged)
  └── Commit B: Tests types.rs (kebab-case serde + roundtrips)
      └── PR #5 -> merge a main

Rama: test/m3-io-safety  (base: main post-PR5)
  └── Commit A: Add tempfile a dev-dependencies + tests journal.rs
  └── Commit B: Tests lock_ext.rs + capabilities.rs
      └── PR #6 -> merge a main

Rama: test/m3-structures  (base: main post-PR6)
  └── Commit A: Tests daemon_state.rs + user_profile + wake_storm
  └── Commit B: Test de integracion daemon startup
      └── PR #7 -> merge a main
```

---

## Decisiones arquitectonicas criticas

### Por que M1 antes que M2?
Eliminar optimizer.rs/reactor.rs simplifica el grafo de dependencias antes de tocar main.rs. Si lo hacemos despues, el diff de M2 es mas grande y mas dificil de revisar.

### Por que LearningContext es su propia PR?
LearningContext es el cambio mas tocado por el borrow checker. Si falla la compilacion, quiero que sea en un PR pequeno y reversible, no en medio de DecisionStage. Una vez que compile con LearningContext wired, el resto es mecanico.

### Por que NO migramos SharedState a daemon_state.rs todavia?
La migracion del SharedState plano (40 campos) al agrupado (6 dominios) requiere cambiar ~300 sitios de acceso en main.rs, socket_handler.rs y llm_daemon.rs simultaneamente. El riesgo es demasiado alto para hacerlo junto con las otras refactorizaciones. Queda como **DEBT-004 v0.9.0**.

### Por que los tests son PRs independientes?
Los tests de protocol.rs/types.rs NO dependen de M2. Pueden ir en paralelo y no tienen riesgo de romper el daemon.

---

## Invariantes que NUNCA se pueden romper

1. `cargo build --bin apollo-optimizerd` debe compilar siempre
2. El daemon debe pasar el test de smoke: arrancar, dar status, responder a ctl
3. `cargo test` no debe tener nuevas regresiones
4. La serializacion de `DaemonRequest`/`DaemonResponse` NO puede cambiar (protocol.rs)
5. Los archivos de estado en `/var/lib/apollo/` deben seguir siendo legibles entre versiones
6. Los procesos protegidos (Claude, Brave, rustc) jamas deben ser afectados

---

## Metricas de exito

| Metrica | v0.7.0 | Target post-PR7 |
|---------|--------|-----------------|
| Madurez estimada | 62% | 82-85% |
| Dead code (optimizer+reactor) | ~1643 lineas | 0 |
| main.rs lineas | ~5487 | ~4900 (M2 reduce ~600) |
| LearningContext wired | No | Si |
| DecisionStage wired | No | Si |
| run_periodic() stub | Si | No |
| Modulos con tests | 81/100 | 93/100 |
| Protocol serde guard | Ninguno | 25 variant tests |
| journal symlink guard | Ninguno | 1 test dedicado |
| Startup integration test | Ninguno | 5 tests |
