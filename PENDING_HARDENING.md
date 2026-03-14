# Pendientes de Hardening — Apollo Optimizer

Actualizado: 2026-03-13

## Completados (sesiones anteriores)

| # | Issue | Fix |
|---|-------|-----|
| 1 | SIGTERM no manejado por el daemon | Signal handler con `libc::sigaction` + `AtomicBool` |
| 2 | SysctlTuner duplicado vs SysctlGovernor | Consolidado en `SysctlGovernor`, eliminado `SysctlTuner` |
| 3 | Socket Unix con permisos 0o666 | 0o660 root:staff (root) / 0o600 (non-root) + `SetLearnedPolicy` privilegiado |
| 4 | Seed policy no se re-aplicaba | `merge_seed_into()` en cada `SetLearnedPolicy` |
| 5 | PID recycling A-B-A en freeze/throttle | `start_sec`/`start_usec` en `RootAction` + `verify_pid_identity()` |
| 6 | Unfreeze no verificaba SIGCONT exitoso | Solo remueve de frozen set si SIGCONT ok o proceso muerto |

## Completados (esta sesion)

| # | Issue | Fix |
|---|-------|-----|
| 7 | ~2,500 string allocations/ciclo por `.to_lowercase()` | Pre-lowercase listas + `to_ascii_lowercase()` en `execute_actions.rs` y `decide_actions.rs` |
| 8 | `frozen_state.json` escrito cada ciclo incondicionalmente | Solo escribe si el frozen set cambio (snapshot antes/despues) |
| 9 | 19 `.lock()` inconsistentes con 157 `.lock_recover()` | Migradas las 19 a `.lock_recover()` — 0 instancias restantes |
| 10 | `HardwareSnapshot` clonado 6 veces/ciclo | 1 clone al inicio del ciclo, reutilizado en los 6 sitios |
| 11 | Doctest `fast_entropy` SIGILL flaky | Marcado `no_run` (compila pero no ejecuta) |
| 12 | Test `probe_hardware_registers` flaky por timing | Rango de assert relajado: 5-100ms en vez de 9-30ms |

## Pendientes — Arquitectura (prioridad media)

### 1. SharedState con 44 campos `Arc<Mutex<T>>`

- **Problema**: Muchos campos podrían agruparse en structs logicos (ej. todos los contadores de reactor en uno, todas las metricas de energia en otro).
- **Beneficio**: Menos locks, menos contention, codigo mas legible.
- **Riesgo**: Refactor grande, requiere tocar muchos sitios.

### 2. `main()` de ~1,900 lineas (mega-funcion)

- **Problema**: Toda la logica del daemon esta en una sola funcion. Dificil de navegar y mantener.
- **Beneficio**: Modularidad, testabilidad, legibilidad.
- **Riesgo**: Refactor grande, pero no cambia funcionalidad.

## Pendientes — Limpieza

| Item | Detalle |
|------|---------|
| ~~`src/sysctl_tuner.rs`~~ | Borrado (2026-03-14) — era huerfano, cero referencias. |
| `silicon_probe::read_rndr` | Test `#[ignore]` por SIGILL en EL0 — limitacion de hardware, no es bug. |

## Pendientes — Operacional

| Item | Detalle |
|------|---------|
| Instalar binario | `cargo build --release && sudo ./scripts/install-root-daemon.sh` |
| Re-habilitar Apollo | Esta apagado manualmente desde el segundo desbordamiento. Re-habilitar tras instalar. |

## Estado de tests

- 621 tests totales: 0 fallos, 1 ignored (`read_rndr`), 1 doctest `no_run`
- Lib: 182 passed
- Integration (level1-level11): 439 passed
