# V0.9.0 — SHARED_STATE Migration Plan

**PR range:** #8–#15 (continuando desde v0.8.0 PRs #1–#7)
**Deuda resuelta:** DEBT-004, DEBT-010
**Estrategia:** Strangler Fig + Anti-Corruption Layer
**Baseline de medición:** commit `9932d0a` (v0.8.0), 5524 líneas en main.rs, 2130 tests

---

## 1. Resumen del Problema

`src/bin/apollo-optimizerd/main.rs` define un `SharedState` plano con **22 campos Arc<Mutex<>>** independientes (líneas 164–221). Cada acceso a cualquier campo requiere su propio `.lock_recover()`. El archivo tiene 5524 líneas, en parte porque toda la lógica de inicialización y el hot loop están en el mismo archivo que la definición del struct.

`src/engine/daemon_state.rs` ya contiene la versión bien diseñada con **6 grupos de dominio** (MetricsState, PolicyState, ProcessState, HardwareState, LlmDomainState, UsageDomainState). El struct `SharedState` de daemon_state.rs (líneas 204–225) tiene la forma correcta y solo necesita agregar dos campos faltantes (circuit_breaker, degradation).

**El problema no es saber qué hacer — es hacerlo sin tocar 382+ sitios de acceso simultáneamente.**

### Conteo real de sitios de acceso (post-análisis estático)

| Archivo | Accesos totales | Domain más frecuente |
|---------|-----------------|----------------------|
| `main.rs` | ~382 | PolicyState (~106), MetricsState (~61) |
| `socket_handler.rs` | ~64 | LlmDomainState (~21), PolicyState (~27) |
| `llm_daemon.rs` | ~53 | LlmDomainState (~43), PolicyState (~10) |
| **Total** | **~499** | — |

---

## 2. Estrategia: Strangler Fig Incremental

### El patrón

El **Strangler Fig** (Fowler 2004) migra un sistema sin big-bang replacement:
1. Crear el nuevo sistema **al lado** del viejo
2. Mover funcionalidad de viejo a nuevo **de a poco**, sin romper nada
3. Al final, el viejo está vacío y puede borrarse

El **Anti-Corruption Layer (ACL)** (Evans 2003) es la capa de traducción que permite que ambos sistemas coexistan durante la migración. En este caso, el ACL son métodos helper que aceptan el `SharedState` plano actual y devuelven referencias de los campos como si fuera el agrupado — o viceversa.

### Las 3 fases

```
Fase 1 — Infraestructura (PR #8):
  - Completar daemon_state.rs SharedState (agregar circuit_breaker, degradation)
  - Importar daemon_state::SharedState en main.rs con alias DomainState
  - Crear métodos ACL: acl_metrics(&self) -> impl Deref<Target=MetricsState>
  - El daemon todavía usa el flat SharedState — 0 sitios de acceso cambian
  - Estado: dos tipos coexisten, comunicados por ACL

Fase 2 — Migración por dominio (PRs #9–#13):
  - Un dominio por PR, ~50-80 sitios de acceso por PR
  - Cada PR: (1) instanciar el subgrupo, (2) migrar todos sus sitios, (3) tests
  - Orden: ProcessState → MetricsState → PolicyState → HardwareState → LLM → Usage
  - El flat SharedState se va vaciando campo por campo

Fase 3 — Limpieza (PRs #14–#15):
  - Migrar socket_handler.rs + llm_daemon.rs al nuevo tipo
  - Borrar flat SharedState, ReactorStatus local, UsageTrackerState local
  - Renombrar DomainState → SharedState en daemon_state.rs (o viceversa)
```

### Regla de oro del ACL

**Durante la migración, nunca se tienen dos locks del mismo dato.** El ACL traduce entre sistemas pero no duplica datos. Si MetricsState ya fue migrado, el campo `metrics: Arc<Mutex<RuntimeMetrics>>` del flat struct desaparece — el flat struct apunta al mismo `Arc` que el DomainState.

### Orden de migración por dominio

El orden importa porque algunos dominios tienen más dependencias cruzadas (un lock de PolicyState y un lock de MetricsState en el mismo bloque sería deadlock potencial). Migrar de menor a mayor interdependencia:

```
1. ProcessState      — frozen_state, last_blockers, wake_state (26 accesos en main.rs)
2. MetricsState      — metrics + thermal_state + throttle_level + reactor_* (61 accesos)
3. HardwareState     — mach_qos + sysctl_governor_status + last_hw_snapshot (18 accesos)
4. LlmDomainState    — llm_cfg + llm_state + paths (2 accesos main, 43 llm_daemon)
5. UsageDomainState  — usage_model + usage_tracker + paths (2 accesos main, 2 socket)
6. PolicyState       — profile + governor + learned_policy + adaptive_governor (106 accesos) [last: highest blast radius]
```

---

## 3. Inventario de Campos

### 3a. Campos del flat SharedState en main.rs (líneas 164–221)

| Campo plano | Tipo | Dominio target | Accesos main.rs | Accesos socket_handler.rs | Accesos llm_daemon.rs |
|-------------|------|----------------|-----------------|--------------------------|----------------------|
| `profile` | `Arc<Mutex<OptimizationProfile>>` | PolicyState | ~40 | 1 | 0 |
| `latency_target` | `Arc<Mutex<LatencyTarget>>` | PolicyState | ~12 | 2 | 0 |
| `metrics` | `Arc<Mutex<RuntimeMetrics>>` | MetricsState | ~30 | 2 | 0 |
| `frozen_state` | `Arc<Mutex<HashMap<u32, FrozenEntry>>>` | ProcessState | ~15 | 3 | 0 |
| `last_blockers` | `Arc<Mutex<Vec<BlockerScore>>>` | ProcessState | ~4 | 2 | 0 |
| `thermal_state` | `Arc<Mutex<String>>` | MetricsState | ~10 | 1 | 0 |
| `throttle_level` | `Arc<Mutex<String>>` | MetricsState | ~8 | 1 | 0 |
| `reactor_event_weight` | `Arc<Mutex<f64>>` | MetricsState | ~4 | 0 | 0 |
| `fast_tick_until` | `Arc<Mutex<Option<Instant>>>` | MetricsState | ~4 | 0 | 0 |
| `thermal_level_real` | `Arc<Mutex<String>>` | MetricsState | ~3 | 0 | 0 |
| `reactor_status` | `Arc<Mutex<ReactorStatus>>` | MetricsState | ~10 | 3 | 0 |
| `governor` | `Arc<Mutex<ProfileGovernor>>` | PolicyState | ~20 | 5 | 0 |
| `timeline` | `Arc<Mutex<VecDeque<ProfileTransition>>>` | PolicyState | ~8 | 1 | 0 |
| `wake_state` | `Arc<Mutex<WakeRuntimeState>>` | ProcessState | ~7 | 1 | 0 |
| `stop` | `Arc<AtomicBool>` | Infrastructure (sin cambio) | ~4 | 1 | 0 |
| `llm_cfg` | `Arc<LlmConfig>` | LlmDomainState | ~1 | 0 | 1 |
| `llm_state` | `Arc<Mutex<LlmState>>` | LlmDomainState | ~1 | 0 | ~15 |
| `learned_policy` | `Arc<Mutex<LearnedPolicy>>` | PolicyState | ~15 | 0 | 0 |
| `llm_state_path` | `PathBuf` | LlmDomainState | 0 | 1 | ~5 |
| `llm_key_path` | `PathBuf` | LlmDomainState | 0 | 1 | ~3 |
| `learned_policy_path` | `PathBuf` | LlmDomainState | ~1 | 0 | 0 |
| `feedback_path` | `PathBuf` | LlmDomainState | 0 | 0 | 0 |
| `suggestions_path` | `PathBuf` | LlmDomainState | 0 | ~10 | 0 |
| `config_path` | `PathBuf` | Infrastructure (read-only, sin cambio) | 0 | 0 | ~1 |
| `usage_model` | `Arc<Mutex<UsageModel>>` | UsageDomainState | ~1 | 1 | 0 |
| `usage_model_path` | `PathBuf` | UsageDomainState | ~1 | 0 | 0 |
| `usage_events_path` | `PathBuf` | UsageDomainState | ~1 | 0 | 0 |
| `usage_tracker` | `Arc<Mutex<UsageTrackerState>>` | UsageDomainState | ~1 | 0 | 0 |
| `adaptive_governor` | `Arc<Mutex<AdaptiveGovernor>>` | PolicyState | ~8 | 0 | 0 |
| `mach_qos` | `Arc<Mutex<MachQoSManager>>` | HardwareState | ~12 | 0 | 0 |
| `last_hw_snapshot` | `Arc<Mutex<Option<HardwareSnapshot>>>` | HardwareState | ~4 | 0 | 0 |
| `discrepancy_log_path` | `PathBuf` | Infrastructure (read-only, sin cambio) | ~2 | 0 | 0 |
| `user_profile_path` | `PathBuf` | Infrastructure (read-only, sin cambio) | ~1 | 0 | 0 |
| `sysctl_governor_status` | `Arc<Mutex<SysctlGovernorStatus>>` | HardwareState | ~2 | 0 | 0 |
| `cycle_condvar` | `Arc<(Mutex<bool>, Condvar)>` | Infrastructure (sin cambio) | ~4 | 0 | 0 |
| `resource_interrupt` | `Arc<ResourceInterruptState>` | Infrastructure (sin cambio) | ~8 | 0 | 0 |
| `subscribers` | `Arc<Mutex<Vec<UnixStream>>>` | Infrastructure (sin cambio) | 0 | 2 | 0 |
| `circuit_breaker` | `Arc<Mutex<CircuitBreaker>>` | PolicyState (**agregar a daemon_state.rs**) | 4 | 2 | 0 |
| `degradation` | `Arc<Mutex<DegradationController>>` | PolicyState (**agregar a daemon_state.rs**) | 4 | 2 | 0 |

### 3b. Diferencias entre flat SharedState y daemon_state.rs

Campos en `daemon_state.rs` que están en el flat struct con diferente agrupación:

- `learned_policy` → en flat está como `Arc<Mutex<LearnedPolicy>>` independiente. En daemon_state.rs está dentro de `PolicyState`. Sin diferencia semántica.
- `llm_cfg` → en flat es `Arc<LlmConfig>` (no Mutex, inmutable). En daemon_state.rs `LlmDomainState.llm_cfg` es `LlmConfig` directo (dentro del Mutex de LlmDomainState). Esto es un trade-off menor — ver nota en PR #12.

Campos **faltantes** en daemon_state.rs que hay que agregar:
- `circuit_breaker: Arc<Mutex<CircuitBreaker>>` → en PolicyState (o nuevo ResilienceState)
- `degradation: Arc<Mutex<DegradationController>>` → en PolicyState (o nuevo ResilienceState)

> **Decisión de diseño:** `circuit_breaker` y `degradation` van en `PolicyState` porque son evaluados inmediatamente antes/después de `decide_actions()` (líneas 4563–4682 de main.rs) y conceptualmente son "governors de política". Si en el futuro crecen, se pueden extraer a un `ResilienceState` propio.

### 3c. Campos que NO migran (infrastructure group — ya correctos)

Estos campos del flat SharedState ya tienen el tipo correcto en daemon_state.rs y no requieren cambio conceptual:

| Campo | Razón para no mover |
|-------|---------------------|
| `stop: Arc<AtomicBool>` | Lock-free, ya está en daemon_state.rs SharedState |
| `cycle_condvar: Arc<(Mutex<bool>, Condvar)>` | Ya está en daemon_state.rs SharedState |
| `resource_interrupt: Arc<ResourceInterruptState>` | Ya está en daemon_state.rs SharedState |
| `subscribers: Arc<Mutex<Vec<UnixStream>>>` | Ya está en daemon_state.rs SharedState |
| `config_path: PathBuf` | Ya está en daemon_state.rs SharedState |
| `discrepancy_log_path: PathBuf` | Ya está en daemon_state.rs SharedState |
| `user_profile_path: PathBuf` | Ya está en daemon_state.rs SharedState |

---

## 4. PRs Detalladas (#8–#15)

---

### PR #8 — Infraestructura: Completar daemon_state.rs y ACL methods

**Commit message:**
```
feat(arch): DEBT-004 infra — complete DomainSharedState + ACL methods
```

**Objetivo:** Preparar el terreno sin cambiar ningún sitio de acceso. El daemon sigue compilando y corriendo exactamente igual.

**Archivos tocados:**
- `src/engine/daemon_state.rs` — agregar circuit_breaker + degradation a PolicyState; verificar que SharedState en daemon_state.rs tiene todos los campos de infraestructura
- `src/bin/apollo-optimizerd/main.rs` — solo imports, alias `use crate::engine::daemon_state::SharedState as DomainSharedState`

**Cambios específicos en daemon_state.rs:**

1. Agregar a `PolicyState`:
```rust
pub circuit_breaker: apollo_optimizer::engine::circuit_breaker::CircuitBreaker,
pub degradation: apollo_optimizer::engine::degradation::DegradationController,
```

2. Agregar `Default` impl a `PolicyState` (necesario para construcción incremental).

3. Agregar métodos ACL al `SharedState` de daemon_state.rs como `impl block` en un módulo `acl` inline:
```rust
// Anti-Corruption Layer: bridge between flat SharedState (main.rs) and
// grouped DomainSharedState (daemon_state.rs). Remove these when migration completes.
impl crate::SharedState {
    // Ninguno por ahora — se agregan en PRs #9-#13 conforme avanzan.
}
```

**Test plan:**
- `cargo build --bin apollo-optimizerd` — debe compilar sin warnings nuevos
- `cargo test --lib` — debe pasar los tests existentes de daemon_state.rs
- Verificar que el daemon arranca: `cargo run --bin apollo-optimizerd -- daemon --profile balanced-root`

**Risk level:** Bajo — solo se agrega código, no se elimina ni modifica lógica existente.

---

### PR #9 — Migrar ProcessState (frozen_state, last_blockers, wake_state)

**Commit message:**
```
refactor(arch): DEBT-004 PR#9 — migrate ProcessState domain (3 fields, ~26 sites)
```

**Objetivo:** Migrar los 3 campos de gestión de procesos. Son los de menor blast radius (~26 sitios en main.rs, ~4 en socket_handler.rs) y los más cohesivos conceptualmente.

**Campos migrados:**
- `frozen_state: Arc<Mutex<HashMap<u32, FrozenEntry>>>` → `ProcessState.frozen_state`
- `last_blockers: Arc<Mutex<Vec<BlockerScore>>>` → `ProcessState.last_blockers`
- `wake_state: Arc<Mutex<WakeRuntimeState>>` → `ProcessState.wake_state`

**Archivos tocados:**
- `src/bin/apollo-optimizerd/main.rs` — 3 campos del flat SharedState reemplazados por `process: Arc<Mutex<ProcessState>>`; ~30 sitios de acceso migrados de `state.frozen_state.lock_recover()` a `state.process.lock_recover().frozen_state`
- `src/engine/daemon_state.rs` — verificar que ProcessState tiene `WakeRuntimeState` (actualmente importado desde daemon_helpers — resolver alias)

**Nota de implementación:** `WakeRuntimeState` actualmente se define en `daemon_helpers.rs` y también se define en `daemon_state.rs`. Antes de esta PR, verificar cuál es la fuente canónica y eliminar el duplicado. El flat SharedState en main.rs importa desde daemon_helpers (línea 129: `WakeRuntimeState`).

**Patrón de migración en main.rs:**
```rust
// ANTES:
let mut frozen_state = state.frozen_state.lock_recover();
frozen_state.insert(pid, entry);

// DESPUÉS:
let mut proc = state.process.lock_recover();
proc.frozen_state.insert(pid, entry);
```

**Regla de lock:** Si un bloque necesita `frozen_state` Y `metrics` simultáneamente:
```rust
// INCORRECTO (deadlock potencial):
let mut proc = state.process.lock_recover();
let mut met = state.metrics.lock_recover();  // <-- puede deadlockear

// CORRECTO: extraer primero lo que necesitas, liberar el lock:
let frozen_count = state.process.lock_recover().frozen_state.len();
state.metrics.lock_recover().frozen_count = frozen_count as u32;
```

**Test plan:**
- `cargo build --bin apollo-optimizerd`
- `cargo test` — ≥2130 tests passing
- Verificar arranque del daemon
- `cargo run --bin apollo-optimizerctl -- status` — debe mostrar estado correcto
- Verificar que frozen_state persiste correctamente en `/var/lib/apollo/frozen_state.json`

**Risk level:** Bajo-Medio. Los 3 campos son cohesivos y rara vez se acceden junto a otros dominios en el mismo bloque. Mayor riesgo: spawn_resource_sentinel recibe `state.frozen_state.clone()` (línea 932) — esto debe actualizarse para clonar el Arc<Mutex<ProcessState>> o extraer el Arc interior.

**Sitios especiales a verificar:**
- Línea ~932: `spawn_resource_sentinel(..., state.frozen_state.clone(), ...)` — el sentinel toma ownership del Arc. Post-migración, pasar `state.process.clone()` y que el sentinel lo llame como `process.lock().frozen_state`.
- Líneas ~785–816: startup unfreeze loop — accede `frozen_state` durante inicialización, antes del hot loop.
- Líneas ~1007–1028: merge de learned_state frozen PIDs en startup.

---

### PR #10 — Migrar MetricsState (metrics + thermal + reactor, ~61 sitios)

**Commit message:**
```
refactor(arch): DEBT-004 PR#10 — migrate MetricsState domain (6 fields, ~61 sites)
```

**Objetivo:** Migrar los 6 campos de métricas runtime. Este es el dominio con más accesos individuales en el hot loop — consolidarlos en un solo lock es el mayor gain de contención.

**Campos migrados:**
- `metrics: Arc<Mutex<RuntimeMetrics>>` → `MetricsState.metrics`
- `thermal_state: Arc<Mutex<String>>` → `MetricsState.thermal_state`
- `throttle_level: Arc<Mutex<String>>` → `MetricsState.throttle_level`
- `reactor_event_weight: Arc<Mutex<f64>>` → `MetricsState.reactor_event_weight`
- `fast_tick_until: Arc<Mutex<Option<Instant>>>` → `MetricsState.fast_tick_until`
- `thermal_level_real: Arc<Mutex<String>>` → `MetricsState.thermal_level_real`
- `reactor_status: Arc<Mutex<ReactorStatus>>` → `MetricsState.reactor_status`

**Nota sobre ReactorStatus:** El flat SharedState define `ReactorStatus` inline en main.rs (líneas 224–251). `daemon_state.rs` también define `ReactorStatus` (líneas 52–80). Son idénticos. En esta PR: eliminar la definición en main.rs y usar la de daemon_state.rs. Actualizar `pub(crate) struct ReactorStatus` → `pub struct ReactorStatus`.

**Archivos tocados:**
- `src/bin/apollo-optimizerd/main.rs` — 7 campos → `metrics: Arc<Mutex<MetricsState>>`; ~70 sitios migrados; eliminar definición local de `ReactorStatus`
- `src/engine/daemon_state.rs` — MetricsState ya completo; verificar que ReactorStatus tiene todos los campos (`reactor_pulses` está en RuntimeMetrics, no en ReactorStatus — confirmar)

**Patrón de acceso optimizado (el mayor beneficio):**
```rust
// ANTES: 3 locks separados en la misma sección lógica del hot loop:
let t = state.thermal_state.lock_recover().clone();
let tl = state.throttle_level.lock_recover().clone();
state.metrics.lock_recover().thermal_state = t.clone();

// DESPUÉS: 1 lock, múltiples campos:
let mut met = state.metrics.lock_recover();
met.metrics.thermal_state = met.thermal_state.clone();  // coherente
// o más idiomáticamente:
{
    let mut ms = state.metrics.lock_recover();
    ms.thermal_state = new_thermal;
    ms.throttle_level = new_throttle;
    ms.metrics.thermal_state = new_thermal.clone();
}
```

**Sitio especial: run_reactor()** (líneas 273–501)
El reactor thread accede `state.reactor_status`, `state.thermal_level_real`, `state.reactor_event_weight`, `state.fast_tick_until`, y `state.metrics` en un loop. Post-migración, todos estos son campos de `MetricsState`. El reactor puede agrupar sus updates en un solo `state.metrics.lock_recover()`:
```rust
// ANTES: 4-5 locks en el reactor loop
*state.thermal_level_real.lock_recover() = level.to_string();
state.reactor_status.lock_recover().events_thermal += 1;
*state.reactor_event_weight.lock_recover() = 1.0;

// DESPUÉS: 1 lock
let mut ms = state.metrics.lock_recover();
ms.thermal_level_real = level.to_string();
ms.reactor_status.events_thermal += 1;
ms.reactor_event_weight = 1.0;
```

**Test plan:**
- `cargo build --bin apollo-optimizerd`
- `cargo test` — ≥2130 tests passing
- Verificar que `ctl status` muestra metrics correctas
- Verificar que `ctl status --health` muestra reactor_status correctamente
- Verificar ausencia de deadlocks: el reactor thread y el hot loop acceden metrics concurrentemente — asegurar que no hay hold-and-wait entre MetricsState y ningún otro dominio

**Risk level:** Medio. La mayor complejidad es el reactor thread que accede este dominio desde un hilo separado. `try_lock()` en socket_handler.rs para metrics debe mantenerse (línea 160 del socket_handler actual).

---

### PR #11 — Migrar HardwareState (mach_qos, sysctl_governor_status, last_hw_snapshot)

**Commit message:**
```
refactor(arch): DEBT-004 PR#11 — migrate HardwareState domain (3 fields, ~18 sites)
```

**Campos migrados:**
- `mach_qos: Arc<Mutex<MachQoSManager>>` → `HardwareState.mach_qos`
- `sysctl_governor_status: Arc<Mutex<SysctlGovernorStatus>>` → `HardwareState.sysctl_governor_status`
- `last_hw_snapshot: Arc<Mutex<Option<HardwareSnapshot>>>` → `HardwareState.last_hw_snapshot`

**Archivos tocados:**
- `src/bin/apollo-optimizerd/main.rs` — 3 campos → `hardware: Arc<Mutex<HardwareState>>`; ~18 sitios migrados
- `src/engine/daemon_state.rs` — HardwareState ya completo

**Sitio especial:** `spawn_resource_sentinel` (línea 936) recibe `Some(state.mach_qos.clone())`. Post-migración, el sentinel debe recibir el Arc de HardwareState o se refactoriza para no necesitar el Arc. Opción preferida: pasar `state.hardware.clone()` al sentinel y que internamente llame `hardware.lock().mach_qos`.

**Test plan:**
- `cargo build --bin apollo-optimizerd`
- `cargo test` — ≥2130 tests
- Verificar que `ctl status` muestra sysctl_governor_status
- Verificar que mach_qos QoS tiers funcionan (wake suppression test manual)

**Risk level:** Bajo. Estos campos se acceden principalmente en el hot loop, raramente concurrentes con otros dominios.

---

### PR #12 — Migrar LlmDomainState (llm_cfg, llm_state, paths)

**Commit message:**
```
refactor(arch): DEBT-004 PR#12 — migrate LlmDomainState (7 fields, ~46 sites llm_daemon+socket)
```

**Campos migrados:**
- `llm_cfg: Arc<LlmConfig>` → `LlmDomainState.llm_cfg` (dentro del Mutex de LlmDomainState)
- `llm_state: Arc<Mutex<LlmState>>` → `LlmDomainState.llm_state`
- `llm_state_path: PathBuf` → `LlmDomainState.llm_state_path`
- `llm_key_path: PathBuf` → `LlmDomainState.llm_key_path`
- `learned_policy_path: PathBuf` → `LlmDomainState.learned_policy_path`
- `feedback_path: PathBuf` → `LlmDomainState.feedback_path`
- `suggestions_path: PathBuf` → `LlmDomainState.suggestions_path`

**Nota sobre llm_cfg:** En el flat SharedState, `llm_cfg` es `Arc<LlmConfig>` (sin Mutex, porque es inmutable tras init). En daemon_state.rs `LlmDomainState` lo contiene dentro del Mutex de LlmDomainState. Para acceder a `llm_cfg` post-migración, se debe lockear `state.llm.lock_recover()` aunque solo se lea `llm_cfg`. Esto es una penalización pequeña porque `llm_cfg` se accede ~2 veces por ciclo. Alternativa aceptable: mantener `llm_cfg: Arc<LlmConfig>` como campo de infrastructura (sin migrar) en el SharedState de daemon_state.rs, al igual que `config_path`. Decisión final en PR implementation.

**Archivos tocados:**
- `src/bin/apollo-optimizerd/llm_daemon.rs` — 43 sitios migrados (mayor concentración de accesos LLM)
- `src/bin/apollo-optimizerd/socket_handler.rs` — 21 sitios migrados (LLM status, key path, suggestions)
- `src/bin/apollo-optimizerd/main.rs` — ~2 sitios

**Patrón en llm_daemon.rs:**
```rust
// ANTES:
let mut llm_state = state.llm_state.lock_recover();
write_json(&state.llm_state_path, &*llm_state, Some(0o600));

// DESPUÉS:
let mut llm = state.llm.lock_recover();
write_json(&llm.llm_state_path, &llm.llm_state, Some(0o600));
```

**Test plan:**
- `cargo build --bin apollo-optimizerd`
- `cargo test` — ≥2130 tests
- Test manual: `ctl llm-status` muestra estado correcto
- Test manual: ciclo de training LLM (si disponible)
- Verificar que `llm_key_path.exists()` checks funcionan correctamente

**Risk level:** Medio. llm_daemon.rs tiene alta densidad de accesos LLM y algunos patrones de acceso compuestos (lock llm_state + read llm_state_path en el mismo bloque).

---

### PR #13 — Migrar UsageDomainState + PolicyState (campos más frecuentes)

**Commit message:**
```
refactor(arch): DEBT-004 PR#13 — migrate UsageDomainState + PolicyState (highest blast radius)
```

**Objetivo:** Esta es la PR más grande. PolicyState tiene ~106 accesos en main.rs (el campo `profile` se lee cada ciclo en docenas de lugares).

**Campos migrados — UsageDomainState:**
- `usage_model: Arc<Mutex<UsageModel>>` → `UsageDomainState.usage_model`
- `usage_tracker: Arc<Mutex<UsageTrackerState>>` → `UsageDomainState.usage_tracker`
- `usage_model_path: PathBuf` → `UsageDomainState.usage_model_path`
- `usage_events_path: PathBuf` → `UsageDomainState.usage_events_path`

**Campos migrados — PolicyState:**
- `profile: Arc<Mutex<OptimizationProfile>>` → `PolicyState.profile`
- `latency_target: Arc<Mutex<LatencyTarget>>` → `PolicyState.latency_target`
- `governor: Arc<Mutex<ProfileGovernor>>` → `PolicyState.governor`
- `timeline: Arc<Mutex<VecDeque<ProfileTransition>>>` → `PolicyState.timeline`
- `learned_policy: Arc<Mutex<LearnedPolicy>>` → `PolicyState.learned_policy`
- `adaptive_governor: Arc<Mutex<AdaptiveGovernor>>` → `PolicyState.adaptive_governor`
- `circuit_breaker` (nuevo en PolicyState) → `PolicyState.circuit_breaker`
- `degradation` (nuevo en PolicyState) → `PolicyState.degradation`

**Nota:** `UsageTrackerState` se define localmente en main.rs (líneas 255–260). Debe moverse a daemon_state.rs (donde ya existe, ver líneas 146–151). Resolver el duplicado antes de migrar.

**Patrón de acceso a profile (más frecuente):**
```rust
// ANTES (cada ciclo, decenas de veces):
let profile = *state.profile.lock_recover();

// DESPUÉS:
let profile = state.policy.lock_recover().profile;
```

**Regla crítica para PolicyState:** Este es el dominio con más contención potencial. Nunca lockear `state.policy` mientras se tiene `state.metrics` lockeado (o viceversa). En todos los sitios donde actualmente se leen ambos en el mismo bloque, extraer los valores necesarios con locks breves y separados.

**Patrón para circuit_breaker + degradation (líneas 4563–4682 en main.rs):**
```rust
// ANTES:
let cb = state.circuit_breaker.lock_recover();
// ... leer cb ...
drop(cb);
let mut deg = state.degradation.lock_recover();

// DESPUÉS (ambos en PolicyState — cuidado: un solo lock para ambos):
let should_skip = {
    let pol = state.policy.lock_recover();
    pol.circuit_breaker.is_open() || pol.degradation.blocks_actions()
};
// Procesar sin lock...
{
    let mut pol = state.policy.lock_recover();
    pol.circuit_breaker.record_result(result);
    pol.degradation.update(inputs);
}
```

**Archivos tocados:**
- `src/bin/apollo-optimizerd/main.rs` — ~108 sitios migrados (UsageDomainState: ~4, PolicyState: ~106, circuit_breaker/degradation: 8)
- `src/bin/apollo-optimizerd/socket_handler.rs` — ~29 sitios (PolicyState: 27, Usage: 2)
- `src/engine/daemon_state.rs` — agregar circuit_breaker, degradation a PolicyState; eliminar duplicado UsageTrackerState

**Test plan:**
- `cargo build --bin apollo-optimizerd`
- `cargo test` — ≥2130 tests
- `cargo clippy --all-targets` — 0 warnings nuevos
- Test manual: profile switching via ctl (`ctl profile set performance`)
- Test manual: `ctl status` muestra profile, latency_target, governor correctamente
- Verificar circuit_breaker + degradation: ejecutar con alta presión y verificar que el daemon no falla

**Risk level:** Alto. PolicyState es el dominio con mayor blast radius. Estrategia: hacer esta PR en 2 commits:
  - Commit A: Migrar UsageDomainState (4 campos, ~4 sitios) — simple
  - Commit B: Migrar PolicyState (8 campos, ~140 sitios) — un commit grande pero atómico

---

### PR #14 — Migrar socket_handler.rs y llm_daemon.rs al nuevo SharedState

**Commit message:**
```
refactor(arch): DEBT-004 PR#14 — migrate socket_handler + llm_daemon to grouped SharedState
```

**Objetivo:** Actualizar los archivos secundarios para usar el nuevo SharedState de daemon_state.rs. En este punto, si los PRs #9–#13 se completaron correctamente, el flat SharedState en main.rs ya no debería tener campos de dominio — solo los de infraestructura.

**Nota:** Si los PRs #9–#13 ya migraron los sitios de socket_handler.rs y llm_daemon.rs en sus respectivos PRs (opción recomendada), esta PR es principalmente de limpieza de tipos. Si se decidió no tocar esos archivos en los PRs anteriores, esta PR los migra completamente.

**Archivos tocados:**
- `src/bin/apollo-optimizerd/socket_handler.rs` — actualizar import de `super::SharedState` al tipo correcto
- `src/bin/apollo-optimizerd/llm_daemon.rs` — ídem
- `src/bin/apollo-optimizerd/main.rs` — verificar que el flat SharedState ya solo tiene campos de infraestructura

**Cambio de tipo en socket_handler.rs:**
```rust
// Antes importa el tipo local:
use super::{SharedState, STOP_REQUESTED};

// Después importa el tipo de daemon_state:
use apollo_optimizer::engine::daemon_state::SharedState;
use super::STOP_REQUESTED;
```

**Test plan:**
- `cargo build --bin apollo-optimizerd`
- `cargo build --bin apollo-optimizerctl`
- `cargo test` — ≥2130 tests
- Test manual completo: daemon start → ctl status → ctl profile set → ctl thaw-all
- Test manual LLM: `ctl llm-status`, training cycle si disponible

**Risk level:** Bajo si los PRs anteriores completaron la migración de sitios. Alto si hay sitios residuales no migrados (build fallará, lo que es el safeguard correcto).

---

### PR #15 — Eliminar flat SharedState, limpieza final

**Commit message:**
```
refactor(arch): DEBT-004+DEBT-010 resolved — delete flat SharedState, finalize DomainSharedState
```

**Objetivo:** Borrar el flat SharedState de main.rs. Renombrar el tipo de daemon_state.rs como el SharedState canónico. main.rs queda < 4500 líneas.

**Cambios:**
1. Eliminar struct `SharedState` de main.rs (líneas 164–221)
2. Eliminar struct `ReactorStatus` de main.rs (si no se eliminó en PR #10)
3. Eliminar struct `UsageTrackerState` de main.rs (si no se eliminó en PR #13)
4. En daemon_state.rs: si el SharedState de daemon_state.rs fue el que se usó directamente, ya es el canónico. Si se usó un alias, hacer el rename definitivo.
5. Actualizar `use` statements en todos los archivos para referenciar `daemon_state::SharedState` directamente en lugar del alias o import desde main.
6. Actualizar DEBT_REGISTER.md: marcar DEBT-004 y DEBT-010 como `resolved`.

**Líneas eliminadas esperadas:**
- Flat SharedState struct: ~58 líneas (164–221)
- ReactorStatus en main.rs: ~29 líneas (224–252)
- UsageTrackerState en main.rs: ~5 líneas (255–260)
- Código de inicialización simplificado: ~30 líneas (construcción del flat SharedState más compacta)
- **Total eliminado: ~122 líneas**

Con la consolidación de locks en el hot loop que viene naturalmente de los PRs anteriores, la reducción total esperada es ~200–300 líneas adicionales por simplificación de bloques lock.

**Test plan:**
- `cargo build` — compila los 3 binarios
- `cargo test` — ≥2179 tests (baseline mínimo; añadir tests de construcción de SharedState si no existen)
- `cargo clippy --all-targets -- -D warnings` — 0 warnings
- `cargo fmt --all -- --check` — código formateado
- Verificar `wc -l src/bin/apollo-optimizerd/main.rs` — < 4500
- Verificar daemon arranca como root: `sudo cargo run --bin apollo-optimizerd -- daemon`
- `ctl status` → `ctl profile set performance` → `ctl status` → `ctl profile set balanced-root`
- Verificar que `/var/lib/apollo/` archivos persisten correctamente post-restart

**Risk level:** Bajo si PRs #9–#14 fueron correctos (es un cleanup). El build es el safeguard.

---

## 5. Invariantes que NUNCA se pueden romper

### 5.1 Compilación siempre verde

```bash
cargo build --bin apollo-optimizerd
```

Debe pasar **en cada commit individual**, no solo al final de cada PR. Si un commit intermedio rompe la compilación, ese commit no puede mergearse.

### 5.2 El daemon arranca y responde

```bash
cargo run --bin apollo-optimizerd -- daemon --profile balanced-root &
sleep 2
cargo run --bin apollo-optimizerctl -- status
```

El `status` debe devolver una respuesta válida (no timeout, no error de socket).

### 5.3 No deadlocks: regla de dos locks

**Nunca se adquieren dos locks de dominio en el mismo scope.**

Si un bloque de código necesita datos de dos dominios, la secuencia obligatoria es:

```rust
// PATRÓN CORRECTO:
let dato_a = {
    let guard = state.domain_a.lock_recover();
    guard.campo.clone()  // clonar/copiar lo necesario
};
// guard de domain_a se libera aquí
let mut guard_b = state.domain_b.lock_recover();
guard_b.campo = dato_a;  // usar el valor copiado

// PATRÓN INCORRECTO (deadlock potencial):
let guard_a = state.domain_a.lock_recover();
let guard_b = state.domain_b.lock_recover();  // <-- PROHIBIDO mientras guard_a vive
```

### 5.4 Orden de locks (si eventualmente necesitas dos)

Si hay un caso excepcional que realmente requiera dos locks simultáneos (justificado en el PR), la regla de adquisición es **orden alfabético de nombre de dominio**:

```
hardware < llm < metrics < policy < process < usage
```

Ejemplo: si necesitas hardware + policy, siempre adquirir hardware primero, luego policy. Nunca al revés.

### 5.5 reactor thread: solo MetricsState

El `run_reactor()` thread solo accede campos de `MetricsState` (metrics, reactor_status, thermal_level_real, reactor_event_weight, fast_tick_until). Esta invariante simplifica el análisis de concurrencia del reactor. Si se necesita acceder a otro dominio desde el reactor, se debe justificar en code review.

### 5.6 No duplicar estado

El ACL solo traduce referencias, nunca copia state. Un campo vive en exactamente un lugar en el nuevo esquema. Durante la migración, el campo plano del flat SharedState se reemplaza por un delegate al campo del dominio agrupado (mismo Arc), no por una copia.

---

## 6. Métricas de Éxito

| Métrica | Baseline (v0.8.0) | Target (v0.9.0) |
|---------|-------------------|-----------------|
| Líneas en main.rs | 5524 | < 4500 |
| Campos Arc<Mutex<>> en flat SharedState | 22 | 0 (struct eliminado) |
| Campos PathBuf standalone en flat SharedState | 7 | 0 |
| Grupos de dominio en SharedState | 0 | 6 |
| cargo test | ≥2130 passing | ≥2179 passing |
| cargo clippy warnings nuevos | 0 | 0 |
| DEBT-004 status | deferred | resolved |
| DEBT-010 status | in_progress | resolved |

---

## 7. Checklist Pre-Merge por PR

### Checklist universal (toda PR)

```bash
# 1. Compilación del binario afectado
cargo build --bin apollo-optimizerd

# 2. Suite completa de tests
cargo test 2>&1 | tail -5  # verificar "X passed, 0 failed"

# 3. Clippy sin warnings nuevos
cargo clippy --all-targets 2>&1 | grep "^warning" | grep -v "generated"

# 4. Formato correcto
cargo fmt --all -- --check
```

### PR #8 — Infra

```bash
cargo build --lib                   # daemon_state.rs compila como lib
cargo test --lib                    # tests de daemon_state.rs pasan
cargo test daemon_state             # tests específicos de daemon_state
# Verificar que no hay import de daemon_state::SharedState en main.rs aún
grep "daemon_state::SharedState" src/bin/apollo-optimizerd/main.rs | wc -l  # debe ser 0
```

### PR #9 — ProcessState

```bash
# Verificar que frozen_state solo se accede via state.process
grep "state\.frozen_state" src/bin/apollo-optimizerd/main.rs | wc -l  # debe ser 0
grep "state\.last_blockers" src/bin/apollo-optimizerd/main.rs | wc -l  # debe ser 0
grep "state\.wake_state" src/bin/apollo-optimizerd/main.rs | wc -l    # debe ser 0
# Y en los otros archivos:
grep "state\.frozen_state\|state\.last_blockers\|state\.wake_state" \
  src/bin/apollo-optimizerd/socket_handler.rs | wc -l  # debe ser 0
```

### PR #10 — MetricsState

```bash
# Verificar migración completa de todos los campos de metrics
for field in metrics thermal_state throttle_level reactor_event_weight fast_tick_until thermal_level_real reactor_status; do
  count=$(grep "state\.$field" src/bin/apollo-optimizerd/main.rs | wc -l)
  echo "$field: $count residual accesses (should be 0)"
done
# Verificar que ReactorStatus no está definido en main.rs
grep "^pub(crate) struct ReactorStatus" src/bin/apollo-optimizerd/main.rs | wc -l  # debe ser 0
```

### PR #11 — HardwareState

```bash
for field in mach_qos sysctl_governor_status last_hw_snapshot; do
  count=$(grep "state\.$field" src/bin/apollo-optimizerd/main.rs | wc -l)
  echo "$field: $count residual (should be 0)"
done
```

### PR #12 — LlmDomainState

```bash
for field in llm_cfg llm_state llm_state_path llm_key_path learned_policy_path feedback_path suggestions_path; do
  for file in src/bin/apollo-optimizerd/{main,socket_handler,llm_daemon}.rs; do
    count=$(grep "state\.$field" "$file" | wc -l)
    echo "$file::$field: $count residual"
  done
done
# Todos deben ser 0
```

### PR #13 — UsageDomainState + PolicyState

```bash
# Verificar migración de todos los campos de policy y usage
for field in profile latency_target governor timeline learned_policy adaptive_governor circuit_breaker degradation usage_model usage_tracker usage_model_path usage_events_path; do
  for file in src/bin/apollo-optimizerd/{main,socket_handler,llm_daemon}.rs; do
    count=$(grep "state\.$field" "$file" | wc -l)
    [ $count -gt 0 ] && echo "PENDIENTE: $file::$field ($count)"
  done
done
# Verificar que UsageTrackerState no está definido en main.rs
grep "^struct UsageTrackerState\|^#\[derive.*\]" src/bin/apollo-optimizerd/main.rs | grep -A1 "UsageTracker" | wc -l  # debe ser 0
```

### PR #14 — socket_handler + llm_daemon

```bash
# Verificar que el tipo SharedState importado ya no es el local de main.rs
grep "use super::SharedState" src/bin/apollo-optimizerd/socket_handler.rs | wc -l  # debe ser 0
grep "use super::SharedState" src/bin/apollo-optimizerd/llm_daemon.rs | wc -l     # debe ser 0
# Test manual de roundtrip:
# cargo run --bin apollo-optimizerd -- daemon &
# sleep 1
# cargo run --bin apollo-optimizerctl -- status
# cargo run --bin apollo-optimizerctl -- profile set performance
# cargo run --bin apollo-optimizerctl -- status
```

### PR #15 — Cleanup final

```bash
# Verificar eliminación del flat SharedState
grep "^pub(crate) struct SharedState" src/bin/apollo-optimizerd/main.rs | wc -l  # debe ser 0
# Verificar conteo de líneas
wc -l src/bin/apollo-optimizerd/main.rs  # debe ser < 4500

# Verificar 0 campos Arc<Mutex<>> residuales en el flat struct (no debería existir)
# Verificar tests
cargo test 2>&1 | grep "test result"
# Verificar que DEBT_REGISTER.md está actualizado
grep "DEBT-004" .plan/DEBT_REGISTER.md | grep "resolved"
grep "DEBT-010" .plan/DEBT_REGISTER.md | grep "resolved"
```

---

## 8. Riesgos Identificados y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| Deadlock MetricsState + PolicyState en hot loop | Media | Alto | Regla de 2 locks; extraer valores con lock breve |
| spawn_resource_sentinel recibe Arc de campo migrado | Alta | Medio | PR #9 cubre este sitio explícitamente |
| WakeRuntimeState duplicado (main.rs + daemon_state.rs) | Alta | Bajo | Resolver antes de PR #9 |
| UsageTrackerState duplicado | Alta | Bajo | Resolver antes de PR #13 |
| llm_cfg semántica diferente (Arc<LlmConfig> vs dentro de Mutex) | Media | Bajo | Decidir en PR #12: mantener como infraestructura o asumir penalización mínima |
| PolicyState muy grande → contención alta | Media | Medio | Medir con `cargo bench` post-v0.9.0 si hay regresión de latencia de ciclo |
| PR #13 demasiado grande (140 sitios en 1 PR) | Alta | Medio | Dividir en PR #13a (UsageDomainState) + PR #13b (PolicyState) si es necesario |

---

## 9. Notas de Implementación

### Compatibilidad con spawn_resource_sentinel

`spawn_resource_sentinel` (línea 928 de main.rs) toma `state.frozen_state.clone()` y `state.mach_qos.clone()`. Tras la migración:

```rust
// Opción A (preferida): pasar el Arc del dominio completo
spawn_resource_sentinel(
    smc_reader.cache_arc(),
    pressure_collector.cache_arc(),
    state.resource_interrupt.clone(),
    state.process.clone(),    // Arc<Mutex<ProcessState>>
    state.stop.clone(),
    SentinelConfig::default(),
    fg_detector.clone(),
    Some(state.hardware.clone()),  // Arc<Mutex<HardwareState>>
);

// La firma de spawn_resource_sentinel debe actualizarse en thermal_interrupt.rs
```

### Construcción del nuevo SharedState en main()

Post-migración, la inicialización del SharedState será más legible:

```rust
let state = daemon_state::SharedState {
    metrics: Arc::new(Mutex::new(MetricsState {
        metrics: RuntimeMetrics { effective_profile: profile, .. RuntimeMetrics::default() },
        throttle_level: "balanced".to_string(),
        thermal_state: "nominal".to_string(),
        // ...
    })),
    policy: Arc::new(Mutex::new(PolicyState {
        profile,
        governor,
        // ...
    })),
    // ... otros dominios
    stop: Arc::new(AtomicBool::new(false)),
    // ... infraestructura
};
```

### Alias durante la migración (PRs #8–#14)

Para evitar colisión de nombres durante la coexistencia de los dos tipos:

```rust
// En main.rs durante la migración:
use apollo_optimizer::engine::daemon_state::SharedState as DomainSharedState;
// El flat struct sigue siendo `SharedState` (definido en main.rs)
```

En PR #15, cuando el flat struct se elimina, el alias se convierte en el import directo y se renombra el tipo.

---

*Spec escrito por AG-ARCH (Claude Sonnet 4.6). Fecha: 2026-04-02. Basado en análisis estático de main.rs (5524 líneas), socket_handler.rs (798 líneas), llm_daemon.rs (743 líneas), daemon_state.rs (226 líneas). Conteo de sitios de acceso: ~499 total, ~382 en main.rs.*
