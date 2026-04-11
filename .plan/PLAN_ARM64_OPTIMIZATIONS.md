# Plan: Optimizaciones ARM64/Apple Silicon de Bajo Nivel para Apollo

## Contexto

Apollo actualmente opera a nivel de **proceso completo** (task-level): congela, throttlea o boostea procesos enteros usando `task_policy_set()` y el CLI `taskpolicy`. En Apple Silicon M1 (big.LITTLE: 4 P-cores Firestorm + 4 E-cores Icestorm), esto es un instrumento grueso -- un proceso marcado "background" envia TODOS sus threads a E-cores, incluso si tiene un thread de UI critico.

**Problema**: Se desperdicia capacidad de optimizacion al no distinguir threads calientes de frios dentro del mismo proceso, y se paga ~5ms de overhead por cada `fork()/exec()` de `taskpolicy` CLI.

**Objetivo**: Llevar Apollo del nivel proceso al nivel thread, reemplazar subprocesos CLI por syscalls directos Mach, y optimizar estructuras internas -- todo respaldado por metodo cientifico con medicion antes/despues.

---

## Metodo Cientifico Aplicado

Cada fase sigue: **Hipotesis -> Experimento -> Medicion -> Conclusion**

---

## Fase 1: Thread-Level Scheduling (IMPACTO ALTO)

### Hipotesis

> En arquitectura big.LITTLE, la granularidad por-thread permite rutear threads calientes (UI, GPU compositing) a P-cores y threads frios (GC, telemetria) a E-cores dentro del mismo proceso, mejorando latencia interactiva 8-15% sin aumentar consumo energetico.

### Papers/Docs de referencia

- **ARM "big.LITTLE Technology" whitepaper (2013)** -- scheduling heterogeneo, como el kernel decide que threads van a que cores
- **Apple WWDC 2020 "Tune your app's performance on Apple Silicon"** -- QoS classes mapean a P-cores vs E-cores
- **XNU source `osfmk/kern/thread_policy.c`** -- implementacion interna de `thread_policy_set()`
- **ARM Architecture Reference Manual ARMv8-A (DDI 0487)** -- registros de sistema, niveles de excepcion EL0-EL3

### Estado actual en Apollo

| Que existe | Donde | Limitacion |
|------------|-------|-----------|
| `task_for_pid()` + `task_policy_set(TASK_CATEGORY_POLICY)` | `mach_qos.rs` | Solo nivel proceso, no thread |
| `proc_pidinfo(PROC_PIDTASKINFO)` | `proc_taskinfo.rs` | Reporta `thread_count` agregado, NO enumera threads individuales |
| `task_threads()` + `ThreadBasicInfo` | `optimizer.rs:1360-1392` | Solo para el propio proceso de Apollo, nunca sobre otros |
| Deteccion SIP + cache de PIDs bloqueados | `mach_qos.rs` | Reutilizable para threads |

### Que falta (gap critico)

1. `task_threads()` sobre procesos ajenos
2. `thread_policy_set()` per-thread
3. `THREAD_AFFINITY_POLICY` -- thread-to-core affinity hints
4. `THREAD_LATENCY_QOS_POLICY` / `THREAD_THROUGHPUT_QOS_POLICY` -- QoS por thread
5. Delta tracking de CPU por thread para clasificar hot/cold

### Implementacion detallada

#### Paso 1: Nuevas declaraciones FFI en `src/engine/mach_qos.rs`

```rust
// ========== Thread Policy Flavors ==========
pub const THREAD_AFFINITY_POLICY: i32 = 4;
pub const THREAD_AFFINITY_POLICY_COUNT: u32 = 1;
pub const THREAD_THROUGHPUT_QOS_POLICY: i32 = 5;
pub const THREAD_THROUGHPUT_QOS_POLICY_COUNT: u32 = 1;
pub const THREAD_LATENCY_QOS_POLICY: i32 = 7;
pub const THREAD_LATENCY_QOS_POLICY_COUNT: u32 = 1;

// Thread states (de osfmk/kern/thread.h)
pub const TH_STATE_RUNNING: i32 = 1;
pub const TH_STATE_STOPPED: i32 = 2;
pub const TH_STATE_WAITING: i32 = 3;
pub const TH_STATE_UNINTERRUPTIBLE: i32 = 4;
pub const TH_STATE_HALTED: i32 = 5;

#[repr(C)]
pub struct ThreadAffinityPolicy {
    pub affinity_tag: i32,  // threads con mismo tag -> co-scheduled
}

#[repr(C)]
pub struct ThreadThroughputQosPolicy {
    pub tier: i32,  // THROUGHPUT_QOS_TIER_0..4
}

#[repr(C)]
pub struct ThreadLatencyQosPolicy {
    pub tier: i32,  // LATENCY_QOS_TIER_0..4
}

extern "C" {
    // Enumerar threads de un task (requiere task port via task_for_pid)
    pub fn task_threads(
        task: u32,           // task port (de task_for_pid)
        thread_list: *mut *mut u32,  // OUT: array de thread ports
        thread_count: *mut u32,      // OUT: cuantos threads
    ) -> i32;  // KERN_SUCCESS = 0

    // Aplicar politica a un thread individual
    pub fn thread_policy_set(
        thread: u32,         // thread port (de task_threads)
        flavor: i32,         // THREAD_AFFINITY_POLICY, etc.
        policy_info: *const libc::c_void,
        count: u32,
    ) -> i32;

    // Obtener info de un thread individual
    pub fn thread_info(
        thread: u32,
        flavor: u32,         // THREAD_BASIC_INFO = 3
        thread_info_out: *mut i32,
        count: *mut u32,     // IN/OUT: size del buffer
    ) -> i32;

    // Liberar memoria del kernel (para el array de thread ports)
    pub fn mach_vm_deallocate(
        target: u64,         // mach_task_self()
        address: u64,        // ptr del array
        size: u64,           // count * sizeof(u32)
    ) -> i32;
}
```

#### Paso 2: Estructuras de introspection por thread

```rust
/// Snapshot de un thread individual dentro de un proceso
pub struct ThreadSnapshot {
    pub thread_port: u32,
    pub user_time_us: u64,     // microsegundos en userspace
    pub system_time_us: u64,   // microsegundos en kernel
    pub cpu_usage_raw: i32,    // 0-1000 fixed-point del kernel (por core)
    pub run_state: i32,        // TH_STATE_RUNNING, WAITING, etc.
    pub flags: i32,            // TH_FLAGS_SWAPPED, TH_FLAGS_IDLE
}

/// Todos los threads de un proceso
pub struct ProcessThreads {
    pub pid: u32,
    pub threads: Vec<ThreadSnapshot>,
    pub hot_count: usize,      // threads con cpu_delta > 5% wall-clock
    pub cold_count: usize,     // threads en WAITING >90% del ciclo
}
```

#### Paso 3: Metodo `enumerate_threads()` en MachQoSManager

```rust
pub fn enumerate_threads(&self, pid: u32) -> Option<ProcessThreads> {
    // 1. Verificar que no es SIP-protected (reusar is_sip_protected())
    // 2. task_for_pid(pid) -> task_port
    // 3. task_threads(task_port) -> [thread_port_0, thread_port_1, ...]
    // 4. Para cada thread: thread_info(THREAD_BASIC_INFO) -> ThreadSnapshot
    // 5. CRITICO: deallocar cada thread port + el array del kernel
    // 6. Clasificar hot/cold basado en prev_thread_cpu deltas
}
```

**Cleanup obligatorio** (evitar leak de Mach ports):
```rust
// RAII guard para cleanup automatico
struct ThreadListGuard {
    thread_list: *mut u32,
    count: u32,
    self_task: u32,
}
impl Drop for ThreadListGuard {
    fn drop(&mut self) {
        for i in 0..self.count {
            mach_port_deallocate(self.self_task, *self.thread_list.add(i));
        }
        mach_vm_deallocate(self.self_task as u64,
                          self.thread_list as u64,
                          (self.count * 4) as u64);
    }
}
```

#### Paso 4: Hot/Cold thread classification con delta tracking

```rust
// En MachQoSManager:
prev_thread_cpu: HashMap<(u32, usize), u64>,  // (pid, thread_idx) -> CPU time acumulado

// Clasificacion:
// - "Hot": delta CPU > 5% del wall-clock del ciclo -> merece P-core
// - "Cold": >90% del ciclo en TH_STATE_WAITING -> puede ir a E-core
// - "Mixed": ni hot ni cold -> dejar al scheduler del kernel decidir
```

#### Paso 5: Per-thread QoS application

```rust
pub fn set_thread_qos(&mut self, pid: u32, thread_idx: u32, tier: ThreadTier) -> bool {
    // 1. enumerate_threads(pid) para obtener thread ports
    // 2. Seleccionar thread por indice
    // 3. thread_policy_set(thread_port, THREAD_LATENCY_QOS_POLICY, tier)
    // 4. Cleanup de todos los ports
}

pub fn set_thread_affinity(&mut self, pid: u32, indices: &[u32], tag: i32) -> bool {
    // Agrupa threads con mismo affinity_tag -> scheduler los co-schedula
    // Util para threads de un mismo pipeline (productor/consumidor)
}
```

#### Paso 6: Nuevos RootAction en `src/engine/types.rs`

```rust
pub enum RootAction {
    // ... variantes existentes ...

    /// Cambiar QoS de un thread individual dentro de un proceso
    SetThreadQoS {
        pid: u32,
        name: String,
        thread_index: u32,
        tier: String,       // "interactive", "background", "utility"
        reason: String,
    },

    /// Agrupar threads con mismo affinity tag para co-scheduling
    SetThreadAffinity {
        pid: u32,
        name: String,
        affinity_tag: i32,
        thread_indices: Vec<u32>,
        reason: String,
    },
}
```

#### Paso 7: Safety en `src/engine/safety.rs`

```rust
// Nuevo campo en SafetyPolicy:
pub max_thread_qos_per_cycle: usize,

// Limites por perfil:
// BalancedRoot:    10 thread QoS changes por ciclo
// AggressiveRoot:  20
// SafeRoot:         4

// Reglas:
// - Procesos protegidos (kernel_task, launchd, etc.): RECHAZAR siempre
// - Procesos learned_interactive: SOLO boost, nunca demote
// - SIP-protected: falla silenciosamente en task_for_pid (ya manejado)
// - Contar thread actions hacia el budget total del ciclo
```

#### Paso 8: Integracion en `src/engine/decide_actions.rs`

```rust
// Para procesos en tier BackgroundVisible o AppHelper con CPU > 15%:
fn decide_thread_actions(pid: u32, threads: &ProcessThreads) -> Vec<RootAction> {
    let mut actions = vec![];
    for (idx, thread) in threads.threads.iter().enumerate() {
        if is_hot(thread) {
            // Thread caliente en proceso background -> boost a P-core
            actions.push(RootAction::SetThreadQoS {
                pid, thread_index: idx as u32,
                tier: "interactive".into(),
                reason: "hot-thread-in-mixed-process".into(),
                ..
            });
        } else if is_cold(thread) {
            // Thread frio en proceso boosted -> demote a E-core
            actions.push(RootAction::SetThreadQoS {
                pid, thread_index: idx as u32,
                tier: "background".into(),
                reason: "cold-thread-save-energy".into(),
                ..
            });
        }
    }
    actions
}
```

### Medicion (antes/despues)

| Metrica | Como medir | Esperado |
|---------|-----------|----------|
| Latencia interactiva P95 | Key-to-screen en terminal durante compilacion | -8 a -15% |
| Energia total | `proc_pid_rusage(RUSAGE_INFO_V4).billed_energy` | Neutral o -5% |
| Tiempo compilacion | `time cargo build --release` con browser+Slack abiertos | -3 a -5% |
| E-core utilization | IOKit `IOCLK_BUSY_PERCENT` para ECPU cluster | +10-20% (mejor uso) |

### Seguridad critica

1. **Port leak**: cada `task_threads()` DEBE deallocar todos los thread ports + el array. RAII guard obligatorio
2. **Threads efimeros**: `thread_policy_set()` retorna `KERN_INVALID_ARGUMENT` si el thread murio entre enumeracion y aplicacion -> manejar sin contar como fallo
3. **Race condition**: proceso puede morir entre `task_for_pid()` y `task_threads()` -> ya manejado por `permanently_blocked` logic
4. **Budget**: thread actions cuentan hacia limites por ciclo para evitar work unbounded

---

## Fase 2: Syscalls Mach Directos (Reemplazar CLI)

### Hipotesis

> Cada `Command::new("/usr/sbin/taskpolicy")` cuesta ~5ms (fork+exec+wait). Reemplazar con `task_policy_set()` directo reduce a ~50us (100x). Con 8-12 acciones por ciclo, se ahorran ~40-60ms por ciclo.

### Referencia

- Stevens & Rago, "Advanced Programming in the UNIX Environment" -- fork/exec overhead medido
- Apple `task_policy_set()` man page -- interfaz directa al kernel
- Mediciones empiricas en XNU userspace tests

### Ubicaciones a modificar

| Archivo | Linea | Que hace | Reemplazo |
|---------|-------|---------|-----------|
| `execute_actions.rs` | 132-133 | `taskpolicy -l 0 -p PID` (boost) | `qos_mgr.set_tier(pid, Foreground)` |
| `execute_actions.rs` | 134 | `taskpolicy -t 0 -p PID` (throughput) | `qos_mgr.set_throughput_qos(pid, Tier0)` |
| `execute_actions.rs` | 187 | `taskpolicy -l <tier> -p PID` (throttle) | `qos_mgr.set_tier(pid, Background)` |
| `io_tiering.rs` | 60 | `taskpolicy -d <tier>` (I/O) | `task_policy_set()` con I/O category |
| `thermal_interrupt.rs` | 622 | `taskpolicy -b` (sentinel) | `qos_mgr.set_tier(pid, Background)` |
| `optimizer.rs` | 962-967 | `taskpolicy` legacy | `qos_mgr` directo |

### Implementacion

#### Paso 1: Nuevos metodos en `src/engine/mach_qos.rs`

```rust
pub fn set_latency_qos(&mut self, pid: u32, tier: LatencyTier) -> QoSOutcome {
    // task_for_pid() -> task_policy_set(TASK_QOS_POLICY, latency_tier)
    // Constantes ya existen en optimizer.rs lineas 1418-1420
}

pub fn set_throughput_qos(&mut self, pid: u32, tier: ThroughputTier) -> QoSOutcome {
    // task_for_pid() -> task_policy_set(TASK_QOS_POLICY, throughput_tier)
}
```

#### Paso 2: Modificar `execute_actions()` signature

```rust
// Antes:
pub fn execute_actions(actions, caps, frozen, learned_protected, learned_interactive) -> ExecuteOutcomes

// Despues:
pub fn execute_actions(actions, caps, frozen, learned_protected, learned_interactive,
                       qos_mgr: &mut MachQoSManager) -> ExecuteOutcomes
```

#### Paso 3: Reemplazar llamadas CLI

```rust
// Antes (BoostProcess):
let _ = run("/usr/sbin/taskpolicy", &["-l", "0", "-p", &pid_s]);
let _ = run("/usr/sbin/taskpolicy", &["-t", "0", "-p", &pid_s]);
let _ = run("/usr/bin/renice", &["-10", "-p", &pid_s]);

// Despues:
qos_mgr.set_tier(pid, SchedulingTier::Foreground);  // ~50us vs ~5ms
qos_mgr.set_latency_qos(pid, LatencyTier::Interactive);
// renice se mantiene (no hay equivalente Mach directo para nice value)
```

#### Paso 4: Sentinel thread

`thermal_interrupt.rs` linea 622: pasar `Arc<Mutex<MachQoSManager>>` al sentinel thread para que use `qos_mgr` directo.

### Medicion

| Benchmark | Antes | Despues | Mejora |
|-----------|-------|---------|--------|
| 100 BoostProcess seguidos | ~500ms | ~5ms | 100x |
| Cycle time P95 (1000 ciclos, 10+ acciones) | ~80ms | ~40ms | 2x |
| Sentinel response time | ~30ms | ~6ms | 5x |

### Fallback

El CLI `taskpolicy` se mantiene como fallback cuando `task_for_pid()` falla (proceso SIP-protected o sin root).

---

## Fase 3: Lock Optimization (Mutex -> RwLock)

### Hipotesis

> Los ~30 campos `Arc<Mutex<T>>` en `SharedState` son leidos por socket handlers (cada `status` request) y escritos 1x por ciclo (30s). `RwLock` permite lecturas concurrentes sin bloqueo mutuo, eliminando latency spikes en respuestas de status bajo carga concurrente.

### Referencia

- Herlihy & Shavit, "The Art of Multiprocessor Programming" (2012), Capitulo 8
- `pthread_rwlock_t` en Darwin kernel -- implementacion lock-free para readers

### Campos a convertir

| Campo en SharedState | Patron de acceso | Convertir? |
|---------------------|-------------------|-----------|
| `profile` | read: cada status, write: cambio perfil (raro) | SI -> RwLock |
| `metrics` | read: cada status, write: 1x/ciclo | SI -> RwLock |
| `thermal_state` | read: cada status + sentinel, write: 1x/ciclo | SI -> RwLock |
| `throttle_level` | read: cada status, write: 1x/ciclo | SI -> RwLock |
| `last_blockers` | read: cada status, write: 1x/ciclo | SI -> RwLock |
| `governor` | read: cada status, write: cambio perfil | SI -> RwLock |
| `learned_policy` | read: cada ciclo, write: LLM suggestion (raro) | SI -> RwLock |
| `frozen_state` | read+write: cada freeze/unfreeze, main loop | NO (write-heavy) |
| `mach_qos` | write: cada accion | NO (exclusive access needed) |

### Implementacion

#### Paso 1: `src/engine/lock_ext.rs` -- nuevo trait

```rust
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub trait RwLockRecover<T> {
    fn read_recover(&self) -> RwLockReadGuard<'_, T>;
    fn write_recover(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T> RwLockRecover<T> for RwLock<T> {
    fn read_recover(&self) -> RwLockReadGuard<'_, T> {
        self.read().unwrap_or_else(|e| e.into_inner())
    }
    fn write_recover(&self) -> RwLockWriteGuard<'_, T> {
        self.write().unwrap_or_else(|e| e.into_inner())
    }
}
```

#### Paso 2: `src/bin/apollo-optimizerd.rs` -- migrar campos

```rust
// Antes:
pub profile: Arc<Mutex<OptimizationProfile>>,
// Despues:
pub profile: Arc<RwLock<OptimizationProfile>>,

// Socket handlers (lectura):
let p = state.profile.read_recover();
// Main loop (escritura):
let mut p = state.profile.write_recover();
```

### Medicion

- 8 threads concurrentes leyendo `metrics` mientras main loop escribe
- **Esperado**: latency spike P99 eliminado en status responses

### Riesgo: writer starvation

`pthread_rwlock_t` en macOS tiene writer starvation por defecto. Dado que writes son raros (1x/30s), esto NO es un problema real.

---

## Fase 4: NEON SIMD para Batch Metrics (CONDICIONAL)

> **SOLO implementar si mediciones de Fases 1-3 muestran que el decision loop es bottleneck (>5ms).**

### Hipotesis

> Con ~400 procesos, vectorizar comparaciones de threshold (4 f32 por instruccion NEON) reduce el tiempo de decision ~4x, de ~3ms a ~0.8ms.

### Referencia

- ARM NEON Programmer's Guide (DEN0018A) -- intrinsics y uso optimo
- Apple M1: 4 unidades NEON por P-core, 128-bit SIMD
- Rust `std::arch::aarch64` -- API estable para NEON intrinsics

### Implementacion

#### Nuevo archivo: `src/engine/simd_batch.rs`

```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Encuentra indices de procesos con CPU > threshold
/// Procesa 4 f32 por instruccion NEON (128-bit)
pub fn batch_threshold_check(cpu_usages: &[f32], threshold: f32) -> Vec<usize> {
    let mut result = Vec::new();
    let thresh_v = unsafe { vdupq_n_f32(threshold) };
    let chunks = cpu_usages.len() / 4;

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(cpu_usages.as_ptr().add(i * 4)) };
        let cmp = unsafe { vcgtq_f32(v, thresh_v) };
        // Extraer lanes con resultados positivos
        for lane in 0..4 {
            if unsafe { vgetq_lane_u32(vreinterpretq_u32_f32(cmp), lane) } != 0 {
                result.push(i * 4 + lane as usize);
            }
        }
    }
    // Remainder escalar
    for j in (chunks * 4)..cpu_usages.len() {
        if cpu_usages[j] > threshold { result.push(j); }
    }
    result
}

/// Calcula scores de blocker usando fused multiply-add vectorizado
/// score = wait_ratio * 0.45 + cpu_spike * 0.35 + seen * 0.10 + reactor * 0.10
pub fn batch_blocker_score(
    wait_ratios: &[f32], cpu_spikes: &[f32],
    seen_flags: &[f32], reactor_weights: &[f32],
) -> Vec<f32> {
    let mut scores = vec![0.0f32; wait_ratios.len()];
    let chunks = wait_ratios.len() / 4;

    unsafe {
        let w1 = vdupq_n_f32(0.45);
        let w2 = vdupq_n_f32(0.35);
        let w3 = vdupq_n_f32(0.10);

        for i in 0..chunks {
            let off = i * 4;
            let wr = vld1q_f32(wait_ratios.as_ptr().add(off));
            let cs = vld1q_f32(cpu_spikes.as_ptr().add(off));
            let sf = vld1q_f32(seen_flags.as_ptr().add(off));
            let rw = vld1q_f32(reactor_weights.as_ptr().add(off));

            // FMA: score = wr*0.45 + cs*0.35 + (sf+rw)*0.10
            let s = vfmaq_f32(
                vfmaq_f32(
                    vmulq_f32(vaddq_f32(sf, rw), w3),
                    wr, w1
                ),
                cs, w2
            );
            vst1q_f32(scores.as_mut_ptr().add(off), s);
        }
    }
    // Remainder escalar
    for j in (chunks * 4)..wait_ratios.len() {
        scores[j] = wait_ratios[j] * 0.45
                   + cpu_spikes[j] * 0.35
                   + (seen_flags[j] + reactor_weights[j]) * 0.10;
    }
    scores
}
```

### Impacto esperado

- Ahorro: ~1-2ms por ciclo en decision logic
- **Marginal** comparado con Fases 1-2 (el bottleneck real son los syscalls de `proc_pidinfo`)

---

## Fase 5: Cache & Prefetch (CONDICIONAL)

> **SOLO si Instruments muestra >5% L1D cache miss rate en process scanning.**

### Hipotesis

> Structure-of-Arrays (SoA) para datos de procesos mejora cache utilization -- 1 cache line (64 bytes M1) cubre 16 procesos (f32 cada uno) en vez de 1 proceso (struct grande).

### Referencia

- ARM Cortex-A Series Programmer's Guide, Capitulo 17: Caches
- Apple M1: L1D 128KB/core, 64-byte cache lines, hardware prefetcher agresivo
- ARM Architecture Reference Manual -- instruccion PRFM

### Implementacion

#### Paso 1: SoA layout en `src/engine/proc_taskinfo.rs`

```rust
/// Structure-of-Arrays para batch processing de procesos
/// 1 cache line (64 bytes) = 16 f32 = 16 procesos para threshold checks
pub struct ProcessBatch {
    pub pids: Vec<u32>,         // hot: iterado siempre
    pub cpu_usages: Vec<f32>,   // hot: threshold checks
    pub mem_usages: Vec<u64>,   // warm: pressure checks
    pub names: Vec<String>,     // cold: solo cuando necesitamos el nombre
    pub thread_counts: Vec<u32>,// cold: solo para thread-level decisions
}
```

#### Paso 2: Cache-align hot structures

```rust
#[repr(align(64))]  // 1 cache line M1
pub struct ThreadSnapshot { ... }

#[repr(align(64))]
pub struct PressureData { ... }
```

#### Paso 3: Software prefetch (impacto probablemente minimo)

```rust
// En bulk_process_scan(), prefetch 2 iteraciones adelante
#[cfg(target_arch = "aarch64")]
unsafe {
    std::arch::asm!(
        "prfm pldl1keep, [{addr}]",
        addr = in(reg) &pids[i + 2],
        options(nostack, nomem, preserves_flags)
    );
}
```

**Nota**: El hardware prefetcher del M1 es muy agresivo con accesos secuenciales. Software prefetch probablemente no ayude aqui. El verdadero beneficio es el SoA layout.

### Impacto esperado

- Cache miss rate: de ~5-8% a ~1-2% (si el bottleneck es cache, no syscalls)
- **Probablemente negligible** dado que `proc_pidinfo()` syscall (~2us) domina

---

## Resumen: Orden de Implementacion

```
                           IMPACTO
                             |
Fase 1: Thread Scheduling ████████████████████ ALTO   (8-15% latencia)
                             |
Fase 2: Direct Mach        ██████████████      MEDIO  (100x per-action)
                             |
Fase 3: RwLock             ████████            BAJO+  (P99 status)
                             |
Fase 4: NEON SIMD          ████                BAJO   (solo si justificado)
                             |
Fase 5: Cache/SoA          ██                  MINIMO (solo si justificado)
```

### Dependencias

```
Fase 1 (Thread Scheduling)     <-- Independiente, hacer PRIMERO
  |
  v
Fase 2 (Direct Mach Syscalls)  <-- Usa infraestructura FFI de Fase 1
  |
  v
Fase 3 (RwLock Migration)      <-- Independiente de 1 y 2, rapido
  |
  v
Fase 4 (NEON SIMD)             <-- MEDIR primero, implementar si justificado
  |
  v
Fase 5 (Cache/Prefetch)        <-- MEDIR primero, implementar si justificado
```

## Archivos Criticos a Modificar

| Archivo | Fases | Cambios principales |
|---------|-------|-------------------|
| `src/engine/mach_qos.rs` | 1, 2 | FFI threads, enumerate_threads, QoS directo, affinity, RAII guard |
| `src/engine/types.rs` | 1 | SetThreadQoS, SetThreadAffinity en RootAction; thread budget en SafetyPolicy |
| `src/engine/execute_actions.rs` | 1, 2 | Match arms para thread actions; reemplazar CLI con qos_mgr |
| `src/engine/decide_actions.rs` | 1, 4 | Thread classification hot/cold; SIMD scoring |
| `src/engine/safety.rs` | 1 | max_thread_qos_per_cycle, validacion thread actions |
| `src/engine/lock_ext.rs` | 3 | RwLockRecover trait |
| `src/bin/apollo-optimizerd.rs` | 2, 3 | Pass MachQoSManager a execute; Mutex->RwLock |
| `src/engine/io_tiering.rs` | 2 | Direct Mach I/O policy (fallback CLI) |
| `src/engine/thermal_interrupt.rs` | 2 | Direct QoS en sentinel thread |
| `src/engine/proc_taskinfo.rs` | 5 | ProcessBatch SoA layout |
| `src/engine/simd_batch.rs` (NUEVO) | 4 | NEON batch threshold + scoring |

## Verificacion End-to-End

1. `cargo test` -- todos los 182+ tests deben pasar
2. `cargo clippy --all-targets` -- sin warnings nuevos
3. **Benchmark manual**: compilar proyecto Rust mediano con browser + Slack abiertos
   - Medir `cycle_time_us`, latencia interactiva, energia
4. **Stress test**: 8 clientes `apollo-optimizerctl status` concurrentes (validar RwLock)
5. **Safety test**: verificar que procesos protegidos/interactivos NUNCA reciben thread-level demotion
6. **Port leak test**: correr daemon 1 hora, verificar con `sudo lsof -p <daemon_pid>` que Mach ports no crecen
7. `cargo build --release` + instalar daemon y validar en produccion
