# Apollo Optimizer — Arquitectura del Sistema

> **Versión:** 0.2.0 · **Lenguaje:** Rust 2021 · **Plataforma:** macOS (Apple Silicon nativo)
> **Commit de referencia:** `951bb98` · **Actualizado:** 2026-03-13
> **Hardware objetivo:** MacBook Air M1 · 8 GB RAM (Unified Memory Architecture)

---

## Índice

1. [Filosofía de diseño](#1-filosofía-de-diseño)
2. [Vista general del sistema](#2-vista-general-del-sistema)
3. [Modelo de inteligencia de tres niveles](#3-modelo-de-inteligencia-de-tres-niveles)
4. [Arquitectura de binarios](#4-arquitectura-de-binarios)
5. [Módulos del motor (~60 módulos)](#5-módulos-del-motor)
6. [Pipeline de decisión](#6-pipeline-de-decisión)
7. [Sistema de seguridad y restricciones](#7-sistema-de-seguridad-y-restricciones)
8. [Sistema reactivo (kqueue)](#8-sistema-reactivo-kqueue)
9. [Máquina de estados del gobernador](#9-máquina-de-estados-del-gobernador)
10. [Overflow Guard — Aprendizaje de desbordamientos](#10-overflow-guard)
11. [Gestión avanzada de memoria](#11-gestión-avanzada-de-memoria)
12. [Térmica y energía](#12-térmica-y-energía)
13. [I/O Tiering granular](#13-io-tiering-granular)
14. [Wait-Graph — Prevención de deadlocks](#14-wait-graph)
15. [Integración LLM Teacher](#15-integración-llm-teacher)
16. [Capa de telemetría hardware](#16-capa-de-telemetría-hardware)
17. [Persistencia y estado](#17-persistencia-y-estado)
18. [Protocolo IPC](#18-protocolo-ipc)
19. [Suite de tests](#19-suite-de-tests)
20. [Instalación y despliegue](#20-instalación-y-despliegue)
21. [Issues de hardening pendientes](#21-issues-de-hardening-pendientes)

---

## 1. Filosofía de diseño

Apollo Optimizer es un **árbitro de recursos de nivel de sistema** para macOS Apple Silicon. Actúa donde el kernel XNU es generalista: prioriza energía pero no protege activamente las apps de primer plano contra el ruido de fondo (Electron, telemetría, indexadores).

**Cuatro principios arquitectónicos:**

1. **Observar, no adivinar.** Cada decisión se basa en estado medido: presión CPU, tendencias de memoria, sensores térmicos, velocidad de swap, tasa de wakeups y patrones de interacción de usuario.

2. **Inteligencia por niveles con latencia acotada.** Tres niveles de decisión con contratos estrictos: heurísticas (<1ms), ML bayesiano ligero (<5ms) y LLM cloud opcional (async, rate-limited). El hot-path nunca espera ML ni red.

3. **Conservador por defecto, agresivo por evidencia.** El daemon arranca en `BalancedRoot`. Escala a `AggressiveRoot` solo tras 3 ciclos consecutivos con presión sostenida >0.72. Baja tras 6 ciclos <0.55. La lógica anti-thrash bloquea el perfil si detecta oscilación. El `OverflowGuard` baja los thresholds automáticamente si ocurren desbordamientos reales.

4. **Reversibilidad como propiedad de primera clase.** Toda acción de optimización (SIGSTOP, renice, sysctl) se registra en un journal append-only con estado antes/después. El sistema puede restaurar el estado previo en cualquier momento. Si el daemon crashea, descongela todos los procesos al reiniciar.

---

## 2. Vista general del sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Apollo Optimizer System                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐  ┌───────┐  │
│  │ apollo-      │  │ apollo-          │  │ apollo-        │  │apollo-│  │
│  │ optimizer    │  │ optimizerd       │  │ optimizerctl   │  │menubar│  │
│  │ (CLI)        │  │ (Daemon)         │  │ (Client)       │  │ (UI)  │  │
│  │              │  │                  │  │                │  │       │  │
│  │ Comandos     │  │ Optimización     │  │ IPC queries    │  │ macOS │  │
│  │ puntuales    │  │ continua         │  │ & control      │  │ menu  │  │
│  └──────┬───────┘  └────────┬─────────┘  └───────┬────────┘  └───────┘  │
│         │                   │                     │                      │
│         │                   │   Unix Socket IPC   │                      │
│         │                   │◄───────────────────►│                      │
│         ▼                   ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Core Engine (~60 módulos)                      │   │
│  │                                                                   │   │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────────────┐   │   │
│  │  │ Nivel 1    │  │ Nivel 2    │  │ Nivel 3                  │   │   │
│  │  │ Heurísticas│  │ ML Ligero  │  │ LLM Teacher (opcional)   │   │   │
│  │  │ <1ms       │  │ <5ms       │  │ async, rate-limited      │   │   │
│  │  └────────────┘  └────────────┘  └──────────────────────────┘   │   │
│  │                                                                   │   │
│  │  ┌───────────────────────────────────────────────────────────┐   │   │
│  │  │              Subsistemas especializados                    │   │   │
│  │  │  Thermal · Memory · Swap · GPU · Power · Network          │   │   │
│  │  │  WakeStorm · ProcessRecovery · Analytics · OverflowGuard  │   │   │
│  │  │  WaitGraph · CompressorAware · ThermalBailout · IOTiering  │   │   │
│  │  └───────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   Interfaz con el kernel macOS                    │   │
│  │  kqueue · Mach task_policy_set · SIGSTOP/SIGCONT · sysctl        │   │
│  │  IOKit sensors · powermetrics · mdutil · memorystatus_control    │   │
│  │  proc_pidinfo · task_info(TASK_VM_INFO) · host_statistics64      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Modelo de inteligencia de tres niveles

### Nivel 1 — Heurísticas (<1ms)

| Módulo | Responsabilidad |
|--------|----------------|
| `adaptive_governor.rs` | Motor central de decisión por proceso (Allow / Throttle / Freeze / Kill) |
| `process_classifier.rs` | Categoriza procesos en 8 tiers por comportamiento |
| `zombie_hunter.rs` | Identifica 5 clases de procesos muertos o improductivos |

**Tiers de procesos:**

| Tier | Criterio |
|------|----------|
| `ActiveForeground` | Interacción de usuario en los últimos 30s |
| `BackgroundVisible` | App abierta, no enfocada, usada en últimos 5 min |
| `AppHelper` | Subproceso XPC/helper de una app activa |
| `SystemEssential` | Kernel, audio, display server |
| `SilentDaemon` | Servicio de fondo sin valor observable para el usuario |
| `Stale` | Sin uso en >24 horas |
| `ZombieOrphan` | Proceso muerto u huérfano |
| `Telemetry` | Analytics, crash reporters, diagnósticos |

**Fórmula de utilidad (0.0–1.0):**
```
utility = base_interaction_score      // 1.0 si <30s, 0.7 si <5min, 0.0 si >1h
        + gui_window_bonus    (0.3)
        + network_active      (0.2)
        - high_wakeup_penalty (0.5)   // >50 wakeups/sec
        + user_profile_boost  (var)   // de comportamiento aprendido
```

**Umbrales de decisión:**
- `utility < 0.1` Y carga pesada → **Freeze** (SIGSTOP)
- `utility < 0.4` → **Throttle** (renice +10)
- `waste_score > 0.9` → **Throttle** (override)
- Zombie/Huérfano → **Kill** (SIGKILL, tras 3 ciclos de confirmación)

### Nivel 2 — ML Ligero (<5ms)

| Módulo | Responsabilidad |
|--------|----------------|
| `workload_classifier.rs` | Clasificación bayesiana de carga de trabajo (5 fuentes de evidencia) |
| `user_profile.rs` | Aprendizaje conductual: estadísticas por app, modelo por hora del día |
| `hw_predictor.rs` | Predicción de tendencias de hardware |
| `hw_bayes.rs` | Modelo bayesiano de hardware |

**Fuentes de evidencia bayesiana:**

| Fuente | Peso | Descripción |
|--------|------|-------------|
| ForegroundApp | 2.0 | Coincidencia directa contra categorías conocidas |
| HourPrior | 0.3 | Distribución probabilística de 24h desde historial |
| AppRecency | 0.1–0.8 | Qué tan reciente fue la última interacción con apps relevantes |
| ProcessMix | variable | Conteo de procesos de fondo coincidentes |
| LlmLearned | 1.5 / -0.5 | Patrones de `LearnedPolicy` (interactivos/ruido) |

**Tipos de carga de trabajo:** Coding, VideoCall, MediaPlayback, VideoEdit, OfficeWork, CommandLine, Idle, General

### Nivel 3 — LLM Teacher (Opcional, Async)

| Módulo | Responsabilidad |
|--------|----------------|
| `llm.rs` | Refinamiento de políticas via API compatible OpenAI |

El LLM Teacher opera en una escala de tiempo diferente — no toma decisiones en tiempo real. Observa patrones del sistema y actualiza la `LearnedPolicy` que consume el Nivel 2. Rate-limited: 2 llamadas/hora, intervalo mínimo 15 min.

---

## 4. Arquitectura de binarios

### `apollo-optimizer` (CLI) — `src/main.rs`

| Comando | Acción |
|---------|--------|
| `snapshot` | Recoge métricas del sistema → guarda JSON |
| `optimize` | Ejecuta el motor de optimización una vez |
| `clean` | Limpieza de disco |
| `turbo` | Modo máximo rendimiento (deshabilita animaciones, tuning extremo) |
| `daemon` | Inicia el bucle de optimización continua |
| `startup` | Configura arranque inteligente (previene reapertura de apps) |
| `llm` | Optimización agresiva para cargas de trabajo IA/LLM |
| `restore` | Revierte todas las optimizaciones, descongela todo |

### `apollo-optimizerd` (Daemon) — `src/bin/apollo-optimizerd.rs`

Servicio de fondo de larga duración (~3,500 líneas). Responsabilidades principales:

- **Bucle de optimización principal** con tick rate adaptivo (2s–60s)
- **Servidor Unix socket** para IPC con `apollo-optimizerctl`
- **Hilo reactor** para respuesta basada en eventos (kqueue)
- **Persistencia de estado** entre reinicios (10 archivos de estado)
- **Gobernador de perfiles** con transiciones automáticas
- **Período de gracia post-wake** (60s tras sueño/despertar)
- **Kill switch** (`/var/run/apollo.disable`)

**SharedState** (~44 campos, `Arc<Mutex<T>>`):
```rust
SharedState {
    // Perfil y política
    profile: OptimizationProfile,
    latency_target: LatencyTarget,
    governor: ProfileGovernor,
    overflow_guard: OverflowGuard,      // NUEVO

    // Seguimiento de procesos
    frozen: HashSet<u32>,
    frozen_since: HashMap<u32, DateTime>,
    last_blockers: Vec<BlockerScore>,

    // Estado del sistema
    thermal_state: String,
    throttle_level: String,
    wake_state: WakeRuntimeState,

    // Módulos de inteligencia
    adaptive_governor: AdaptiveGovernor,
    workload_classifier: WorkloadClassifier,
    user_profile: UserProfile,
    llm_state: LlmState,
    learned_policy: LearnedPolicy,
    usage_model: UsageModel,

    // Interfaces de hardware
    mach_qos: MachQoSManager,
    iokit_reader: IOKitSensorReader,

    // Métricas y reactor
    metrics: RuntimeMetrics,            // 50+ contadores
    reactor_event_weight: f64,
    reactor_mode: String,
    reactor_health: String,
}
```

> **Deuda técnica conocida:** `SharedState` tiene ~44 campos `Arc<Mutex<T>>` — candidatos a agruparse en structs lógicos para reducir contención. La función `main()` tiene ~1,900 líneas — candidata a modularización.

### `apollo-optimizerctl` (Cliente) — `src/bin/apollo-optimizerctl.rs`

Cliente CLI ligero. Conecta al socket del daemon, envía peticiones JSON, muestra respuestas.

### `apollo-menubar` (UI nativa) — `src/bin/apollo-menubar.rs`

Interfaz de menú macOS nativa. Acceso rápido a estado y control desde la barra de menú.

---

## 5. Módulos del motor

Los módulos en `src/engine/` se organizan por función:

### Tipos y protocolo

| Módulo | Tipos clave |
|--------|------------|
| `types.rs` | `OptimizationProfile`, `RootAction`, `SafetyPolicy`, `RuntimeMetrics`, `BlockerScore`, `DaemonStatus` |
| `protocol.rs` | `DaemonRequest` (23 variantes), `DaemonResponse` (12 variantes) |
| `journal.rs` | `JournalEntry` — audit trail JSONL append-only |
| `lock_ext.rs` | Trait `LockRecover` — acceso uniforme a mutex con recuperación de poison |

### Seguridad y capacidades

| Módulo | Funciones clave |
|--------|----------------|
| `safety.rs` | `protected_processes()`, `critical_background_processes()`, `allowlisted_sysctls()`, `enforce_limits()` |
| `capabilities.rs` | `can_taskpolicy()`, `can_sysctl()`, `can_memorystatus()`, `can_mdutil()`, `is_root()` |
| `process_identity.rs` | Verificación de identidad PID (previene A-B-A recycling con `start_sec`/`start_usec`) |

### Decisión y ejecución

| Módulo | Propósito |
|--------|-----------|
| `decide_actions.rs` | Clasificación de contexto → detección de bloqueadores → generación de acciones |
| `execute_actions.rs` | Validación de existencia del proceso → ejecución de señales/renice/sysctl |
| `profile_governor.rs` | Puntuación de presión → transiciones de perfil → anti-thrash → overrides |
| `sysctl_governor.rs` | Gobernador de parámetros sysctl con allowlist estricta |

### Inteligencia (Nivel 1 y 2)

| Módulo | Latencia | Propósito |
|--------|----------|-----------|
| `adaptive_governor.rs` | <1ms | Motor heurístico central (Nivel 1) |
| `process_classifier.rs` | <2ms | Categorización en 8 tiers + utility score |
| `zombie_hunter.rs` | <5ms | Detección de dead-weight (3 ciclos de confirmación) |
| `workload_classifier.rs` | <1ms | Clasificación bayesiana de carga de trabajo (Nivel 2) |
| `user_profile.rs` | <1ms | Aprendizaje conductual: stats por app, modelo por hora |
| `hw_predictor.rs` | <1ms | Predicción de tendencias de hardware |
| `hw_bayes.rs` | <1ms | Modelo bayesiano de hardware |
| `predictive_agent.rs` | <5ms | Agente predictivo de estado del sistema |
| `signal_intelligence.rs` | <1ms | Inteligencia de señales del proceso |
| `outcome_tracker.rs` | <1ms | Seguimiento de resultados de acciones |
| `entropy_anomaly.rs` | <1ms | Detección de anomalías por entropía |
| `cusum.rs` | <1ms | CUSUM: detección de cambios de tendencia |
| `kalman.rs` | <1ms | Filtro de Kalman para señales sucias |
| `lotka_volterra.rs` | <1ms | Modelo depredador-presa para procesos compitiendo |
| `mpc_horizon.rs` | <5ms | Control predictivo de modelo (horizonte finito) |
| `hazard_model.rs` | <1ms | Modelo de riesgo por proceso |
| `activity_sensor.rs` | <1ms | Sensor de actividad de usuario |
| `foreground.rs` | <1ms | Detección de app en primer plano |
| `process_tree.rs` | <1ms | Árbol de procesos padre/hijo |

### Hardware y sistema

| Módulo | Propósito |
|--------|-----------|
| `iokit_sensors.rs` | Telemetría hardware via `powermetrics` (temps, potencia, utilización) |
| `mach_qos.rs` | Clases QoS de Mach: enrutamiento P-Core vs E-Core |
| `thermal_manager.rs` | Gestión térmica predictiva con historial de 60 muestras |
| `thermal_interrupt.rs` | Interrupciones térmicas con atomic ordering correcto (Release) |
| `thermal_bailout.rs` | Estrategia de enfriamiento graduada de 4 fases |
| `power_management.rs` | Modos de batería, estimación de potencia, acciones críticas |
| `energy.rs` | Cálculo de consumo energético en mW |
| `silicon_probe.rs` | Sonda de características del Silicon (AMX, LSE, RNDR) |
| `amx_detector.rs` | Detección del acelerador AMX (Apple Matrix coprocessor) |
| `lse_counters.rs` | Contadores de instrucciones LSE (Large System Extensions ARM64) |
| `smc_reader.rs` | Lectura de sensores SMC directamente |

### Gestión de memoria

| Módulo | Propósito |
|--------|-----------|
| `memory_analyzer.rs` | Profiling RSS/VMS/WSS, detección de leaks (>70% crecimiento → leak) |
| `swap_predictor.rs` | Predicción lineal de swap (30s adelante), tiempo hasta crítico |
| `compressor_aware.rs` | Gestión de memoria consciente del compresor (Freeze vs Hint) |
| `vm_surgeon.rs` | Cirujano de memoria virtual: análisis de footprint |
| `overflow_guard.rs` | Aprendizaje de desbordamientos — ajuste dinámico de thresholds |
| `jetsam_control.rs` | Control directo de Jetsam (memoria emergencia) |
| `kqueue_pressure.rs` | Presión de memoria via kqueue EVFILT_VM |

### Subsistemas especializados

| Módulo | Propósito |
|--------|-----------|
| `gpu_manager.rs` | Estados de potencia GPU, optimización por carga de trabajo |
| `network_optimizer.rs` | Perfiles de tuning TCP (HighThroughput, LowLatency, Balanced, Battery) |
| `network_monitor.rs` | Monitor de actividad de red por proceso |
| `wake_storm_detector.rs` | Detección de anomalías en tasa de wakeups (>10/s = tormenta) |
| `process_recovery.rs` | Kill automático + reinicio de procesos con memory leak |
| `analytics.rs` | Métricas de impacto acumuladas, estimados de energía/CO₂ |
| `usage_model.rs` | Seguimiento de uso por proceso para `usage top` / `usage explain` |
| `wait_graph.rs` | Grafo de dependencias IPC — veto de freeze si hay deadlock |
| `io_tiering.rs` | 5 niveles de prioridad I/O de Darwin via `taskpolicy -d` |
| `background_collectors.rs` | Colectores de fondo con watchdog |
| `proc_taskinfo.rs` | Información de tarea de proceso a nivel Mach |
| `mach_qos.rs` | Control fino de scheduling QoS en el kernel Mach |

---

## 6. Pipeline de decisión

```
                         ┌──────────────────────┐
                         │   Snapshot del sistema│
                         │   (sysinfo + IOKit +  │
                         │    host_statistics64) │
                         └──────────┬────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │  OverflowGuard         │
                         │  Aplica thresholds     │
                         │  dinámicos (bajados    │
                         │  si hubo overflows)    │
                         └──────────┬────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │  Clasificación contexto│
                         │                        │
                         │  CPU>88% O Mem>90%     │
                         │  → ThermalConstrained  │
                         │  CPU>72% O Mem>78%     │
                         │  → BackgroundPressure  │
                         │  Resto                 │
                         │  → InteractiveFocus    │
                         └──────────┬────────────┘
                                    │
           ┌────────────────────────┼──────────────────────┐
           │                        │                      │
┌──────────▼──────────┐             │          ┌───────────▼─────────┐
│  Detección bloqueador│             │          │  Clasificador proc.  │
│                      │             │          │                      │
│  Wait-graph scoring: │             │          │  8 tiers × utility   │
│  interactive_wait    │             │          │  score → decisión    │
│  × 0.45 +            │             │          │  por proceso         │
│  cpu_spike × 0.35 +  │             │          │                      │
│  seen_recent × 0.10  │             │          │  Zombie Hunter en    │
│  + reactor × 0.10    │             │          │  paralelo            │
│                      │             │          │                      │
│  score > 0.30        │             │          └───────────┬─────────┘
│  → Boost             │             │                      │
└──────────┬──────────┘             │                      │
           │                        │                      │
           └────────────────────────┼──────────────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │  Clasificador de carga │
                         │  (Bayesiano, Nivel 2)  │
                         │                        │
                         │  Confirma/ajusta el    │
                         │  nivel de agresión     │
                         │  según tipo de trabajo │
                         └──────────┬────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │  Seguridad             │
                         │                        │
                         │  • Procesos protegidos │
                         │  • Budgets de acciones │
                         │  • Allowlist sysctl    │
                         │  • Procs críticos bg   │
                         │  • Interactivos apren. │
                         └──────────┬────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │  Ejecutar acciones     │
                         │                        │
                         │  Validar PID →         │
                         │  verificar identidad → │
                         │  taskpolicy / renice / │
                         │  SIGSTOP / SIGCONT /   │
                         │  sysctl / mdutil       │
                         └──────────┬────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │  Journal + Métricas    │
                         │                        │
                         │  Append JSONL →        │
                         │  contadores atómicos   │
                         └───────────────────────┘
```

### Tipos de acción (`RootAction`)

| Acción | Mecanismo | Reversible |
|--------|-----------|------------|
| `BoostProcess` | `taskpolicy -l 0 -t 0` + `renice -10` | Sí (renice 0) |
| `ThrottleProcess` | `taskpolicy -l {2\|4} -d 4` + `renice {+10\|+20}` | Sí (renice 0) |
| `FreezeProcess` | `taskpolicy -d 4` + `SIGSTOP` | Sí (SIGCONT) |
| `UnfreezeProcess` | `SIGCONT` | N/A |
| `SetSysctl` | `sysctl -w key=value` (solo allowlist) | Sí (guardado previo) |
| `SetMemorystatus` | `sysctl kern.memorystatus_vm_pressure_send=PID` | Sí |
| `ToggleSpotlight` | `mdutil -i {on\|off} /` | Sí |
| `QuarantineDaemon` | Demote I/O + throttle CPU | Sí |

---

## 7. Sistema de seguridad y restricciones

### Procesos protegidos (nunca se tocan)

```
kernel_task   launchd       WindowServer   loginwindow
configd       securityd     tccd           syspolicyd
notifyd       hidd          UserEventAgent
Spotlight     mds           mds_stores     mdworker     mdworker_shared
```

### Procesos críticos de fondo (throttle ligero, nunca freeze)

```
podman   docker   colima   qemu-system          // Contenedores
postgres mysqld   redis-server   mongod          // Bases de datos
node     python   java     nginx                 // Servidores dev
go       ruby     php                            // Runtimes de lenguaje
rustc    cargo                                   // Compilación
```

### Invariante de interactivos aprendidos

Nunca se throttlea ni congela ningún proceso cuyo nombre coincida con `learned_interactive` (de `LearnedPolicy`) ni con la lista estática hardcoded. Esto aplica en **toda** condición de presión, incluyendo extrema. Los patrones confirmados incluyen: Antigravity, Claude, Brave, rustc/cargo.

### Sysctl permitidos (16 parámetros)

```
net.inet.tcp.sendspace          net.inet.tcp.recvspace
net.inet.tcp.delayed_ack        net.inet.tcp.win_scale_factor
net.inet.tcp.autorcvbufmax      net.inet.tcp.autosndbufmax
vm.compressor_poll_interval     vm.compressor_sample_min
kern.maxvnodes                  kern.maxfiles
kern.ipc.somaxconn              kern.ipc.maxsockbuf
iogpu.wired_limit_mb            debug.iogpu.wired_limit
debug.lowpri_throttle_enabled   kern.memorystatus_vm_pressure_send
```

### Budgets de acciones por ciclo

| Perfil | Boosts | Throttles | Hints | Freezes | Cooldown |
|--------|--------|-----------|-------|---------|----------|
| AggressiveRoot | 10 | 20 | 12 | 8 | 10s |
| BalancedRoot | 6 | 12 | 6 | 4 | 20s |
| SafeRoot | 3 | 6 | 3 | 2 | 45s |

### Invariantes de seguridad

1. Nunca congelar procesos críticos del sistema (`protected_processes()`)
2. Nunca congelar trabajo crítico de fondo (`critical_background_processes()`)
3. Todos los comandos externos usan `std::process::Command` — sin inyección de shell
4. Escrituras sysctl estrictamente en allowlist — solo 16 claves
5. Cooldown de transiciones de perfil: 90 segundos
6. Anti-thrash: >4 transiciones en 10 min → bloquear BalancedRoot por 5 min
7. Developer floor: nunca bajar a SafeRoot en sesiones interactivas/dev activas
8. Gracia post-wake: 60s de agresión suprimida tras despertar del sistema
9. Patrones LLM saneados: máx 80 chars, sin saltos de línea, confianza ≥ 0.80
10. PIDs congelados en `frozen_state.json` — descongelados al reiniciar daemon
11. Verificación de identidad PID: `start_sec`/`start_usec` — previene A-B-A recycling

---

## 8. Sistema reactivo (kqueue)

```
┌─────────────────────────────────────────────────────────┐
│                   Bucle de eventos kqueue                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Nervio 1: EVFILT_VM (Presión de memoria)               │
│    └─ NOTE_VM_PRESSURE → re-optimización inmediata      │
│                                                          │
│  Nervio 2: Darwin Notification (Térmica)                 │
│    └─ com.apple.system.thermalpressurelevel              │
│    └─ Cambio de temperatura → cascada térmica           │
│                                                          │
│  Nervio 3: Darwin Notification (Ciclo de vida)           │
│    └─ com.apple.launchd.spawn                            │
│    └─ Lanzamiento de nuevo proceso → clasificar y decidir│
│                                                          │
│  Nervio 4: Darwin Notification (Energía)                 │
│    └─ com.apple.system.powersources.source               │
│    └─ Conectar/desconectar AC → cambio de modo de energía│
│                                                          │
├─────────────────────────────────────────────────────────┤
│  Ante cualquier evento:                                  │
│    1. Incrementar contadores de evento                   │
│    2. Establecer fast_tick_until (acelerar bucle a 2s)   │
│    3. Recoger snapshot fresco                            │
│    4. Ejecutar ciclo de optimización inmediatamente      │
└─────────────────────────────────────────────────────────┘
```

**Tick rate adaptivo:**
- Normal: 60s entre ciclos
- Carga de trabajo pro detectada: 15s
- Evento reactor: 2s (durante `fast_tick_duration`)

---

## 9. Máquina de estados del gobernador

```
                 ┌──────────────────────────┐
                 │        SafeRoot           │
                 │    (conservador: 3/6/2)   │
                 └──────────┬───────────────┘
                            │
             presión ≥ 0.40 │ 3 consecutivos
             ───────────────►│
                            │◄──────────────
             presión ≤ 0.28 │ 6 consecutivos
                            │
                 ┌──────────▼───────────────┐
                 │       BalancedRoot        │
                 │    (defecto: 6/12/4)      │
                 └──────────┬───────────────┘
                            │
             presión ≥ 0.72 │ 3 consecutivos
             ───────────────►│
                            │◄──────────────
             presión ≤ 0.55 │ 6 consecutivos
                            │
                 ┌──────────▼───────────────┐
                 │      AggressiveRoot       │
                 │    (máximo: 10/20/8)      │
                 └──────────────────────────┘

    Anti-thrash: >4 transiciones en 10 min → bloquear BalancedRoot 5 min
    Developer floor: sesión dev activa → nunca bajar de BalancedRoot
    Override manual: perfil fijado por usuario con TTL (expira automáticamente)
    OverflowGuard: thresholds reales = base - overflow_penalty - build_penalty
```

**Fórmula de puntuación de presión:**
```
score = 0.35 × cpu_pressure
      + 0.35 × ram_pressure
      + 0.20 × interactive_wait_ratio
      + 0.10 × reactor_event_weight
```

**Presión RAM real (M1 con compresor):**
```
ram_pressure = max(kern.memorystatus_level, compressor_ratio × 0.85)
```
donde `compressor_ratio = total_uncompressed_pages_in_compressor / total_ram_pages`, leído via `host_statistics64` con `VmStats64` (size=152, offset=144).

---

## 10. Overflow Guard

`src/engine/overflow_guard.rs` — aprende de desbordamientos anteriores para prevenir futuros.

**Comportamiento:**
- **Por evento de overflow:** baja thresholds 5pp (piso: -20pp acumulados)
- **Recuperación:** +1pp por hora de sistema estable sin overflow
- **Build mode:** si ≥2 compiladores activos (rustc, cargo, cc, clang) → -8pp adicionales sobre el threshold actual
- **Persistencia:** `/var/lib/apollo/overflow_history.json`

**Resultado práctico en M1 8GB:**
- Threshold de presión para AggressiveRoot puede bajar de 0.72 a ~0.52 si el sistema ha tenido 4 overflows recientes
- Durante compilación activa: threshold efectivo ~0.44

---

## 11. Gestión avanzada de memoria

### Compressor-Aware (`compressor_aware.rs`)

Antes de congelar un proceso, Apollo analiza el ratio de compresión de su memoria via `task_info(TASK_VM_INFO)`:

- **Ratio alto (texto/datos):** → **Freeze** via SIGSTOP. El kernel mantiene el proceso comprimido en RAM; la recuperación es casi instantánea.
- **Ratio bajo (media/cifrado):** → **PressureHint** via `memorystatus_control`. Le pide a la app que libere sus cachés internas de forma segura, evitando Swap I/O costoso.

### Predictor de Swap (`swap_predictor.rs`)

Regresión lineal sobre el uso de swap para predecir colapsos de responsividad con 30 segundos de antelación. Calcula `time_to_critical`.

### Memory Analyzer (`memory_analyzer.rs`)

- Profiling RSS/VMS/WSS por proceso
- Detección de leaks: crecimiento >70% sostenido → clasificado como leak
- Reporta `leak_probability` (0.0–1.0)

### Wait-Graph y Freeze Safety

Antes de congelar cualquier proceso, `wait_graph.rs` verifica:
- Que ningún proceso en primer plano esté esperando un mensaje Mach del candidato
- Si hay dependencia IPC activa → **veto del freeze** o descongelamiento preventivo del waiter

---

## 12. Térmica y energía

### Gestión térmica predictiva (`thermal_manager.rs`)

- Historial de 60 muestras de temperatura
- Tendencia EMA para predicción
- Coordinado con `thermal_interrupt.rs` (atómica Release ordering)

### Estrategia de enfriamiento de 4 fases (`thermal_bailout.rs`)

| Fase | Rango | Acción |
|------|-------|--------|
| 1 - Suave | 80–85°C | Reduce I/O de fondo, hints de memoria purgeable |
| 2 - Moderado | 85–90°C | Fuerza tareas de fondo a E-Cores, throttle GPU |
| 3 - Agresivo | 90–95°C | Congela todos los daemons no esenciales, limita P-Cores al 40% |
| 4 - Emergencia | >95°C | Congela todo excepto servicios protegidos y app activa; P-Cores al 10% |

### Mach QoS Manager (`mach_qos.rs`)

Control directo del scheduler del kernel via `task_policy_set()`:

| Clase QoS | Target | Efecto |
|-----------|--------|--------|
| USER_INTERACTIVE | P-Cores (Firestorm) | Máximo throughput, mínima latencia |
| USER_INITIATED | P-Cores, menor prioridad | Alto throughput |
| DEFAULT | Decisión del scheduler | Balanceado |
| UTILITY | Cores mixtos | Menor impacto energético |
| BACKGROUND | Solo E-Cores (Icestorm) | I/O throttled, energía mínima |

---

## 13. I/O Tiering granular

`io_tiering.rs` utiliza los 5 niveles de prioridad I/O de Darwin via `taskpolicy -d`:

| Tier | Nivel | Para |
|------|-------|------|
| 0 - Interactive | Máxima prioridad | Swap paging, compilación activa |
| 1 - Standard | Normal | Apps de fondo visibles |
| 2 - Utility | Reducida | Spotlight, Time Machine |
| 3 - Throttle | Baja | Daemons silenciosos |
| 4 - Passive | Mínima | Telemetría diferible — solo ejecuta si el SSD está idle |

---

## 14. Wait-Graph

`wait_graph.rs` — prevención de deadlocks IPC:

**Análisis de hilos Mach:**
- Usa `proc_pidinfo` con `PROC_PIDLISTTHREADS` y `PROC_PIDTHREADINFO`
- Inspecciona `pth_run_state` de cada hilo

**Veto de freeze:**
- Si el proceso candidato a congelar tiene hilos en `TH_STATE_WAITING` y es probable lock-holder → **veto**
- Si el proceso en primer plano espera a un proceso de fondo → descongelamiento preventivo del waiter

**Stuck-detection:**
- Identifica periódicamente PIDs "stuck-frozen" atrapados en mitad de IPC y los recupera

---

## 15. Integración LLM Teacher

```
┌──────────────────────────────────────────────────────────────┐
│                    Modo LLM Teacher                           │
├──────────────────────────────────────────────────────────────┤
│  Configuración:                                               │
│    model: gpt-4.1-mini (compatible OpenAI)                   │
│    min_confidence: 0.80                                       │
│    max_calls_per_hour: 2                                      │
│    min_interval: 15 minutos                                   │
│    timeout: 5 segundos                                        │
│    training_window: 2 semanas (configurable)                  │
│                                                               │
│  Entrada (resumen del sistema):                               │
│    Presión CPU, estado memoria, nivel térmico,               │
│    top 10 procesos, perfil actual, patrones actuales         │
│                                                               │
│  Salida (JSON estructurado):                                  │
│    suggest_profile: OptimizationProfile                       │
│    suggest_latency_target: LatencyTarget                      │
│    add_interactive_patterns: Vec<String>  (máx 6)            │
│    add_noise_patterns: Vec<String>  (máx 6)                  │
│    add_protected_patterns: Vec<String>  (máx 6)              │
│    confidence: f64  (debe ser ≥ 0.80)                        │
│    rationale: String                                          │
│                                                               │
│  Salvaguardas:                                                │
│    • Patrones de Spotlight NUNCA aceptados                    │
│    • Patrones saneados: máx 80 chars, sin saltos de línea    │
│    • Máx 6 patrones por categoría por llamada                 │
│    • Gate de confianza: ≥ 0.80 requerido                     │
│    • Rate limiting: 2 llamadas/hora, 15 min mínimo           │
│    • Ventana de training expira → vuelve a solo heurísticas  │
│                                                               │
│  Almacenamiento:                                              │
│    /var/lib/apollo/learned_policy.json  (600, root:root)     │
│    /var/lib/apollo/suggestions.jsonl    (log de respuestas)  │
│    /var/lib/apollo/feedback.jsonl       (log de ratings)     │
│    /var/lib/apollo/llm_key_secret      (600, root:root)     │
└──────────────────────────────────────────────────────────────┘
```

---

## 16. Capa de telemetría hardware

### IOKit Sensor Reader (`iokit_sensors.rs`)

Datos via `powermetrics` (requiere root):

| Sensor | Fuente |
|--------|--------|
| Temperatura P-Cluster | Núcleos Firestorm (rendimiento) |
| Temperatura E-Cluster | Núcleos Icestorm (eficiencia) |
| Temperatura GPU | Apple GPU |
| Temperatura NAND | Controlador de almacenamiento |
| Potencia del paquete (W) | Consumo total del SoC |
| Potencia CPU (W) | Subsistema CPU |
| Potencia GPU (W) | Subsistema GPU |
| Potencia DRAM (W) | Subsistema de memoria |
| Utilización P-Core (%) | Carga de núcleos de rendimiento |
| Utilización E-Core (%) | Carga de núcleos de eficiencia |
| Carga batería (%) | Nivel actual de batería |
| Tasa de descarga (W) | Drenaje de batería |

---

## 17. Persistencia y estado

### Archivos de estado (root: `/var/lib/apollo/`, non-root: `/tmp/`)

| Archivo | Formato | Propósito | Frecuencia |
|---------|---------|-----------|------------|
| `journal.jsonl` | JSONL (append) | Audit trail de cada acción (antes/después) | Cada acción |
| `runtime_metrics.json` | JSON | 50+ contadores | Cada ciclo |
| `governor_state.json` | JSON | Perfil activo, cooldowns, conteo de transiciones | En transición |
| `profile_timeline.jsonl` | JSONL (append) | Historial de cambios de perfil | En transición |
| `frozen_state.json` | JSON | PIDs actualmente congelados | Solo si cambia el set |
| `wake_state.json` | JSON | Seguimiento de eventos sleep/wake | En sleep/wake |
| `learned_policy.json` | JSON | Patrones aprendidos por ML | En actualización LLM |
| `usage_model.json` | JSON | Estadísticas de uso por proceso | Periódico |
| `overflow_history.json` | JSON | Historial de overflows y thresholds ajustados | En overflow |
| `suggestions.jsonl` | JSONL (append) | Historial de sugerencias LLM | En llamada LLM |
| `feedback.jsonl` | JSONL (append) | Ratings del usuario | En feedback |

**Técnica de escritura:** Write-then-Rename — los archivos JSON nunca se corrompen durante apagón súbito.

### Métricas rastreadas (RuntimeMetrics — 50+ campos)

**Contadores de optimización:** cycles, boosts_applied, throttles_applied, freezes_applied, unfreezes_applied, paging_hints_applied, sysctl_applied

**Contadores de seguridad:** failures, invalid_sysctl_denied, critical_background_skips, heuristic_kills_downgraded

**Estado del sistema:** effective_profile, thermal_state, throttle_level, current_workload, ml_confidence

**Térmico/energía:** iokit_p_cluster_temp, iokit_e_cluster_temp, iokit_package_watts

**Reactor:** reactor_pulses, reactor_mode, reactor_health, reactor_events_total

**Supervivencia:** survival_mode_activations, kills_applied, zombies_detected

---

## 18. Protocolo IPC

**Socket Unix:** `/var/run/apollo-optimizer.sock` (root) / `/tmp/apollo-optimizer.sock` (non-root)

**Wire format (JSON con tags):**
```json
// Petición
{"type": "SetProfile", "payload": {"profile": "aggressive-root", "ttl_minutes": 60}}

// Respuesta
{"type": "Ok"}
```

**23 variantes de `DaemonRequest`** · **12 variantes de `DaemonResponse`**

**Permisos del socket:**
- Root: `0o660` (root:staff) — `SetLearnedPolicy` requiere root
- Non-root: `0o600`

---

## 19. Suite de tests

**621 tests totales: 0 fallos, 1 ignored** (`silicon_probe::read_rndr` → SIGILL en EL0, limitación de hardware)

| Nivel | Enfoque | Tests |
|-------|---------|-------|
| 1 - Unit | Seguridad, convergencia EMA, bounds | ~20 |
| 2 - Integration | Módulo safety, enforcement de acciones | ~25 |
| 3 - Concurrent | Acciones concurrentes, race conditions | ~20 |
| 4 - Advanced | Restricciones avanzadas, edge cases | ~15 |
| 5 - Tier1 Extended | Heurísticas extendidas | ~25 |
| 6 - Tier2 Features | Clasificación ML de carga de trabajo | ~20 |
| 7 - Tier3 Features | Modo LLM teacher | ~25 |
| 8 - Adaptive Intelligence | Gobernador adaptivo, recuperación | ~30 |
| 9 - M1 Native | Características nativas M1 (QoS, sensores) | ~20 |
| 10 - ML Ligero | Clasificador bayesiano, políticas aprendidas | ~28 |
| 11 - Subatomic | Tests de bajo nivel, primitivas | ~21 |

```bash
cargo test                  # Todos los tests
cargo test level1           # Nivel específico
cargo test test_nombre      # Test individual
```

---

## 20. Instalación y despliegue

### Compilación

```bash
cargo build --release     # LTO + native CPU (M1) + panic=abort
```

Produce 4 binarios:
- `target/release/apollo-optimizer` (CLI)
- `target/release/apollo-optimizerd` (Daemon)
- `target/release/apollo-optimizerctl` (Cliente)
- `target/release/apollo-menubar` (UI menú nativo)

### Instalación (launchd)

```bash
./scripts/install-root-daemon.sh
```

Instala:
- `/usr/local/libexec/apollo-optimizerd` (binario daemon)
- `/usr/local/bin/apollo-optimizerctl` (binario cliente)
- `/Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist` (servicio launchd)
- `/var/lib/apollo/` (directorio de estado, modo 700)
- `/etc/apollo-optimizer/config.toml` (configuración, modo 600)

### Kill switch

Crear `/var/run/apollo.disable` pausa toda optimización sin desinstalar.

---

## 21. Issues de hardening pendientes

### Completados

| # | Issue | Fix aplicado |
|---|-------|-------------|
| 1–6 | SIGTERM, permisos socket, seed policy, PID recycling, SIGCONT | Implementados en sesiones anteriores |
| 7 | ~2,500 string allocations/ciclo por `.to_lowercase()` | Pre-lowercase listas + `to_ascii_lowercase()` |
| 8 | `frozen_state.json` escrito incondicionalmente cada ciclo | Solo escribe si el frozen set cambió |
| 9 | 19 `.lock()` inconsistentes | Migradas a `.lock_recover()` — 0 instancias sin trait |
| 10 | `HardwareSnapshot` clonado 6 veces/ciclo | 1 clone al inicio, reutilizado |
| 11–12 | Tests flaky (doctest SIGILL, timing) | Marcados `no_run` / rangos relajados |

### Pendientes — Arquitectura (prioridad media)

| # | Issue | Impacto |
|---|-------|---------|
| A | SharedState con ~44 campos `Arc<Mutex<T>>` — candidatos a agruparse | Menos locks, menos contención |
| B | `main()` de ~1,900 líneas | Modularidad, testabilidad |

### Pendientes — Limpieza

| Item | Detalle |
|------|---------|
| `src/sysctl_tuner.rs` | Archivo huérfano en disco — ya no se compila. Borrar. |
| `silicon_probe::read_rndr` | Test `#[ignore]` por SIGILL en EL0 — limitación de hardware M1, no es bug. |

---

*Documento consolidado — reemplaza: `ARQUITECTURA_Y_MANUAL_COMPLETO.md`, `TECHNICAL_DEEP_DIVE.md`, `AGENT_CONTEXT.md`*
*Refleja el estado real del sistema en commit `951bb98` (2026-03-13)*
