# Navegación
[<- System Overview](./01_System_Overview.md) | [Volver al Índice](./00_Index.md) | [Siguiente: Cognitive Architecture ->](./03_Cognitive_Architecture.md)

# 02. Core Execution Engine — El Sistema Reactivo (L0)

Este documento describe el **motor de ejecución reactivo** de Apollo: la capa que opera cada tick del daemon recolectando sensores, calculando presión, clasificando procesos, emitiendo decisiones y ejecutándolas con restricciones de seguridad. Comprende la mayor parte de los 126 módulos  en `src/engine/` (~77,178 LOC).

---

## 1. Ciclo del Reactor (Tick Lifecycle Detallado)

Cada tick ejecuta 8 etapas secuenciales. El daemon completo consume <10ms por ciclo en un M1 8GB:

### Etapa 1: Recolección de Sensores (~30ms)

| Sensor | Fuente | Archivo | Datos extraídos |
|--------|--------|---------|-----------------|
| Árbol de procesos | `sysinfo` crate | `collector.rs` | CPU%, RSS, PID, PPID, nombre, estado, wakeups, por cada uno de ~400 procesos |
| RAM del kernel | `host_statistics64()` Mach | `host_vm_info.rs` | `free_count`, `active_count`, `inactive_count`, `compressor_page_count`, `pageins`, `pageouts`, `swapins`, `swapouts` |
| Sensores IOKit | IOKit framework | `iokit_sensors.rs` | Temperatura GPU, watts del paquete, GPU%, fan RPM, battery mWh |
| SMC directo | AppleSMC (via C bridge) | `smc_direct.rs` | CPU temp (clave `Tc0P`), board temp, battery temp, battery overheat flag |
| Presión VM | kqueue `EVFILT_VM` | `kqueue_pressure.rs` | `NOTE_VM_PRESSURE` events, Darwin notifications (thermal, spawn, power) |
| Silicon info | Detección de chip | `silicon_probe.rs` | Generación M1/M2/M3/M4, core count (P+E), RAM total, capacidades LSE/RNDR |
| AMX detector | Coprocessor IA | `amx_detector.rs` | Si el AMX está activo (CoreML/PyTorch workloads) |
| LSE/KPC | Hardware counters | `lse_counters.rs`, `kpc_counters.rs` | ARM64 atomics, PMC performance counters EL0 |
| IOReport per-cluster | Apple CPU clusters | `ioreport.rs` | Utilización ECPU vs PCPU por cluster, con patrón de nombre broadened para macOS 26 |
| CPU saturation | `host_processor_info()` | `cpu_saturation.rs` | Per-core utilization, detección de saturación |
| Contention tracker | PSI-style stall | `contention_tracker.rs` | Per-process `cpu_contention` ratio (queue-wait / total) |

**Optimización clave (commit `6c0da93`):** Se eliminó el refresh de disco y red de `sysinfo` porque causaba spam de `CacheDelete` en `logd`. Solo se refreshan procesos y CPU.

### Etapa 2: Presión Efectiva (~1ms)

`effective_pressure.rs` calcula la **presión autoritativa** del sistema como una métrica compuesta [0.0, 1.0]:

```
effective = base_kernel_pressure
          + hardware_boost        (Warning=+0.15, Critical=+0.30)
          + battery_boost         (Normal=+0.04, LowPower=+0.10, Critical=+0.18)
          + thermal_boost         (Phase1=+0.07, Phase2=+0.15, Phase3=+0.25, Phase4=+0.40)
          + llm_workload_boost    (ollama/llama detected=+0.20)
          + charging_stress       (>8W while charging=+0.06)
          + battery_low           (TTE <20min=+0.08)
          + memory_bandwidth      (AMC >80% saturated=+0.10)
          + smc_thermal           (≥80°C=+0.05, ≥90°C=+0.15, ≥100°C=+0.30)
          + battery_overheat      (flag=+0.12)
```

**Resultado:** `clamp(0.0, 1.0)`. Ejemplo real: `base=0.60 + hw=0.15 + batt=0.04 + thermal=0.07 = 0.86 → BackgroundPressure` (sin los boosts, 0.60 sería `InteractiveFocus` y no haría nada).

**Contexto forzado por Swap Exhaustion (commit `e0cd030`):** Si swap ≥ 4GB, la presión se fuerza a `BackgroundPressure` independientemente del cálculo base.

**Tratamiento de páginas comprimidas (commit `ebf7fa2`):** Las compressor pages se tratan como 30% disponibles (no como memoria ocupada), alineándose con la semántica real del kernel macOS. Esto corrigió un ciclo de pánico donde Apollo interpretaba RAM comprimida como presión extrema, generando cascadas de acciones innecesarias.

### Etapa 3: OverflowGuard (~1ms)

`overflow_guard.rs` (23,741 bytes) — Aprendizaje adaptativo para prevenir OOM:

**Thresholds base:**
- `bg_pressure = 0.78` — Umbral para acciones de background.
- `critical = 0.88` — Umbral de presión crítica.
- `extreme = 0.90` — Umbral de presión extrema.

**Ajustes aditivos:**
- `overflow_offset`: Cada overflow detectado baja -5pp, piso -20pp, half-life 8 horas.
- `workload_bonus`: Idle=+3pp, Interactive=+1pp, Build=-3pp, HeavyBuild=-5pp.
- `rl_adjustment`: Corrección aprendida online por Q-learning (Phase 4, ver doc 04).
- `device_offset`: ≤8GB=-5pp, ≤16GB=0pp, >16GB=+5pp.

**Deduplicación:** Ventana de 60s entre eventos del mismo overflow.
**Persistencia:** `overflow_history.json` sobrevive reboots (máx 20 eventos).
**Pattern matching:** `resembles_past_overflow()` compara la composición actual de procesos con el historial.

### Etapa 4: Clasificación y Decisión (~5ms)

#### 4a. ProcessClassifier (`process_classifier.rs`, 20,770 bytes)

Clasifica cada proceso en uno de 8 tiers jerárquicos:

| Tier | Prioridad | Criterio | Ejemplos |
|------|-----------|----------|----------|
| `SystemEssential` | Más Alta | Hardcoded: kernel, WindowServer, launchd | `kernel_task`, `launchd`, `WindowServer`, `coreaudiod`, `configd` |
| `ActiveForeground` | Alta | GUI visible + interacción < 30 segundos | El navegador que estás usando ahora mismo |
| `BackgroundVisible` | Media-Alta | GUI visible pero sin interacción reciente | App en otra pestaña del Dock |
| `AppHelper` | Media | Chrome Helper, WebKit, Electron renderers | `Brave Helper (Renderer)`, `Slack Helper`, `plugin-container` |
| `SilentDaemon` | Media-Baja | Sin GUI, CPU > 0.5%, wakeups > 1/s | Daemons del sistema activos |
| `Stale` | Baja | Sin GUI, CPU < 0.5%, wakeups < 1/s, idle > 300s | Procesos olvidados |
| `Telemetry` | Baja | Conocidos: analytics, Siri, rapportd | `DiagnosticReporter`, `analyticsd` |
| `ZombieOrphan` | Más Baja | `is_zombie` OR padre muerto && ppid ≠ 1 | Procesos huérfanos/zombies del kernel |

**Scores por proceso:**
```
utility_score [0,1] = f(GUI, interacción, red, CPU, wakeups, Rosetta, idle)
  GUI + interacción<10s + red activa = 0.50+0.25+0.20+0.05 = 1.00
  sin GUI + idle>1h + wakeups>50 − penalización = 0.50-0.40-0.20 = -0.10 → 0.0

waste_score [0,1] = f(tier, wakeups, RSS>200MB, idle>1h)
```

#### 4b. ZombieHunter (`zombie_hunter.rs`, 25,678 bytes)

Detecta 5 clases de "peso muerto":

| Clase | Detección | Acción | Confirmación |
|-------|-----------|--------|-------------|
| `TrueZombie` | Kernel SZOMB state | Kill inmediato | 0 (estado del kernel) |
| `Orphan` | Padre muerto, ppid ≠ 1 | Kill inmediato | 0 |
| `GhostHelper` | Host app ausente > 24h | Suspend | 3 ciclos consecutivos |
| `WakeupBurner` | > 20 wakeups/s, sin GUI | NiceToMax | 3 ciclos consecutivos |
| `MemoryHoarder` | > 256MB RSS, idle > 30min | Suspend | 3 ciclos consecutivos |

Las reglas blandas (clases 3-5) requieren **3 observaciones consecutivas** antes de actuar, previniendo falsos positivos por picos momentáneos.

#### 4c. Adaptive Governor (`adaptive_governor.rs`, 49,653 bytes / 1,243 LOC)

Emite el vector de `ProcessDecision` para cada proceso. La cascada tiene **21 reglas** evaluadas en orden (primera que matchea gana):

| # | Condición | Decisión |
|---|-----------|----------|
| 1 | ZombieOrphan | → Kill |
| 2 | SystemEssential O ActiveForeground | → Allow (protegido absoluto) |
| 3 | Uptime < 8 segundos | → Allow (efímero XPC) |
| 4 | AppHelper con audio/video/red activa | → Allow (romperlo crashea tab) |
| 5 | AppHelper inactivo | → Throttle (nunca Freeze) |
| 6 | Telemetry | → Throttle (Freeze si workload pesado) |
| 7 | Mach ports > 80 | → Allow (IPC hub — throttle causa beachball) |
| 8 | LLM cargado (ollama >1GB RSS) | → Allow si idle < 12h |
| 9 | I/O activo (pageins > 50K, CPU > 5%) | → Allow (backup/encode en curso) |
| 10 | SilentDaemon idle (CPU < 0.5%, sin fg > 1h) | → Freeze si Rosetta O RSS>1GB, else Throttle |
| 11 | Idle graduado sin GUI: >6h → Throttle, >12h → Freeze | |
| 12 | Helper del foreground activo (Safari→WebKit) | → Allow |
| 13 | Modo nocturno (00:00-06:00) | → Throttle daemons idle>15min |
| 14 | Stale + utility < 0.05 | → Freeze |
| 15 | Render pipeline (GPU buffer/faults) | → Allow |
| 16 | Waste override (waste ≥ 0.90) | → Throttle si utility < 0.60 |
| 17 | Swarm (>30 procs, waste ≥ 0.30) | → Throttle (Freeze si Rosetta) |
| 18 | Wakeup hog (>100 wakeups/s, sin GUI) | → Throttle |
| 19 | utility < 0.05 | → Freeze |
| 20 | utility < 0.20 | → Throttle |
| 21 | else | → Allow |

**Calibración por hardware al inicio:** M1 8GB usa `waste_override = 0.80` (más agresivo); M3 Max usa `0.90` (más tolerante).

#### 4d. Profile Governor (`profile_governor.rs`, 23,278 bytes)

Máquina de estados para el perfil global: `SafeRoot ↔ BalancedRoot ↔ AggressiveRoot`.

**Fórmula de presión:**
```
score = 0.35×cpu + 0.35×ram + 0.20×interactive_wait + 0.10×reactor_events
      + swap_boost (min(swap_GB/2, 1.0) × 0.12)
```

**Crisis override:** `ram≥0.60 && swap≥1.5GB` → `crisis_score = 0.60 + clamp(swap-1.5, 0, 1.5)/1.5 × 0.25` → garantiza cruzar 0.72.

**Transiciones (con histéresis):**
- Balanced → Aggressive: score ≥ 0.72 × 3 ciclos (2 en Build mode)
- Balanced → Safe: score ≤ 0.28 × 6 ciclos (4 en Idle mode)
- Aggressive → Balanced: score ≤ 0.55 × 6 ciclos
- Safe → Balanced: score ≥ 0.40 × 3 ciclos

**Overrides (prioridad descendente):**
1. ManualOverride con TTL (vía `apolloctl set-override`)
2. `thermal_constrained` → cap en BalancedRoot
3. Anti-thrash lock (>4 transiciones/10min → Balanced 5min, se rompe si `ram≥0.60 && swap≥2GB`)
4. `workload_onset` (cargo/rustc detectado) → AggressiveRoot proactivo
5. `context_switch_burst` (3+ cambios/5min, ram<0.70) → AggressiveRoot
6. Dev/interactive floor → mínimo BalancedRoot

#### 4e. Lotka-Volterra (`lotka_volterra.rs`, 9,898 bytes)

Modelo ecológico de competencia por RAM basado en Volterra (1926):

```
dx/dt = r₁·x·(1 - (x + α₁₂·y)/K)    ← proceso dominante
dy/dt = r₂·y·(1 - (y + α₂₁·x)/K)    ← resto del sistema
```

- **monopoly_risk()** [0,1] = `∛(share × growth × competition)` — Media geométrica; los tres factores deben ser altos para alarma.
- **simulate_forward(horizon_secs):** Euler explícito, paso 1s, máx 120 pasos. Predice fracción de RAM dominante.
- **Simplificación:** Solo 2 "especies" (dominante vs resto), no N (evita O(N²)).

### Etapa 5: Seguridad y Filtrado (~1ms)

`safety.rs` (53,672 bytes / 1,320 LOC) — **13 invariantes de seguridad:**

| # | Invariante | Consecuencia de violarla |
|---|-----------|--------------------------|
| 1 | Nunca congelar `protected_processes()` | Apollo no puede paralizar: `kernel_task`, `launchd`, `WindowServer`, `loginwindow`, `configd`, `securityd`, `tccd`, `syspolicyd`, `notifyd`, `hidd`, `UserEventAgent`, `Spotlight`, `mds`, `mds_stores`, `mdworker`, `mdworker_shared` |
| 2 | Nunca congelar `critical_background_processes()` | Contenedores (docker, podman, colima, qemu), DBs (postgres, mysql, redis, mongo), runtimes (node, python, java, nginx, go, ruby, php), compiladores (rustc, cargo) — solo throttle ligero |
| 3 | Comandos vía `std::process::Command`, sin shell | Previene shell injection |
| 4 | Sysctl solo sobre allowlist de 16 claves exactas | No se puede escribir a claves fuera de la lista |
| 5 | Cooldown 90s entre transiciones de perfil | Previene oscilación de perfiles |
| 6 | Anti-thrash: >4 transiciones/10min → BalancedRoot lock 5min | Con escape valve si `ram≥0.60 && swap≥2GB` |
| 7 | Dev floor: sesión activa → nunca SafeRoot | Previene que un developer pierda performance |
| 8 | Gracia post-wake: 60s de agresión suprimida | Post-sleep el sistema necesita estabilizarse |
| 9 | LLM patterns saneados: max 80 chars, sin newlines, conf≥0.80 | Previene inyección vía LLM |
| 10 | PIDs congelados en `frozen_state.json` → descongelados al reiniciar | Crash safety |
| 11 | PID identity check: `start_sec`/`start_usec` | Previene A-B-A recycling del PID |
| 12 | AppHelper = throttle-only (nunca freeze) | Chromium watchdog crashea tab si helper congelado |
| 13 | IPC hubs (>80 Mach ports) = Allow siempre | Throttle causa beachballs |

**Budgets por ciclo por perfil:**

| Perfil | Boosts | Throttles | Freezes | Cooldown |
|--------|--------|-----------|---------|----------|
| AggressiveRoot | 10 | 20 | 8 | 10s |
| BalancedRoot | 6 | 12 | 4 | 20s |
| SafeRoot | 3 | 6 | 2 | 45s |

### Etapa 6: Ejecución (~10ms)

`execute_actions.rs` (45,968 bytes / 1,003 LOC):

| Acción | Implementación | Efecto |
|--------|----------------|--------|
| **Boost** | `task_policy_set(LATENCY_QOS_TIER_0)` + `renice -10` | Proceso promovido a P-cores con prioridad alta |
| **Throttle** | `PRIO_DARWIN_BG` + `task_policy_set(THROUGHPUT)` | Proceso relegado a E-cores con prioridad baja |
| **Freeze** | `SIGSTOP` + registro en `frozen_state.json` | Proceso completamente pausado |
| **Kill** | `SIGKILL` (solo zombies confirmados 3+ ciclos) | Proceso terminado |
| **Unfreeze** | `SIGCONT` con PID identity check | Proceso reanudado tras verificar identidad |
| **Sysctl** | `sysctl -w` sobre allowlist exacta | Parámetro de kernel ajustado |
| **Spotlight** | `mdutil -i on/off` | Indexación activada/desactivada |
| **Jetsam** | `memorystatus_control()` | Hint de prioridad jetsam al kernel |

**Protección especial LLM servers (commit `5372de6`):** Tiers adaptativos de protección para servidores LLM (ollama, llama) basándose en presión actual.

### Etapa 7: Aprendizaje (~2ms)

Ver **[04_Learning_Pipeline_and_Metrics.md](./04_Learning_Pipeline_and_Metrics.md)** para detalle completo.

Resumen: Cada acción genera una `LearningObservation`. Cuando el batch acumula 8 observaciones, se hace flush a los 3 subsistemas (OutcomeTracker, CausalGraph, SkillRegistry) con cross-feed rules.

### Etapa 8: Persistencia (~5ms, no cada tick)

Solo se escribe a disco cuando hay cambios relevantes. Ver tabla de archivos en [01_System_Overview.md](./01_System_Overview.md#31-archivos-de-estado-del-daemon).

---

## 2. Reactor kqueue (Hilo de Eventos)

`reactor.rs` + `kqueue_pressure.rs` — Hilo separado que escucha eventos del kernel:

| Evento | Fuente | Efecto |
|--------|--------|--------|
| `EVFILT_VM / NOTE_VM_PRESSURE` | Kernel: presión de memoria | Re-optimización inmediata |
| Darwin Notif thermal | `com.apple.system.thermalpressurelevel` | Ajuste térmico |
| Darwin Notif spawn | `com.apple.launchd.spawn` | Nuevo proceso detectado |
| Darwin Notif power | `com.apple.system.powersources.source` | Cambio fuente de energía |

**Efecto de cualquier evento:**
```
fast_tick_until = now + fast_tick_duration
reactor_event_weight += 1  (alimenta pressure_score del governor)
→ Dispara ciclo de optimización inmediato
```

**Tick rate adaptivo:**
- Idle normal: 60s
- Workload (coding/build): 15s
- Post-evento kqueue: **2s** (durante `fast_tick_duration`)

---

## 3. Chromium Manager — Subsistema Especializado

`chromium_manager.rs` es el mayor módulo del engine (101,811 bytes / 2,402 LOC). Gestiona específicamente los procesos helper de Chromium (Chrome, Brave, Edge, Slack, Discord, VS Code, etc.):

### Problema que resuelve
Los navegadores Chromium spawnan docenas de helpers (renderers, GPU, utility). Congelarlos incorrectamente mata tabs permanentemente porque el watchdog del browser detecta el helper como unresponsive.

### Solución
- **Inventario de renderers** desacoplado del freeze gate (commit `6ae268f`).
- **Visibility-aware freeze** via `CGWindowListCopyWindowInfo` (commit `1874659`): Solo congela renderers de tabs no visibles.
- **Grace period para renderers nuevos:** `NEW_RENDERER_GRACE_CYCLES` evita congelar renderers recién creados.
- **Pressure-adaptive `max_freeze_ratio`** (commit `9ef180a`): En heavy workloads, permite congelar más renderers.
- **Jetsam background demotion** (commit `59b449d`): En survival mode, los renderers de background se degradan a jetsam `BACKGROUND`.
- **SIGSTOP permanentemente deshabilitado para renderers** (commit `712b927`): Después de múltiples reverts y oscilaciones, se determinó que SIGSTOP en renderers es inherentemente inseguro en producción.

### Historial de oscilación (visible en los commits)

```
dfee139  feat(chromium): re-enable renderer freeze ← Intento 1
2b45016  revert(chromium): disable — tabs stay frozen  ← No funcionó
21bcb7d  fix(chromium): guard via app-name fallback + re-enable ← Intento 2
712b927  revert(chromium): permanently disable renderer SIGSTOP ← Decisión final
```

Este historial muestra el proceso empírico de ingeniería: la teoría de que congelar renderers inactivos sería seguro fue refutada repetidamente por la realidad de los watchdogs de Chromium.

---

## 4. Subsistemas Especializados Adicionales

| Módulo | LOC | Función |
|--------|-----|---------|
| `thermal_interrupt.rs` | 1,297 | Gestión de interrupciones y bailouts térmicos. Compute de fases térmicas (1-4). Propagación de `Option<f32>` para cuando no hay sensor (refactored en commit `64d045a`). |
| `sysctl_governor.rs` | 1,480 | Governor para 16 claves sysctl. Cooldown por wall-clock (no monotonic — sobrevive sleep/wake). |
| `foreground.rs` | 1,104 | Detección de la app foreground, boost de su familia de procesos, wait-graph awareness. |
| `display_turbo.rs` | LOC | Freeze/unfreeze de procesos no-esenciales cuando el display se apaga. A-B-A PID identity check (commit `6f068b6`). |
| `energy.rs` + `energy_pid.rs` | 970+LOC | Estimación de energía ahorrada, PID controller para eficiencia energética. |
| `compressor_aware.rs` | 915 | Lógica especializada para interpretar páginas del compresor macOS correctamente. |
| `wait_graph.rs` | LOC | Detección de procesos que bloquean la interactividad (blocker scoring). |
| `window_sensor.rs` | 776 | Monitoring de ventanas visibles para decisions context-aware. |
