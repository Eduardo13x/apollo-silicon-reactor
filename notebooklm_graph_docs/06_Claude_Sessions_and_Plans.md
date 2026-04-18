# Navegación
[<- Evolución Histórica](./05_Evolution_and_Commit_History.md) | [Volver al Índice](./00_Index.md)

# 06. Claude Sessions & Planning — Decisiones de Diseño y Deuda Técnica

Este documento captura las sesiones de análisis colaborativo entre el desarrollador y Claude Code que shaped la arquitectura de Apollo. Incluye planes activos, deudas técnicas abiertas/cerradas, y las decisiones de diseño que no son obvias desde el código.

---

## 1. DEBT-SENSOR-01/02 — La "God-Sensor Session" (2026-04-08)

### Contexto

La sesión del 8 de abril añadió 5 nuevos ejes de sensor en un día (`VmRate`, `thrashing_score`, `CpuSaturation`, `cpu_contention`, `stall_fraction`). Sin embargo, **dos consumers de decisión propuestos fueron deliberadamente diferidos**.

### DEBT-SENSOR-01 — Boost foreground pid on high `cpu_contention` ✅ CERRADO 2026-04-09

**Propuesta original:** Si un pid de familia foreground tiene `cpu_contention > 0.6` sostenido por N ciclos, promoverlo a `SchedulingTier::Foreground` vía `mach_qos.set_tier()`.

**Por qué se difirió:** La familia foreground YA es promovida por el path `boost_foreground_family`. Un segundo boost por contención es un no-op para procesos que ya están en Foreground tier.

**Resolución:** Cerrado como "ya cubierto" — el sensor es útil para observabilidad/métricas pero no necesita consumer de decisión.

### DEBT-SENSOR-02 — Throttle background pids on `stall_fraction > threshold` ✅ CERRADO

**Propuesta original:** Si `stall_fraction > 0.7` por 5+ ciclos, throttlear automáticamente procesos background sin importar su historial Bayesiano.

**Por qué se difirió:** El análisis de 11h de datos reales (331 rows) mostró que en M1 8GB el stall fraccional nunca superó 0.3 excepto durante compilaciones de Rust, y en esos casos el `build_tracker.rs` ya aplica protección. Una regla de stall sin calibración empírica generaría falsos positivos.

**Resolución:** Sensor permanece en `StabilityOracle` → RL reward. Consumer de decisión descartado para hardware actual M1.

### Lección Extraída

> Los M1 con 8 GB no se asfixian por "CPU Stall" (0 veces en simulaciones de 11h) sino por MEMORIA (RAM bottleneck). Promover throttle por "CPU starvation" es redundante para Apple Silicon M1. La feature tiene sentido para M3 Ultra / Mac Studio con múltiples sockets, no para el hardware objetivo actual.

---

## 2. V1.1.0 — Targets Pendientes Post v1.0.0 (`.plan/V110_PENDING.md`)

Estado al 2026-04-03. Tres targets identificados:

### Target 1 — `main.rs` ≤ 4100L (hoy: ~4962L, gap: ~-862L)

Estrategia de extracción en 5 módulos nuevos:

| Módulo propuesto | LOC aprox | Líneas fuente en main.rs | Qué extrae |
|------------------|-----------|--------------------------|------------|
| `daemon_process_collector.rs` | ~641 | 1500–2140 | Árbol de procesos, enriquecimiento GUI/net/CPU, memory scan top-50 |
| `daemon_freeze_executor.rs` | ~291 | 4109–4400 | TTL unfreeze, confirmación candidatos 2+ ciclos, budget enforcement |
| `daemon_action_safety.rs` | ~183 | 3278–3400, 4046–4107 | Sysctl governor, filtrado de acciones seguras |
| `daemon_wake_handler.rs` | ~127 | 1124–1250 | Post-wake grace, wake state management |
| `daemon_turbo_manager.rs` | ~79 | 1252–1330 | Display-off turbo freeze/unfreeze |

**Riesgo alto:** `daemon_process_collector.rs` (641L) tiene muchas dependencias — requiere pase de fix post-extracción.

**Lo que NO se mueve:** Control flow del hot loop (cycle_count, condvar wait, last_cycle_instant), guardas de lock ordering sobre SharedState, reactor pulse monitoring.

**Estrategia de ejecución:** Wave 1 (3 agentes paralelos): daemon_action_safety + daemon_wake_handler + daemon_turbo_manager. Wave 2 (2 agentes): daemon_process_collector + daemon_freeze_executor + agente de tests.

Guard de compilación: `cargo check --tests` (NO `cargo test` — tarda ~20 min).

### Target 2 — Tests ≥ 2500 (hoy: ~2263, gap: ~+237)

Cuatro módulos extraídos en v1.0.0 con 0 tests sobre 1763L de código:

| Módulo | LOC | Tests hoy | Dificultad |
|--------|-----|-----------|------------|
| `socket_handler.rs` | 878 | 0 | Alta — depende de SharedState completo |
| `metrics_reporter.rs` | 385 | 0 | Media |
| `learning_tick.rs` | 373 | 0 | Alta — depende de SharedState + LearningContext |
| `daemon_init.rs` | 127 | 0 | Baja |

**Riesgo:** Tests de módulos bin requieren instanciar `SharedState` (muchos `Arc<Mutex<>>`). Resultado real estimado: 150–180 tests nuevos en vez de 237 si el setup es muy costoso.

### Target 3 — Workspace Split (opcional, v1.2.0)

Ver §3 abajo para detalle completo.

---

## 3. Workspace Split Plan (`.plan/WORKSPACE_SPLIT.md`)

### El Problema

Single crate → full recompile en cualquier cambio → 20 minutos para `cargo test`.

**Estado actual (baseline):**

| Ubicación | Archivos | LOC |
|-----------|----------|-----|
| `src/engine/` | 126 archivos | ~77,178 |
| `src/*.rs` (lib root + collector etc.) | 4 | ~1,381 |
| `src/bin/` (daemon, ctl, main) | ~12 | ~8,660 |
| **Total** | **~142** | **~87,219** |

### Opción B (Recomendada) — Mínimo Disruptiva

```
Cargo.toml              ← workspace [workspace] + mantiene [package] apollo-optimizer
crates/apollo-engine/
    Cargo.toml          ← crate puro: solo lógica, sin deps macOS-specific
    src/lib.rs          ← pub mod declarations (mirror de src/engine/mod.rs)
    src/…               ← moved from src/engine/ (verbatim)
```

**Beneficio principal:** `cargo test -p apollo-engine learning_pipeline` → 3–5 min vs 20 min.

**Lo que NO cambia:** `src/bin/`, `src/main.rs`, `src/collector.rs` — solo la lógica del engine se mueve.

**Bloqueantes:** Ninguno. Puede ejecutarse después de cerrar Targets 1 y 2 de V110.

**Por qué está diferido:** El split masivo requiere rewrite de todos los imports (`crate::engine::X` → `apollo_engine::X`). Actualmente ~1200+ import paths. Se hace en una sola sesión para evitar inconsistencias parciales.

---

## 4. ARM64 / Apple Silicon Thread-Level Plan (`.plan/PLAN_ARM64_OPTIMIZATIONS.md`)

### El Problema Actual

Apollo opera a nivel de **proceso completo** (task-level): congela/throttlea procesos enteros via `task_policy_set()`. En Apple Silicon M1 (big.LITTLE: 4 P-cores Firestorm + 4 E-cores Icestorm), esto es un instrumento grueso — un proceso marcado "background" envía TODOS sus threads a E-cores, incluso si tiene un thread de UI crítico.

### Fase 1: Thread-Level Scheduling

**Hipótesis:** En arquitectura big.LITTLE, la granularidad por-thread permite rutear threads calientes (UI, GPU compositing) a P-cores y threads fríos (GC, telemetría) a E-cores dentro del mismo proceso, mejorando latencia interactiva 8–15% sin aumentar consumo energético.

**Papers de referencia:**
- ARM "big.LITTLE Technology" whitepaper (2013)
- Apple WWDC 2020 "Tune your app's performance on Apple Silicon"
- XNU source `osfmk/kern/thread_policy.c`

**Gap crítico en Apollo actual:**

| Qué existe | Dónde | Limitación |
|------------|-------|------------|
| `task_for_pid()` + `task_policy_set(TASK_CATEGORY_POLICY)` | `mach_qos.rs` | Solo nivel proceso |
| `proc_pidinfo(PROC_PIDTASKINFO)` | `proc_taskinfo.rs` | Reporta `thread_count` agregado, NO enumera threads |
| `task_threads()` + `ThreadBasicInfo` | `optimizer.rs` | Solo para el propio proceso de Apollo |

**Lo que falta:**
1. `task_threads()` sobre procesos ajenos
2. `thread_policy_set()` per-thread
3. `THREAD_AFFINITY_POLICY` — thread-to-core affinity hints
4. `THREAD_LATENCY_QOS_POLICY` / `THREAD_THROUGHPUT_QOS_POLICY` — QoS por thread
5. Delta tracking de CPU por thread para clasificar hot/cold

**Estado:** Plan documentado, no iniciado. Prerequisito: V110 Targets 1+2 cerrados.

---

## 5. Sesiones de Análisis Arquitectónico

### Sesión: Chromium Manager Oscillation Analysis (Abr 13–18)

**Problema identificado:** `chromium_manager.rs` tenía 3 rutas de freeze con semántica inconsistente:
1. Freeze por presión alta
2. Freeze por long-idle (30 ciclos ~60s)
3. Freeze reactivo por survival mode

El problema: el path de "pressure-triggered thaw-all" (Iteración 1 del Chromium Evolve) interfería con los renderers congelados por long-idle. La solución definitiva fue eliminar el thaw-all reactivo y usar CGWindowList para visibilidad real.

**Diseño final decidido en sesión:**
- Solo congelar renderers cuya ventana es NOT visible según `CGWindowList`
- Nunca congelar el browser foreground (verificar por app-name como fallback si CGWindowList falla)
- Durante survival mode: usar jetsam demotion en vez de SIGSTOP para renderers background

### Sesión: Swap Exhaustion Root Cause (Abr 17)

**Diagnóstico colaborativo:** La presión de memoria en M1 tiene tres capas que Apollo no distinguía:

```
Layer 1: vm_pressure()          → kVMPressureLevel (warning/urgent/critical)
Layer 2: compressor ratio       → páginas comprimidas / páginas activas
Layer 3: swap file utilization  → bytes usados del swap file en disco
```

`vm_pressure()` puede reportar "normal" mientras swap está ≥4GB porque el kernel considera que el compressor tiene capacidad. Apollo necesitaba monitorear Layer 3 independientemente.

**Decisión de diseño:**
- Swap exhaustion = `swap_used_bytes >= 0.80 * swap_total_bytes` (relativo, no hardcoded 4GB)
- Threshold absoluto: 4GB como piso para M1 8GB
- Threshold relativo: escala con `swap_total_bytes` para compatibilidad con M2/M3 con más RAM

### Sesión: NARS Code Audit + 9 Bugs (Abr 10)

Sesión de análisis profundo usando NARS + graphify (4020 nodos/7897 aristas/85 comunidades):

**5 bugs Round 1:**
- B006: Hazard saturation (valor se estabilizaba en 1.0 sin decaer)
- B002: Sentinel -1 en ThermalManager (temperatura inválida propagada)
- B009: Overflow en SwapPredictor (sentinel -1 convertido a u64 = MAX)
- B003+display_turbo: Ghost PIDs (PID reusado después de freeze)
- Missing pre-sleep unfreeze

**4 bugs Round 2:**
- B007: OomKill bypass velocity gate (procesos con OOM score alto evitaban throttle)
- H-1: `validate_after_restore` denominator drift
- B010: Wake detection missing display_turbo thaw
- B011: `avg_delta` frozen under failure (EMA no se actualizaba en path de error)

**Meta-patterns extraídos:**
- Fix-N-minus-1: fix expone bug en componente anterior de la cadena
- Coupled-ratio-managed-independently: dos ratios relacionados se actualizan en paths distintos
- Arithmetic-overflow-without-saturation: sentinel -1 convertido a tipo unsigned
- Asymmetric-state-management: freeze sin unfreeze correspondiente en todos los paths
- Hardcoded-placeholder-never-wired: valor placeholder que nunca fue reemplazado por cálculo real

### Sesión: Paper AGI 2026-04-10

**Objetivo:** Cerrar gaps entre implementación y paper `apollo_agi_paper_draft.md`.

**3 gaps cerrados:**
- §5.2: Definición formal de NARS revision derivation + cognitive stack
- §6.2: Proposition 1 sobre convergencia del NestedLearner
- §8.3: Wiring de ThermalManager return value a metrics (era un gap de observabilidad)

**Resultado:** Tests 3672, commits `ee6b9a4`→`b38d8ca`, paper sync con código ✓.

---

## 6. Decisiones de Diseño No-Obvias (Design Rationale)

### Por qué `sudo cp` para deployment (NO `python3 open().write()`)

Crítico: usar `python3 open().write()` para reemplazar el binario STRIPA el linker-signed flag, causando "Launch Constraint Violation" en launchctl.

**Procedure correcto:**
```bash
sudo cp target/release/apollo-optimizerd /usr/local/libexec/apollo-optimizerd
sudo launchctl bootout system/com.eduardocortez.systemoptimizerd
sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist
```

**Nota:** `kickstart` solo puede no ser suficiente si hay codesign cacheado — usar bootout+bootstrap.

### Por qué `max(kernel_pressure, compressor_signal)` para presión efectiva

`vm_pressure()` solo reporta el nivel del kernel. El compressor puede estar saturado sin que `vm_pressure()` lo refleje inmediatamente (lag de ~2 ciclos). Tomar el máximo de ambas señales da una presión efectiva más conservadora.

### Por qué batch_size=8 en LearningPipeline (no 1, no 32)

- batch=1: demasiado frecuente, HashMap lookups sin amortizar.
- batch=32: hasta 64 segundos de delay a 2s/tick — demasiado lento para adaptar.
- batch=8: ~16 segundos, amoriza costo, suficientemente frecuente para responder a cambios de workload.

### Por qué `LlmConfig.always_on` bypasa TTL

Los modelos locales (Gemma en localhost:8080) tienen latencia predecible (~124s fija). El TTL de 1800s (30 min) fue diseñado para modelos remotos (costos de API, rate limits). Para modelos locales, el TTL era contraproducente — el sistema aprendía menos. `always_on: true` + TTL de 10 años efectivamente desactiva el throttle para inferencia local.

### Por qué no usar `cgroups` para freeze (usar SIGSTOP)

macOS no expone cgroups para procesos de usuario. Las alternativas evaluadas:
- `SIGSTOP/SIGCONT`: Funciona, instantáneo, reversible, sin privilegios especiales más allá de ownership.
- `task_suspend()/task_resume()`: Equivalente a SIGSTOP pero por Mach API — no más granular.
- Jetsam demotion: Solo indica preferencia al kernel, no garantiza freeze.

Decisión: SIGSTOP para freeze explícito, jetsam demotion para "candidato-a-kill" bajo presión extrema.

---

## 7. Estado de Planes al 2026-04-18

| Plan | Estado | Próximo paso |
|------|--------|-------------|
| V110_PENDING Target 1 (main.rs ≤4100L) | 🔴 No iniciado | Wave 1: extraer 3 módulos pequeños |
| V110_PENDING Target 2 (tests ≥2500) | 🔴 No iniciado | Empezar por daemon_init.rs (baja dificultad) |
| Workspace Split (v1.2.0) | 🟡 Diferido | Después de V110 |
| ARM64 Thread-Level (Fase 1) | 🟡 Diferido | Después de Workspace Split |
| DEBT-SENSOR-01 | ✅ Cerrado | — |
| DEBT-SENSOR-02 | ✅ Cerrado | — |
| Paper AGI gaps §5.2/§6.2/§8.3 | ✅ Cerrado | — |
