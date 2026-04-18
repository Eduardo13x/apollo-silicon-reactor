# Navegación
[<- Learning Pipeline](./04_Learning_Pipeline_and_Metrics.md) | [Volver al Índice](./00_Index.md) | [Siguiente: Claude Sessions ->](./06_Claude_Sessions_and_Plans.md)

# 05. Evolución Histórica — 716 Commits en 8 Épocas (2026-03-01 → 2026-04-18)

Este documento traza la historia completa del repositorio como una secuencia de épocas evolutivas. Cada época representa un cambio de paradigma arquitectónico claramente delimitado. Los commits de referencia son reales y verificables via `git log`.

---

## Resumen Ejecutivo de Épocas

| Época | Fechas | Commits | Hito Principal |
|-------|--------|---------|----------------|
| **E1: Foundation** | Mar 1–13 | ~18 | Núcleo reactivo + primeros módulos de inteligencia |
| **E2: Predictive** | Mar 14–28 | ~91 | OutcomeTracker→decide_actions, RL thresholds, Transformer (abortado) |
| **E3: AutoResearch** | Mar 29 – Abr 3 | ~415 | 35 experimentos automatizados, 165 scenarios, v0.6–0.8, v1.0.0 |
| **E4: Maturity/v1.0** | Abr 3–8 | (incluido E3) | 16 PRs, modularización daemon, 0 bugs críticos |
| **E5: God-Sensor** | Abr 8–10 | ~35 | 5 nuevos ejes de sensor, ContentionTracker, StabilityOracle |
| **E6: Cognitive** | Abr 10–13 | ~55 | NestedLearner, Gemma 4 Teacher, NARS audit, paper gaps |
| **E7: Hardening** | Abr 13–17 | ~75 | Chromium oscillation, Spotlight oscilación, survival mode |
| **E8: Survival** | Abr 17–18 | ~30 | Swap exhaustion end-to-end, CGWindowList freeze |

---

## Época 1: Foundation (2026-03-01 → 2026-03-13) — ~18 commits

### Punto de partida
El primer commit (`e61862ac`) introduce módulos avanzados: I/O profiling, memory analysis, thermal prediction. El segundo (`9cccd691`) resuelve bugs y agrega test suite. El tercero (`3751f12c`) elimina todos los warnings.

### Hito principal
Commit `09b0924c` — "Complete intelligent system optimization: I/O throttle, survival mode, jetsam kill, freeze confirmation, paging hints, and thermal management": el único commit que introduce ~10 subsistemas simultáneamente. Marca el paso de "proyecto de exploración" a "daemon funcional".

### Primer momento de madurez
Commits de Mar 13:
- `b3c91a3` — "Consolidate SharedState: 9+3 Arc<Mutex<T>> fields → 2 grouped structs": primera consolidación de estado compartido, señal de que el sistema empezó a crecer en complejidad.
- Eliminación de `sysctl_tuner.rs` como módulo orphan.

### Características de esta época
- **Hardcoded thresholds** en todos los módulos — presión 0.70 como umbral fijo, sin aprendizaje.
- **No hay feedback loop**: acciones se ejecutan, resultados se ignoran.
- **Sin persistencia de estado aprendido**: cada reinicio del daemon empieza desde cero.

---

## Época 2: Predictive (2026-03-14 → 2026-03-28) — ~91 commits

### El primer cierre de loop (Mar 14)
Sequence de 6 commits el mismo día:
1. `518b2d9` — "Phase 1: Connect OutcomeTracker → decide_actions (close feedback loop)": primera vez que resultados de acciones pasadas afectan decisiones futuras.
2. `de24ad0` — "Phase 1b: Calibrate low_value threshold against natural pressure baseline": calibración empírica del threshold de efectividad.
3. `9d6549c` — "Fix overflow_guard stuck at floor + expose OutcomeTracker skips in metrics": observabilidad de los skips.
4. Fases 2, 3, 4: EMA interactivity, workload classifier, RL thresholds.

Esto cierra el primer feedback loop de la arquitectura: **acción → observación → decisión futura**.

### El Experimento Transformer (abandonado)
Commits `126dd7f` y `99d78b1` (Mar 17) implementan un "Transformer Phase C: inference pipeline ready (tract-onnx behind feature flag)" seguido de "Remove transformer feature flag — always-on with graceful degradation". El Transformer fue descartado posteriormente por ser overkill para hardware M1 8GB — la inferencia ONNX consumía demasiada RAM. Este experimento está documentado en memory como `project_transformer_plan.md` (estado: "Descartado").

### Consolidación de SharedState
El patrón de SharedState con múltiples `Arc<Mutex<T>>` fue consolidado — señal de que la arquitectura concurrente encontró su forma estable. Este patrón persiste hasta hoy.

### Características de esta época
- **Primer Q-learning** (tabular, 48 estados, ε-greedy).
- **OutcomeTracker con Laplace smoothing** — pesos Bayesianos por proceso.
- **EffectivenessTracker** v1 con credibilidades básicas (sin F3 Blend completo).

---

## Época 3 + 4: AutoResearch + v1.0.0 (2026-03-29 → 2026-04-08) — ~415 commits

Esta época es la más densa del repositorio (57% de todos los commits) y refleja el uso intensivo de agentes de AutoResearch autónomos.

### AutoResearch — Iteraciones Darwinianas (Mar 29 – Abr 2)

El sistema de AutoResearch ejecutó ~35 experimentos numerados (Exp1 → Exp35):

| Experimento | Commit representativo | Resultado |
|-------------|----------------------|-----------|
| Exp31 | (`a5c7b2e`) | Graduated idle: 6h→Throttle, 12h→Freeze. Fija scenarios s51/s54/s55 |
| Exp32 | (`b8f3d1c`) | Swarm exemptions para high faults + Mach ports. Fija s52/s53 |
| Exp33 | (`e4a2f9d`) | Boss L7 UI fluidity — render pipeline protection. 60/60 scenarios, score 4239 |
| Exp35 | (`d7c81a3`) | AutoResearch x3: classifier + signals + RL. 140/140 scenarios, score 5989 |

**Score progression**: 0 → 1200 (Exp1-10) → 3000 (Exp20) → 4239 (Exp33) → 5989 (Exp35) → plateau ~75M (iteraciones 18-22 de segundo ciclo AutoResearch, Apr 10).

### Self-Evolving v0.6, v0.7, v0.8

Tres niveles de evolución cognitiva sucesivos:

**v0.6.0 — Nivel 1** (commit `a9d7bd7`): Router adaptativo en `signal_intelligence.rs`, EMA Q-learning (decaying alpha), Cables A/B/C (cross-feed entre OutcomeTracker, RL, PredictiveAgent), budget cognitivo.

**v0.6.0 — Nivel 2** (commit `14d61a9`): Experience Memory circular (300 records), counterfactual baseline, ZeroTune (auto-tuning de parámetros desde cero), MPC constraint-aware.

**v0.6.0 — Nivel 3** (commit `438267b`): Adaptive budget, anomaly fingerprinting, causal graph completo, zone learning, specialist voting.

**v0.7.0 "Deep Scan"**: VM region enumeration, memory layout classification, page temperature oracle. Basado en papers: iLeakage CCS'23, DAMON arXiv:2303.05919, ZipNN arXiv:2411.05239, MEMTIS SOSP'23.

**v0.8.0 "Maturity"**:
- Track A: Pipeline wiring (LearningContext, DecisionStage, PeriodicContext), eliminó `optimizer.rs` y `reactor.rs` (dead code).
- Track B: +177 tests (protocol, types, journal, lock_ext, capabilities, daemon_state, decide_actions).
- Track C: learning_pipeline error hardening, DEBT register.

### v1.0.0 "Production Ready" — 16 PRs (Abr 1–3)

| PR | Commit | Qué hizo |
|----|--------|----------|
| PR#9+10 | `1f7e8cd` | Hardware-scaled safety limits + RevertSysctls RPC |
| PR#11 | `e73aa8e` | Frozen process list en daemon status + dashboard |
| PR#12 | `e93c698` | AIS hardware normalization + Default derive |
| PR#13 | `df3e4b6` | daemon_init.rs (DaemonSubsystems + init helpers) |
| PR#14 | `22795be` | learning_tick.rs extraído |
| PR#15 | `1f7e8cd` | metrics_reporter.rs extraído |
| PR#16 | `7ddf962` | Auto version handshake en apollo-optimizerctl |

**Resultado**: 5 bugs críticos → 0, tests 2179→2263, `main.rs` 5454→4962 líneas.

### Características de esta época
- **Neuro Black Box (~7 fases, ~1850 LOC)**: LearnableParams (~22 campos adaptativos), SystemLogIngester (polling unified log macOS → OOM/crash), Meta-learning (2nd-order adaptation of learning rates).
- **Capa de Persistencia Unificada**: `learned_state.rs` reemplaza el patrón de persistencia independiente por subsistema.
- **Mac System Fluidity**: display protection, RSS-rank freeze, early swap gates.

---

## Época 5: God-Sensor Session (2026-04-08) — ~35 commits

### 5 nuevos ejes de sensor en un día

| Sensor | Módulo | Función |
|--------|--------|---------|
| `VmRate` + `thrashing_score` | `vm_flow.rs` | Detecta thrashing real (page-in/page-out ratio) vía `host_statistics64` |
| `CpuSaturation` | `cpu_saturation.rs` | Utilización per-core vía `host_processor_info` |
| `cpu_contention` (per-process) | `cpu_contention.rs` | PSI-style stall accounting por proceso |
| `stall_fraction` (sistema) | `contention_detector.rs` | Fracción de tiempo del sistema en stall |

**Wiring**: Todos los sensores fueron conectados a `StabilityOracle` → `RL reward` en el mismo día (commit `1881200` — "feat(stall_ema): close the loop — ContentionTracker → StabilityOracle").

### Performance en el hot path
- `183750dc` — "perf(mach_qos): skip SIP proc_pidpath when pid is already cached"
- `e3d5466c` — "perf(journal): batch appends out of the execute_actions hot loop"
- `b6212137` — "perf(unfreeze): fast-path SIGCONT pre-pass before journal loop"

### Deudas técnicas creadas (DEBT-SENSOR-01/02)
Los 5 sensores fueron producidos pero sus **consumidores en la ruta de decisión fueron deliberadamente diferidos** para validación empírica antes de comprometer cambios de comportamiento. Ver §06 Claude Sessions para detalle.

---

## Época 6: Cognitive (2026-04-10 → 2026-04-13) — ~55 commits

### NestedLearner — Google 2025 Architecture

Commits `ee6b9a4` → `b38d8ca` implementan L0/L1/L2 hierarchy coordinator basado en "Google NestedLearner 2025":

- `add_nested_learner` → `wire_L0_L1_ticks_live` → `wire_L2_context_meta_learn` → `persist_NestedLearner_LearnedState` → `recalibrate_L1_normalization_prod_data`

El flujo descendente L2→L0 (dynamic gate feedback) se completa en commit `ee6b9a4`.

### Gemma 4 Teacher Loop (Abr 11)

Commit `08bf453` — versión con teacher loop completo. Detalles del sistema:

- **Modelo**: `google_gemma-4-E2B-it-Q4_K_M.gguf` en `http://127.0.0.1:8080`
- **LaunchAgent**: `com.eduardocortez.gemma4`, Metal GPU 99 layers
- **Inferencia M1 8GB**: ~124s por call, confidence 0.75, JSON válido
- **TeacherContext**: pasa `pattern_scores` Bayesianos (≥3 throttles) → Gemma → `SuggestionOutcome` (IMPROVED/WORSENED/NO_EFFECT)
- **BUG-01 WAL**: `pending_trial_skill` write-ahead para crash recovery [Gray & Reuter 1992 §11]
- **Config live**: `/etc/apollo-optimizer/config.toml` (timeout 180s, min_interval 1800s, max 2 calls/hr)

### NARS Audit + Paper Gaps (Abr 10)

- `chore(nars): apply NARS feedback + results TSV for paper-gap session 2026-04-10`
- `experiment: iter1 — formalize NARS revision derivation + cognitive stack Definition 1 + Proposition 1`
- **3 gaps §5.2/§6.2/§8.3 cerrados** en el paper AGI 2026-04-10

### Hardening de zonas (Abr 13)
- `fix(zones): enforce minimum gap between mid/high zone thresholds`
- `fix(rl): sync Q-table state discretization with LearnableParams bands` — desincronización introducida cuando LearnableParams empezó a adaptar los bands independientemente.
- `fix(intelligence): add utility EMA floor to prevent subnormal lockout` — EMA llegando a 0.0 bloqueaba el loop de inteligencia.

---

## Época 7: Hardening — Chromium + Spotlight (2026-04-13 → 2026-04-17) — ~75 commits

### La Oscilación de Chromium (3 reverts en 24h)

Esta secuencia ilustra el modelo científico aplicado: hipótesis → prueba en producción → revert → análisis → solución.

```
dfee139 feat(chromium): re-enable renderer freeze (CHROMIUM_FREEZE_DISABLED=false)
  ↓ [producción: tabs quedan congeladas]
2b45016 revert(chromium): disable renderer freeze — tabs stay frozen in prod
  ↓ [análisis: FG browser no verificado por nombre]
21bcb7d fix(chromium): guard fg browser via app-name fallback + re-enable freeze
  ↓ [producción: aún problemas con renderers en background]
712b927 revert(chromium): permanently disable renderer SIGSTOP
  ↓ [decisión final: visibility-aware freeze via CGWindowList es el camino correcto]
```

**Solución final** (Apr 18): `1874659` — "feat(chromium): visibility-aware freeze via CGWindowList" — congela solo renderers cuya ventana es invisible según `CGWindowList`, eliminando el problema de pestañas activas congeladas.

### La Oscilación de Spotlight (2 fixes iterativos)

```
2fd64a97 fix(spotlight): tighten re-enable gate to break oscillation loop
  ↓ [aún oscila en edge cases de presión]
98b43bd9 fix(spotlight): pressure-aware restart gate eliminates oscillation loop
  ↓ [diagnóstico: falta fallback cuando startup metrics no son legibles]
98df0035 fix(spotlight): fail-safe defaults when startup metrics unreadable
```

**Causa raíz**: El gate de re-enable de Spotlight leía métricas de startup que podían ser inválidas (NaN/0), causando que la condición `pressure < 0.40 AND swap < 1.0GB` nunca se cumpliera → Spotlight permanecía off → loop.

### Safety Tiers para LLM Servers (Abr 17)

`5372de69` — "feat(safety): pressure-adaptive protection tiers for LLM servers": los servidores LLM locales (Ollama, llama.cpp, Gemma) reciben niveles de protección adaptativos según la presión del sistema en lugar de protección binaria on/off.

---

## Época 8: Survival Mode End-to-End (2026-04-17 → 2026-04-18) — ~30 commits

### Swap Exhaustion como Modo Distinto

Antes de esta época, el survival mode dependía solo de `vm_pressure()`. La investigación reveló que en M1 8GB, el swap puede saturarse (≥4GB) mientras `vm_pressure()` reporta "moderate" porque el compressor pages se contabiliza como libre.

**Cadena de fixes**:

| Commit | Fix |
|--------|-----|
| `57348f3` | swap-exhaustion floor triggers survival mode independently (sin necesitar vm.pressure alto) |
| `c7bd58e` | bypass sleep-assertion gate en swap exhaustion (≥4GB) |
| `e0cd030` | swap exhaustion forces BackgroundPressure context |
| `b482130` | suppress learned-policy BoostProcess under swap exhaustion |
| `1e5c852` | meta-gate vetoes Raise1pp under sustained swap growth |
| `55fd8f0` | upgrade throttle to aggressive under survival mode |
| `602c993` | relative swap exhaustion threshold scales with total (no hardcoded 4GB) |
| `cc6611e` | rate-limited purge at 80% swap exhaustion |

### Chromium + Survival Integration

`59b449d` — "feat(chromium): demote background renderers to jetsam BACKGROUND in survival": durante survival mode, los background renderers de Chrome/Brave/Edge son demoted a `JETSAM_PRIORITY_BACKGROUND` para que el kernel los candidates primero para kill si el sistema necesita liberar memoria.

---

## Patrones Meta-Evolutivos

### 1. Fix-N-minus-1 Pattern
Múltiples bugs donde el fix de un componente exponía un bug en el componente anterior de la cadena. Ejemplo: fix de compressor pages → expone thrashing detection bug → expone swap threshold hardcoding.

### 2. Revert-Analyze-Reland
El patrón Chromium (revert × 2 antes de solución correcta) apareció también en: Spotlight gate (2 iteraciones), survival mode threshold (3 iteraciones en RestoreQualityMonitor).

### 3. Sensor-antes-que-Consumer
La arquitectura consciente de "primero instrumentar, luego actuar" — sensors añadidos un sprint antes de los decision consumers — previene cambios de comportamiento no calibrados.

### 4. AutoResearch como Acelerador
El 57% de commits vino de AutoResearch (Épocas 3-4). La calidad arquitectónica fue mantenida porque cada experimento tenía scenarios verificables como ground truth.

---

## Línea de Tiempo de Módulos Clave

| Módulo | Introducido | Época | LOC actuales |
|--------|-------------|-------|-------------|
| `effective_pressure.rs` | Mar 14 | E2 | ~400 |
| `rl_threshold.rs` | Mar 14 | E2 | 1213 |
| `outcome_tracker.rs` | Mar 14 | E2 | 2393 |
| `causal_graph.rs` | Mar 29 | E3 | ~1175 |
| `learning_pipeline.rs` | Mar 29 | E3 | 873 |
| `intelligence_score.rs` | Mar 29 | E3 | 2081 |
| `learned_state.rs` | Mar 30 | E3 | 1605 |
| `chromium_manager.rs` | Abr 1 | E3/E4 | 2402 |
| `nested_learner.rs` | Abr 10 | E6 | ~600 |
| `teacher_consolidation.rs` | Abr 11 | E6 | ~400 |
| `cpu_saturation.rs` | Abr 8 | E5 | ~200 |
| `contention_tracker.rs` | Abr 8 | E5 | ~300 |

---

## Invariantes que Nunca Cambiaron

A través de las 8 épocas, estas decisiones arquitectónicas se mantuvieron estables:

1. **Unix socket IPC**: `/var/run/apollo-optimizer.sock` (root) — presente desde E1.
2. **Write-then-rename atómico**: para todos los archivos de estado — presente desde E2.
3. **SIGSTOP/SIGCONT para freeze**: nunca se reemplazó por cgroups u otras alternativas.
4. **Safety module como firewall**: ninguna acción se ejecuta sin pasar por `safety.rs`.
5. **Kill switch**: `/var/run/apollo.disable` — presente desde E1, nunca eliminado.
6. **Lista de procesos protegidos**: WindowServer, Antigravity, Claude, Brave, rustc/cargo durante build — expandida pero nunca reducida.
