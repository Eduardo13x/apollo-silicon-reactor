# 🧠 Apollo Optimizer — Mapa 360° de Músculos del Core

> **Base:** 782 commits · Rango: 2026-03-01 → 2026-04-20 · Velocidad: ~188 commits/marzo, ~594 commits/abril  
> **Distribución de tipos:** `feat`×207 · `fix`×169 · `experiment`×94 · `refactor`×51 · `test`×35 · `perf`×25 · `revert`×17

---

## 🔬 Función de Relevancia de Módulo (MRF)

La **Module Relevance Function** cuantifica cuánto "trabajo gravitacional" concentra cada módulo. Se calcula como:

```
MRF(m) = (feat_count × 2) + (fix_count × 3) + (experiment_count × 1.5)
        + (revert_count × 4) + (refactor_count × 1) - (test_cover_ratio × 5)
```

> `revert × 4` porque un revert indica oscilación: el módulo no tiene modelo mental estable.  
> `test_cover_ratio - 5` como **bono**: a mayor cobertura de tests, menor urgencia de fortalecimiento.

| Módulo / Subsistema | `feat` | `fix` | `exp` | `rev` | `ref` | MRF est. | 💪 Músculo |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **chromium_manager** | 8 | 5 | 0 | 3 | 6 | **54** | 🔴 CRÍTICO |
| **intelligence_score (AIS)** | 7 | 12 | 10 | 2 | 0 | **65** | 🔴 CRÍTICO |
| **safety / sentinel** | 10 | 15 | 0 | 0 | 0 | **65** | 🔴 CRÍTICO |
| **learned_state** | 6 | 5 | 2 | 0 | 0 | **32** | 🟠 ALTO |
| **neuromodulator / neuro** | 13 | 6 | 0 | 0 | 1 | **45** | 🟠 ALTO |
| **cognitive stack (NARS+bus)** | 13 | 2 | 10 | 0 | 2 | **43** | 🟠 ALTO |
| **decide_actions** | 4 | 2 | 0 | 0 | 3 | **19** | 🟡 MEDIO |
| **causal_graph** | 9 | 3 | 3 | 1 | 0 | **31** | 🟠 ALTO |
| **signal_intelligence** | 5 | 4 | 10 | 2 | 0 | **36** | 🟠 ALTO |
| **fluidity** | 11 | 1 | 0 | 0 | 0 | **25** | 🟡 MEDIO |
| **spotlight** | 0 | 5 | 0 | 1 | 0 | **19** | 🟡 MEDIO |
| **daemon cycle** | 0 | 4 | 24 | 3 | 11 | **59** | 🔴 CRÍTICO |
| **unfreeze_decay / ODE** | 5 | 3 | 0 | 0 | 0 | **19** | 🟡 MEDIO |
| **swap_reclaim** | 3 | 3 | 0 | 0 | 0 | **15** | 🟢 BAJO |
| **overflow_guard / rl_threshold** | 3 | 4 | 2 | 1 | 0 | **23** | 🟡 MEDIO |
| **mach_qos** | 3 | 0 | 0 | 0 | 1 | **7** | 🟢 BAJO |
| **outcome_tracker** | 2 | 2 | 0 | 0 | 0 | **10** | 🟢 BAJO |
| **hazard_model** | 2 | 4 | 0 | 0 | 0 | **16** | 🟡 MEDIO |
| **planner / hierarchical** | 1 | 0 | 0 | 0 | 0 | **2** | 🟢 BAJO* |

> \* `planner` tiene MRF bajo porque es un **Strangler Fig Phase 0** — recién nacido, músculo atrofiado por diseño.

---

## 📊 Evolución Arquitectónica: Las 5 Eras

### Era 1 — Bootstrap & Kernel Direct (mar 2026, semanas 1–2)
**Commits clave:** `Kernel-direct: replace all subprocess calls`, `Real-time memory pressure via sysctl poller`, `IPC-aware throttling via thread_selfcounts`

El sistema arrancó con una filosofía correcta pero inconsistente: mezcla de llamadas subprocess (`mdutil`) con accesos directos al kernel. El primer músculo grande que se definió fue **colección de datos**, pero sin pipeline de decisión real. Los módulos `energy.rs`, `thermal_manager.rs`, y `memory_analyzer.rs` surgieron en este período como monolitos de conveniencia.

**Debt fundacional:** la ausencia de un modelo de presión unificado obligó a re-inventarlo 3 veces (`effective_pressure.rs` en Era 3, luego `compressor_aware.rs`, luego el aggregator extraído en Era 5).

---

### Era 2 — Learning Stack & Feedback Loops (mar semanas 3–4)
**Commits clave:** `Phase 1–6` `v0.5.0`, `OutcomeTracker → decide_actions`, `RL thresholds`, `Holt-Winters`

Este período define el **corazón retroalimentado**: se conecta `OutcomeTracker → decide_actions` por primera vez. Los Phases 1–6 son una secuencia lineal que indica que el equipo estaba iterando en caliente, sin staging. Se descubren varios false positives en `overflow_guard`.

**Patrón recurrente:** wiring desconectado. Los commits `Wire 7 disconnected systems` y `Wire remaining 4 disconnected systems` muestran que se construían módulos sin conectarlos al ciclo de decisión real. Esto es **deuda de integración crónica**.

---

### Era 3 — v0.6/v0.7 Self-Evolving + Deep Scan (mar semanas 5–6)
**Commits clave:** `v0.6.0 Self-Evolving`, `Level 2/3 Alien X`, `v0.7.0 Deep Scan`

Explosión de complejidad: se ingestan 3 loops de aprendizaje aislados que se reconectan via adaptive router. Aparecen `causal_graph`, `zone_classifier`, y los primeros `experiment(nars)`. El `chromium_manager` nace aquí y **inmediatamente** genera 3 reverts seguidos por bugs de tabs congelados.

**Patrón bug Era 3:** funcionalidad "Level N" añadida sin tests de regresión. Los `BUG-01` a `BUG-09` rastreables están todos en esta era.

---

### Era 4 — Cognitive Architecture (abr semanas 1–2)
**Commits clave:** `ITER1→ITER10 cognitive`, `NARS beliefs`, `ArousalState`, `SelfReward`, `ReptileMeta`

10 iteraciones en cadena de la pila cognitiva. Es el período más denso en `feat(cognitive)` y también donde aparecen los primeros commits `fix(cognitive)`. Se introduce NARS como motor de creencias, pero su wiring es inconsistente: hay `experiment(nars)` con reverts múltiples porque el sistema de creencias aún no tiene modelo estable de convergencia.

**Pattern:** `experiment → revert → experiment` en `nars`, `kalman`, `cusum`, `salience`. El MRF sube porque el costo de la oscilación experimental es alto.

---

### Era 5 — ODE Physics + G-Gap Closure (abr semanas 3–4)
**Commits clave:** `G10→G21 gap closure`, `KalmanMV8`, `OdeDivergenceResilient`, `τ-informed freeze`, `thermal bulkhead`

El período más maduro. Se adopta un sistema de **G-numbers** (gaps identificados) para cierre sistemático. Los ODE (Ordinary Differential Equations) para decay de memoria, swap reclaim, y arousal se consolidan bajo el trait `CyberPhysicalSignal`. El daemon se refactoriza en submódulos extraídos (~11 PR-extractos).

**Strength:** el patrón `feat(G-N):` indica disciplina de closure. **Debilidad:** G1–G5 no están en los commits visibles, lo que sugiere que o se cerraron ad-hoc sin tracking o quedaron sin resolver.

---

## 🏋️ Análisis por Módulo — Puntos de Fortaleza y Debilidad

---

### 🔴 `intelligence_score.rs` (AIS — Adaptive Intelligence Score)
**Tamaño:** 91 KB · **Commits totales:** ~29 (12 fix + 7 feat + 10 experiment)

#### Músculos fuertes ✅
- Sistema de 5 dimensiones (D1–D5) bien definidas con referencias académicas (Kalman 1960, Cover 2006, Jain 1991)
- Riccati-adaptive threshold en D2 ajusta el Kalman dinámicamente según carga térmica
- Benchmarks vivos: AIS floor subió 87.0→90.0 después de 7 iteraciones de evolución

#### Músculos débiles ❌
- **Oscillación en D2:** el commit `fix: D2 kalman ^2→^3` fue revertido y reaplicado — el exponent correcto de scoring no tiene test de regresión que lo ancle
- **D5 safety gradient:** `fix: D5 overflow scoring 6-30=0.15 replaces near-zero bucket` corrige un bucket de scoring que producía scores casi cero — error de calibración no detectado por tests previos
- **D1 vacuous truth:** `fix: protected_rate=1.0 when no protected processes` — el sistema daba score perfecto en estado vacío
- **Sin test de integración E2E:** los benchmarks son unitarios; no hay un test que valide que un AIS de 90 realmente correlaciona con comportamiento de sistema saludable

#### 🎯 Recomendación
Crear una **AIS Golden Dataset**: capturas de estado del daemon en condiciones conocidas (presión normal, pico de swap, build activo) y assertar que el AIS score cae dentro de rangos esperados. Esto convierte el AIS de un "score de autovalidación" a un "contrato observable".

---

### 🔴 `chromium_manager.rs`
**Tamaño:** 101 KB (el más grande del engine) · **Commits:** ~29 (8 feat + 5 fix + 3 revert + 6 refactor)

#### Músculos fuertes ✅
- Inventario de renderers separado del gate de freeze (decoupling correcto post-fix)
- Visibilidad via CGWindowList — freeze solo afecta renderers no visibles
- Build-preemption mode como bulkhead entre rustc y renderers
- ITER1–ITER3 conectan FocusMarkov + ArousalState + NARS para decisiones contextuales

#### Músculos débiles ❌
- **Triple revert en renderer SIGSTOP:** `revert: disable renderer freeze — tabs stay frozen in prod` → `re-enable` → `revert permanently disable` indica que el modelo mental del ciclo de vida de tab aún es inestable
- **Decouple tardío:** el bug `chromium_renderers=0` (inventario desconectado del gate) debería haberse detectado en PR review — sugiere falta de tests de integración reales
- **101 KB sin submódulo:** el módulo más grande del sistema es un monolito. La refactorización solo redujo LOC en tests, no extrajo responsabilidades
- **Thaw responsiveness:** aparece en commit `stabilize chromium freezer and improve thaw responsiveness` — en producción los tabs quedaban congelados indefinidamente bajo ciertas condiciones

#### 🎯 Recomendación
Extraer `ChromiumRenderer`, `ChromiumFreezePolicyEngine`, y `ChromiumThawOrchestrator` como structs separados dentro del archivo o en submódulos. El 101 KB actual mezcla 4 responsabilidades (inventario, visibility, policy, learning integration). Un test de contrato `freeze → thaw → assert_responsive` con mock de CGWindowList es la pieza más faltante.

---

### 🔴 `safety.rs` / `sentinel` / `freeze_gate`
**Tamaño:** 53 KB · **Commits:** ~25 (10 fix + 4 feat + 1 revert)

#### Músculos fuertes ✅
- Protección estática de GUI apps con `OnceLock` cache (~900 eliminaciones de alloc/ciclo)
- PID identity check (A-B-A) antes de SIGSTOP — previene señal a PID reciclado
- `recover()` consolidado con lock único elimina TOCTOU en freeze
- Denylist de behavioral (antipattern: boost Apple background daemons)

#### Músculos débiles ❌
- **Bypass de freeze repetidos:** `fix: close 3 bypass paths`, `fix: Close freeze bypass in heuristic merge`, `fix: Complete Mediation` — el mismo tipo de bug (bypass de protección) aparece ≥3 veces, lo que indica que el modelo de "quién puede ser congelado" no está centralizado
- **Oscillación Spotlight:** 5 fixes + 1 revert en el gate de Spotlight sugieren que la lógica de presión para re-enable/disable no tiene un modelo correcto de equilibrio
- **Jetsam hint falso:** `fix: disable send_memory_pressure_hint — was triggering launchd jetsam` — se activaba jetsam incorrectamente hasta que se deshabilitó manualmente
- **Protección de `log` CLI:** `fix: protect log CLI from throttle (self-targeting cascade)` — el daemon se auto-throttleaba

#### 🎯 Recomendación
Centralizar el modelo de protección en una sola función pura `is_protected(pid, context) -> ProtectionLevel` que sea el **único** punto de decisión. Hoy la lógica está distribuida en `safety.rs`, `sentinel`, `freeze_gate`, y partes de `decide_actions`. Los bypass repetidos son síntoma de que hay múltiples caminos que no consultan esa función central.

---

### 🟠 `learned_state.rs`
**Tamaño:** 69 KB · **Commits:** ~18 (6 feat + 5 fix + 2 experiment)

#### Músculos fuertes ✅
- Escritura atómica previene pérdida de estado en crash
- `RestoreQualityMonitor` detecta estado corrupto
- Forward-compat serde con `#[serde(other)]` en `FreezeSource` — evita panic en restart tras upgrade

#### Músculos débiles ❌
- **False stale detection:** `fix: root-cause fix for RestoreQualityMonitor false stale detection` — el monitor que detecta corrupción generaba falsos positivos, descartando estado válido
- **Double-increment:** `fix: persist_generations increments once per persist, not twice` — bug de conteo que inflaba el número de generaciones
- **Warm baselines perdían state al reiniciar** hasta el commit `feat: persist ProcessBaselineMap` — el aprendizaje por proceso no sobrevivía reinicios durante meses de desarrollo
- **22+ parámetros adaptativos en LearnableParams sin schema versioning** — cualquier campo nuevo puede romper deserialización silenciosamente

#### 🎯 Recomendación
Añadir `schema_version: u32` al `LearnedState` serializado y una función `migrate(v_old: u32, raw: serde_json::Value) -> LearnedState` para cada migración. La combinación de 22 parámetros adaptativos + persistencia sin versioning es una bomba de tiempo en upgrades silenciosos del daemon.

---

### 🟠 `neuromodulator.rs` / stack neurocognitivo
**Tamaño:** 15 KB base + integración en ~20 módulos · **Commits neuro:** ~25 (13 feat + 6 fix)

#### Músculos fuertes ✅
- Dopamina (DA), Acetylcolina (ACh), Norepinefrina (NA), Serotonina (5-HT) como señales diferenciadas
- ODE prediction-error signals para DA/ACh (gaps 4+5)
- LearnableParams propaga NARS decay + CUSUM k/h + Kalman Q/R como live parameters
- Graded thermal stress reemplaza binario emergency

#### Músculos débiles ❌
- **G11 requirió 2 fixes:** el proxy de entropía para `contention_stall_fraction` en M1 necesitó corrección de saturación post-implementación — el M1 no expone `contention_stall_fraction` directamente via KPC y el fallback era incorrecto
- **`fix: cognitive_snr + self_eval_quality` — key mismatch:** un key de HashMap incorrecto hacía que la señal neurocognitiva no se actualizara; el módulo era silenciosamente no-operativo
- **Coupling neuromodulador→daemon:** el `refactor(neuro)` extrae el tick pero el estado de DA/ACh/NA aún vive en SharedState — si el daemon se reinicia, las señales parten de cero sin warm-start
- **Sin tests de convergencia:** no hay test que verifique que DA sube ante sorpresa y decae correctamente

#### 🎯 Recomendación
Añadir `NeuroState` a `LearnedState` para warm-start de señales neuromodulatorias. Un test de **impulse-response**: inyectar evento de sorpresa, assertar que DA > baseline en ciclo N+1, DA ≈ baseline en ciclo N+30. Sin esta prueba, es imposible saber si el neuromodulator está convergiendo o divergiendo en producción.

---

### 🟠 `signal_intelligence.rs` + `kalman.rs` + `cusum.rs`
**Tamaño agregado:** 81 KB + 32 KB + 8 KB · **Commits:** ~31 (12 AIS + señales directas + experiments)

#### Músculos fuertes ✅
- `KalmanMV8` — 8-state multivariate filter fusionando presión + ODE signals
- `CyberPhysicalSignal` trait unifica normalización ODE (gap 9)
- CUSUM k/h + Kalman Q/R ajustables via `LearnableParams`
- 5 benchmarks de signal pipeline añadidos

#### Músculos débiles ❌
- **Revert de Kalman exponent:** el exponent correcto para scoring de Cramér-Rao fue revertido tras análisis — el módulo fue incorrecto en producción por un período
- **Experiment oscillation:** experiments de `kalman`, `cusum`, `signal` fueron revertidos porque "regressed". Sin un baseline metric grabado automáticamente, la comparación de regresión es manual y propensa a error
- **NaN sin guards hasta fix tardío:** `fix: add NaN guards to Kalman/CUSUM` llegó tarde — el sistema podía divergir silenciosamente con NaN en los filtros
- **KalmanMV8 es nuevo:** introducido recientemente, aún no tiene tests más allá de los de señal básica. Con 8 dimensiones de estado, las condiciones de divergencia son difíciles de detectar unitariamente

#### 🎯 Recomendación
Implementar `SignalHealthMonitor` que detecte NaN/Inf/subnormal en cualquier salida de Kalman/CUSUM y emita una métrica `signal_health_violations_total`. Actualmente una divergencia numérica solo se detecta si hace crash o produce comportamiento observable. El `adversarial_probe.rs` ya tiene `OdeDivergenceResilient` — el patrón existe, hay que replicarlo para el pipeline de señal.

---

### 🟠 `causal_graph.rs`
**Tamaño:** 47 KB · **Commits:** ~15 (9 feat + 3 fix + 3 experiment + 1 revert)

#### Músculos fuertes ✅
- Edges clasificados como `solid`, `weak`, `ambiguous` con crédito diferenciado (3/4 para ambiguous)
- `self-improvement decay` + validación de edges persistidos
- Integrado con `LearningPipeline` para score de efectividad de skills
- +30 tests: multi-horizon, mechanisms, clusters, NARS

#### Músculos débiles ❌
- **Ambiguous edge credit cambiado de 1/2 → 3/4:** calibración ad-hoc sin modelo formal de por qué ese coeficiente es correcto
- **Causal graph solo graba newly-frozen PIDs:** `fix: causal graph records only newly-frozen PIDs` — estaba grabando todos los PIDs procesados, inflando los datos de causalidad
- **Causal depth contaba edges incorrectamente:** `improve: causal depth counts resolved edges (solid + weak)` — el score de profundidad era incorrecto en producción
- **Sin test de falsos positivos causal:** no hay test que verifique que una correlación espuria no se éleva a `solid` edge

#### 🎯 Recomendación
Añadir un test de **counterfactual validity**: si Apollo ejecuta acción A en contexto C y la métrica mejora, verificar que el causal edge C→A se refuerza, y que si repite la acción en contexto C' (sin mejora), el edge se debilita. Esto requiere un harness de simulación, no solo unit tests de estructura.

---

### 🟠 `cognitive_bus.rs` / NARS stack
**Tamaño:** 15 KB (bus) + `nars_belief.rs` 57 KB · **Commits NARS:** ~25 (10 experiment + 3 chore)

#### Músculos fuertes ✅
- NARS provee beliefs con frecuencia/confianza para decisions de monopoly specialist
- `observations_to_reach` — inversa algebraica de `confidence_from_count` para planificación
- `DriftDetector` integrado vía OutcomeTrackerPersisted
- Contextual beliefs por pressure bucket + workload-aware

#### Músculos débiles ❌
- **10 experiments con reverts:** la mayor densidad de experimentos sin convergencia. NARS no tiene un "test de verdad" que confirme que las beliefs se actualizan correctamente
- **Belief maturity gate mal cableada:** `fix: key monopoly maturity gate to real NARS belief` — la gate usaba una métrica proxy en lugar de la belief real durante semanas
- **Sin modelo de olvido explícito bajo cold-start:** al reiniciar el daemon, NARS empieza con creencias frías e inmediatamente tomará decisiones de alto riesgo basadas en prior débil
- **El `nars_drift_threshold` wire llega tarde:** la conexión del threshold de NARS al `DriftDetector` no existía hasta un commit específico — el drift se detectaba pero no se usaba en NARS

#### 🎯 Recomendación
Definir un **NARS Convergence Contract**: dado N observaciones de un patrón estable, `confidence` debe superar 0.6. Dado un cambio de régimen, `confidence` debe caer por debajo de 0.3 dentro de M ciclos. Sin este contrato expresado como test, las 10+ iteraciones de experimentación no tienen criterio de "terminado".

---

### 🟡 `decide_actions.rs`
**Tamaño:** 82 KB · **Commits:** ~10 (4 feat + 2 fix + 3 refactor + 1 test batch)

#### Músculos fuertes ✅
- `DecisionStage::run()` + `PolicyContext` extraídos del hot path
- 22 tests cubriendo context, blocker score, classification
- `effective_context` reportado en salida — transparencia de por qué se tomó la decisión
- Anomaly hints + IO burst hints integrados correctamente

#### Músculos débiles ❌
- **82 KB con múltiples responsabilidades:** context classification, blocker scoring, anomaly gating, IO burst detection, sleep assertion handling — demasiado acoplado
- **`fix: effective_context reports raw context`** — por un período reportaba el contexto antes de aplicar modificadores (LLM mode, user context), generando telemetría engañosa
- **Interactive app immunity** no tenía test hasta `test(decide_actions): verify interactive apps are immune` — una invariante de seguridad sin prueba

#### 🎯 Recomendación
Extraer `ContextClassifier`, `BlockerScorer`, y `ActionGate` como structs independientes dentro de `decide_actions.rs`. El módulo es el cuello de botella de decisión del sistema y actualmente mezcla sensing, scoring, y execution gating en la misma función.

---

### 🟡 `fluidity.rs`
**Tamaño:** 35 KB · **Commits:** ~13 (11 feat + 1 fix)

#### Músculos fuertes ✅
- Kalman prediction fields expuestos en RuntimeMetrics
- Launch acceleration: defer background freezes durante app launch
- WindowServer CPU integrado como señal primaria
- 4 micro-benchmarks añadidos

#### Músculos débiles ❌
- **Módulo completamente nuevo en Era 5:** 11 feat commits en pocos días sugieren implementación acelerada. La velocidad de features sin fix pattern es señal de que los bugs aún no han emergido
- **`fix: add fluidity fields to all SignalDigest struct literals`** — se olvidaron campos en varias inicializaciones de struct, causando datos incompletos. Error típico de implementación apresurada
- **`ws_spike_threshold` y `fluidity_degraded_threshold` llegaron tarde desde LearnableParams** — los umbrales eran hardcoded hasta otro commit

#### 🎯 Recomendación
El fluidity module necesita **pruebas de regresión de latencia percibida**: simular WindowServer CPU spike y verificar que `fluidity_score` cae dentro de 1 ciclo y recupera en ≤5 ciclos tras la resolución. Sin benchmark de respuesta temporal, el módulo puede parecer funcional pero ser lento para reaccionar.

---

### 🟡 `overflow_guard.rs` / `rl_threshold.rs`
**Tamaño:** 26 KB + 46 KB · **Commits:** ~15 (3 feat + 4 fix + 2 experiment + 1 revert)

#### Músculos fuertes ✅
- Q-table con α clamped a [0,1] — previene divergencia
- `fix: band_to_pressure maps all 4 bands` — corrección de bug crítico donde band 3 colapsaba en band 2
- `fix: bandit: remove -0.1 penalty that locked agent on Observe forever` — el agente aprendía a no actuar

#### Músculos débiles ❌
- **Bandit atrapado en Observe:** el penalty negativo bloqueó al agente en un estado de inacción — error de diseño de reward tardíamente detectado
- **G15 complacency fix:** el RL no detectaba cuando Observe precedía un overflow — el agente era recompensado por no actuar incluso cuando debería haber actuado
- **`fix: overflow_guard stuck at floor`** — el guard se quedaba en su valor mínimo y no respondía — existía desde Era 2 y se detectó tardíamente

#### 🎯 Recomendación
El RL agent necesita una suite de **adversarial scenarios** dedicados: (1) presión que sube sin intervención para verificar que Observe no es elegido, (2) presión que mejora sola para verificar que el agente no recibe crédito incorrecto. Los escenarios "Boss Level" ya existen para el módulo cognitive — replicar ese patrón aquí.

---

### 🔴 `daemon cycle` (apollo-optimizerd principal)
**Commits:** ~50 (24 e2e experiments + 11 refactor + 4 fix)

#### Músculos fuertes ✅  
- Ciclo ahora descompuesto en 12 submódulos extraídos (Phase 1–4 refactor + 11 PR extracts)
- `run_periodic()` abstracto para GC, eliminando bloques `% 500` inline
- `LearningContext<'a>` + `DecisionStage::run()` formalizan el hot path
- `cycle_dt_secs` corregido para abarcar el intervalo completo (no solo parte del loop)

#### Músculos débiles ❌
- **24 experiments e2e con reverts**: la mayor concentración de experimentos en el sistema. El ciclo principal fue el banco de prueba de optimizaciones de throughput, pero `skip refresh_processes()` fue revertido dos veces porque omitir el refresh rompe invariantes de decisión
- **`Revert: SharedState grouping`** — la consolidación de 20 Mutex→10 fue revertida, dejando deuda de SharedState sin resolver
- **`cycle_dt_secs` mid-loop bug**: el dt se reseteaba a mitad del ciclo, corrompiendo los modelos basados en tiempo (ODE, decay). Este bug es sutil e impactó todos los módulos que usan tiempo relativo
- **Socket handler en el hot path**: hasta el refactor, el socket handler bloqueaba el ciclo principal en ciertos casos

#### 🎯 Recomendación
Definir un **Daemon Cycle Contract**: el ciclo debe completarse en <X ms bajo presión normal, <Y ms bajo survival mode, con una lista explícita de qué pasos son **mandatorios** vs **deferibles** (skippable bajo carga). El experimento `skip refresh_processes()` evidencia que este contrato no existe formalmente — cada experimento lo redefine ad-hoc.

---

## 🗺️ Mapa de Calor de Bugs por Módulo

```
Módulo                   │ fix count │ revert │ Peligrosidad │ Estado
─────────────────────────┼───────────┼────────┼──────────────┼────────────────
intelligence_score (AIS) │    12     │   2    │   🔴 ALTA    │ Mejorando (7-iter)
safety / sentinel        │    15     │   0    │   🔴 ALTA    │ Estable post-Era4
chromium_manager         │     5     │   3    │   🔴 ALTA    │ Inestable (thaw)
learned_state            │     5     │   0    │   🟠 MEDIA   │ Mejorando
neuromodulator           │     6     │   0    │   🟠 MEDIA   │ Mejorando (G11)
signal_intelligence      │     4     │   2    │   🟠 MEDIA   │ Mejorando (KalmanMV8)
spotlight                │     5     │   1    │   🟠 MEDIA   │ Oscillando
overflow_guard / rl      │     4     │   1    │   🟡 MEDIA   │ Mejorando
causal_graph             │     3     │   1    │   🟡 MEDIA   │ Estable
decide_actions           │     2     │   0    │   🟡 BAJA    │ Estable
fluidity                 │     1     │   0    │   🟢 BAJA    │ Nuevo — sin datos
swap_reclaim             │     3     │   0    │   🟢 BAJA    │ Estable
```

---

## 🔁 Patrones de Bug Recurrentes (Cross-Module)

### ① El bug de Wiring Desconectado
**Prevalencia:** ≥15 commits con "wire" en el título  
**Síntoma:** Un módulo existe y tiene lógica correcta pero no está conectado al ciclo de decisión. Puede pasar meses sin que sus outputs lleguen a `decide_actions`.  
**Módulos afectados:** stale_apps, purgeable purge, specialist weights, workload hourly bias  
**Fix:** Checklist de integración en PR: **¿dónde se consume este módulo? ¿está el campo en RuntimeMetrics? ¿hay un test que falle si se desconecta?**

---

### ② El bug de NaN / Subnormal Silencioso
**Prevalencia:** NaN guards añadidos tardíamente en Kalman, CUSUM, utility EMA  
**Síntoma:** Un valor numérico diverge a NaN/Inf/subnormal y el módulo produce outputs que parecen válidos pero están corrompidos.  
**Módulos afectados:** `kalman.rs`, `ais`, `intelligence_score` (utility EMA floor)  
**Fix:** `SignalHealthMonitor` global que detecte valores patológicos en cualquier output de filtro.

---

### ③ El bug de PID Reciclado
**Prevalencia:** 3 fixes explícitos (pre-sleep, wake-unfreeze, sentinel)  
**Síntoma:** Un PID es reutilizado por un nuevo proceso; Apollo envía SIGSTOP/SIGCONT al proceso equivocado.  
**Módulos afectados:** `sentinel`, `daemon_freeze_executor`, `lifecycle`  
**Fix:** PID identity check (start_usec + name) ya implementado — **falta propagarlo a todos los paths de freeze/unfreeze**, especialmente los de wake_unfreeze.

---

### ④ El bug de Cooldown con Sleep
**Prevalencia:** `fix: use wall-clock for cooldown timers — survive sleep/wake`  
**Síntoma:** Un timer basado en `Instant` no avanza durante sleep del sistema; al despertar, el cooldown parece no haber pasado.  
**Módulos afectados:** `sysctl_governor`, cualquier módulo con `min_interval_secs`  
**Fix:** Auditar todos los módulos que usan `Instant::now()` para cooldowns y migrarlos a `SystemTime` o wall-clock equivalente.

---

### ⑤ La Oscilación de Feature Toggle
**Prevalencia:** Multiple reverts de features completas (Transformer, Chromium freeze, Spotlight pressure gate)  
**Síntoma:** Una feature se implementa, se despliega, produce efectos negativos, se revierte. Luego se vuelve a intentar con modificaciones.  
**Root cause:** Falta de **feature flags** con degradation graceful. El sistema es all-or-nothing para features nuevas.  
**Fix:** Usar el patrón de `CHROMIUM_FREEZE_DISABLED=false` env-var que ya existe — generalizarlo a un sistema de feature flags en config.toml.

---

## 🎯 Top 10 Músculos a Fortalecer (Priorizado)

| # | Módulo | Acción Concreta | Impacto | Esfuerzo |
|---|--------|-----------------|---------|----------|
| 1 | **safety (protección centralizada)** | Función única `is_protected()` como verdad absoluta | 🔴 Crítico | 🟡 Medio |
| 2 | **daemon cycle (contrato de ciclo)** | Definir qué pasos son mandatorios vs deferibles en spec | 🔴 Crítico | 🟡 Medio |
| 3 | **learned_state (schema versioning)** | Añadir `schema_version` + migrations para LearnableParams | 🔴 Crítico | 🟢 Bajo |
| 4 | **signal pipeline (heath monitor)** | `SignalHealthMonitor` detecta NaN/Inf/subnormal | 🔴 Crítico | 🟢 Bajo |
| 5 | **chromium_manager (extracción)** | Separar Inventory / FreezePolicyEngine / ThawOrchestrator | 🟠 Alto | 🟠 Medio |
| 6 | **NARS (convergence contract)** | Test formal: N obs → confidence > 0.6 | 🟠 Alto | 🟢 Bajo |
| 7 | **RL/overflow (adversarial coverage)** | Scenarios "Observe inaction under rising pressure" | 🟠 Alto | 🟢 Bajo |
| 8 | **neuromodulator (warm-start)** | Añadir NeuroState a LearnedState | 🟠 Alto | 🟡 Medio |
| 9 | **cooldown timers (wall-clock audit)** | Auditar todos los `Instant` en cooldowns + migrar | 🟡 Medio | 🟢 Bajo |
| 10 | **fluidity (regression tests)** | Benchmark de respuesta temporal a WindowServer spike | 🟡 Medio | 🟢 Bajo |

---

## 📐 Arquitectura de Dependencias — Módulos con Máximo Fan-In

Los módulos que más frecuentemente reciben "wire" de otros indican el **núcleo de acoplamiento**:

```
decide_actions    ←── recibe de: 12+ módulos (máximo fan-in del sistema)
learned_state     ←── recibe de: 8+ módulos (persistencia global)
signal_intelligence ←── recibe de: 6+ módulos (filtros de señal)
neuromodulator    ←── recibe de: 5+ módulos (modulación global)
outcome_tracker   ←── recibe de: 4+ módulos (observación de efectividad)
```

> Regla: módulos con fan-in alto son los más costosos de cambiar y los que concentran más bugs de integración. Cada uno debería tener un **integration smoke test** que valide que todos sus inputs están conectados.

---

## 🧬 Evolución del Vocabulary de Commits — Señal de Madurez

| Era | Vocabulario dominante | Señal de madurez |
|-----|----------------------|------------------|
| Mar semana 1–2 | `Fix all warnings`, `Complete intelligent system` | 🟡 Monolítico |
| Mar semana 3–4 | `Phase N`, `Fix false positives`, `Wire systems` | 🟡 Iterativo sin tests |
| Mar semana 5 – Abr semana 1 | `Level N`, `BUG-0N`, `experiment(X): revert Y` | 🟠 Experimental con debt |
| Abr semana 2–3 | `ITER-N`, `feat(cognitive)`, `fix(nars)` | 🟠 Convergiendo |
| Abr semana 4 | `G[0-9]+`, `gap N`, `PR#N extraction` | 🟢 Disciplinado |

La evolución del vocabulario es positiva: del "fix all" monolítico inicial al sistema de G-numbers de cierre sistemático. El riesgo actual es que los G-numbers terminaron en G21 pero hay gaps no numerados anteriores.

---

## 📋 Checklist de PR — Función de Mejora Continua

Para cada nuevo módulo o integración, usar esta función antes de merge:

```
fn module_review_score(module: &str) -> ReviewScore {
    check!(has_integration_test(module));           // ¿Falla si se desconecta?
    check!(is_protected_via_safety_central(module));// ¿Usa is_protected() central?
    check!(outputs_in_runtime_metrics(module));     // ¿Observable en telemetría?
    check!(cooldowns_use_wallclock(module));        // ¿Cooldowns sobreviven sleep?
    check!(no_nan_paths(module));                   // ¿Guards numéricos?
    check!(warm_start_tested(module));              // ¿Recupera estado tras restart?
    check!(feature_flag_if_risky(module));          // ¿Tiene flag de degradación?
}
```

---

*Generado el 2026-04-20 · base: 782 commits · `/Users/eduardocortez/proyectos/system-optimizer`*
