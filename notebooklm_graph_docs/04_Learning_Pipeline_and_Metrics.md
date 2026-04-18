# Navegación
[<- Cognitive Architecture](./03_Cognitive_Architecture.md) | [Volver al Índice](./00_Index.md) | [Siguiente: Historical Evolution ->](./05_Evolution_and_Commit_History.md)

# 04. Learning Pipeline & Metrics — Orquestación del Conocimiento Empírico

Este documento describe los mecanismos de aprendizaje operativo de Apollo: cómo las acciones generan observaciones, cómo esas observaciones se destilan en conocimiento persistente vía 3 subsistemas, cómo se fusionan en un score unificado (F3 Blend), y cómo el sistema se auto-evalúa (AIS, UCHS). Comprende ~8 módulos que suman ~350K bytes de código fuente.

---

## 1. LearningPipeline — El Coordinador Mini-Batch

`learning_pipeline.rs` (34,386 bytes / 873 LOC) — Coordinador de batch para los 3 subsistemas de aprendizaje.

### 1.1 LearningObservation

Cada acción ejecutada por el Core Engine genera una observación:

```rust
pub struct LearningObservation {
    pub process_name: String,
    pub skill_name: Option<String>,   // Si fue disparado por un Skill registrado
    pub pre_pressure: f32,            // Presión antes de la acción
    pub post_pressure: f32,           // Presión después (medida ~3 ciclos después)
    pub workload: WorkloadMode,       // Idle, Interactive, Build, HeavyBuild
    pub cycle: u64,                   // Número de ciclo global
}

impl LearningObservation {
    pub fn effective(&self) -> bool {
        (self.pre_pressure - self.post_pressure) >= 0.01
    }
}
```

**Threshold de efectividad:** Un drop de presión ≥ 0.01 se considera "efectivo". Este umbral fue calibrado contra la línea base natural de presión (commit `de24ad0` — "Phase 1b: Calibrate low_value threshold against natural pressure baseline").

### 1.2 Flujo de Batch

```
push(observation) → buffer.push(obs)
                    │
                    ├── Si buffer.len() >= 8  →  flush():
                    │   1. Ordena por process_name (cache locality en HashMap lookups)
                    │   2. Fan-out a 3 subsistemas:
                    │      ├── OutcomeTracker::record(obs)
                    │      ├── CausalGraph::record_action(obs)
                    │      └── SkillRegistry::record_outcome(obs)
                    │   3. Cross-feed rules A, B, C (ver §1.5)
                    │   4. Sync → EffectivenessTracker::update(process, blended_score)
                    │
                    └── Si buffer.len() < 8  →  espera más observaciones
```

**¿Por qué batch_size = 8?** Suficiente para amortizar el costo de lookups en HashMaps y ordenamiento, pero lo bastante pequeño para no retrasar el aprendizaje más de ~16 segundos (a tick rate de 2s). Configurado como default, no hardcoded.

### 1.3 Evolución del Pipeline

El pipeline fue introducido en commit `72847bc` ("feat(learning): LearningPipeline trait, mini-batch, cross-feed OutcomeTracker→SkillRegistry") y luego fusionado via swarm merge en `dc2f9f3` ("merge(swarm-Arq5): LearningPipeline trait, mini-batch, 3-way cross-feed, 24 tests"). Originalmente los 3 subsistemas funcionaban independientemente sin comunicación (problema documentado en CLAUDE.md §Current Development: "Three independent learning loops never cross-feed").

---

## 2. Subsistema A: OutcomeTracker — Pesos Bayesianos Per-Proceso

`outcome_tracker.rs` (94,638 bytes / 2,393 LOC) — El mayor módulo del engine después de `chromium_manager.rs`.

### 2.1 Modelo Bayesiano

Para cada proceso, mantiene un prior Laplace suavizado:

```
effectiveness(process) = (effective_count + 1) / (throttle_count + 2)
```

- **Laplace smoothing** (+1/+2): Previene extremos (0% o 100%) con pocas observaciones.
- **Cold start:** Con 0 observaciones, `effectiveness = 0.5` (neutral).
- **Interpretación:** `> 0.7` = throttling este proceso funciona confiablamente. `< 0.3` = throttling este proceso no sirve.

### 2.2 Co-occurrence Matrix

Detecta qué procesos tienden a aparecer juntos durante presión alta:

```rust
// Estructura interna simplificada
co_occurrence: HashMap<(String, String), CoOccurrenceEntry>

struct CoOccurrenceEntry {
    count: u32,
    last_seen: u64,  // ciclo
}
```

- **GC (Garbage Collection):** Cada ~500 ciclos, elimina entradas con `last_seen` > `max_stale_cycles` AND `count < min_count`.
- **Bug fix (commit `8073af4`):** "BUG-02 co_occurrence eviction" — Corrigió el fallback de eviction que fallaba con conteos homogéneos. Se implementó eviction basada en `last_seen` más antiguo.

### 2.3 Experience Memory Buffer

Mantiene un buffer circular de máximo 300 records de experiencias pasadas con auto-pruning:

```
self_improve() → antes de cada persist:
  1. Prune stale co-occurrence entries
  2. Remove noisy weights (demasiado pocas observaciones + desviación alta)
  3. Cap experience memory at 300 records
```

### 2.4 Integración con el Core Engine

Conectado directamente a `decide_actions.rs` (commit `518b2d9` — "Phase 1: Connect OutcomeTracker → decide_actions (close feedback loop)"):

- Si un proceso tiene `effectiveness < low_value_threshold` (calibrado contra baseline natural), el Adaptive Governor puede **skip** el throttle porque históricamente no tiene impacto.
- Los skips se exponen en `runtime_metrics.json` para observabilidad (commit `9d6549c`).

---

## 3. Subsistema B: CausalGraph — Inferencia Causal (Pearl)

`causal_graph.rs` (47,437 bytes / 1,175 LOC) — [Pearl 2009 "Causality: Models, Reasoning and Inference"].

### 3.1 Estructura del Grafo

```rust
edges: HashMap<(String, String), CausalEdge>  // (cause, effect) → edge

pub struct CausalEdge {
    pub confidence: f32,   // EMA α=0.10 (Bayesian update)
    pub avg_delta: f32,    // EMA α=0.15 (magnitud del pressure drop)
    pub evidence: u32,     // Número total de observaciones
    pub last_updated: u64, // Ciclo
}
```

- **cause:** `"throttle:{process_name}"` o `"freeze:{process_name}"`.
- **effect:** `"pressure_drop"` o `"pressure_neutral"`.

### 3.2 Evaluación Retrasada

El CausalGraph no evalúa el resultado inmediatamente. Espera `eval_delay = 3 ciclos` antes de comparar la presión actual con la presión al momento de la acción:

```
record_action(obs) → pending_queue.push(PendingEval { action, cycle, pressure_at_action })
evaluate(current_pressure, current_cycle):
  para cada pending donde (current_cycle - pending.cycle) >= eval_delay:
    delta = pending.pressure_at_action - current_pressure
    success = delta >= 0.01
    update_edge(cause, effect, success, delta)
```

- **pending queue:** Máximo 200 entradas (overflow silencioso).

### 3.3 Clasificación de Bordes

| Condición | Clasificación | Significado |
|-----------|---------------|-------------|
| `confidence > 0.7 && evidence ≥ 5` | **Solid** | Relación causal establecida con alta confianza |
| `confidence < 0.25 && evidence ≥ 5` | **Weak** | Relación refutada — la acción no causa el efecto |
| Otro | Indeterminate | Datos insuficientes o mixtos |

### 3.4 Impact Score

```
impact_score(edge) = confidence × avg_delta
```

Ranking real-world: un borde con `confidence=0.9, avg_delta=0.05` (impact=0.045) es más valioso que uno con `confidence=0.95, avg_delta=0.01` (impact=0.0095). Esto significa que "throttlear a Dropbox baja la presión significativamente" es más útil que "throttlear a Siri baja la presión imperceptiblemente".

### 3.5 Mechanism Breakdown (commit `bbb0964`)

```rust
pub fn mechanism_breakdown(&self) -> Vec<MechanismChannel> {
    // Expone Pearl mediation por canal: qué fracción del efecto total
    // pasa por cada intermediario (ej. "pressure_drop via memory_reclaim"
    // vs "pressure_drop via cpu_relief")
}
```

Permite al dashboard de runtime mostrar **por qué** una acción fue efectiva, no solo **si** fue efectiva.

---

## 4. Subsistema C: SkillRegistry — Recetas Aprendidas

`optimization_skills.rs` (27,418 bytes) + `rule_inducer.rs` (18,548 bytes)

### 4.1 Estructura de un Skill

```rust
pub struct OptimizationSkill {
    pub name: String,                // Ej: "throttle_dropbox_under_build"
    pub min_pressure: f32,           // Presión mínima para activar
    pub workload_hint: Option<WorkloadMode>, // Contexto de workload
    pub throttle_targets: Vec<String>, // Procesos a throttlear
    pub success_rate: f32,           // (success_count / apply_count)
    pub apply_count: u32,
    pub success_count: u32,
    pub origin: SkillOrigin,         // Individual, Induced (group:/batch:)
}
```

### 4.2 Tipos de Skills

- **Individual:** Aprendido de throttles directos sobre un proceso específico.
- **Induced (group:/batch:):** Generado por `rule_inducer.rs` que encuentra patrones de co-ocurrencia entre procesos que tienden a ser throttleados juntos exitosamente.

### 4.3 Ciclo de Vida

```
Creación:            record_outcome(obs) → nuevo skill individual si no existe
Inducción:           rule_inducer::induce() → group skills de co-ocurrencia
Maduración:          apply_count ≥ 5 && success_rate ≥ 0.60 → is_reliable()
Jubilación:          (≥10 apps, <35%) || (≥20 apps, <50%) → should_retire()
Auto-calibración:    adapt_pressure: EMA α=0.20 → auto-ajusta min_pressure
Exploración:         next_trial_skill() → round-robin de skills unproven
Limpieza:            purge_unexecutable() → elimina si todos los targets son protegidos
```

### 4.4 Rule Inducer

Introducido en commits `dd2911b`→`a5ffb50`→`b38c767` (abril 2026):

```
Iteración 1: Unlock 22 real skills from live data (a5ffb50)
Iteración 2: Batch detector + pressure filter (b38c767)
Iteración 3: Group skills de co-ocurrencia madurados
```

El Rule Inducer cristaliza la experiencia del daemon en "recetas" reutilizables. Si Apollo descubre que throttlear `softwareupdated` + `photolibraryd` + `Spotlight` juntos reduce la presión un 0.08 durante workloads de Build, genera un skill `batch:build_relief` que aplica esta combinación automáticamente la próxima vez.

---

## 5. EffectivenessTracker — Fusión F3 Blend (Thompson Sampling)

`effectiveness_tracker.rs` (22,859 bytes) — [Thompson 1933], [Russo et al. 2018 arXiv:1707.02038], [Auer 2002].

### 5.1 El Número Autoritativo

El EffectivenessTracker produce **un único score [0,1] por proceso** que es el número final usado por el Adaptive Governor para decidir si vale la pena throttlear/freezear ese proceso:

```
cred_bayesian = min(bayesian_obs / 20, 1.0)     ← satura a 20 obs
cred_causal   = min(causal_obs / 5, 1.0)        ← satura a 5 (Pearl dominance)
cred_skill    = min(skill_obs / 10, 1.0)        ← satura a 10

blended = (cred_b × bayes + cred_c × causal + cred_s × skill)
        / (cred_b + cred_c + cred_s)

Cold start (0 obs) → 0.5 (neutral). NaN guard + clamp [0,1].
```

### 5.2 Dominancia Causal

El diseño intencionalmente da más peso al CausalGraph cuando tiene suficiente evidencia:
- Causal: 5 obs → cred=1.0 (satura rápido)
- Bayes: 20 obs → cred=1.0 (satura lento)

**Ejemplo real:**
```
Causal: 5 obs → cred=1.0, conf=0.90
Bayes:  2 obs → cred=0.10, eff=0.30
Score = (0.10×0.30 + 1.0×0.90) / 1.10 = 0.845 ← causal gana
```

Esto es deliberado: la evidencia causal (Pearl) es más robusta que la correlación bayesiana.

### 5.3 Interpretación del Score

| Score | Significado | Efecto en el Governor |
|-------|-------------|----------------------|
| ≥ 0.6 | Objetivo fiable de throttling | Candidato prioritario |
| 0.4–0.6 | Neutral / datos insuficientes | Sin preferencia |
| < 0.4 | Throttling históricamente inefectivo | Skip (Allow) |

### 5.4 Garbage Collection

Cada ~500 ciclos: elimina entradas con `age > max_stale_cycles && obs < min_obs`.
Persistencia: `snapshot()` / `restore_from_map()` para `LearnedState`.

---

## 6. Cross-feed Rules — Transferencia Entre Subsistemas

Las 3 reglas de cross-feed ejecutan al final de cada flush del batch:

### Rule A: OutcomeTracker → SkillRegistry
```
Si effectiveness(process) > 0.7 con ≥3 throttles:
  → skill.success_count += 1
  (Acelera convergencia de skills nuevos con evidencia bayesiana fuerte)
```

### Rule B: CausalGraph → SkillRegistry
```
Si borde sólido (confidence > 0.7, evidence ≥ 5) && skill.success_rate < 0.5:
  → skill.success_count += 1 (artificial)
  (Corrige trials con failures anómalos — el grafo causal "rescata" un skill que debería funcionar)
```

### Rule C: SkillRegistry → OutcomeTracker
```
Si skill.success_rate > 0.8 con ≥20 aplicaciones:
  → Siembra el prior bayesiano en OutcomeTracker
  (Sabiduría persistente sobrevive reinicios del daemon — la experiencia de skills maduros se propaga)
```

---

## 7. RL Threshold Agent — Q-Learning con Dyna-Q

`rl_threshold.rs` (46,827 bytes / 1,213 LOC) — [Sutton & Barto 2018 §6.3, §6.5], [Sutton 1991 Dyna Architecture].

### 7.1 Espacio de Estados y Acciones

```
Estado (discretizado):
  pressure_band: 6 niveles ([0,0.3), [0.3,0.5), [0.5,0.65), [0.65,0.78), [0.78,0.88), [0.88,1.0])
  workload: 4 niveles (Idle, Interactive, Build, HeavyBuild)
  regime: 2 niveles (Stable, Volatile)
  Total: 6 × 4 × 2 = 48 estados

Acciones:
  Lower5pp:    bajar threshold -5pp
  Lower1pp:    bajar threshold -1pp
  Hold:        mantener
  Raise1pp:    subir threshold +1pp
  Raise5pp:    subir threshold +5pp
  Total: 5 acciones

Q-table: 48 × 5 = 240 entradas
```

### 7.2 Dyna-Q (Model-Based RL)

Además del Q-learning estándar, el agente ejecuta `dyna_steps` transiciones simuladas por tick:

```
Real experience: Q(s,a) ← Q(s,a) + α × (r + γ×max_a'Q(s',a') - Q(s,a))
Simulated:       replay de transiciones pasadas almacenadas en model

dyna_steps es modulado por Noradrenaline:
  Baseline (NA=0.5): 10 steps
  Alta urgencia (NA=1.0): 20 steps (planifica más cuando hay estrés)
  Baja urgencia (NA=0.0): 4 steps
```

### 7.3 Decaying Alpha (EMA Q-Learning)

Introducido en v0.6.0 (CLAUDE.md §Plan Nivel 1, punto 2): En vez de un learning rate fijo `α=0.10`, se usa un EMA decaying que converge con la experiencia:

```
α_effective = α_base × dopamine_multiplier
dopamine_multiplier ∈ [0.5, 1.5] (del Neuromodulator)
```

### 7.4 Meta-gate (commit `1e5c852`)

El RL puede proponer `Raise1pp` (relajar umbrales). Pero bajo swap sostenido, un meta-gate **veta** esta acción:

```
fix(rl): meta-gate vetoes Raise1pp under sustained swap growth
```

Esto previene que el agente aprenda a relajar protecciones justo cuando el sistema más las necesita.

### 7.5 Persistencia

`rl_threshold.json` cada 50 ticks con las 240 entradas de la Q-table, epsilon actual, y modelo Dyna-Q.

---

## 8. AIS (Apollo Intelligence Score) — La Métrica Compuesta [0, 100]

`intelligence_score.rs` (91,010 bytes / 2,081 LOC) — El segundo archivo más grande del engine.

### 8.1 Fórmula

```
AIS = Σ wᵢ × Dᵢ(x) × 100
```

### 8.2 Las 6 Dimensiones

| # | Dimensión | Peso | Qué mide | Fuente de datos | Nota clave |
|---|-----------|------|----------|-----------------|------------|
| D1 | Decision Precision | 0.25 | F1 score: preserved=40%, noise=30%, interactive=30% | Adaptive Governor outcomes | La más pesada: si las decisiones son erróneas, nada más importa |
| D2 | Signal Quality | 0.20 | Kalman RMSE (threshold √P* = 0.0884, Riccati steady-state), CUSUM Fβ (β=2, recall 4× más que precision), Hazard calibration | `kalman.rs`, `cusum.rs`, `hazard_model.rs` | CUSUM con β=2 penaliza fuerte no detectar un cambio de régimen |
| D3 | Learning Velocity | 0.20 | RL convergence speed, Q-variance, causal depth, skill maturation rate | `rl_threshold.rs`, `causal_graph.rs`, `optimization_skills.rs` | Post-convergence (`total_ticks ≥ max_ticks`) → speed = 1.0 (stability reward) |
| D4 | Resource Efficiency | 0.15 | P75 cycle < 100ms, skip-rate ~40%, habituation | Daemon cycle timer | `pressure ≥ 0.55` → `budget_score = 1.0` (running all subsystems is correct under load) |
| D5 | Safety Compliance | 0.12 | 0 frozen critical, kills, survival acts, failures, overflows | `safety.rs`, runtime counters | **HARD KILL SWITCH: `frozen_critical > 0` → score = 0.0** |
| D6 | Adaptability | 0.08 | Profile switch accuracy, workload classification, regime detection | `profile_governor.rs`, `workload_classifier.rs` | Menor peso porque la adaptación tiene alta inercia natural |

### 8.3 Grades y Floors

```
S(≥90)  A(≥80)  B(≥70)  C(≥60)  D(≥50)  F(<50)
```

**Regression floor:** Score ≥ 87.0 en runtime benchmark (daemon M1 estable). Si cae por debajo, trigger de recalibration automática vía MetaCognition.

**Pareto balanced:** Todas las dimensiones ≥ 0.30 → ninguna puede mejorar sin degradar otra.

### 8.4 Hardware Normalization (commit `e93c698`)

El AIS aplica normalización por hardware: un M1 8GB operando bajo presión constante merece un score comparable a un M3 Max 64GB con baja presión. Los scores se normalizan por la capacidad del dispositivo.

### 8.5 Wired to Runtime Metrics (commit `97c8ed6`)

```rust
// En runtime_metrics.json:
"intelligence_score": 91.3,
"intelligence_grade": "S",
"intelligence_dimensions": {
  "decision_precision": 0.92,
  "signal_quality": 0.88,
  "learning_velocity": 0.94,
  "resource_efficiency": 0.89,
  "safety_compliance": 1.00,
  "adaptability": 0.86
}
```

---

## 9. LearnedState — Persistencia Unificada

`learned_state.rs` (67,835 bytes / 1,605 LOC)

### 9.1 Motivación

Antes de v0.8.0, cada subsistema persistía independientemente, generando inconsistencias temporales y complejidad de restauración. La migración a `LearnedState` unificado se documentó en commit `b5ea203` ("docs(plan): add v0.9.0 SharedState migration spec + pattern mapping").

### 9.2 Campos Persistidos

```rust
pub struct LearnedState {
    // Signal Intelligence
    pub hazard_model: HazardModelState,
    pub mpc_state: MpcState,
    pub kalman_filters: HashMap<String, KalmanState>,
    pub learned_zones: Vec<ZoneDefinition>,
    pub utility_emas: HashMap<String, f32>,
    
    // Outcome Tracker
    pub bayesian_weights: HashMap<String, BayesianWeight>,
    pub experience_memory: Vec<ExperienceRecord>,  // max 300
    pub co_occurrence_graph: HashMap<(String,String), CoEntry>,
    pub hrpo_groups: Vec<HrpoGroup>,
    
    // Specialist Accuracy
    pub specialist_ema_weights: HashMap<String, f32>,
    
    // NestedLearner
    pub nested_l0_quality: f32,
    pub nested_l1_aggregate: f32,
    pub nested_l2_context: f32,
    pub nested_meta_velocity: f32,
    
    // NARS Beliefs
    pub nars_beliefs: HashMap<String, NarsTruthValue>,
    pub drift_priors: HashMap<String, f32>,
    
    // TeacherConsolidation
    pub gemma_trust: HashMap<String, GemmaTrustEntry>,
    pub pattern_weights: HashMap<String, PatternWeight>,
    
    // Overflow + Frozen state (commit 64d175d)
    pub overflow_history: Vec<OverflowEvent>,
    pub overflow_offset: f32,
    
    // RL Agent
    pub q_table: Vec<f32>,  // 240 entries (48 states × 5 actions)
    pub rl_epsilon: f32,
    pub rl_adjustment: f32,
    
    // Skills
    pub optimization_skills: HashMap<String, SkillState>,
    
    #[serde(default)]  // Backward-compatible additions
    pub _version: u32,
}
```

### 9.3 Self-Improvement Pre-Persist

```rust
pub fn self_improve(&mut self) {
    // 1. Prune stale co-occurrence (age > threshold)
    // 2. Remove noisy weights (low obs + high variance)
    // 3. Cap experience memory at 300
    // 4. Clamp out-of-range values
}
```

### 9.4 Restore Quality Monitor

`RestoreQualityMonitor` rastrea efectividad por 50 ciclos post-restore:
- Si `quality < 0.35` → zonas reseteadas a defaults (el estado guardado era stale).
- Esto previene que un `learned_state.json` de hace 2 semanas (con otro workload) degrade el rendimiento actual.

### 9.5 Adding a New Component

Patrón documentado en CLAUDE.md:
```
1. Agregar campo #[serde(default)] a LearnedState
2. Poblar en collect()
3. Restaurar en apply()
```

---

## 10. SignalIntelligence — Procesamiento de Señales

`signal_intelligence.rs` (78,938 bytes / 2,025 LOC) — El cerebro de procesamiento de señal:

### Subsistemas Internos

| Subsistema | Referencia | Función |
|------------|-----------|---------|
| **Kalman Filter** (`kalman.rs`, 15,779 bytes) | [Welch & Bishop 2006] | Suavizado de ruido en señales de presión. Threshold = √P* = 0.0884 (Riccati steady-state). |
| **CUSUM** (`cusum.rs`, 8,083 bytes) | [Page 1954] | Detección de cambios de régimen (shift detection). Fβ con β=2 (recall 4× más importante que precision). |
| **Hazard Model** (`hazard_model.rs`, 20,959 bytes) | [Cox 1972] | Cox proportional hazards — predice probabilidad de OOM. Velocity gate para OomKill handler (commit `036dfe9`). |
| **Entropy Anomaly** (`entropy_anomaly.rs`, 17,775 bytes) | [Shannon 1948] | Detección de anomalías en la distribución de uso de CPU/RAM. |
| **Holt-Winters** (`holt_winters.rs`, 11,079 bytes) | Forecasting estacional | Predicción de tendencias de uso de memoria con estacionalidad horaria. |

### Router Adaptativo (v0.6.0)

Introducido en commit `a9d7bd7`. Cuando la presión es baja (< 0.40), el router skip los subsistemas pesados para ahorrar CPU:

```
Si pressure < 0.40:
  → Skip Hazard, CUSUM, Holt-Winters
  → Solo Kalman (ligero) + Entropy check
```

---

## 11. Resumen del Flujo de Datos Completo

```
                    ACCIÓN EJECUTADA
                          │
                          v
              LearningObservation
                          │
                    batch_size=8?
                    /           \
                  no            yes → flush()
                  │                    │
               buffer              ┌───┴───────────────────────┐
                                   │                           │
                             OutcomeTracker          CausalGraph
                             (Bayesian per-       (Pearl edges,
                              process weights)     eval_delay=3)
                                   │                    │
                                   │     SkillRegistry  │
                                   │    (group/batch    │
                                   │     induction)     │
                                   │         │          │
                                   └────┬────┘──────────┘
                                        │
                                  Cross-feed A,B,C
                                        │
                                        v
                              EffectivenessTracker
                              (F3 Blend: Thompson)
                                        │
                                        v
                              Blended Score [0,1]
                                   /         \
                           Governor           AIS D1
                           (skip/allow)     (Decision
                                             Precision)
```

---

## 12. Referencias Académicas (Pipeline de Aprendizaje)

| Referencia | Dónde se usa |
|---|---|
| Pearl (2009) "Causality: Models, Reasoning and Inference" | `causal_graph.rs`, `effectiveness_tracker.rs` |
| Thompson (1933) "On the likelihood that one unknown probability exceeds another" | `effectiveness_tracker.rs` |
| Russo et al. (2018) "A Tutorial on Thompson Sampling" arXiv:1707.02038 | `effectiveness_tracker.rs` |
| Auer et al. (2002) "Finite-time Analysis of the Multiarmed Bandit Problem" | `effectiveness_tracker.rs` |
| Sutton & Barto (2018) "Reinforcement Learning" §6.3, §6.5 | `rl_threshold.rs` |
| Sutton (1991) "Dyna Architecture" | `rl_threshold.rs` |
| Shannon (1948) Information Theory | `intelligence_score.rs`, `entropy_anomaly.rs` |
| Bellman (1957) Optimality Principle | `intelligence_score.rs` |
| Welch & Bishop (2006) Kalman filter (Riccati P*) | `intelligence_score.rs`, `kalman.rs` |
| Page (1954) "CUSUM schemes" | `cusum.rs`, `intelligence_score.rs` |
| Cox (1972) "Regression Models and Life Tables" | `hazard_model.rs`, `intelligence_score.rs` |
| Hellerstein (2004) "Feedback Control of Computing Systems" | `intelligence_score.rs` |
| Jain (1991) "Art of Computer Systems Performance Analysis" | `intelligence_score.rs` |
| Jaynes (2003) "Probability Theory" (MaxEnt neutral prior) | `intelligence_score.rs` |
