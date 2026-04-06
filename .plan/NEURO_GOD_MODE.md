# NEURO GOD MODE — Plan Maestro
## Apollo Neurocognitive v2.0: Super-Agente Auto-Mejorante

**Baseline:** 1287 tests | 7807 LOC cognitive | AIS sim 99.5
**Objetivo:** Sistema que sabe cuando se equivoca, se auto-evalúa, meta-aprende, y es a prueba de balas
**Disciplina:** commit-per-phase, measure after each ITER, no regressions

---

## Diagnóstico: Gaps del sistema actual

| Gap | Impacto | Origen |
|-----|---------|--------|
| Loops de aprendizaje aislados (RL, NARS, LinUCB sin cross-feed) | Conflicto entre agentes, señales contradictorias | Architecture |
| Sin metacognición — no sabe "cuán confiado es en su confianza" | No puede entrar en modo conservador automáticamente | Design |
| Sin evaluación retroactiva propia | No puede generar señal de entrenamiento sin OOM event | Missing |
| DriftDetector reactivo — espera acumulación de drift | Reacciona tarde, no previene | Reactive |
| Sin cuantificación de incertidumbre por decisión | Decide igual con 3 observaciones que con 300 | Missing |
| Sin meta-aprendizaje por tipo de workload | LinUCB/RL comienza desde cero en nuevo fingerprint | Missing |
| Sin adversarial self-testing | Cognitive safety no verificada bajo presión extrema | Missing |
| Sin health score unificado | No hay señal de "sistema cognitivo sano/enfermo" | Missing |

---

## Papers Base (todos open-access o arXiv)

| Código | Paper | Aplicación |
|--------|-------|------------|
| **DR-ZERO** | Meta AI: "Self-Rewarding Language Models" [Yuan et al. 2024] arXiv:2401.10020 | Self-evaluation sin oráculo externo — Apollo evalúa sus propias decisiones pasadas |
| **PPO-SHAPE** | Schulman et al. "Proximal Policy Optimization" [2017] arXiv:1707.06347 | Reward shaping normalizado para el bus unificado |
| **CALIBR** | Guo et al. "On Calibration of Modern Neural Networks" [2017] ICML | Metacognición: temperatura de calibración para confianza de segundo orden |
| **ENSEMBLE** | Lakshminarayanan et al. "Simple Scalable Predictive Uncertainty" [2017] NeurIPS | Epistemic uncertainty via spread de predicciones independientes |
| **REPTILE** | Nichol et al. "On First-Order Meta-Learning" [2018] arXiv:1803.02999 | Meta-learning ligero (no MAML completo) — θ_fast por fingerprint workload |
| **CHANGEPOINT** | Adams & MacKay "Bayesian Online Changepoint Detection" [2007] arXiv:0710.3742 | Early warning antes de drift threshold breach |
| **ADVERSARIAL** | Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" [2018] | Self-testing bajo stress sintético extremo |
| **OEL** | Doncieux et al. "Open-ended Learning: Conceptual Framework" [2018] Front. Neurorobotics | Unified Cognitive Health Score — auto-regulación sistémica |

---

## ITER 1: CognitiveRewardBus — Bus de Recompensas Unificado
**Archivo:** `src/engine/cognitive_bus.rs` (nuevo, ~100 líneas)
**Paper base:** [PPO-SHAPE] + [DR-ZERO §3.2 reward normalization]

### Problema que resuelve
RL recibe -10 solo en OOM. LinUCB aprende de regret local. NARS no recibe reward de RL.
→ Agentes en conflicto, señales contradictorias, convergencia lenta.

### Diseño
```rust
pub struct CognitiveRewardBus {
    signals: VecDeque<RewardSignal>,  // ring buffer 200
    rl_ema: f64,      // EMA normalizada para RlThresholdAgent
    linucb_ema: f64,  // EMA normalizada para PredictiveAgent
    nars_ema: f64,    // EMA normalizada para DriftDetector weights
}

pub struct RewardSignal {
    source: RewardSource,  // OutcomeTracker | RlAgent | CausalGraph | SelfEval
    value: f64,            // raw [-1, +1]
    confidence: f32,       // peso del emisor
    cycle: u64,
}
```

### Mecánica
1. `OutcomeTracker::batch_resolve()` publica `RewardSignal { source: Outcome, value: Δpressure_normalized }`
2. `RlThresholdAgent::step()` publica `RewardSignal { source: RlAgent, value: q_improvement }`
3. `CausalGraph::evaluate_edges()` publica `RewardSignal { source: Causal, value: causal_confidence_delta }`
4. Bus normaliza con PPO-style: `normalized = tanh(raw / std_dev)`
5. Downstream: `RlAgent.inject_bus_reward()`, `PredictiveAgent.inject_bus_reward()`

### Tests objetivo: +12 tests
- Bus normalización, cross-feed RL←Outcome, cross-feed NARS←Causal, ring buffer FIFO

---

## ITER 2: MetaCognition — Capa de Metacognición
**Archivo:** `src/engine/meta_cognition.rs` (nuevo, ~130 líneas)
**Paper base:** [CALIBR] + [ENSEMBLE §2.1 predictive entropy]

### Problema que resuelve
Sistema decide con la misma "confianza" cuando tiene 5 observaciones que cuando tiene 500.
No hay señal de "estoy aprendiendo bien vs. estoy dando vueltas".

### Diseño
```rust
pub struct MetaCognition {
    subsystem_accuracy: HashMap<SubsystemId, AccuracyEma>,
    calibration_error: f32,    // ECE (Expected Calibration Error) proxy
    meta_confidence: f32,      // confianza en la confianza [0,1]
    humble_mode: bool,         // true → más exploración, thresholds más suaves
    humble_cycles_remaining: u32,
}

pub enum SubsystemId { RlAgent, LinUcb, NarsBelief, CausalGraph, SignalKalman }

pub struct AccuracyEma {
    predicted: f32,
    actual_ema: f32,
    calibration_gap: f32,  // |predicted - actual|
}
```

### Mecánica
- Cada subsistema reporta `(predicted_confidence, actual_outcome)` al finalizar
- ECE proxy: `Σ |predicted - actual| / N` por bucket de confianza
- Si `calibration_error > 0.20` → `humble_mode = true` por 50 ciclos
- Humble mode: RL epsilon ×2, LinUCB alpha ×1.5, freeze MIN_CONFIDENCE → 0.45
- [CALIBR]: temperatura óptima T* minimiza `calibration_error`
- `meta_confidence = 1.0 - calibration_error.clamp(0.0, 1.0)`

### Tests objetivo: +15 tests
- ECE cómputo, humble mode trigger/exit, accuracy EMA, per-subsystem tracking

---

## ITER 3: SelfRewardingEvaluator — Evaluación Retroactiva Propia
**Archivo:** `src/engine/self_reward.rs` (nuevo, ~120 líneas)
**Paper base:** [DR-ZERO] — Meta AI Self-Rewarding Language Models §3 "LLM-as-a-Judge"

### Problema que resuelve
Apollo solo aprende cuando hay un OOM event (sparse reward).
**DR Zero key insight:** el modelo puede juzgar sus propias decisiones sin oráculo externo.

### Adaptación a Apollo
DR Zero usa LLM para auto-evaluar. Apollo usa `CausalGraph` como juez interno:
```
JuicyScore(decision_t) = causal_confidence(action_t)
                        × pressure_improvement(t → t+10)
                        / cycles_to_effect(t)
                        × (1 - arousal_penalty)
```

### Diseño
```rust
pub struct SelfRewardingEvaluator {
    decision_log: VecDeque<DecisionRecord>,  // últimas 20 decisiones
    eval_scores: VecDeque<f32>,              // JuicyScore por decisión
    reward_ema: f32,                          // EMA de evaluaciones
    self_eval_accuracy: f32,                  // calibrado vs. external outcomes
}

pub struct DecisionRecord {
    cycle: u64,
    action_taken: String,
    predicted_score: f32,   // confianza al tomar la decisión
    actual_score: Option<f32>,  // evaluado 10 ciclos después
    causal_edge_confidence: f32,
}
```

### Mecánica
1. Cada acción → `log_decision()` con predicted_score = LinUCB confidence
2. 10 ciclos después → `evaluate_past(cycle-10)` usando CausalGraph
3. `actual_score = JuicyScore(...)`
4. Delta = `actual_score - predicted_score` → feed al CognitiveRewardBus
5. Acumular `self_eval_accuracy` = EMA de |delta| (calibración del evaluador mismo)
6. Si `self_eval_accuracy < 0.20` → auto-evaluador confiable → señal tiene más peso

### Tests objetivo: +14 tests
- JuicyScore cómputo, log/evaluate cycle, feed al bus, accuracy tracking

---

## ITER 4: EpistemicUncertainty — Incertidumbre por Decisión
**Archivo:** `src/engine/epistemic.rs` (nuevo, ~90 líneas)
**Paper base:** [ENSEMBLE §3 epistemic vs. aleatoric uncertainty]

### Problema que resuelve
LinUCB exploration bonus solo mide "no conozco bien este brazo".
Falta: cuantificar incertidumbre TOTAL en la decisión (suma de todas las incertidumbres).

### Diseño
```rust
pub struct EpistemicUncertainty {
    rl_q_variance: f64,         // varianza de Q-values en el estado actual
    linucb_exploration: f64,    // √(x'A⁻¹x) del brazo elegido
    nars_confidence_spread: f32, // 1 - min(confidences) entre creencias relevantes
    drift_score: f32,            // DriftDetector.score()
    composite: f32,              // [0,1] incertidumbre total
    high_uncertainty_mode: bool,
}
```

### Fórmula composite (inspirada en [ENSEMBLE] predictive entropy)
```
composite = w1×rl_q_variance + w2×linucb_exploration + w3×nars_confidence_spread + w4×drift_score
          = 0.30×RL + 0.30×LinUCB + 0.25×NARS + 0.15×Drift
```

### Mecánica
- Calculado una vez por ciclo antes de apply_actions
- `composite > 0.70` → `high_uncertainty_mode = true` → bloquear freezes agresivos
- `composite > 0.85` → SOLO observar (Observe arm forzado en LinUCB)
- Exposed en RuntimeMetrics para dashboard

### Tests objetivo: +12 tests
- Composite formula, threshold gates, mode transitions, edge cases (all=0, all=1)

---

## ITER 5: ReptileMeta — Meta-Aprendizaje Ligero por Workload
**Archivo:** `src/engine/reptile_meta.rs` (nuevo, ~120 líneas)
**Paper base:** [REPTILE] — OpenAI "On First-Order Meta-Learning Algorithms"

### Problema que resuelve
Cuando el fingerprint de workload cambia (dev→LLM, browser→build), los agentes comienzan desde cero.
**Reptile insight:** mantenemos θ_slow (global) + θ_fast (por workload) con interpolación lineal.

### Diseño
```rust
pub struct ReptileMeta {
    global_params: MetaParams,               // θ_slow — aprende lento
    workload_params: HashMap<u64, MetaParams>, // θ_fast por fingerprint hash
    current_fingerprint: u64,
    adaptation_steps: u32,
    meta_lr: f64,  // ε en paper Reptile
}

pub struct MetaParams {
    rl_q_bias: [f64; 48],    // bias sobre Q-table actual
    linucb_arm_biases: [f64; 5], // bias sobre arm scores
    nars_confidence_floor: f32,
    last_updated: u64,  // cycle
}
```

### Mecánica (Reptile update rule)
```
θ_slow ← θ_slow + ε × (θ_current - θ_slow)   // outer loop, ε=0.01
θ_fast[fp] ← θ_current  // inner loop — save on workload end
```
- On fingerprint change: `θ_current = θ_slow + 0.5 × (θ_fast[new_fp] - θ_slow)` (interpolate)
- Warmup: si θ_fast[fp] no existe → usar θ_slow directo
- Stale params: si `cycle - last_updated > 10000` → decay θ_fast[fp] hacia θ_slow

### Tests objetivo: +14 tests
- Reptile update rule, interpolación, warmup, stale decay, fingerprint isolation

---

## ITER 6: ProactiveDrift — Early Warning con Bayesian Changepoint
**Archivo:** Extensión de `src/engine/nars_belief.rs` (~80 líneas nuevas)
**Paper base:** [CHANGEPOINT] — Adams & MacKay "Bayesian Online Changepoint Detection"

### Problema que resuelve
`DriftDetector` actual: espera a que `drifted_count >= 2 OR drift_score > 0.08`.
Reacciona DESPUÉS del drift. Necesitamos detectar CUANDO EL DRIFT EMPIEZA.

### Diseño (extensión de DriftDetector)
```rust
// En DriftDetector, nuevos campos:
gradient_ema: f32,       // d(drift_score)/dt — velocidad del drift
gradient_acceleration: f32, // d²(drift_score)/dt² — aceleración
early_warning_score: f32,   // [0,1] — señal proactiva
changepoint_posterior: f32, // P(changepoint en los últimos 5 ciclos)
```

### Mecánica
- Cada `observe_salient()`:
  ```
  gradient = drift_score_new - drift_score_old
  gradient_ema = 0.3×gradient + 0.7×gradient_ema
  changepoint_posterior = bayesian_run_length_update(gradient_ema)
  early_warning_score = 0.6×gradient_ema + 0.4×changepoint_posterior
  ```
- `early_warning_score > 0.05` → emitir `EarlyDriftWarning` al MetaCognition
- MetaCognition puede activar humble_mode ANTES del drift real
- Expuesto en RuntimeMetrics como `nars_drift_early_warning`

### Tests objetivo: +10 tests
- Gradient tracking, changepoint posterior update, early warning threshold, integration con MetaCognition

---

## ITER 7: AdversarialProbe — Self-Testing Adversarial
**Archivo:** `src/engine/adversarial_probe.rs` (nuevo, ~130 líneas)
**Paper base:** [ADVERSARIAL] — Madry et al. + [DR-ZERO §4.2 self-consistency checks]

### Problema que resuelve
No existe verificación de que el sistema cognitivo falla graciosamente bajo condiciones extremas.
Bugs sutiles (como el thaw bug con 0% CPU) solo se detectan en producción.

### Diseño
```rust
pub struct AdversarialProbe {
    probe_interval: u32,   // cada 500 ciclos
    last_probe_cycle: u64,
    pass_rate_ema: f32,     // [0,1] — % de probes que pasan
    failure_log: VecDeque<ProbeFailure>,  // últimas 10 fallas
    cognitive_safety_score: f32,  // contribuye al UCHS
}

pub struct SyntheticScenario {
    pressure: f32,       // forzado al valor extremo
    p_oom: f32,          // P(OOM) forzado a 0.95
    protected_pids: Vec<u32>,  // procesos protegidos que NO deben congelarse
    expected: ProbeExpectation,
}

pub enum ProbeExpectation {
    NoFreezeProtected,         // nunca congelar protegidos
    SafetyFloorRespected,      // RL floor nunca < 0.45
    NarsDriftRecovery,         // tras drift, sistema se recalibra en ≤20 ciclos
    EpistemicBlocksFreeze,     // high uncertainty → no freezes agresivos
}
```

### Mecánica
- Cada 500 ciclos, corre N=4 scenarios sintéticos (sin side effects)
- Cada scenario verifica UN invariante de seguridad cognitiva
- Pass/fail actualiza `pass_rate_ema`
- `pass_rate_ema < 0.75` → COGNITIVE_SAFETY_ALERT en metrics + dashboard
- Fallas se loggean para debugging
- NO afectan estado real — probe corre sobre copias de structs

### Tests objetivo: +16 tests
- Cada ProbeExpectation, pass_rate_ema, alert threshold, no side effects

---

## ITER 8: CognitiveHealthScore — UCHS Unificado
**Archivo:** `src/engine/cognitive_health.rs` (nuevo, ~90 líneas)
**Paper base:** [OEL] + DR-ZERO §5 "emergent self-regulation"

### Problema que resuelve
No hay una señal de "¿está el sistema cognitivo sano?".
AIS mide qué tan bien optimiza el sistema. UCHS mide qué tan bien APRENDE el sistema.

### Diseño (6 dimensiones, igual que AIS)
```rust
pub struct CognitiveHealthScore {
    // D1: Calibración — ¿confía bien cuando debe?
    d1_calibration: f32,    // 1 - MetaCognition.calibration_error
    // D2: Señal de aprendizaje — ¿el reward bus tiene señal limpia?
    d2_reward_quality: f32, // CognitiveRewardBus.signal_to_noise
    // D3: Estabilidad de creencias — ¿cuánto drift hay?
    d3_belief_stability: f32, // 1 - DriftDetector.score()
    // D4: Metacognición — ¿se detectan errores a tiempo?
    d4_self_awareness: f32,   // ProactiveDrift.early_warning_accuracy
    // D5: Adaptabilidad — ¿meta-aprendizaje funcionando?
    d5_adaptability: f32,    // ReptileMeta.adaptation_quality
    // D6: Seguridad cognitiva — ¿constraints se respetan?
    d6_safety: f32,          // AdversarialProbe.pass_rate_ema

    composite: f32,          // weighted average
    recovery_mode: bool,     // composite < 0.40 → disable learning 10 ciclos
}
```

### Fórmula
```
UCHS = 0.20×D1 + 0.20×D2 + 0.15×D3 + 0.20×D4 + 0.10×D5 + 0.15×D6
```

### Mecánica
- `composite < 0.40` → `recovery_mode = true` (10 ciclos) — todos los learning agents pausan
- `composite > 0.80` → "Cognitive S-tier" log
- Expuesto en dashboard: `COGNITIVO: 87.3% [D1:92 D2:88 D3:91 D4:85 D5:78 D6:90]`
- Persiste en `learned_state.json` via LearnedState

### Tests objetivo: +14 tests
- Score computation, recovery mode trigger/exit, dimension isolation, persistence

---

## Secuencia de ejecución con apollo-evolve

```
ITER 1: CognitiveRewardBus  → src/engine/cognitive_bus.rs        (+12 tests)
ITER 2: MetaCognition        → src/engine/meta_cognition.rs       (+15 tests)
ITER 3: SelfRewardingEval    → src/engine/self_reward.rs          (+14 tests)
ITER 4: EpistemicUncert.     → src/engine/epistemic.rs            (+12 tests)
ITER 5: ReptileMeta          → src/engine/reptile_meta.rs         (+14 tests)
ITER 6: ProactiveDrift       → nars_belief.rs extension           (+10 tests)
ITER 7: AdversarialProbe     → src/engine/adversarial_probe.rs    (+16 tests)
ITER 8: CognitiveHealthScore → src/engine/cognitive_health.rs     (+14 tests)
ITER 9: Wiring daemon        → main.rs + metrics + dashboard       (+10 tests wiring)
ITER10: Integration tests    → full cognitive pipeline tests       (+15 tests)
```

**Target total: +122 tests → ~1409 tests**

---

## Medidas de éxito

| Métrica | Before | Target |
|---------|--------|--------|
| Tests lib | 1287 | ≥ 1409 |
| Cognitive files LOC | 7807 | ~9800 |
| New cognitive modules | 0 | 8 |
| "Knows when wrong" signals | 2 (DriftDetector, low_value) | 8+ |
| Self-evaluation cycle | None | 10 cycles |
| Meta-learning | None | Per-workload θ |
| Adversarial pass rate | N/A | ≥ 95% |
| AIS sim | 99.5 | ≥ 99.5 (no regression) |

---

## Invariantes de seguridad NO negociables

1. `RL_ABSOLUTE_FLOOR = 0.45` — nunca se reduce
2. Protected processes nunca se congelan — ni siquiera bajo AdversarialProbe
3. `recovery_mode` pausa LEARNING, no SAFETY — safety constraints siempre activos
4. EpistemicUncertainty > 0.85 → solo Observe arm — sin side effects externos
5. AdversarialProbe NO modifica estado real — corre sobre snapshots

---

*Generado: 2026-04-05 | Base papers: DR-ZERO (Meta), REPTILE (OpenAI), CALIBR, ENSEMBLE, CHANGEPOINT, ADVERSARIAL, OEL*
