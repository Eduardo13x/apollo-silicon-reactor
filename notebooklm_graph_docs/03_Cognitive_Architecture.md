# Navegación
[<- Core Execution Engine](./02_Core_Execution_Engine.md) | [Volver al Índice](./00_Index.md) | [Siguiente: Learning Pipeline ->](./04_Learning_Pipeline_and_Metrics.md)

# 03. Cognitive Architecture — El Cerebro Post-1.0 (L1-L2)

A partir de la v0.6.0 "Self-Evolving" (commit `a9d7bd7`, 2026-03-28), Apollo dejó de ser puramente reactivo y adquirió capacidades metacognitivas: aprender de sus propias decisiones, detectar cuándo su modelo del mundo diverge de la realidad, y modular su comportamiento como un organismo biológico. Esta capa está construida sobre 14+ módulos que suman ~280K bytes de código fuente, con 25+ referencias académicas citadas directamente en el código.

---

## 1. NestedLearner — Coordinación Jerárquica de 3 Niveles

`nested_learner.rs` (18,962 bytes) — Inspirado en [Google Nested Learning 2025] y [Hochreiter & Schmidhuber 1997 LSTM multi-timescale memory].

### Concepto Central
Tres frecuencias de aprendizaje coordinadas bidireccionalmente. La información fluye de alta frecuencia (L0, cada ciclo) a baja (L2, periódico), y el contexto de largo plazo retroalimenta las decisiones instantáneas:

```
           ┌─────────────────────────────────────────────────────────────┐
           │                   NESTED LEARNER                            │
           │                                                             │
           │  L0 (cada ciclo)          L1 (por outcome)      L2 (periódico)
           │  ┌──────────────┐        ┌──────────────┐      ┌──────────────┐
           │  │ SignalIntel. │──gate──▶│ OutcomeTrack │──20──▶│ LearningPipe│
           │  │ Kalman/CUSUM │        │ CausalGraph  │flush │ ReptileMeta │
           │  │ Entropía     │        │              │      │             │
           │  └──────────────┘        └──────────────┘      └──────┬───────┘
           │         ▲                                              │
           │         │              L2→L0 feedback                  │
           │         └──────────────────────────────────────────────┘
           │         meta_velocity alta → eleva L1_GATE_THRESHOLD
           └─────────────────────────────────────────────────────────────┘
```

### Detalle por Nivel

| Nivel | Frecuencia | Subsistemas que alimenta | Métrica | Qué controla |
|-------|------------|--------------------------|---------|--------------|
| **L0** | Cada ciclo | `SignalIntelligence` (Kalman, CUSUM, Entropía Shannon, Hazard model) | `l0_quality` EMA [0,1] | Gate de L1: si la señal L0 es ruidosa, L1 no procesa outcomes. Previene "aprender de basura". |
| **L1** | Por outcome | `OutcomeTracker`, `CausalGraph` | `l1_aggregate` EMA | Ponderado por `l0_quality`. Acumula data limpia para L2. |
| **L2** | Cada 20 flushes de L1 | `LearningPipeline`, `ReptileMeta` | `l2_context` | Meta-learning rate. Alimenta Reptile updates y cross-workload adaptation. |

### Retroalimentación L2→L0 (Google NL 2025 §6.2)
```
l2_meta_velocity = EMA de |Δl2_context| por flush
dynamic_l1_gate = L1_GATE_THRESHOLD + L2_VELOCITY_GATE_SCALE × l2_meta_velocity
Clamped a [0.25, 0.60]
```
**Interpretación:** Si el meta-aprendizaje (L2) oscila mucho, automáticamente exige del L0 señales más limpias antes de permitir actualizaciones. Auto-regulación bio-inspirada.

### Persistencia
Serializado dentro de `learned_state.json`. Los EMAs de L0/L1/L2, la meta_velocity, y el gate threshold sobreviven reinicios del daemon. Implementado en commits `91351ec` y `a1aa6bf`.

---

## 2. TeacherConsolidator — Consolidación S2 → S1 (LLM a Kernel)

`teacher_consolidation.rs` (30,309 bytes) — [McGaugh 2004 "Amygdala modulates consolidation"], [Yerkes-Dodson 1908], [Kahneman 2011 "Thinking, Fast and Slow"], [Rubin 1974 Potential Outcomes].

### Modelo Conceptual
- **Sistema 2 (Pensamiento Lento):** Gemma 4 (LLM local). Costoso (~100ms por llamada HTTP). Observa el sistema, deduce contextos que las reglas hardcoded no captan.
- **Sistema 1 (Reflejo Kernel):** Reglas nativas del engine + pattern_weights + NARS beliefs. Sub-milisegundo.

El TeacherConsolidator traduce sugerencias del LLM (S2) a reflejos nativos (S1) mediante consolidación **afectiva** (usando simulación neurobiológica):

### Flujo de Consolidación

```
  Gemma sugiere → Apollo aplica → OutcomeTracker mide → SuggestionOutcome
       │
       └─→ TeacherConsolidator::consolidate()
            ├─ Compute causal_effect = observed_drop - natural_drift_ema
            │   (Rubin 1974: contrafactual — quitar lo que habría pasado solo)
            │
            ├─ Build Salience { arousal, valence }
            │   arousal = f(pressure, p_oom, swap_gb)
            │   valence = sign(causal_effect)
            │
            ├─ Yerkes-Dodson gate: arousal ∈ [0.20, 0.70] → amplificación óptima
            │   fuera de banda → dampening (no consolidar en crisis extrema ni calma total)
            │
            ├─ Si valence > 0 (DOPAMINE BURST):
            │   ├─ pattern_weights[proc].effective_count += ceil(amplification)
            │   │   amplification = 1.0 + valence × arousal × yerkes × 3.0
            │   │   max = 4.0× (MAX_DOPAMINE_BURST)
            │   ├─ NARS: drift_detector.observe_salient(proc, success=true, salience)
            │   └─ GemmaTrust[categoría] ← EMA(α=0.20) hacia 1.0
            │
            ├─ Si valence < 0 (ACETYLCHOLINE SPIKE):
            │   ├─ pattern_weights[proc].throttle_count += 1 (sin effective_count)
            │   │   → effectiveness baja automáticamente por Laplace prior
            │   ├─ NARS: observe_salient(proc, success=false, salience)
            │   └─ GemmaTrust[categoría] ← EMA hacia 0.0
            │
            └─ Deadband: |causal_effect| < 0.015 → BELOW_DEADBAND, sin update
               (calibrado contra noise floor de ~0.01 en M1 8GB)
```

### GemmaTrust por Categoría
5 categorías de confianza independientes: `Interactive`, `Noise`, `Protected`, `Profile`, `Latency`.
- `is_reliable()`: count ≥ 3 && trust ≥ 0.70.
- Apollo puede ignorar sugerencias de una categoría con bajo trust permanentemente.
- Persistencia: sobrevive reinicios (commit `b940ac9`).

### Performance
`consolidate()` < 100µs/call — Hot-path safe. Zero allocations en el path crítico.

---

## 3. NARS Belief System — Creencias No-Axiomáticas

`nars_belief.rs` (53,991 bytes / 1,407 LOC) — [Wang 2013 NARS §3.3.3], [McGaugh 2004].

### TruthValue
```rust
pub struct TruthValue {
    pub frequency: f32,   // ∈ [0,1] — fracción de éxitos observados (Bayesian: (pos+1)/(pos+neg+2))
    pub confidence: f32,  // ∈ [0,1] — certeza (decay × función del total de evidencia)
}
```

### Salience (Peso Afectivo)
```rust
pub struct Salience {
    pub arousal: f32,     // ∈ [0,1] — qué tan importante es este evento
    pub valence: f32,     // ∈ [-1,1] — positivo (éxito) o negativo (fracaso)
}
```
- **Arousal alta → evidencia pesa 4× más** (via `evidence_weight()`)
- `arousal = f(pressure, p_oom, swap_gb)` — Eventos durante alta presión importan más
- Esto implementa el fenómeno neurobiológico de "flashbulb memories": recuerdos formados bajo estrés se consolidan con más fuerza.

### DriftDetector — Detección de Concept Drift
El sistema operativo del usuario cambia: nuevas apps, actualizaciones de macOS, cambios de workflow. NARS detecta cuando las "verdades" aprendidas dejan de ser válidas:

- Rastrea frecuencia de éxito por proceso. Si cae significativamente vs `prior_frequency` → señal de drift.
- `drift_score = media de |frequency - prior|` ponderada por confianza.
- `needs_recalibration()`: `drift_score > 0.05 && drifted_count ≥ 2`.
- `acknowledge_recalibration()`: resetea priors al estado actual. Los pesos empiezan a re-aprender.

### ArousalState — Estado Emocional Global
EMA de arousal y valence promedio del sistema. Alimenta al `Neuromodulator` y a la dimensión D3 del UCHS.

---

## 4. FreezeIntelligence — NARS Aplicado a Freeze/Thaw

`freeze_intelligence.rs` (17,301 bytes) — [Wang 2013], [Altmann & Trafton 2002 pre-activation before task switch].

Reemplaza la lógica per-app hardcodeada con creencias NARS por **categoría de proceso**:

| Categoría | Ejemplos | Default confidence |
|-----------|----------|-------------------|
| `chromium-renderer` | Brave Helper (Renderer), Slack Helper (Renderer) | 0.70 |
| `chromium-gpu` | Code Helper (GPU), Brave Helper (GPU) | 0.70 |
| `ide-lsp` | sourcekit-lsp, clangd, rust-analyzer | 0.70 |
| `xpc-service` | *.XPCService | 0.70 |
| `media-helper` | Spotify Helper, Music Helper | 0.70 |
| `app-helper` | SomeApp Helper (plain) | 0.70 |
| `generic` | Todo lo demás | 0.70 |

**Comportamiento clave:**
- `observe(process_name, success, salience)` → actualiza la belief de su categoría.
- `should_freeze(name)` → `false` si `confidence < 0.35` (`MIN_FREEZE_CONFIDENCE`).
- **Humble Mode override:** Si MetaCognition está en Humble Mode, el floor sube a 0.45.
- `pre_thaw_hint(predicted_app)` → categorías que deben thaw antes de un switch predicho. Ej: predicción "Brave Browser" → `["chromium-renderer", "chromium-gpu"]`.
- **Aislamiento por diseño:** Failures en una categoría NO afectan a otra. Si `chromium-gpu` tiene 100% de failures, `ide-lsp` mantiene su confianza intacta.

---

## 5. MetaCognition — Calibración de Segundo Orden

`meta_cognition.rs` (16,605 bytes) — [Guo 2017 ECE calibration, ICML], [Lakshminarayanan 2017].

### Qué Mide
Rastrea la brecha entre lo que cada subsistema **predice** y lo que **realmente ocurre**:

**Subsistemas rastreados:** `RlAgent`, `LinUcb`, `NarsBelief`, `CausalGraph`, `SignalKalman`, `FreezeIntelligence`.

```
AccuracyEMA por subsistema:
  predicted_ema = EMA(α=0.05) de confianzas predichas
  actual_ema    = EMA(α=0.05) de resultados reales
  calibration_gap = EMA(α=0.05) de |predicted - actual|

Aggregate ECE:
  calibration_error = Σ(gap_i × √obs_i) / Σ(√obs_i)
  meta_confidence   = 1.0 - calibration_error
```

### Humble Mode
**Activación:** `calibration_error > 0.20` AND `≥ 10 observaciones por subsistema`.
**Duración:** Mínimo 50 ciclos.
**Efectos:**
- `humble_exploration_mult() = 2.0×` — Duplica ε-greedy (RL) y LinUCB α (exploración).
- `humble_freeze_confidence_floor() = 0.45` (vs 0.35 normal) — Más cauteloso con freezes.
- **Sale solo si:** ciclos expirados AND `calibration_error < 0.20`.

**Direccionalidad:**
- `predicted > actual` → Overconfident → Apollo "cree que sabe más de lo que sabe".
- `predicted < actual` → Underconfident → Apollo descarta señales buenas.

---

## 6. SelfRewardingEvaluator — Auto-Recompensa Retroactiva

`self_reward.rs` (14,135 bytes) — [Yuan 2024 DR-ZERO], [Pearl 2009].

### Problema
Entre eventos OOM, no hay señal de aprendizaje (sparse reward). El RL Agent puede pasar horas sin feedback.

### Solución
Apollo juzga sus propias decisiones pasadas usando el CausalGraph como oráculo:

```
1. log_decision(cycle, "throttle:Firefox", predicted=0.80, pressure=0.75)
2. ... esperan EVAL_DELAY_CYCLES=10 ciclos ...
3. evaluate_past():
     causal_conf = CausalGraph.confidence("throttle:Firefox")
     pressure_improvement = (pressure_at_decision - current_pressure).max(0)
     JuicyScore = causal_conf × pressure_improvement / (cycles × 0.1 + 1.0)
     prediction_error = JuicyScore - predicted_score
```

- `reward_ema` = EMA de JuicyScore (calidad general de decisiones)
- `self_eval_accuracy` = EMA de |prediction_error|
- `is_well_calibrated()`: ≥10 evals AND accuracy < 0.20
- `evaluator_trust()` [0,1]: calibración (60%) + volumen (40%)

---

## 7. AdversarialProbe — Stress Testing Sintético

`adversarial_probe.rs` (19,377 bytes) — [Madry 2018 adversarial robustness, ICLR], [Yuan 2024 §4.2].

Cada 500 ciclos, ejecuta 4 escenarios de "peor caso" sobre **copias** del estado cognitivo (zero side effects en producción):

| Escenario | Expectation | Qué prueba |
|-----------|-------------|------------|
| Presión 0.98, P(OOM)=0.95, protegidos=[kernel_task, WindowServer, Claude] | `NoFreezeProtected` | Jamás congelar procesos protegidos, ni bajo presión extrema |
| Presión 0.99, P(OOM)=0.99 | `SafetyFloorRespected` | RL threshold ≥ 0.45 (piso de seguridad) |
| Inject drift: 15 obs negativas con crisis salience | `NarsDriftRecovery` | Debe detectar AND recuperarse en ≤20 ciclos |
| Incertidumbre máxima en todas las dimensiones | `EpistemicBlocksAggressive` | Composite uncertainty > 0.70 → DEBE bloquear acciones agresivas |

- `pass_rate_ema` (EMA α=0.10) alimenta dimensión D6 del UCHS.
- `safety_alert = true` cuando `pass_rate < 0.75`.
- Log de hasta 20 failures (newest first vía `recent_failures()`).

---

## 8. Neuromodulator — 4 Señales Bio-Inspiradas

`neuromodulator.rs` (10,378 bytes) — Adaptado de memoria-core. **Costo: ~50ns/ciclo, 0 allocations, 0 dependencias.**

| Señal | Inputs | Parámetro derivado | Rango |
|-------|--------|--------------------|-------|
| **Dopamine** (recompensa) | pressure_drop, outcome_penalty, !overflow | `alpha_multiplier` (RL learning rate) | [0.5, 1.5] |
| **Noradrenaline** (estrés) | urgency, regime_shift_up, pressure_velocity, thermal_emergency | `dyna_steps` (Dyna-Q planning steps) | [4, 20] |
| **Serotonin** (estabilidad) | low_pressure_streak, !urgency, regime_shift_down, !overflow | `serotonin_shift` (zone threshold shift) | [-0.05, +0.05] |
| **Acetylcholine** (novedad) | process_churn, entropy_anomaly, rl_exploring | `epsilon_bonus` (exploración ε-greedy) | [0.0, 0.05] |

**Leaky integration** con τ≈10 ticks. Baseline (todos en 0.5): parámetros derivados igualan los valores hardcodeados originales. El sistema comienza "neutro" y se modula con la experiencia.

---

## 9. Incertidumbre Epistémica

`epistemic.rs` (9,949 bytes) — [Lakshminarayanan 2017 §3].

```
composite = 0.30 × rl_q_variance          ← Q-values muy dispersos
          + 0.30 × linucb_exploration      ← brazo con pocas observaciones
          + 0.25 × nars_confidence_spread  ← alguna creencia NARS con confianza baja
          + 0.15 × drift_score             ← modelo diverge de realidad
```

| Composite | Modo | Efecto |
|-----------|------|--------|
| < 0.40 | LOW | Operación normal |
| 0.40–0.70 | MODERATE | Sin restricciones |
| 0.70–0.85 | HIGH | Bloquea freezes agresivos, SIGSTOP vetado |
| > 0.85 | OBSERVE-ONLY | Fuerza brazo "Observe" (zero side effects) |

---

## 10. ReptileMeta — Meta-Learning entre Workloads

`reptile_meta.rs` (15,595 bytes) — [Nichol 2018 "On First-Order Meta-Learning"], [Finn 2017 MAML].

Mantiene θ_slow (global) + θ_fast (por workload fingerprint):

```
On workload change (fingerprint A → B):
  1. Save θ_current → workload_params[A]
  2. Reptile update: θ_slow ← θ_slow + ε × (θ_current - θ_slow)   [ε = 0.01]
  3. Si B es conocido: θ_current = θ_slow + 0.5 × (θ_fast[B] - θ_slow)
     Si B es nuevo:    θ_current = θ_slow  (warm start desde experiencia global)
```

**MetaParams** (biases, no copias completas):
- `rl_q_bias[48]`: corrección aditiva por estado de Q-table.
- `linucb_arm_biases[5]`: corrección por brazo LinUCB.
- `nars_confidence_adj`: ajuste al piso de confianza NARS [−0.15, +0.30].

**Cache:** máx 16 workload fingerprints. Eviction por LRU. Stale > 10,000 ciclos → prune.

---

## 11. PredictiveAgent — Contextual Bandit (LinUCB)

`predictive_agent.rs` (66,437 bytes / 1,789 LOC) — [Li 2010 LinUCB, WWW], [Auer 2002].

5 brazos para intervención **proactiva** de memoria:

| Brazo | Acción |
|-------|--------|
| 0: Observe | Solo observar, no actuar |
| 1: TightenThresholds | Bajar bg_pressure -5pp |
| 2: SuggestAggressive | Recomendar perfil AggressiveRoot |
| 3: PreemptiveThrottle | Throttle top-3 waste processes |
| 4: WarnUser | Emitir alerta de presión |

**Contexto (12 dimensiones):**
```
[pressure, pressure_velocity, p_oom, compressor_ratio, swap_gb,
 lv_monopoly_risk, lv_predicted, hour_sin, hour_cos,
 workload_encoded, profile_encoded, effectiveness_top3]
```

---

## 12. Unified Cognitive Health Score (UCHS)

`cognitive_health.rs` (13,186 bytes) — [Doncieux 2018 Open-ended Learning], [Yuan 2024 §5].

Mide **qué tan bien aprende Apollo** (vs AIS que mide qué tan bien *optimiza*):

```
UCHS = Σ wᵢ × Dᵢ     (composite ∈ [0, 1])

D1: Calibration    (0.20) ← MetaCognition.meta_confidence
D2: Reward Quality (0.20) ← CognitiveRewardBus.signal_to_noise (tanh(SNR/3))
D3: Belief Stability (0.15) ← 1 - DriftDetector.drift_score
D4: Self-Awareness  (0.20) ← SelfRewardingEvaluator.evaluator_trust
D5: Adaptability    (0.10) ← ReptileMeta.adaptation_quality
D6: Safety          (0.15) ← AdversarialProbe.pass_rate_ema
```

**Recovery Mode** (`composite < 0.40`): Pausa TODO el aprendizaje por 10 ciclos. Previene que el sistema "aprenda basura" cuando su cognición está degradada.

---

## 13. StabilityOracle

`stability_oracle.rs` (16,721 bytes) — [Kuncheva 2004], [Schulman 2017 PPO].

Agrega 5 señales de estabilidad perceptual:

| Señal | Fuente | Normalización |
|-------|--------|---------------|
| Display jank | DisplayTurbo deactivate | 0/1 evento |
| Zombie rate | heuristic_stats.zombies | count/5, cap 1 |
| Swap spike | Δswap ≥ 512MB/ciclo | 0/1 evento |
| VM thrashing | VmRate.thrashing_score | score/5000, cap 1 |
| CPU stall | ContentionTracker.stall_fraction | [0,1] directo |

`stability_score = 1 - mean(5 EMAs)`. Penalty inyectado al RL vía `NeuroSignals::outcome_penalty`.

**Dampener post-boot:** Los primeros 300 segundos atenúan el penalty linealmente (Spotlight reindexing y launchd warmup no son culpa de Apollo).

---

## 14. Pipeline de Tick Cognitivo (2 Stages)

### Stage 1: learning_tick (`learning_tick.rs`)
```
1. NestedLearner::tick_l0(signal_quality)    → gating de outcomes
2. OutcomeTracker feed + CausalGraph evaluate
3. NestedLearner::tick_l1(effectiveness)     → acumula para L2
4. Si L2 gate period alcanzado → NestedLearner::flush_l2()
5. LearningPipeline flush (fan-out + cross-feed)
6. ReptileMeta::apply_learning_delta()
7. TeacherConsolidator::consolidate() (si hay SuggestionOutcome pendiente)
```

### Stage 2: cognitive_tick
```
1. CognitiveRewardBus::collect_rewards()     → normalización PPO-style
2. MetaCognition::observe() + tick()         → ECE, humble mode
3. SelfRewardingEvaluator::evaluate_past()   → dense reward signal
4. EpistemicUncertainty::update()            → action gating
5. CognitiveHealthScore::update()            → UCHS composite
6. AdversarialProbe (si should_probe)        → safety invariants
7. Neuromodulator::tick()                    → parameter modulation
8. StabilityOracle::record_*()              → instability penalty para RL
```

---

## 15. Referencias Académicas (Capa Cognitiva)

| Referencia | Dónde se usa |
|---|---|
| Google (2025) "Nested Learning" | `nested_learner.rs` |
| Hochreiter & Schmidhuber (1997) LSTM | `nested_learner.rs` |
| McGaugh (2004) "Amygdala modulates consolidation" | `teacher_consolidation.rs`, `nars_belief.rs` |
| Yerkes & Dodson (1908) Inverted-U arousal/learning | `teacher_consolidation.rs` |
| Kahneman (2011) "Thinking, Fast and Slow" (S1/S2) | `teacher_consolidation.rs` |
| Rubin (1974) Potential Outcomes (counterfactual) | `teacher_consolidation.rs` |
| Wang (2013) NARS §3.3.3 | `nars_belief.rs`, `freeze_intelligence.rs` |
| Guo (2017) ECE calibration, ICML | `meta_cognition.rs` |
| Lakshminarayanan (2017) Predictive Uncertainty, NeurIPS | `meta_cognition.rs`, `epistemic.rs` |
| Yuan (2024) "Self-Rewarding LMs" arXiv:2401.10020 | `self_reward.rs`, `adversarial_probe.rs` |
| Madry (2018) Adversarial Robustness, ICLR | `adversarial_probe.rs` |
| Nichol (2018) First-Order Meta-Learning | `reptile_meta.rs` |
| Finn (2017) MAML, ICML | `reptile_meta.rs` |
| Li (2010) LinUCB, WWW | `predictive_agent.rs` |
| Doncieux (2018) Open-ended Learning | `cognitive_health.rs` |
| Schulman (2017) PPO | `cognitive_bus.rs`, `stability_oracle.rs` |
| Altmann & Trafton (2002) Pre-activation | `freeze_intelligence.rs` |
| Kuncheva (2004) Non-stationary signal tracking | `stability_oracle.rs` |
