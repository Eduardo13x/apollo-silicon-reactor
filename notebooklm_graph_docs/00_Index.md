# Knowledge Graph: Apollo System Optimizer
**Versión del análisis:** 2026-04-18 · **Commits analizados:** 716 · **Módulos del Engine:** 126 archivos · **LOC engine:** 77,178 · **LOC total:** ~85,000+

Este conjunto de documentos conforma el **grafo de conocimiento completo** del proyecto Apollo System Optimizer. Está diseñado para ser ingerido por sistemas LLM con Retrieval-Augmented Generation (RAG) como Google NotebookLM. Cada nodo corresponde a una capa arquitectónica, con interconexiones explícitas entre conceptos.

---

## Mapa del Grafo de Conocimiento

```
                        ┌──────────────────────────┐
                        │   00_Index (este archivo) │
                        │   Nodo raíz y navegación  │
                        └────────────┬─────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              v                      v                      v
    ┌─────────────────┐   ┌──────────────────┐   ┌───────────────────┐
    │ 01_System       │   │ 05_Evolution     │   │ 06_Claude         │
    │ Overview        │   │ & Commit History │   │ Sessions & Plans  │
    │ (Macro Arq.)    │   │ (716 commits)    │   │ (Debt & Design)   │
    └────────┬────────┘   └──────────────────┘   └───────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    v                 v
┌────────────────┐  ┌───────────────────┐
│ 02_Core        │  │ 03_Cognitive      │
│ Execution      │  │ Architecture      │
│ Engine (L0)    │  │ (L1-L2, NARS, LLM)│
│ [Reactivo]     │  │ [Predictivo]      │
└───────┬────────┘  └────────┬──────────┘
        │                    │
        └───────┬────────────┘
                v
       ┌─────────────────────┐
       │ 04_Learning         │
       │ Pipeline & Metrics  │
       │ (F3, AIS, UCHS)     │
       └─────────────────────┘
```

---

## Tabla de Nodos con Resumen Detallado

| Nodo | Archivo | Qué Cubre | Módulos Rust Referenciados |
|------|---------|-----------|---------------------------|
| **01** | [01_System_Overview.md](./01_System_Overview.md) | Los 3 binarios, IPC Unix, persistencia atómica, restricciones de seguridad, dependencias Cargo, perfil de compilación | `main.rs`, `apollo-optimizerd.rs`, `apollo-optimizerctl.rs`, `protocol.rs`, `Cargo.toml` |
| **02** | [02_Core_Execution_Engine.md](./02_Core_Execution_Engine.md) | Tick lifecycle (8 pasos), telemetría de sensores, presión efectiva con 9 boosts, clasificador 8-tier, zombie hunter 5-class, Adaptive Governor (21 reglas), Profile Governor (3 perfiles), OverflowGuard, Lotka-Volterra, Safety Layer (13 invariantes), budgets por ciclo | `effective_pressure.rs`, `process_classifier.rs`, `zombie_hunter.rs`, `adaptive_governor.rs`, `profile_governor.rs`, `overflow_guard.rs`, `lotka_volterra.rs`, `safety.rs`, `execute_actions.rs`, `iokit_sensors.rs`, `smc_direct.rs`, `kqueue_pressure.rs`, `silicon_probe.rs` |
| **03** | [03_Cognitive_Architecture.md](./03_Cognitive_Architecture.md) | NestedLearner (L0/L1/L2), TeacherConsolidator (S2→S1), NARS Beliefs, DriftDetector, FreezeIntelligence, MetaCognition (ECE, Humble Mode), SelfRewardingEvaluator, AdversarialProbe (4 escenarios), Neuromodulator (4 señales), Incertidumbre Epistémica, Reptile Meta-Learning, PredictiveAgent (LinUCB), UCHS, StabilityOracle | `nested_learner.rs`, `teacher_consolidation.rs`, `nars_belief.rs`, `freeze_intelligence.rs`, `meta_cognition.rs`, `self_reward.rs`, `adversarial_probe.rs`, `neuromodulator.rs`, `epistemic.rs`, `reptile_meta.rs`, `predictive_agent.rs`, `cognitive_health.rs`, `stability_oracle.rs`, `cognitive_bus.rs` |
| **04** | [04_Learning_Pipeline_and_Metrics.md](./04_Learning_Pipeline_and_Metrics.md) | LearningPipeline mini-batch, OutcomeTracker (Bayesian), CausalGraph (Pearl), SkillRegistry (inducción), EffectivenessTracker (F3 Blend Thompson), Cross-feed rules (A/B/C), AIS (6 dimensiones), LearnedState (persistencia unificada), RL Q-Learning con Dyna-Q | `learning_pipeline.rs`, `outcome_tracker.rs`, `causal_graph.rs`, `optimization_skills.rs`, `effectiveness_tracker.rs`, `intelligence_score.rs`, `learned_state.rs`, `rl_threshold.rs`, `rule_inducer.rs` |
| **05** | [05_Evolution_and_Commit_History.md](./05_Evolution_and_Commit_History.md) | 716 commits cronológicos agrupados en 8 épocas evolutivas: Foundation (mar-01), Predictive (mar-14), AutoResearch (mar-29), Self-Evolving v0.6-v0.8 (mar-28 → abr-03), Stability Week (abr-08), Cognitive Epoch (abr-10), Hardening (abr-14-17), Current (abr-18). Incluye diffs, reverts, y oscilaciones del Chromium freezer | 716 commits, `git log`, evolución completa |
| **06** | [06_Claude_Sessions_and_Plans.md](./06_Claude_Sessions_and_Plans.md) | Planes de `.plan/` (V110_PENDING, WORKSPACE_SPLIT, ARM64_OPTIMIZATIONS), resolución de deuda técnica (DEBT-SENSOR-01/02), calibración empírica (11h, 331 rows), sesiones de análisis de commits, diseño cognitivo, y verificación de arquitectura | `.plan/*.md`, sesiones de Claude (10 conversaciones) |

---

## Interconexiones Causales Clave

Estas relaciones cruzadas entre nodos son críticas para la comprensión integral:

1. **02 → 04**: El Core Engine genera `LearningObservation` que alimenta al Learning Pipeline.
2. **04 → 02**: El EffectivenessTracker (F3 Blend) modifica las decisiones del Adaptive Governor via `low_value` skips.
3. **03 → 04**: El `NestedLearner` coordina el flujo L0→L1→L2, que controla cuándo el Learning Pipeline procesa datos.
4. **03 → 02**: El `Neuromodulator` ajusta en tiempo real los parámetros del RL Agent del OverflowGuard.
5. **05 → 02/03**: La evolución de commits muestra cómo se pasó de hardcode (02) a cognición (03).
6. **06 → 03**: Las sesiones de Claude diseñaron los módulos NARS, TeacherConsolidator, y NestedLearner.

---

## Estadísticas del Repositorio

| Métrica | Valor |
|---------|-------|
| **Commits totales** | 716 |
| **Primer commit** | 2026-03-01 |
| **Último commit** | 2026-04-18 |
| **Módulos en `src/engine/`** | 126 archivos + 1 subdirectorio |
| **LOC en engine** | 77,178 |
| **Top 5 módulos por tamaño** | `chromium_manager.rs` (2,402), `outcome_tracker.rs` (2,393), `intelligence_score.rs` (2,081), `signal_intelligence.rs` (2,025), `decide_actions.rs` (1,883) |
| **Dependencias principales** | `sysinfo`, `serde/serde_json`, `clap`, `chrono`, `anyhow`, `libc`, `ureq`, `tracing` |
| **Perfil de compilación** | `target-cpu=native`, LTO, `codegen-units=1`, `panic=abort` |
| **Plataforma** | macOS Apple Silicon (M1/M2/M3/M4), Rust edition 2021 |
| **Archivos de persistencia** | 10 archivos operativos (JSON/JSONL), write-then-rename atómico |
| **Referencias académicas** | 25+ papers citados en el código |
