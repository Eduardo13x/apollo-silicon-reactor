# PR #3 — M2b: Wire DecisionStage en el Hot Path

**Rama**: `refactor/m2-decision-stage`
**Base**: `main` post-PR #2 (LearningContext wired)
**Deuda resuelta**: DEBT-006
**Riesgo daemon**: BAJO-MEDIO — el stage ya existe y wrappea decide_actions() sin cambiar su lógica
**Archivos tocados**: `src/bin/apollo-optimizerd/main.rs` (call site), opcionalmente `src/engine/pipeline/decision_stage.rs` (si falta algo)

---

## Contexto

### DecisionStage actual (ya implementado, no es stub)

`src/engine/pipeline/decision_stage.rs` (327 líneas) define:

```rust
pub struct DecisionStage;

impl DecisionStage {
    pub fn new() -> Self { Self }

    pub fn run<'a>(
        &mut self,
        snapshot: &SystemSnapshot,
        sys: &System,
        profile: OptimizationProfile,
        latency_target: LatencyTarget,
        reactor_weight: f64,
        overflow_thresholds: OverflowThresholds,
        qos_mgr: Option<&mut MachQoSManager>,
        policy: &PolicyContext<'a>,
    ) -> DecisionStageOutput {
        let decision = decide_actions(
            snapshot, sys, profile, latency_target, reactor_weight,
            policy.decide_interactive, policy.decide_noise, overflow_thresholds,
            qos_mgr, policy.decide_weights, policy.outcome_baseline,
            policy.behavior_interactive_pids, policy.ipc_hints,
            policy.hop_groups, policy.habituated_pids, policy.causal_confidence,
        );
        DecisionStageOutput { decision }
    }
}
```

```rust
pub struct PolicyContext<'a> {
    pub decide_interactive:         &'a [String],
    pub decide_noise:               &'a [String],
    pub decide_weights:             &'a HashMap<String, PatternWeight>,
    pub outcome_baseline:           f64,
    pub behavior_interactive_pids:  &'a HashSet<u32>,
    pub ipc_hints:                  &'a HashMap<u32, f64>,
    pub hop_groups:                 &'a HashMap<WorkloadHop, HopGroupWeight>,
    pub habituated_pids:            &'a HashSet<u32>,
    pub causal_confidence:          &'a HashMap<String, f32>,
}
```

```rust
pub struct DecisionStageOutput {
    pub decision: DecisionOutput,
}

impl DecisionStageOutput {
    pub fn into_actions(self) -> Vec<RootAction> {
        self.decision.actions
    }
}
```

### Estado actual en main.rs

El daemon llama `decide_actions()` directamente con ~16 parámetros posicionales. `DecisionStage` y `PolicyContext` no se instancian nunca.

---

## ¿Qué cambia y qué NO cambia?

### Cambia
1. Se instancia `DecisionStage::new()` UNA VEZ antes del loop (costo cero, es una struct sin estado)
2. Cada ciclo se construye `PolicyContext { ... }` con los mismos valores que se pasaban como args posicionales
3. Se llama `decision_stage.run(...)` en lugar de `decide_actions(...)` directamente
4. El resultado es `DecisionStageOutput` que contiene `decision: DecisionOutput`

### NO cambia
- La lógica dentro de `decide_actions()` — exactamente igual
- Los valores que se pasan — exactamente los mismos, solo agrupados en PolicyContext
- El resultado del proceso de decisión — exactamente igual
- El comportamiento del daemon — idéntico

---

## Commit A — Wire DecisionStage

**Commit message**: `refactor(daemon): wire DecisionStage::run() + PolicyContext into hot path`

### Paso 1: Agregar imports en main.rs

```rust
// Agregar junto a los otros imports de pipeline:
use apollo_optimizer::engine::pipeline::decision_stage::{
    DecisionStage, DecisionStageOutput, PolicyContext,
};
```

### Paso 2: Instanciar DecisionStage antes del loop

Buscar la línea donde empieza el `loop {` en `run_daemon`. Justo ANTES:

```rust
let mut decision_stage = DecisionStage::new();

loop {
    // ... todo el resto
```

### Paso 3: Identificar los 16+ argumentos de decide_actions

Buscar la llamada actual:
```bash
grep -n "decide_actions(" src/bin/apollo-optimizerd/main.rs
```

La llamada actual se verá similar a:
```rust
let decision = decide_actions(
    &snapshot,
    &sys,
    profile,
    latency_target,
    reactor_weight,
    &decide_interactive,      // viene de state.policy
    &decide_noise,            // viene de state.policy
    overflow_thresholds,
    Some(&mut mach_qos_guard),
    &decide_weights,          // viene de state.policy
    outcome_baseline,         // viene de lctx.outcome_tracker
    &behavior_interactive_pids,
    &ipc_hints,
    &hop_groups,
    &habituated_pids,
    &causal_confidence,       // viene de lctx.causal_graph
);
```

### Paso 4: Reemplazar con PolicyContext + decision_stage.run()

**ANTES**:
```rust
let decision = decide_actions(
    &snapshot, &sys, profile, latency_target, reactor_weight,
    &decide_interactive, &decide_noise, overflow_thresholds,
    Some(&mut mach_qos_guard), &decide_weights, outcome_baseline,
    &behavior_interactive_pids, &ipc_hints, &hop_groups,
    &habituated_pids, &causal_confidence,
);
```

**DESPUÉS**:
```rust
let policy = PolicyContext {
    decide_interactive:        &decide_interactive,
    decide_noise:              &decide_noise,
    decide_weights:            &decide_weights,
    outcome_baseline,
    behavior_interactive_pids: &behavior_interactive_pids,
    ipc_hints:                 &ipc_hints,
    hop_groups:                &hop_groups,
    habituated_pids:           &habituated_pids,
    causal_confidence:         &causal_confidence,
};

let output = decision_stage.run(
    &snapshot,
    &sys,
    profile,
    latency_target,
    reactor_weight,
    overflow_thresholds,
    Some(&mut mach_qos_guard),
    &policy,
);

let decision = output.decision;
```

> **Nota**: Si `causal_confidence` viene de `lctx.causal_graph`, la construcción de `PolicyContext` debe estar DENTRO del scope de `lctx`. Revisar el origen de cada campo.

### Paso 5: Verificar que el post-procesamiento no cambia

Después de la llamada, el código usa `decision.actions`, `decision.blockers`, `decision.top_skipped`, etc. Estos siguen siendo accesibles via `output.decision.actions` o simplemente `decision.actions` si se hace `let decision = output.decision`.

### Verificación post-commit A

```bash
# 1. Compilación:
cargo build --bin apollo-optimizerd

# 2. Tests completos:
cargo test

# 3. Smoke test:
cargo run --bin apollo-optimizerd -- daemon --profile balanced-root &
sleep 5
cargo run --bin apollo-optimizerctl -- status
# Verificar que el output muestra ciclos procesados > 0
kill %1

# 4. Verificar que decide_actions ya no se llama directamente:
grep -n "decide_actions(" src/bin/apollo-optimizerd/main.rs
# Solo debe aparecer en imports, no en el loop body
```

---

## Qué hacer si DecisionStageOutput no tiene todos los campos necesarios

Puede ser que el main.rs use campos del resultado de decide_actions que `DecisionStageOutput` no expone. Si eso pasa:

1. Leer qué campos exactos usa el código DESPUÉS de la llamada a decide_actions:
```bash
grep -A 50 "decide_actions(" src/bin/apollo-optimizerd/main.rs | head -60
```

2. Si faltan campos en `DecisionOutput` o `DecisionStageOutput`, agregarlos en `decision_stage.rs`:
```rust
pub struct DecisionStageOutput {
    pub decision: DecisionOutput,
    // Agregar campos que falten:
    // pub some_field: SomeType,
}
```

3. NO modificar la lógica, solo exponer lo que ya existe en `DecisionOutput`.

---

## PR #3 — Descripción completa

```markdown
## Summary

Wire `DecisionStage::run()` y `PolicyContext` en el hot path del daemon.
La llamada directa a `decide_actions(...)` con 16 parámetros posicionales
se reemplaza por construcción de `PolicyContext` + `decision_stage.run(...)`.

### Qué cambia
- `src/bin/apollo-optimizerd/main.rs`:
  - +1 línea: `let mut decision_stage = DecisionStage::new();` pre-loop
  - +10 líneas: construcción de `PolicyContext` en cada ciclo
  - -1 línea: reemplazo de `decide_actions(...)` call
  - Net: ~+10 líneas pero código más legible

### Qué NO cambia
- La lógica de `decide_actions()` — idéntica
- Los valores pasados — los mismos, solo agrupados
- El comportamiento del daemon — idéntico

### Deuda resuelta
- DEBT-006: DecisionStage no wired

## Test plan
- [ ] `cargo build --bin apollo-optimizerd` — verde
- [ ] `cargo test` — mismo conteo
- [ ] Smoke test daemon + ctl
- [ ] `grep "decide_actions(" src/bin/apollo-optimizerd/main.rs` — no aparece en loop body
```

---

## Checklist antes de mergear PR #3

- [ ] Compilación verde
- [ ] Tests sin regresiones
- [ ] Smoke test ejecutado
- [ ] `decide_actions` solo aparece en imports en main.rs
- [ ] `policy` se construye con los valores correctos (verificar campo por campo)
- [ ] El post-procesamiento del resultado usa `output.decision` o equivalente
- [ ] `decision_stage` instanciado ANTES del loop (no dentro)
