# PR #2 — M2a: Wire LearningContext en el Hot Loop

**Rama**: `refactor/m2-learning-context`
**Base**: `main` post-PR #1
**Deuda resuelta**: DEBT-007 (bloqueador de DEBT-005 y DEBT-006)
**Riesgo daemon**: BAJO — es un refactor de agrupación, sin cambiar lógica
**Archivos tocados**: SOLO `src/bin/apollo-optimizerd/main.rs`

---

## Contexto

### ¿Qué es LearningContext?

`src/engine/pipeline/learning_context.rs` (219 líneas) define:

```rust
pub struct LearningContext<'a> {
    pub outcome_tracker:     &'a mut OutcomeTracker,
    pub signal_intel:        &'a mut SignalIntelligence,
    pub predictive_agent:    &'a mut PredictiveAgent,
    pub specialist_accuracy: &'a mut SpecialistAccuracyTracker,
    pub overflow_guard:      &'a mut OverflowGuard,
    pub causal_graph:        &'a mut CausalGraph,
    pub skill_registry:      &'a mut SkillRegistry,
    pub neuromod:            &'a mut ApolloNeuromodulator,
    pub energy_tracker:      &'a mut EnergyTracker,
}
```

Tiene 2 tests propios que ya pasan. Está 100% listo para usar.

### Estado actual del daemon

En `run_daemon()`, estas 9 variables se pasan individualmente a cada función de decisión y observación:

```rust
// Variables existentes en run_daemon (declaradas antes del loop):
let mut outcome_tracker    = OutcomeTracker::new();
let mut signal_intel       = SignalIntelligence::new();
let mut predictive_agent   = PredictiveAgent::load_or_default(...);
let mut specialist_accuracy = SpecialistAccuracyTracker::new();
let mut overflow_guard     = OverflowGuard::load_or_default(...);
let mut causal_graph       = CausalGraph::new();
let mut skill_registry     = SkillRegistry::new();
let mut neuromod           = ApolloNeuromodulator::new();
let mut energy_tracker     = EnergyTracker::new();

// Y en el loop, se pasan una a una:
decide_actions(
    &snapshot, &sys, profile, latency_target, ...,
    &mut overflow_guard,    // ← separado
    &signal_intel,          // ← separado
    &outcome_tracker,       // ← separado
    // etc.
)
```

### Objetivo del cambio

Agruparlos en `LearningContext` dentro del loop body, sin mover las declaraciones fuera del loop (eso es un cambio separado en PR #3):

```rust
// ANTES (en cada ciclo):
decide_actions(..., &mut outcome_tracker, &signal_intel, ...)

// DESPUÉS (en cada ciclo):
let mut lctx = LearningContext::new(
    &mut outcome_tracker,
    &mut signal_intel,
    &mut predictive_agent,
    &mut specialist_accuracy,
    &mut overflow_guard,
    &mut causal_graph,
    &mut skill_registry,
    &mut neuromod,
    &mut energy_tracker,
);
// lctx disponible para decide, observe, periodic
// se dropea al final del scope antes de cualquier lock
```

---

## El problema del borrow checker (CRÍTICO — leer antes de implementar)

`LearningContext<'a>` toma **borrows mutables exclusivos** de las 9 variables. Mientras `lctx` viva:
- NO puedes acceder directamente a `outcome_tracker`, `signal_intel`, etc.
- NO puedes adquirir locks que internamente usen estas referencias

**Patrón correcto** — usar un bloque `{}` scoped:

```rust
loop {
    // 1. Colección de snapshot, presión (sin lctx aún)
    let snapshot = collector.collect_snapshot();
    let effective_pressure = compute_pressure(&snapshot, ...);

    // 2. Fase de aprendizaje — lctx vive aquí
    {
        let mut lctx = LearningContext::new(
            &mut outcome_tracker,
            &mut signal_intel,
            &mut predictive_agent,
            &mut specialist_accuracy,
            &mut overflow_guard,
            &mut causal_graph,
            &mut skill_registry,
            &mut neuromod,
            &mut energy_tracker,
        );

        // Usar lctx.signal_intel, lctx.outcome_tracker, etc.
        // en lugar de las variables directas

    } // ← lctx se dropea aquí, liberando todos los &mut

    // 3. Locks de estado compartido (DESPUÉS del drop de lctx)
    let mut metrics = state.metrics.lock_recover();
    // etc.
}
```

**¿Por qué el scope `{}`?** Porque el loop más adelante (después de la fase de aprendizaje) adquiere locks sobre `state`. Si `lctx` viviera hasta el final del ciclo y alguna de las 9 variables fuera también prestada dentro de un lock guard (cosa que no pasa hoy, pero podría pasar), el borrow checker lo rechazaría. El scope garantiza que los `&mut` se liberan limpiamente.

---

## Commit A — Wire LearningContext

**Commit message**: `refactor(daemon): wire LearningContext<'a> into optimization cycle`

### Pasos exactos

**Paso 1**: Agregar el import al top de main.rs (en la sección de imports de pipeline):
```rust
use apollo_optimizer::engine::pipeline::learning_context::LearningContext;
```

**Paso 2**: Localizar el inicio del hot loop en main.rs. Buscar el patrón:
```bash
grep -n "cycle_count\s*+=" src/bin/apollo-optimizerd/main.rs | head -5
```
Esto da la línea donde incrementa `cycle_count`. El bloque de aprendizaje es las ~200 líneas ANTES de ese incremento.

**Paso 3**: Identificar exactamente dónde empiezan a usarse las 9 variables en el loop body:
```bash
grep -n "outcome_tracker\.\|signal_intel\.\|predictive_agent\.\|causal_graph\.\|skill_registry\.\|overflow_guard\.\|specialist_accuracy\.\|neuromod\.\|energy_tracker\." \
  src/bin/apollo-optimizerd/main.rs | head -40
```

**Paso 4**: Envolver el bloque de usos con `{` antes del primer uso y `}` antes del primer `state.*.lock_recover()` posterior.

**Paso 5**: Dentro del bloque, reemplazar accesos directos con `lctx.field`:
- `&mut outcome_tracker` → `&mut lctx.outcome_tracker` (o simplemente `lctx.outcome_tracker` como ya es `&mut T`)
- `&signal_intel` → `lctx.signal_intel`
- `&mut predictive_agent` → `lctx.predictive_agent`
- etc.

### Lo que NO cambia
- Las 9 variables siguen declaradas antes del loop como `let mut X = ...`
- Su inicialización / carga de estado no cambia
- La lógica de decisión y observación no cambia
- Los resultados de cada función no cambian

### Cambio mecánico de firma

Las funciones que recibían `&mut outcome_tracker` separado ahora reciben parte de `lctx`. Si alguna función tiene una firma así:

```rust
fn some_fn(ot: &mut OutcomeTracker, si: &mut SignalIntelligence, ...) { ... }
```

Y se llamaba como:
```rust
some_fn(&mut outcome_tracker, &mut signal_intel, ...)
```

Ahora se llama como:
```rust
some_fn(&mut lctx.outcome_tracker, &mut lctx.signal_intel, ...)
```

> **IMPORTANTE**: NO cambiar las firmas de las funciones en este PR. Solo cambiar los call sites. Las firmas se simplificarán en PR #3 cuando wire DecisionStage.

### Verificación post-commit A

```bash
# 1. Debe compilar sin errores de borrow checker:
cargo build --bin apollo-optimizerd

# 2. Tests no deben regresar:
cargo test

# 3. El daemon debe arrancar y responder:
cargo run --bin apollo-optimizerd -- daemon --profile balanced-root &
sleep 3
cargo run --bin apollo-optimizerctl -- status
kill %1

# 4. Verificar que lctx se construye y dropea en el scope correcto:
# (revisar manualmente que no hay acceso a las 9 variables directas dentro del scope de lctx)
grep -n "outcome_tracker\.\|signal_intel\." src/bin/apollo-optimizerd/main.rs \
  | grep -v "let mut\|LearningContext\|lctx\."
# Resultado esperado: solo las líneas de inicialización pre-loop
```

---

## PR #2 — Descripción completa

```markdown
## Summary

Wire `LearningContext<'a>` en el hot loop del daemon. Los 9 subsistemas de
aprendizaje que se pasaban individualmente como `&mut` ahora se agrupan en
`LearningContext` dentro de un bloque scoped por ciclo.

### Qué cambia
- `src/bin/apollo-optimizerd/main.rs`: ~20 líneas de construcción/drop de lctx,
  ~40 sitios de acceso actualizados de `&mut outcome_tracker` → `lctx.outcome_tracker`
- 0 cambios en `src/engine/` (learning_context.rs ya estaba listo)
- 0 cambios en comportamiento en runtime

### Por qué en su propia PR
Este cambio es el fundamento de PR #3 (DecisionStage) y PR #4 (PeriodicStage).
Si hay un problema con el borrow checker, es mejor detectarlo aquí en un diff
pequeño que en medio de un refactor más grande.

### Deuda resuelta
- DEBT-007: LearningContext definido pero nunca instanciado

## Test plan
- [ ] `cargo build --bin apollo-optimizerd` — verde, 0 borrow checker errors
- [ ] `cargo test` — mismo conteo que baseline
- [ ] Smoke test: daemon arranca, `ctl status` responde OK
- [ ] `cargo clippy --bin apollo-optimizerd` — 0 warnings nuevos
```

---

## Checklist antes de mergear PR #2

- [ ] `cargo build --bin apollo-optimizerd` compila sin errores
- [ ] `cargo build --all-targets` compila (los otros binarios no se tocan)
- [ ] `cargo test` — mismo número de tests pasando
- [ ] Smoke test del daemon ejecutado manualmente
- [ ] No hay accesos directos a las 9 variables DENTRO del scope de lctx
- [ ] `lctx` se dropea ANTES de cualquier `state.*.lock_recover()` en el mismo ciclo
