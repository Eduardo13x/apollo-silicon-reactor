# PR #4 — M2c: Implementar + Wire PeriodicStage

**Rama**: `refactor/m2-periodic-stage`
**Base**: `main` post-PR #3 (DecisionStage wired)
**Deuda resuelta**: DEBT-005, contribuye a DEBT-010
**Riesgo daemon**: MEDIO — mueve lógica de GC/persistencia del inline al stage
**Archivos tocados**: `src/engine/pipeline/periodic_stage.rs`, `src/bin/apollo-optimizerd/main.rs`

---

## Contexto

### periodic_stage.rs — estado actual (stub)

`run_periodic()` en la línea 122 es una función stub. Solo setea flags booleanos sin ejecutar ninguna lógica real:

```rust
pub fn run_periodic(ctx: &PeriodicContext) -> PeriodicResult {
    let mut result = PeriodicResult::default();

    if ctx.cycle_count % 100 == 0 {
        result.did_persist = true;
        result.causal_solid_edges = Some(0);  // ← placeholder, no hace nada
        result.induced_skills = Some(0);      // ← placeholder, no hace nada
    }

    if ctx.cycle_count % 500 == 0 {
        result.did_gc = true;  // ← flag, no hace GC real
    }

    if ctx.cycle_count % 7200 == 0 {
        result.did_hourly = true;  // ← flag, no hace nada real
    }

    result
}
```

### PeriodicContext actual

```rust
pub struct PeriodicContext<'a> {
    pub cycle_count:          u64,
    pub current_pressure:     f64,
    pub workload_mode:        &'a str,
    pub skills_path:          &'a std::path::Path,
    pub hop_groups_path:      &'a std::path::Path,
    pub signal_intel_path:    &'a std::path::Path,
    pub learned_state_path:   &'a std::path::Path,
    pub persist_generations:  u32,
    pub last_restore_quality: Option<f64>,
    pub pending_trial_skill:  Option<(String, f64)>,
    // TODO: learning_ctx: &mut LearningContext  ← bloqueado hasta PR #2
}
```

### Lógica inline en main.rs (lo que necesitamos mover)

En main.rs existen bloques `if cycle_count % N == 0` dispersos. Necesitamos identificarlos con exactitud:

```bash
grep -n "cycle_count % " src/bin/apollo-optimizerd/main.rs
```

Los bloques típicos hacen:
- `% 100`: persistir `SignalIntelligence`, `OutcomeTracker`, `SkillRegistry`, `LearnedState`, `EffectivenessTracker`, `SpecialistAccuracyTracker`
- `% 500`: GC de experience memory, pruning de causal graph edges
- `% 7200`: limpieza horaria — cache warmer, temporal predictor, io_shaper housekeeping

---

## Commit A — Implementar el cuerpo real de run_periodic()

**Commit message**: `feat(pipeline): implement run_periodic() body with real persist/GC logic`

### Paso 1: Auditar los bloques inline en main.rs

Antes de escribir una sola línea en periodic_stage.rs, ejecutar:
```bash
grep -n "% 100\|% 500\|% 7200\|% 300\|% 1000" src/bin/apollo-optimizerd/main.rs
```

Documentar en un comentario al inicio de `run_periodic()` exactamente qué hace cada gate (ya que lo moveremos).

### Paso 2: Agregar `LearningContext` al `PeriodicContext`

La razón por la que el stub no puede hacer persistencia real es que necesita acceso a los subsistemas de aprendizaje. Ahora que PR #2 ya está mergeado, podemos agregar el campo:

**En `periodic_stage.rs`**, actualizar `PeriodicContext`:

```rust
pub struct PeriodicContext<'a> {
    pub cycle_count:          u64,
    pub current_pressure:     f64,
    pub workload_mode:        &'a str,
    pub skills_path:          &'a std::path::Path,
    pub hop_groups_path:      &'a std::path::Path,
    pub signal_intel_path:    &'a std::path::Path,
    pub learned_state_path:   &'a std::path::Path,
    pub persist_generations:  u32,
    pub last_restore_quality: Option<f64>,
    pub pending_trial_skill:  Option<(String, f64)>,
    pub lctx:                 &'a mut LearningContext<'a>, // ← NUEVO (ya no es TODO)
}
```

> **Nota sobre lifetimes**: `LearningContext<'a>` ya usa el lifetime `'a` internamente. La referencia `&'a mut LearningContext<'a>` puede causar un "variance issue" con el borrow checker. Si el compilador se queja, usar un lifetime separado:
> ```rust
> pub struct PeriodicContext<'a, 'lctx> {
>     // ...
>     pub lctx: &'lctx mut LearningContext<'a>,
> }
> ```
> Esto es un detalle de implementación que se resuelve al momento de compilar.

### Paso 3: Actualizar la firma de `run_periodic()`

```rust
pub fn run_periodic(ctx: &mut PeriodicContext<'_, '_>) -> PeriodicResult {
```

(La mutabilidad en `ctx` es necesaria porque `ctx.lctx` es `&mut`)

### Paso 4: Implementar el cuerpo

```rust
pub fn run_periodic(ctx: &mut PeriodicContext) -> PeriodicResult {
    let mut result = PeriodicResult::default();

    // ── Cada 100 ciclos: persistencia de estado aprendido ──────────────────
    if ctx.cycle_count % 100 == 0 {
        // Persistir LearnedState unificado (signal_intel, outcome_tracker,
        // specialist_accuracy, skill_registry, causal_graph, overflow_guard, energy_tracker)
        //
        // LearnedState::persist_improved() toma los componentes del lctx y los
        // serializa a learned_state.json via ctx.learned_state_path.
        //
        // El patrón exact depende de cómo se llama en main.rs actualmente.
        // Buscar con:
        // grep -n "LearnedState\|learned_state\|persist" src/bin/apollo-optimizerd/main.rs | head -20

        // Placeholder hasta que se lea el código exact:
        // LearnedState::persist_improved(
        //     &ctx.lctx.signal_intel,
        //     &ctx.lctx.outcome_tracker,
        //     ctx.learned_state_path,
        // )?;

        result.did_persist = true;

        // Contar edges sólidos del causal graph para el log
        let solid_edges = ctx.lctx.causal_graph.solid_edge_count();
        result.causal_solid_edges = Some(solid_edges);

        // Inducción de skills a partir del causal graph
        let new_skills = ctx.lctx.skill_registry
            .induce_from_causal_graph(&ctx.lctx.causal_graph, ctx.workload_mode);
        result.induced_skills = Some(new_skills);
    }

    // ── Cada 500 ciclos: GC de memoria y causal graph ─────────────────────
    if ctx.cycle_count % 500 == 0 {
        ctx.lctx.outcome_tracker.gc_experience_memory();
        ctx.lctx.causal_graph.prune_stale_edges();
        ctx.lctx.skill_registry.gc_low_quality_skills();
        ctx.lctx.specialist_accuracy.decay_weights();
        result.did_gc = true;
    }

    // ── Cada 7200 ciclos (~2 horas): tareas de largo plazo ────────────────
    if ctx.cycle_count % 7200 == 0 {
        // Estas operaciones no necesitan lctx (son cache flushes de otros módulos)
        // Se pasan via ctx.* paths o se refieren a estructuras no en lctx
        // Revisar main.rs para ver exactamente qué se hace cada 7200 ciclos
        result.did_hourly = true;
    }

    result
}
```

> ⚠️ **IMPORTANTE**: Los nombres exactos de métodos (gc_experience_memory, prune_stale_edges, etc.) deben verificarse contra el código real de OutcomeTracker, CausalGraph, etc. antes de implementar. Los nombres arriba son ilustrativos basados en el audit.

### Verificación post-commit A

```bash
# Solo compilar periodic_stage.rs en contexto:
cargo build --lib
# Si hay errores de nombre de método, leer los archivos correspondientes:
# grep -n "pub fn" src/engine/outcome_tracker.rs | head -20
# grep -n "pub fn" src/engine/causal_graph.rs (si existe)
```

---

## Commit B — Wire run_periodic() en el daemon loop

**Commit message**: `refactor(daemon): replace inline periodic gates with run_periodic() call`

### Paso 1: Agregar imports en main.rs

```rust
use apollo_optimizer::engine::pipeline::periodic_stage::{
    PeriodicContext, PeriodicResult, run_periodic,
};
```

### Paso 2: Localizar todos los bloques inline

```bash
grep -n "cycle_count % " src/bin/apollo-optimizerd/main.rs
```

Anotar las líneas de CADA bloque. Verificar que ninguno tiene código que dependa de variables locales del loop que NO estén en `PeriodicContext` ni `LearningContext`. Si los hay, agregarlos a `PeriodicContext`.

### Paso 3: Construir PeriodicContext dentro del scope de lctx

> **CRÍTICO**: `PeriodicContext` contiene `&mut LearningContext`, así que su construcción DEBE estar DENTRO del bloque `{}` donde vive `lctx`.

```rust
// Dentro del scope de lctx:
{
    let mut lctx = LearningContext::new(/* ... */);

    // ... decide, observe ...

    // Al final del scope, antes del drop:
    let mut periodic_ctx = PeriodicContext {
        cycle_count,
        current_pressure: effective_pressure,
        workload_mode: workload_mode_str,
        skills_path: &skills_path,
        hop_groups_path: &hop_groups_path,
        signal_intel_path: &signal_intel_path,
        learned_state_path: &learned_state_path,
        persist_generations,
        last_restore_quality,
        pending_trial_skill: pending_trial_skill.clone(),
        lctx: &mut lctx,
    };

    let _periodic_result = run_periodic(&mut periodic_ctx);

    // lctx se dropea aquí al salir del bloque
}
```

### Paso 4: Eliminar los bloques inline

Después de verificar que `run_periodic()` cubre exactamente la misma lógica, eliminar cada bloque `if cycle_count % N == 0 { ... }` inline del main loop.

**Verificar inline vs periodic que la cobertura es 100%**:
```bash
# Antes: anotar todos los side effects de los bloques inline
# Después: verificar que run_periodic() produce los mismos side effects
```

### Verificación post-commit B

```bash
# 1. Compilar:
cargo build --bin apollo-optimizerd

# 2. Tests:
cargo test

# 3. Verificar que los gates ya no están inline:
grep -n "cycle_count % " src/bin/apollo-optimizerd/main.rs
# Solo debe quedar la construcción de PeriodicContext + llamada a run_periodic

# 4. Smoke test extendido (dejar correr 200+ ciclos para verificar persistencia):
cargo run --bin apollo-optimizerd -- daemon --profile balanced-root &
sleep 120  # ~100 ciclos
ls -la /tmp/apollo-* 2>/dev/null || ls -la /var/lib/apollo/ 2>/dev/null
# learned_state.json debe tener timestamp reciente
kill %1
```

---

## PR #4 — Descripción completa

```markdown
## Summary

Implementa el cuerpo real de `run_periodic()` en `periodic_stage.rs` y lo conecta
al daemon loop. Los 3 bloques `if cycle_count % N == 0` inline en `run_daemon()`
se reemplazan por una sola llamada a `run_periodic()`.

### Commit A — Implementación
- `src/engine/pipeline/periodic_stage.rs`:
  - Agrega `lctx: &mut LearningContext<'a>` a `PeriodicContext`
  - Implementa `run_periodic()` con lógica real de persist/GC/inducción

### Commit B — Wiring
- `src/bin/apollo-optimizerd/main.rs`:
  - Construye `PeriodicContext` dentro del scope de `lctx`
  - Llama `run_periodic()` una vez al final del scope
  - Elimina los 3 bloques inline `if cycle_count % N`
  - Net: ~-150 líneas del loop body

### Qué NO cambia
- La frecuencia de persist/GC — exactamente los mismos gates (100/500/7200)
- Los datos que se persisten — exactamente los mismos
- El comportamiento del daemon — idéntico

### Deuda resuelta
- DEBT-005: run_periodic() stub
- DEBT-010 (parcial): main.rs reducido ~150 líneas

## Test plan
- [ ] `cargo build --bin apollo-optimizerd` verde
- [ ] `cargo test` — sin regresiones
- [ ] Smoke test: correr 100+ ciclos, verificar que learned_state.json se actualiza
- [ ] `grep "cycle_count %" src/bin/apollo-optimizerd/main.rs` — solo aparece en PeriodicContext
```

---

## Checklist antes de mergear PR #4

- [ ] Verificar nombres exactos de métodos en OutcomeTracker, CausalGraph, SkillRegistry antes de Commit A
- [ ] `PeriodicContext` tiene todos los campos necesarios para cubrir los 3 gates
- [ ] `run_periodic()` DENTRO del scope de `lctx` (borrow checker)
- [ ] Bloques inline eliminados DESPUÉS de verificar cobertura equivalente
- [ ] Smoke test de 100+ ciclos con verificación de persistencia
- [ ] `cargo test` sin regresiones

---

## Reducción de líneas esperada post-M2 completo (PRs #2+3+4)

| Cambio | Delta líneas main.rs |
|--------|---------------------|
| PR #2: LearningContext | ~+20 (construcción) -0 (lógica igual) |
| PR #3: DecisionStage | ~+10 (PolicyContext) -1 (call site) |
| PR #4: PeriodicStage | ~+15 (PeriodicContext) -150 (inline blocks) |
| **Total neto** | **~-100 a -120 líneas** |

> **Nota**: La reducción en líneas de main.rs es modesta (~100-120) porque el objetivo de estos PRs es **estructura**, no reducción de líneas. El verdadero beneficio es testabilidad: `run_periodic()` ahora puede ser testeado en aislamiento (PR #7).
