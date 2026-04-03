# PR #1 — M1: Dead Code Elimination + API Prelude

**Rama**: `refactor/m1-dead-code-api`
**Base**: `main` (v0.7.0, commit `1257633`)
**Deuda resuelta**: DEBT-001, DEBT-002, DEBT-003, DEBT-011
**Riesgo daemon**: NINGUNO — el daemon no toca optimizer.rs ni reactor.rs
**Riesgo CLI**: MEDIO — los subcomandos del CLI usan OptimizerEngine; necesitan reemplazos thin

---

## Contexto: qué existe hoy

### optimizer.rs (1418 líneas)
`OptimizerEngine` con 13 métodos públicos, todos llamados solo desde `src/main.rs`:

| Método | Llamado desde | Subcomando CLI |
|--------|--------------|----------------|
| `new()` | main.rs:64,69,73,108,115,120,127 | todos |
| `apply_turbo_mode()` | main.rs:65 | `Turbo` |
| `apply_llm_mode()` | main.rs:70 | `Llm` |
| `restore_background_noise()` | main.rs:74 | `Restore` |
| `optimize(&snapshot)` | main.rs:109,158; reactor.rs:145,152,169,180,191 | `Optimize`, `Daemon` |
| `clean_disk()` | main.rs:116,163 | `Clean`, `Daemon` |
| `configure_startup()` | main.rs:121,132 | `Startup`, `Daemon` |
| `boost_self_once()` | main.rs:131 | `Daemon` |
| `cleanup()` | main.rs:141 | `Daemon` |
| `get_tick_rate()` | main.rs:170 | `Daemon` |

### reactor.rs (225 líneas)
`SystemReactor::new(Arc<OptimizerEngine>)` + `start()`. Llamado solo desde `Daemon` subcommand en main.rs:148-149.

### CLI subcomandos afectados
```
Commands::Turbo    → OptimizerEngine::apply_turbo_mode()
Commands::Llm      → OptimizerEngine::apply_llm_mode()
Commands::Restore  → OptimizerEngine::restore_background_noise()
Commands::Optimize → OptimizerEngine::optimize(&snapshot)
Commands::Clean    → OptimizerEngine::clean_disk()
Commands::Startup  → OptimizerEngine::configure_startup()
Commands::Daemon   → OptimizerEngine + SystemReactor (evento loop completo)
Commands::Snapshot → SystemCollector (NO usa optimizer.rs) ← safe
```

---

## Decisión de diseño: qué hacer con los subcomandos CLI

**Opción A (elegida)**: Reemplazar subcomandos que dependen de optimizer.rs con stubs que imprimen un mensaje de redirección hacia `apollo-optimizerctl` + `apollo-optimizerd`. Los comandos `one-shot` con lógica mínima (Turbo, Restore) se reimplementan thin.

**Opción B (descartada)**: Inline toda la lógica de optimizer.rs en main.rs. Correcto técnicamente pero no reduce complejidad global.

**Opción C (descartada)**: Borrar los subcomandos sin reemplazo. Rompe UX sin aviso.

**Justificación de Opción A**: El CLAUDE.md ya documenta que el CLI es para "one-off commands" y el daemon es para uso continuo. Los usuarios ya usan `apollo-optimizerctl` para interactuar con el daemon. Los subcomandos legados se deprecan con mensajes claros.

---

## Commit A — Preparación: thin replacements en main.rs

**Commit message**: `refactor(cli): replace OptimizerEngine calls with thin OS ops or ctl redirects`

### Cambios en `src/main.rs`

**ANTES** (Commands::Turbo):
```rust
Commands::Turbo => {
    let optimizer = OptimizerEngine::new();
    optimizer.apply_turbo_mode();
}
```

**DESPUÉS**:
```rust
Commands::Turbo => {
    eprintln!("apollo-optimizer turbo está deprecado.");
    eprintln!("Usa: apollo-optimizerctl profile set performance");
    std::process::exit(1);
}
```

**ANTES** (Commands::Llm):
```rust
Commands::Llm => {
    let optimizer = OptimizerEngine::new();
    optimizer.apply_llm_mode();
}
```

**DESPUÉS**:
```rust
Commands::Llm => {
    eprintln!("apollo-optimizer llm está deprecado.");
    eprintln!("Usa: apollo-optimizerctl profile set llm-boost");
    std::process::exit(1);
}
```

**ANTES** (Commands::Restore):
```rust
Commands::Restore => {
    let optimizer = OptimizerEngine::new();
    optimizer.restore_background_noise();
}
```

**DESPUÉS**:
```rust
Commands::Restore => {
    eprintln!("apollo-optimizer restore está deprecado.");
    eprintln!("Usa: apollo-optimizerctl restore");
    std::process::exit(1);
}
```

**ANTES** (Commands::Optimize):
```rust
Commands::Optimize => {
    let mut collector = SystemCollector::new();
    let snapshot = collector.collect_snapshot();
    let optimizer = OptimizerEngine::new();
    optimizer.optimize(&snapshot);
}
```

**DESPUÉS**:
```rust
Commands::Optimize => {
    eprintln!("apollo-optimizer optimize está deprecado.");
    eprintln!("El daemon apollo-optimizerd optimiza continuamente.");
    eprintln!("Usa: apollo-optimizerctl status para ver el estado actual.");
    std::process::exit(1);
}
```

**ANTES** (Commands::Clean):
```rust
Commands::Clean => {
    let optimizer = OptimizerEngine::new();
    optimizer.clean_disk();
}
```

**DESPUÉS**:
```rust
Commands::Clean => {
    eprintln!("apollo-optimizer clean está deprecado.");
    eprintln!("Usa: apollo-optimizerctl doctor");
    std::process::exit(1);
}
```

**ANTES** (Commands::Startup):
```rust
Commands::Startup => {
    let optimizer = OptimizerEngine::new();
    optimizer.configure_startup();
}
```

**DESPUÉS**:
```rust
Commands::Startup => {
    eprintln!("apollo-optimizer startup está deprecado.");
    eprintln!("El daemon se instala vía: ./scripts/install-root-daemon.sh");
    std::process::exit(1);
}
```

**ANTES** (Commands::Daemon — el más complejo):
```rust
Commands::Daemon => {
    // ... usa Arc<OptimizerEngine>, SystemReactor, tick loop entero
}
```

**DESPUÉS**:
```rust
Commands::Daemon => {
    eprintln!("Error: usa apollo-optimizerd directamente para el modo daemon.");
    eprintln!("Ejemplo: apollo-optimizerd daemon --profile balanced-root");
    std::process::exit(1);
}
```

### Imports a eliminar de main.rs después de este commit
```rust
// Eliminar estas líneas:
use apollo_optimizer::optimizer::OptimizerEngine;
use apollo_optimizer::reactor;
// (verificar con grep que no queden otras referencias)
```

### Verificación post-commit A
```bash
cargo build --bin apollo-optimizer   # debe compilar sin optimizer/reactor
cargo build --bin apollo-optimizerd  # NO debe haber cambiado
cargo test                            # no debe haber regresiones
grep -n "OptimizerEngine\|SystemReactor" src/main.rs  # debe ser vacío
```

---

## Commit B — Eliminación: delete optimizer.rs + reactor.rs + cleanup lib.rs

**Commit message**: `refactor(dead-code): delete OptimizerEngine (1418L) and SystemReactor (225L)`

### Pre-condiciones (verificar ANTES de borrar)
```bash
# Confirmar cero call sites fuera de los archivos a borrar:
grep -rn "OptimizerEngine\|HeuristicEngine" src/ tests/ \
  | grep -v "src/optimizer.rs" \
  | grep -v "src/reactor.rs"
# Resultado esperado: vacío

grep -rn "SystemReactor" src/ tests/ \
  | grep -v "src/reactor.rs"
# Resultado esperado: vacío

grep -rn "use apollo_optimizer::optimizer\|use apollo_optimizer::reactor" src/ tests/
# Resultado esperado: vacío
```

### Archivos eliminados
```
DELETE: src/optimizer.rs    (1418 líneas)
DELETE: src/reactor.rs      (225 líneas)
```

### Cambios en `src/lib.rs`

**ANTES**:
```rust
pub mod collector;
pub mod dashboard;
pub mod optimizer;
pub mod reactor;
pub mod engine;
```

**DESPUÉS**:
```rust
pub mod collector;
pub mod dashboard;
pub mod engine;
```

### Verificación post-commit B
```bash
cargo build --all-targets  # todos los binarios deben compilar
cargo test                  # todos los tests deben pasar
# Net change: -1643 líneas
```

---

## Commit C — Prelude: API pública consolidada

**Commit message**: `feat(api): add apollo_optimizer::prelude with stable re-exports`

### Cambios en `src/lib.rs`

Agregar al final del archivo:

```rust
/// Convenient re-exports of the types most commonly needed by consumers
/// of this crate (`apollo-optimizerctl`, `apollo-menubar`, integration tests).
///
/// Import with: `use apollo_optimizer::prelude::*;`
pub mod prelude {
    pub use crate::engine::types::{
        BlockerScore, CapabilityReport, DaemonStatus, FrozenEntry,
        LatencyTarget, OptimizationProfile, RuntimeMetrics, SafetyPolicy,
    };
    pub use crate::engine::protocol::{DaemonRequest, DaemonResponse, PROTOCOL_VERSION};
}
```

> **Nota**: `PROTOCOL_VERSION` debe existir en protocol.rs. Verificar con:
> `grep -n "PROTOCOL_VERSION\|pub const" src/engine/protocol.rs`
> Si no existe como constante pública, omitirlo del prelude y agregar un comentario TODO.

### Verificación post-commit C
```bash
cargo build --all-targets
cargo test
cargo doc --no-deps  # verificar que el módulo prelude aparece en la documentación
# Verificar que optimizerctl y menubar siguen compilando:
cargo build --bin apollo-optimizerctl
```

---

## PR #1 — Descripción completa

```markdown
## Summary

Elimina ~1643 líneas de código muerto (OptimizerEngine + SystemReactor) del
binary CLI legacy y consolida la API pública de la crate.

### Cambios por commit

**Commit A** — Thin replacements en CLI
- Los 7 subcomandos que dependían de OptimizerEngine ahora imprimen mensajes de
  deprecación con el equivalente moderno (`apollo-optimizerctl`).
- `Commands::Snapshot` no cambia (no usaba optimizer.rs).
- Elimina todos los `use apollo_optimizer::optimizer` y `use apollo_optimizer::reactor`
  de `src/main.rs`.

**Commit B** — Delete
- `src/optimizer.rs` eliminado (1418 líneas): OptimizerEngine, HeuristicEngine
- `src/reactor.rs` eliminado (225 líneas): SystemReactor
- `src/lib.rs`: quitadas las líneas `pub mod optimizer` y `pub mod reactor`

**Commit C** — Prelude
- `src/lib.rs`: nuevo módulo `pub mod prelude` con 11 re-exports estables

### Qué NO cambia
- `apollo-optimizerd` (daemon): no importa ni optimizer.rs ni reactor.rs
- `apollo-optimizerctl` (client): no afectado
- `apollo-menubar`: no afectado
- Todos los tests existentes: no afectados (ningún test importaba optimizer/reactor)

### Deuda resuelta
- DEBT-001: OptimizerEngine
- DEBT-002: SystemReactor
- DEBT-003: lib.rs módulos muertos
- DEBT-011: sin prelude

## Test plan
- [ ] `cargo build --all-targets` — verde
- [ ] `cargo test` — mismos resultados que baseline
- [ ] `cargo clippy --all-targets` — 0 warnings nuevos de dead_code
- [ ] `grep -rn "OptimizerEngine" src/` — resultado vacío
- [ ] `grep -rn "SystemReactor" src/` — resultado vacío
- [ ] `apollo-optimizer snapshot` — funciona (este subcomando no fue tocado)
- [ ] `apollo-optimizer turbo` — imprime mensaje de deprecación y exit 1
```

---

## Checklist antes de mergear PR #1

- [ ] Commit A: `cargo build --bin apollo-optimizer` verde
- [ ] Commit A: `grep -n "OptimizerEngine" src/main.rs` vacío
- [ ] Commit B: pre-condiciones de grep verificadas y vacías
- [ ] Commit B: `cargo build --all-targets` verde
- [ ] Commit B: `cargo test` — mismo conteo de tests pasando
- [ ] Commit C: `cargo doc` — prelude aparece
- [ ] Commit C: imports en ctl/menubar siguen funcionando
- [ ] No tocar nada en `src/engine/` ni en `src/bin/apollo-optimizerd/`
