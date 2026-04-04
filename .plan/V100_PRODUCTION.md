# Plan v1.0.0 "Production Ready" — Iteración 2

**Base**: v0.9.0 (commit `df3e4b6`, tag `v0.9.0`)
**Análisis forense**: 4 agentes × 7 preguntas incómodas → bugs confirmados
**Tests base**: 2179 passed, 10 ignored (25 suites)
**main.rs base**: 5434 líneas
**Fecha**: 2026-04-03

---

## Las 7 Preguntas — Respondidas con Evidencia

### Q1 ✅ CONFIRMADO: predictive_agent tiene bugs reales, no solo riesgos

**Hallazgo 1 — CRÍTICO: `LearnedState::collect()` nunca es llamado**

```rust
// src/engine/pipeline/periodic_stage.rs:102-103
// Full persist (signal_intel, LearnedState, skills) remains inline because
// it requires learning_pipeline and effectiveness_tracker from the binary.
```

El comentario dice "remains inline" pero el código inline **no existe**. Grep en main.rs:
`write_json.*learned_state` → **cero resultados**.

**Consecuencia real**: specialist weights, Bayesian weights del OutcomeTracker, experience memory,
causal graph, counterfactual baseline — todo se resetea a valores iniciales en cada restart del daemon.
El AIS de 98.9 fue medido en una sesión continua. Después de un reboot o actualización: cold start.

**Hallazgo 2 — ALTO: Feedback de especialistas usa proxies equivocados**

```rust
// main.rs:2758-2763
let hazard_predicted_high = prev_pressure_smooth > 0.40;  // PROXY
let hazard_correct = (hazard_predicted_high && pressure_spiked)
    || (!hazard_predicted_high && !pressure_spiked);
```

Pero el especialista Hazard vota basado en `p_oom_30s > 0.30` (probabilidad OOM), no en
`prev_pressure_smooth > 0.40`. Son condiciones diferentes. El tracker mide la consistencia
interna del proxy, no la precisión real del especialista.

**Hallazgo 3 — MEDIO: LinUCB acumula crédito falso en sistemas calmados**

Durante warmup exit, LinUCB vota `Observe`. Si el sistema está calmado (la mayoría del tiempo
en M1 idle), se considera "correcto" por no predecir spike. Su peso sube hacia 1.0 sin haber
hecho nada útil.

**Hallazgo 4 — BAJO: `from_index()` fallback silencioso**

```rust
// src/engine/predictive_agent.rs:62-71
_ => Self::Observe,  // SILENT FALLBACK para índice >= 5
```

Si MPC corruption → mpc_recommendation >= 5 → Observe silencioso.

**Veredicto Q1**: El AIS 98.9 mide rendimiento real en sesión continua en M1 8GB.
Es una métrica legítima PERO no mide persistencia de aprendizaje entre restarts.
El bug de `collect()` es el más importante de todo el proyecto.

---

### Q2 ✅ CONFIRMADO: Dos instancias pueden correr simultáneamente

**La falla**: socket bind failure en el thread secundario **no es fatal para el daemon**:

```rust
// main.rs:748-752
thread::spawn(move || {
    if let Err(e) = socket_handler::run_socket_server(socket_state) {
        tracing::error!(err = ?e, "CRITICAL: socket server failed");
        // NO exit, NO panic, daemon continues
    }
});
```

Daemon B arranca, falla al hacer bind del socket, y **entra al loop principal de optimización**.
Ambos daemons leen y escriben `frozen_state.json` concurrentemente.

**Escenario de freeze permanente**:
1. Daemon A: freeze PID 1234 → escribe frozen_state.json
2. Daemon B: lee disco → tiene PID 1234 en memoria
3. Daemon A: unfreeze PID 1234 → envía SIGCONT, escribe {frozen:[]}
4. Daemon B: escribe su estado → re-persiste PID 1234 como frozen
5. Daemon A cae, launchd reinicia, lee frozen_state.json → ve PID 1234
6. Si PID 1234 fue reciclado (nombre diferente), la guarda de nombres PREVIENE el unfreeze
7. **Proceso real en PID 1234 está congelado permanentemente y nunca se libera**

**Veredicto Q2**: Bug real con escenario de daño concreto. Fix: exit(1) si socket bind falla.

---

### Q3 ✅ SEMI-CONFIRMADO: AIS es test-only, pero tiene bugs de portabilidad

**Clarificación**: AIS **NO es código del daemon**. Es un benchmark en el test suite.
No hay llamadas a `compute_ais()` en código de producción.

**Bug en el benchmark** (relevante para futuro multi-machine):

```rust
// src/engine/intelligence_score.rs:254-255
let rl_speed = if input.rl_max_ticks > 0 {
    1.0 - (input.rl_convergence_ticks as f64 / input.rl_max_ticks as f64).clamp(0.0, 1.0)
}
// Test hardcodea: rl_max_ticks = 500
```

En Mac 16GB donde la presión es menor y el RL converge más lento (>500 ticks),
el clamp lo convierte en 0.0 → `learning_velocity` dimension = 0 → AIS < 70.
**El benchmark fallaría en hardware diferente aunque el daemon funcione perfectamente.**

**6 hardcodes M1-específicos**: cycle_time baseline=80ms, rl_max_ticks=500, system_limit=0.88,
pressure baseline 0.50→0.80, Kalman RMSE threshold=0.03, skip_rate expected=40%.

**Veredicto Q3**: Para uso personal → irrelevante. Para otros usuarios → el AIS benchmark
necesita hardware normalization antes de tener significado. No bloquea v1.0.0 pero
debe documentarse como "benchmark calibrado para M1 8GB".

---

### Q4 ✅ MITIGADO pero con mejor solución disponible

El AWK parser es frágil pero **las escrituras JSON son atómicas** (rename-then-write),
así que el escenario de corrupción por crash es muy improbable.

**Problema real**: el daemon ya sabe cómo revertir sysctls (shutdown path), pero no expone
eso como comando RPC. El uninstall depende de AWK cuando podría llamar `apollo-optimizerctl restore`.

**Veredicto Q4**: Bajo riesgo actual. La solución correcta es agregar `DaemonRequest::RevertSysctls`.

---

### Q5 ✅ PROTOCOLO SEGURO hoy, frágil en upgrades futuros

**Serde behavior confirmado**: `#[serde(tag = "type")]` retorna `Err` en variante desconocida,
no panic ni default silencioso. El error es legible:
```
"invalid request: unknown variant NewFeature, expected one of GetStatus, ..."
```

**Gap**: el cliente NO envía `GetVersion` automáticamente antes de cada comando.
Solo lo hace para el subcomando explícito `version`. Un mismatch falla con el error
de serde, no con "version mismatch" útil.

**Veredicto Q5**: Seguro mientras `PROTOCOL_VERSION = 1` y no añadamos variantes incompatibles.
El fix es preventivo: auto-handshake en ctl antes de cualquier comando.

---

### Q6 ✅ BUG REAL EN CAUSAL GRAPH

```rust
// main.rs:4696-4706
if exec_outcomes.freezes_applied > 0 {
    let frozen_state = state.frozen_state.lock_recover();
    for &pid in frozen_state.keys() {  // ← TODOS los PIDs frozen, no solo los NUEVOS
        lctx.causal_graph.record_action(pid, ...);
    }
}
```

Debería iterar solo los PIDs añadidos este ciclo (`frozen_set - frozen_before`).
Al registrar TODOS los congelados cada vez que hay un freeze, los PIDs que llevan
ciclos congelados acumulan crédito causal extra que no se ganaron.

**Efecto**: co-occurrence scores inflatados para procesos que llevan congelados mucho tiempo.
El sistema aprende correlaciones espurias → skills inducidos incorrectos.

Co-occurrence edge-triggered en 0.60 sin histéresis (Q6.8): menor severidad, GC previene leak.

**Veredicto Q6**: Bug real, no catastrófico pero silencioso y degrada calidad del aprendizaje causal.

---

### Q7 ✅ EFFECTIVE_PRESSURE ES CORRECTO, DOCS ENGAÑOSOS

El docstring dice "max(kernel_pressure, compressor_signal)" pero el código hace suma aditiva:
```rust
let effective = (base + hardware + battery + thermal + ...).clamp(0.0, 1.0);
```

Esto es correcto por diseño. El docstring miente. No es un bug de comportamiento.

**Hallazgo menor**: signal_intelligence.rs recibe la presión sin validar (confía en el caller).
Añadir `debug_assert!((0.0..=1.0).contains(&memory_pressure))` es defensa en profundidad.

**Veredicto Q7**: Código correcto, comentario incorrecto. Fix: actualizar docstring.

---

## Prioridad Revisada: Bugs Confirmados vs Riesgos

| # | Tipo | Severidad | Descripción |
|---|------|-----------|-------------|
| B1 | **BUG REAL** | CRÍTICO | `LearnedState::collect()` nunca llamado → aprendizaje no persiste |
| B2 | **BUG REAL** | ALTO | Socket bind failure no es fatal → dos instancias simultáneas |
| B3 | **BUG REAL** | MEDIO | Causal graph registra todos los frozen PIDs, no solo los nuevos |
| B4 | **BUG REAL** | MEDIO | Specialist accuracy feedback usa proxies equivocados |
| B5 | **BUG REAL** | BAJO | Docstring effective_pressure.rs dice "max" pero hace "sum" |
| R1 | Riesgo | ALTO | AIS benchmark no portable a otro hardware |
| R2 | Riesgo | ALTO | Sin PanicRestore sysctls RPC → uninstall depende de AWK |
| R3 | Riesgo | MEDIO | Cliente ctl no hace version handshake automático |
| R4 | Riesgo | BAJO | LinUCB acumula peso falso en sistemas calmados |
| R5 | Riesgo | BAJO | from_index() fallback silencioso a Observe |

---

## Plan Actualizado: 16 PRs en 5 Tracks

### TRACK A — Bug Fixes CRÍTICOS (ejecutar PRIMERO, en orden)

---

#### PR #1 — Fix: LearnedState persiste en cada ciclo periódico [CRÍTICO]

**El bug**: `periodic_stage.rs` tiene un comentario que dice "persist inline" pero el código
inline no existe en main.rs.

**Fix** (main.rs, bloque periódico `% 100`):

Localizar la sección de persist periódico (~línea 4880-4946) y añadir:

```rust
// Dentro del bloque "every 100 cycles":
{
    let ls = LearnedState::collect(
        &mut lctx.signal_intel,
        &mut lctx.outcome_tracker,
        &mut specialist_accuracy,
    );
    write_json(&learned_state_path, &ls, Some(0o600));
    tracing::debug!(cycles = cycle_count, "learned_state persisted");
}
```

Verificar que `LearnedState::collect()` acepta los tipos correctos y que
`learned_state_path` ya está en scope (está: se inicializa en líneas ~850).

**Tests**: unit test que verifica collect() → apply() roundtrip para todos los campos.
Verificar que después de daemon restart los pesos no vuelven a 0.70.

**Tamaño**: ~15 líneas. Riesgo: MÍNIMO — solo activa código que ya existe pero nunca se llamaba.

---

#### PR #2 — Fix: Socket bind failure es fatal [ALTO]

**El bug**: daemon continúa sin socket si bind falla → dos instancias.

**Fix** (main.rs:748-752 + socket_handler.rs):

```rust
// main.rs: si el socket thread falla al bind, detener el daemon
let socket_state = state.clone();
let socket_handle = thread::spawn(move || {
    socket_handler::run_socket_server(socket_state)
});

// Esperar 200ms para que el bind ocurra
thread::sleep(std::time::Duration::from_millis(200));

// Si el thread terminó (bind falló), abortar
if socket_handle.is_finished() {
    tracing::error!("FATAL: socket server failed to start — another instance may be running");
    std::process::exit(1);
}
```

**Alternativa más robusta**: canal de señalización bind_ok:

```rust
let (tx, rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();
thread::spawn(move || {
    match UnixListener::bind(socket_path) {
        Ok(listener) => { let _ = tx.send(Ok(())); /* continue serving */ }
        Err(e) => { let _ = tx.send(Err(anyhow::anyhow!(e))); }
    }
});
match rx.recv_timeout(Duration::from_secs(2)) {
    Ok(Err(e)) => { tracing::error!("socket bind failed: {e}"); std::process::exit(1); }
    Err(_timeout) => { tracing::warn!("socket bind check timed out"); }
    Ok(Ok(())) => { /* continue */ }
}
```

**Tests**: integration test que verifica que el segundo daemon con socket ocupado sale con code 1.

**Tamaño**: ~30 líneas.

---

#### PR #3 — Fix: Causal graph double-counts frozen PIDs [MEDIO]

**El bug** (main.rs:4696-4706):

```rust
// ANTES (buggy):
if exec_outcomes.freezes_applied > 0 {
    for &pid in frozen_state.keys() {  // TODOS los congelados

// DESPUÉS (correcto):
let newly_frozen: Vec<u32> = exec_outcomes.newly_frozen_pids.clone();
if !newly_frozen.is_empty() {
    for &pid in &newly_frozen {        // Solo los NUEVOS este ciclo
```

Requiere que `ExecutionOutcomes` añada campo `newly_frozen_pids: Vec<u32>`.
Este campo ya se puede poblar en `execute_actions.rs` donde se llama `libc::kill(SIGSTOP)`.

**Tests**: test que verifica que un PID congelado en ciclo N no acumula causal credit en ciclo N+5.

**Tamaño**: ~25 líneas (execute_actions.rs + main.rs).

---

#### PR #4 — Fix: Specialist feedback basada en señales reales [MEDIO]

**El bug** (main.rs:2751-2783): todos los especialistas se evalúan contra `prev_pressure_smooth`
como proxy en lugar de sus señales reales.

**Fix**: capturar las señales de decisión del ciclo anterior y comparar con las del ciclo actual:

```rust
// Añadir struct para guardar decisiones previas:
struct PrevSpecialistSignals {
    hazard_fired: bool,      // p_oom_30s > 0.30 el ciclo anterior
    kalman_fired: bool,      // pressure_predicted_5s > 0.85 el ciclo anterior
    linucb_intervention: Intervention,  // lo que eligió LinUCB
    mpc_intervention: Intervention,     // lo que eligió MPC
}

// Al final de cada ciclo, guardar:
prev_specialist_signals = PrevSpecialistSignals {
    hazard_fired: signal_digest.p_oom_30s > 0.30,
    kalman_fired: signal_digest.pressure_predicted_5s > 0.85,
    linucb_intervention: linucb_choice,
    mpc_intervention: Intervention::from_index(signal_digest.mpc_recommendation),
};

// Al inicio del SIGUIENTE ciclo, evaluar si la predicción fue correcta:
let hazard_correct = (prev.hazard_fired && pressure_spiked)
    || (!prev.hazard_fired && !pressure_spiked);
```

**Tamaño**: ~50 líneas. Riesgo: BAJO — solo cambia cómo se mide la precisión, no la lógica de decisión.

---

#### PR #5 — Fix: Panic sites + docstring effective_pressure [BAJO]

1. `src/engine/foreground.rs:701`: `let app = app.unwrap()` → `let Some(app) = app else { continue }`
2. `src/engine/kqueue_pressure.rs:286,293,300`: `.expect()/.unwrap()` en Mach API
   → `match` con log + state degradado, no panic
3. `src/engine/effective_pressure.rs`: corregir docstring "max()" → "additive sum with clamp"
4. Añadir `debug_assert!((0.0..=1.0).contains(&memory_pressure))` al inicio de `signal_intelligence.rs::tick()`

**Tamaño**: ~40 líneas.

---

### TRACK B — Tests Coverage: 3 Módulos Críticos Sin Tests

*(Paralelo con Track A una vez PR#1-2 mergeados)*

---

#### PR #6 — Tests: effective_pressure.rs (≥15 casos)

Módulo: `src/engine/effective_pressure.rs` (186L)

Suite de tests a añadir:
- Base dominates when boosts are zero
- All boosts at max clamp to 1.0 (ya existe pero verificar)
- Each boost individually contributes correctly
- LLM boost doesn't appear twice when wired to signal_intel
- Zero pressure with all boosts → bounded result
- Additive semantics confirmed (not max semantics as docstring claimed)
- PressureComponents.total_boost() refleja todos los boosts activos

---

#### PR #7 — Tests: learning_pipeline.rs (≥20 casos)

Módulo: `src/engine/learning_pipeline.rs` (707L)

Suite:
- Observation fan-out llega a todos los subsistemas wired
- flush() persiste correctamente (mock write + verify)
- GC elimina observaciones stale
- Buffer no crece unboundedly
- Partial failure en un subsistema no bloquea los demás
- LearningContext wiring: todos los 9 campos presentes y accesibles
- collect() → apply() roundtrip: después de restore, estado = antes de persist

---

#### PR #8 — Tests: predictive_agent.rs (≥25 casos)

Módulo: `src/engine/predictive_agent.rs` (1220L)

Suite crítica (debe cubrir los bugs encontrados):
- **Post-fix B4**: feedback con señales reales produce accuracy correcta
- **Post-fix del collect()**: pesos persisten y se restauran correctamente
- Specialist voting con pesos iguales → sum correcto
- Specialist voting con peso dominante → ese especialista determina la decisión
- LinUCB warmup exit → primera selección no es siempre Observe
- LinUCB arm selection: exploración decrece con confianza
- Accuracy EMA: especialista correcto 100 veces → peso ~0.95
- Accuracy EMA: especialista incorrecto 100 veces → peso ~0.05
- Convergence diversity: si todos los especialistas tienen accuracy similar, la variance en pesos es < X
- from_index out-of-range → Observe (documentado, test para coverage)
- cold start meta-seed reduce ciclos de warmup
- pressure_spike_detected retroalimenta el tracker

---

### TRACK C — Portabilidad & Config

*(Paralelo con Track B)*

---

#### PR #9 — Multi-Machine: CPU + RAM Scaling

**Cambios en `src/engine/types.rs`** — scale action budgets por cores:
```rust
impl SafetyPolicy {
    pub fn for_capabilities(cores: u32, ram_gb: f64) -> Self {
        let core_scale = (cores as f64 / 8.0).clamp(0.5, 2.0);
        let ram_scale  = (ram_gb / 8.0).clamp(0.5, 4.0);
        SafetyPolicy {
            max_freezes_per_cycle: (4.0 * core_scale) as u32,
            max_throttles_per_cycle: (12.0 * core_scale) as u32,
            // ... etc
        }
    }
}
```

**Cambios en `src/engine/decide_actions.rs`** — scale process thresholds por RAM:
```rust
lazy_static! {
    static ref INTERACTIVE_THRESHOLD: u64 = {
        let ram_scale = (query_ram_gb() / 8.0).clamp(0.5, 4.0);
        (100.0 * ram_scale * 1024.0 * 1024.0) as u64
    };
}
```

**En M1 8GB**: `core_scale = 1.0`, `ram_scale = 1.0` → idéntico a hoy.

---

#### PR #10 — Config File + DaemonRequest::RevertSysctls

**`src/engine/apollo_config.rs`** (NUEVO, ~120L):
```toml
[thresholds]
ram_gb_override = 0.0

[protected]
extra_names = []
```

**`src/engine/protocol.rs`** — añadir variante:
```rust
RevertSysctls,  // Revierte todos los sysctls a defaults capturados en startup
```

**`socket_handler.rs`** — handler:
```rust
DaemonRequest::RevertSysctls => {
    sysctl_governor.revert_to_defaults();
    DaemonResponse::Ok
}
```

**`scripts/uninstall-root-daemon.sh`** — reemplazar AWK:
```bash
# Antes del binario desaparecer:
sudo /usr/local/bin/apollo-optimizerctl panic-restore
sudo /usr/local/bin/apollo-optimizerctl revert-sysctls  # NUEVO
# Ya no necesitamos AWK
```

---

### TRACK D — Observabilidad & Instalación

*(Paralelo con Track C)*

---

#### PR #11 — Frozen Process List en Status

**`src/engine/protocol.rs`** — añadir a `DaemonStatus`:
```rust
#[serde(default)]
pub frozen_processes: Vec<FrozenProcessInfo>,
```

```rust
pub struct FrozenProcessInfo {
    pub pid: u32,
    pub name: String,
    pub frozen_seconds: u64,
    pub source: String,       // "MainLoop", "ThermalInterrupt", "DisplayTurbo"
    pub pressure_at_freeze: f64,
}
```

**`socket_handler.rs`** — poblar desde `state.frozen_state`.

**`apollo-optimizerctl.rs`** — mostrar tabla en `status` + mensajes contextuales:
```
Frozen processes (2):
  PID    NAME              AGE        SOURCE        PRESSURE@FREEZE
  12345  mediaanalysisd    2m 14s     MainLoop      0.82
  12891  nsurlsessiond     47s        ThermalIntr.  0.91
```

**Error messages mejorados** (distinguir socket-no-existe vs kill-switch vs starting-up).

---

#### PR #12 — AIS Hardware Normalization + Documentación

**`src/engine/intelligence_score.rs`** — añadir hardware profile a AisInput:
```rust
pub struct AisInput {
    // NEW:
    pub hardware_cores: u32,
    pub hardware_memory_gb: u32,
    // resto igual...
}
```

Ajustar:
- `rl_max_ticks` normalization: `500 * (hardware_memory_gb / 8)` en vez de 500 fijo
- `system_limit`: `0.85 + (hardware_memory_gb as f64 / 16.0) * 0.10`
- Documentar en el benchmark que los thresholds son M1-específicos y cómo calibrar

**Nota**: AIS es test-only. Este PR es para que el benchmark sea honesto en hardware diferente.

---

### TRACK E — Monolito Decomposition

*(Último, una vez todos los tests del código a extraer estén verdes)*

---

#### PR #13 — Extract daemon_init.rs (~350L)

Líneas 770–1124 de main.rs → `daemon_init.rs`.

```rust
pub struct InitializedSubsystems { /* ~40 fields */ }
pub fn initialize_subsystems(state: &SharedState, config: &ApolloConfig) -> Result<InitializedSubsystems>
```

Ahorro: ~350L de main.rs.

---

#### PR #14 — Extract learning_tick.rs (~120L)

Líneas 4724–4814 de main.rs → `learning_tick.rs`.

```rust
pub fn run_learning_tick(ctx: &mut LearningContext, state: &SharedState,
                         actions: &[RootAction], pressure: f64, cycle: u64)
```

Una vez extraído, es directamente testeable con unit tests.
Ahorro: ~120L.

---

#### PR #15 — Extract metrics_reporter.rs (~280L)

Líneas 5037–5280 de main.rs → `metrics_reporter.rs`.

```rust
pub fn apply_io_shaping(state: &SharedState, decisions: &[HeuristicDecision], cycle: u64)
pub fn apply_qos_routing(state: &SharedState, decisions: &[HeuristicDecision], cycle: u64)
pub fn merge_cycle_metrics(state: &SharedState, ctx: &CycleMetricsCtx)
```

Ahorro: ~280L.

---

#### PR #16 — Version Handshake Automático en ctl

**`apollo-optimizerctl.rs`** — al inicio de `send_request()`:
```rust
// Auto-check version on first connect (cached per process lifetime)
static VERSION_CHECKED: AtomicBool = AtomicBool::new(false);
if !VERSION_CHECKED.load(Ordering::Relaxed) {
    if let Ok(Response::VersionInfo { protocol, .. }) = try_send(socket, Request::GetVersion) {
        if protocol != PROTOCOL_VERSION {
            eprintln!("⚠ Protocol mismatch: daemon={protocol}, ctl={PROTOCOL_VERSION}");
            if protocol > PROTOCOL_VERSION { std::process::exit(1); }
        }
        VERSION_CHECKED.store(true, Ordering::Relaxed);
    }
}
```

---

## Secuencia de Ejecución

```
SEMANA 1 — Bug fixes (bloqueantes):
  PR#1 → PR#2 → PR#3 → PR#4 → PR#5

SEMANA 2 — Tests coverage (paralelo):
  PR#6 + PR#7 + PR#8 (paralelo)

SEMANA 3 — Portabilidad + observabilidad (paralelo):
  PR#9 + PR#10 + PR#11 + PR#12 (paralelo)

SEMANA 4 — Monolito + polish:
  PR#13 → PR#14 → PR#15 → PR#16 (secuencial)
```

**Regla de oro**: Cada PR → `cargo test` ≥ test count anterior. Ningún PR reduce tests.

---

## Métricas de Éxito v1.0.0

| Métrica | v0.9.0 | Target v1.0.0 |
|---------|--------|---------------|
| Tests | 2179 | ≥ 2500 |
| Bugs confirmados | 5 | 0 |
| Panic sites producción | 2 | 0 |
| LearnedState persiste en restarts | ❌ | ✅ |
| Specialist feedback correcta | ❌ | ✅ |
| Causal graph no double-counts | ❌ | ✅ |
| Dual-instance bloqueado | ❌ | ✅ |
| Frozen list en status | ❌ | ✅ |
| Portabilidad multi-machine (RAM+CPU) | RAM only | ✅ |
| AIS benchmark portable | ❌ | ✅ |
| main.rs líneas | 5434 | ≤ 4100 |
| Clippy warnings | 1471 | ≤ 500 |
| Config file usuario | ❌ | ✅ |
| Version handshake automático | ❌ | ✅ |
| RevertSysctls RPC | ❌ | ✅ |

---

## Riesgos del Plan

| Riesgo | P | Impacto | Mitigation |
|--------|---|---------|-----------|
| PR#1: `collect()` wiring causa panic en tipos | M | A | Leer learned_state.rs primero, verificar firma |
| PR#4: cambio de feedback descubre que accuracy era inflada | A | M | Esperado — es el punto del fix |
| PR#8: tests en predictive_agent revelan bugs adicionales | A | M | Fix en el mismo PR |
| PR#13: extracción daemon_init introduce regresión | M | A | Copiar-primero, verificar tests, luego eliminar inline |
| PR#9: ram_scale diferente en build con menos RAM | B | B | clamp(0.5, 4.0) acota el daño |

---

## Notas Finales

**El hallazgo más importante**: El sistema de aprendizaje (que es el corazón de apollo-optimizer
v0.6.0+) nunca ha persistido correctamente entre restarts. Cada sesión empieza desde cold start.
Esto no afecta el rendimiento en sesiones largas (el M1 idle aprende rápido), pero significa que
la "inteligencia acumulada" documentada en los planes v0.6.0–v0.9.0 existe solo en memoria RAM.

**PR #1 es el más importante del proyecto desde v0.6.0.**

Una vez que `LearnedState::collect()` se llame correctamente en el ciclo periódico, el daemon
comenzará a acumular aprendizaje real que sobrevive restarts. El AIS después de 7 días de uptime
con restarts será significativamente más alto que el AIS medido el 2026-03-29.
