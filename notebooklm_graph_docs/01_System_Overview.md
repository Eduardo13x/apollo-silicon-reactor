# Navegación
[<- Volver al Índice](./00_Index.md) | [Siguiente: Core Execution Engine ->](./02_Core_Execution_Engine.md)

# 01. System Overview — Macro Arquitectura

Apollo System Optimizer es una plataforma de gestión autónoma de recursos en tiempo real para **macOS Apple Silicon (M1/M2/M3/M4)**, escrita completamente en Rust (edición 2021, versión 1.0.0). Opera como un daemon root continuo que observa ~400-600 procesos concurrentes y redistribuye CPU, memoria, I/O y headroom térmico basándose en la intención del usuario, no en la asignación "justa" del scheduler de XNU.

---

## 1. Los Tres Binarios

### 1.1 `apollo-optimizer` (CLI)
- **Ruta fuente:** `src/main.rs`
- **Función:** Punto de entrada para comandos one-shot.
- **Subcomandos:** `snapshot`, `optimize`, `clean`, `turbo`, `daemon`, `startup`, `llm`, `restore`.
- **Ejemplo:** `apollo-optimizer snapshot --output system_snapshot.json` captura el estado completo del sistema en JSON.
- **Puede arrancar el daemon directamente** con `apollo-optimizer daemon`.

### 1.2 `apollo-optimizerd` (Daemon)
- **Ruta fuente:** `src/bin/apollo-optimizerd.rs` (~5,454 LOC en main.rs del daemon)
- **Función:** Proceso long-running administrado por `launchd` (el sistema de init nativo de macOS).
- **Socket IPC Unix:**
  - Root: `/var/run/apollo-optimizer.sock`
  - Non-root: `/tmp/apollo-optimizer.sock`
- **Protocolo:** JSON con tags `type`/`payload` (definido en `src/engine/protocol.rs`). Tags estables para backward-compatibility.
- **Hot loop:** Ciclo de optimización que se repite cada 2-60 segundos (adaptativo según carga).
- **Módulos extraídos del daemon:**
  - `daemon_init.rs` — Constructor de `DaemonSubsystems`
  - `learning_tick.rs` — Pipeline de aprendizaje por ciclo
  - `metrics_reporter.rs` — Reporte de 50+ contadores
  - `socket_handler.rs` — Manejo de conexiones IPC entrantes (878 LOC)

### 1.3 `apollo-optimizerctl` (Client Controller)
- **Ruta fuente:** `src/bin/apollo-optimizerctl.rs`
- **Función:** CLI para control remoto del daemon en ejecución.
- **Comandos disponibles:**
  - `status` — Estado completo del daemon en JSON
  - `profile set <perfil> [--ttl-minutes N]` — Override de perfil con TTL
  - `set-auto-profile on|off` — Activar/desactivar gobernanza automática
  - `doctor` — Diagnóstico del sistema
  - `capabilities` — Capacidades detectadas
  - `top-blockers` — Procesos que bloquean la interactividad
  - `metrics` — 50+ contadores de runtime
  - `profile-timeline` — Historial de cambios de perfil
  - `usage top --limit N` y `usage explain <proceso>` — Análisis de uso
  - `llm set-key|status|test|disable` — Control del LLM teacher
  - `dump-policy` — Volcado de la política aprendida
  - `feedback good|bad --note "..."` — Retroalimentación del usuario
  - `restore` y `panic-restore` — Restauración de emergencia
  - `optimize-spotlight` — Control manual de indexación Spotlight
- **Auto version handshake:** El cliente verifica compatibilidad de protocolo con el daemon al conectarse.

---

## 2. Comunicación IPC (Unix Socket Protocol)

El daemon y el cliente se comunican vía Unix sockets con un protocolo JSON line-delimited:

```
Client → Daemon:  {"type":"status","payload":{}}
Daemon → Client:  {"type":"daemon-status","payload":{"profile":"balanced-root",...}}
```

- **Serialización:** `serde_json` con `#[serde(rename_all = "kebab-case")]` para todos los tipos del protocolo.
- **Tags estables:** Los strings de tipo (`"status"`, `"set-profile"`, `"daemon-status"`) son parte del contrato de API y no deben cambiar sin bumping de versión.
- **Definido en:** `src/engine/protocol.rs` (10,296 bytes).
- **Sanitización:** `protocol::sanitize()` limpia inputs antes de deserialización.
- **Privilegio:** `is_privileged()` verifica si el caller tiene permisos root para comandos de control.

---

## 3. Persistencia y Estado Operativo

Dado que Apollo opera con privilegios root y envía señales a procesos del sistema, la persistencia debe ser **crash-safe**. Todos los archivos se escriben con la técnica **Write-then-Rename** (escritura atómica):

```rust
// Patrón: write_then_rename (nunca corrupción parcial)
1. write(path.tmp, data)
2. fsync(path.tmp)
3. rename(path.tmp, path)  // atómico en POSIX
```

### 3.1 Archivos de Estado del Daemon

| Archivo | Formato | Frecuencia de escritura | Contenido detallado |
|---------|---------|------------------------|---------------------|
| `journal.jsonl` | JSONL append-only | Cada acción ejecutada | Acción, PID, nombre del proceso, estado before/after, timestamp |
| `runtime_metrics.json` | JSON completo | Cada tick (~2-60s) | 50+ contadores: `boosts_applied`, `throttles_applied`, `freezes_applied`, `paging_hints_applied`, `zombies_detected`, `kills_applied`, `survival_mode_activations`, `profile_switches`, `cycle_durations` (ring buffer), `swap_total_bytes`, `intelligence_score` |
| `governor_state.json` | JSON completo | En transición de perfil | Perfil activo, cooldown restante, override manual, anti-thrash state |
| `profile_timeline.jsonl` | JSONL append | En transición | `{from, to, reason, score, timestamp}` |
| `frozen_state.json` | JSON completo | Cambio en set congelados | Set de PIDs congelados + `start_sec`/`start_usec` para PID identity check |
| `wake_state.json` | JSON completo | Eventos sleep/wake | Estado de gracia post-wake, timestamps de sleep/wake |
| `learned_state.json` | JSON completo | Periódico (~cada 500 ciclos) | **Persistencia unificada** de todo el estado aprendido: Kalman filters, CUSUM, Hazard model, OutcomeTracker weights, co-occurrence graph, specialist accuracy, NestedLearner (L0/L1/L2 EMAs), GemmaTrust scores, NARS beliefs, TeacherConsolidator state |
| `overflow_history.json` | JSON completo | En evento overflow | Últimos 20 eventos de overflow + offset aprendido |
| `optimization_skills.json` | JSON completo | Flush del pipeline | Mapa de skills con success_rate, apply_count, min_pressure |
| `rl_threshold.json` | JSON completo | Cada 50 ticks | Q-table (48 entradas), current_adjustment, epsilon, Dyna-Q model |

### 3.2 Ubicaciones

- **Root daemon:** `/var/lib/apollo/` (estado), `/var/run/` (socket + kill switch)
- **Non-root:** `/tmp/` (equivalentes)
- **Config:** `/etc/apollo-optimizer/config.toml` (creado por `install-root-daemon.sh`)
- **Kill switch:** `/var/run/apollo.disable` (presencia → daemon pausa toda optimización)

### 3.3 Recovery de Crash

Si el daemon colisiona mientras tiene procesos congelados (SIGSTOP):
1. Al reiniciar, lee `frozen_state.json`
2. **Verifica identidad de cada PID** con `start_sec`/`start_usec` (evita confundir PIDs reciclados)
3. Envía `SIGCONT` a cada PID válido
4. Limpia `frozen_state.json`

Este comportamiento es un invariante de seguridad crítico: **Apollo nunca puede dejar procesos congelados tras un crash**.

---

## 4. Dependencias (Cargo.toml)

| Crate | Versión | Propósito | Notas |
|-------|---------|-----------|-------|
| `sysinfo` | 0.30 | CPU, memoria, procesos | `default-features = false` (sin rayon, threading innecesario para ~400 procesos) |
| `serde` + `serde_json` | 1.0 | Serialización de estado y protocolo IPC | `features = ["derive"]` |
| `clap` | 4.4 | Parsing de argumentos CLI | `features = ["derive"]` |
| `chrono` | 0.4 | Timestamps con serialización | `features = ["serde"]` |
| `anyhow` | 1.0 | Manejo de errores con contexto | Usado en todos los binarios como `fn main() -> anyhow::Result<()>` |
| `libc` | 0.2 | System calls directos | kqueue, SIGSTOP/SIGCONT, mach APIs, proc_pidinfo |
| `ctrlc` | 3.4 | Manejo graceful de SIGINT | `features = ["termination"]` |
| `toml` | 0.8 | Parsing de config.toml | |
| `ureq` | 2.12 | HTTP client para LLM integration | `features = ["json"]`, solo llamadas a Gemma/GPT |
| `tracing` + `tracing-subscriber` | 0.1/0.3 | Logging estructurado | `features = ["json", "env-filter"]` |
| `tray-icon` | 0.14 | Icono de barra de menú macOS | Solo `cfg(target_os = "macos")` |
| `winit` | 0.29 | Event loop de ventana | Solo para tray icon |
| `cocoa` + `objc` | 0.25/0.2 | Bindings Objective-C nativos | Solo para tray icon |
| `cc` | 1.0 (build-dep) | Compilación de código C/FFI | Para `build.rs` (puentes a IOKit, SMC, Mach) |
| `tempfile` | 3 (dev-dep) | Tests: archivos temporales | |

---

## 5. Perfiles de Compilación

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=native"]   # SIMD específico del chip M1/M2/M3

[profile.release]
lto = true              # Link-Time Optimization cross-crate
codegen-units = 1        # Máxima optimización (sacrifica tiempo de compilación)
panic = "abort"          # Sin unwinding en daemon de producción

[profile.test]
opt-level = 2            # Tests compilados con optimización (e2e benchmarks)

[profile.menubar]
inherits = "release"
lto = false              # Compilación rápida para el menubar icon
codegen-units = 16
```

---

## 6. Estructura del Código Fuente

```
src/
├── main.rs                     # CLI (apollo-optimizer)
├── lib.rs                      # Librería: re-exports de engine
├── collector.rs                # Recolección de métricas vía sysinfo
├── reactor.rs                  # Event loop kqueue del daemon
├── sysctl_tuner.rs             # [DEPRECATED - código muerto eliminado]
├── bin/
│   ├── apollo-optimizerd/
│   │   ├── main.rs             # Daemon principal (~5,454 LOC)
│   │   ├── daemon_init.rs      # Constructor DaemonSubsystems
│   │   ├── learning_tick.rs    # Pipeline de aprendizaje por ciclo
│   │   ├── metrics_reporter.rs # Reporte de métricas
│   │   └── socket_handler.rs   # Manejo IPC (878 LOC)
│   └── apollo-optimizerctl.rs  # Client CLI
└── engine/                     # ← 126 archivos, 77,178 LOC
    ├── mod.rs                  # Declaraciones pub mod (3,282 bytes)
    ├── types.rs                # Tipos core: Profile, RootAction, etc. (1,394 LOC)
    ├── protocol.rs             # Wire protocol JSON (10,296 bytes)
    ├── safety.rs               # Invariantes de seguridad (1,320 LOC)
    ├── ...                     # Ver documentos 02, 03, 04 para detalle
    └── pipeline/               # Subdirectorio para stages del pipeline
```

---

## 7. Restricciones Operativas Fundamentales

### 7.1 Conservadurismo ante todo
- **Nunca prompts interactivos** en paths del daemon (no hanging en password requests).
- Si se necesita `sudo`, usar `sudo -n` (no-interactive, falla en vez de esperar).
- **Preferir `std::process::Command`** sin shell (evita shell injection, PATH manipulation).

### 7.2 Mutex Poisoning Convention
```rust
// En todo el repositorio:
lock().unwrap_or_else(|e| e.into_inner())
// Helper trait en lock_ext.rs: safe_lock()
```
Un mutex poisoned (por panic en otro thread) se recupera en vez de propagar el panic. Esto es crítico para un daemon long-running.

### 7.3 Rendimiento del Hot-Path
- **Per-cycle budget:** <10ms de wall time para los 126 módulos.
- **Zero allocation en hot-path** donde posible (pre-allocated buffers, borrow over clone).
- **Mutex guards dropped early** antes de I/O (escritura a disco, socket responses).
- **RSS del daemon:** ~8 MB.
- **CPU promedio:** ~0.02% de un solo core.

### 7.4 Sysctl Safety
- **Solo 16 claves allowlisted** para escritura (TCP buffers, file cache, compresión).
- El allowlist es hardcoded en el código; ni la configuración ni el LLM pueden añadir claves.
- Cada cambio loguea el valor anterior. `restore` revierte todo.

---

## 8. Instalación como LaunchDaemon

```bash
./scripts/install-root-daemon.sh
# 1. cargo build --release
# 2. cp target/release/apollo-optimizer* /usr/local/bin/
# 3. cp com.eduardocortez.systemoptimizer.plist /Library/LaunchDaemons/
# 4. launchctl load /Library/LaunchDaemons/com.eduardocortez.systemoptimizer.plist
# 5. Crea /etc/apollo-optimizer/config.toml si no existe

./scripts/uninstall-root-daemon.sh
# 1. launchctl unload
# 2. Trigger safety restore (descongela PIDs, revierte sysctls)
# 3. Remove binarios y plist
```

**Best practice:** Compilar como usuario normal, ejecutar el binario resultante como root vía launchd.

---

## 9. Interacciones con el Kernel macOS

Apollo utiliza las siguientes APIs del kernel XNU directamente (sin subprocesos CLI cuando es posible):

| API | Qué hace Apollo con ella | Archivo fuente |
|-----|--------------------------|----------------|
| `kqueue` + `EVFILT_VM` + `NOTE_VM_PRESSURE` | Detección instantánea de presión de memoria del kernel | `kqueue_pressure.rs` |
| Darwin Notifications (`com.apple.system.thermalpressurelevel`) | Alertas térmicas en tiempo real | `kqueue_pressure.rs` |
| `host_statistics64()` (Mach) | RAM: free, active, inactive, compressor, pageins, pageouts | `host_vm_info.rs` |
| `task_for_pid()` + `task_policy_set()` | Scheduling QoS por proceso (P-cores vs E-cores) | `mach_qos.rs` |
| `task_threads()` + `thread_info()` | Introspección per-thread para clasificación hot/cold | `mach_qos.rs` |
| `proc_pidinfo(PROC_PIDTASKINFO)` | CPU, memoria, threads por proceso individual | `proc_taskinfo.rs` |
| `SIGSTOP` / `SIGCONT` | Freeze/thaw de procesos | `execute_actions.rs` |
| `setpriority()` / `PRIO_DARWIN_BG` | Renice + background scheduling nativo Darwin | `execute_actions.rs` |
| IOKit (`IOServiceGetMatchingServices`) | Temperatura CPU/GPU, watts, fan RPM, battery | `iokit_sensors.rs` |
| IOReport (framework) | Contadores de rendimiento per-cluster CPU (ECPU/PCPU) | `ioreport.rs` |
| SMC (System Management Controller) | Temp CPU directa (Tc0P), temp board, battery overheat | `smc_direct.rs` |
| `sysctl` (lectura/escritura) | Parámetros TCP, file cache, compresión | `sysctl_governor.rs` |
| `mdutil` | Control on/off de Spotlight indexing | `execute_actions.rs` |
| `CGWindowListCopyWindowInfo` | Detección de ventanas visibles para freeze gate | `cg_window.rs` |
| `host_processor_info()` | Utilización per-core (saturación CPU) | `cpu_saturation.rs` |
