# Apollo Optimizer — Arquitectura Interna Detallada

> **Versión:** 1.0.0 · **Lenguaje:** Rust 2021 · **Plataforma:** macOS Apple Silicon (M1/M2/M3/M4)
> **AIS:** Apollo Intelligence Score — métrica compuesta de 6 dimensiones con escala 0–100.

Cada afirmación de este documento tiene correspondencia directa en un archivo `.rs` del proyecto. Validado contra el código fuente el 04 de Abril de 2026.

---

## 0. ¿Cómo funciona Apollo? (Explicación No Técnica)

### Para alguien que no programa

Imagina que tu Mac es un restaurante con una cocina (el procesador) y mesas limitadas (la memoria RAM). Los clientes son las aplicaciones: Safari, Xcode, Spotify, etc.

**El problema:** A veces llegan tantos clientes que la cocina se satura y la comida sale lenta para todos — incluso para el cliente VIP que pagó más (la app que estás usando activamente). macOS intenta manejar esto por sí solo, pero es como un mesero que no distingue entre un cliente esperando su postre y uno que ya se fue hace 2 horas y dejó la taza de café en la mesa ocupando espacio.

**Lo que hace Apollo:** Es un gerente de restaurante inteligente que trabaja en segundo plano, las 24 horas:

1. **Observa constantemente** quién está en el restaurante, cuánto come (CPU) y cuánto espacio ocupa en la mesa (RAM). Lee sensores físicos reales del chip: temperatura, presión de memoria, voltaje de batería, velocidad del ventilador.

2. **Clasifica a cada cliente** en categorías:
   - 🟢 **VIP (Intocable):** La app que estás usando ahora mismo — jamás se toca.
   - 🟢 **Personal del restaurante:** Procesos del sistema (WindowServer, launchd) — sin ellos el restaurante cierra. Nunca se tocan.
   - 🟡 **Cliente normal:** Apps abiertas en segundo plano — se les puede pedir que esperen un poco.
   - 🟠 **Cliente dormido:** Apps que llevan horas sin usarse, ocupando una mesa entera. Candidatas a que se les pida que se levanten.
   - 🔴 **Fantasma:** Procesos huérfanos o zombies que ya no sirven a nadie — se les retira la mesa.

3. **Toma decisiones automáticas:**
   - **Boost (subir prioridad):** "Pon al VIP más cerca de la cocina."
   - **Throttle (bajar prioridad):** "Ese cliente que descarga torrents en segundo plano puede esperar."
   - **Freeze (pausar):** "Dropbox lleva 3 horas sincronizando en la nube y estás compilando — que espere hasta que termines."
   - **Kill (solo zombies):** "Ese proceso está muerto en la tabla del kernel — hay que limpiar."

4. **Aprende de sus propias decisiones:**
   - Si congeló a Firefox y la presión de memoria bajó → lo recuerda para la próxima vez.
   - Si throttleó a un daemon y no cambió nada → lo anota como inefectivo.
   - Combina tres fuentes de aprendizaje para tener un "score" único por proceso.

5. **Se protege contra errores:**
   - Tiene una lista de procesos que **jamás** puede tocar (el kernel, la pantalla, el audio).
   - Si una app visible en pantalla depende de un proceso en segundo plano (por ejemplo, Safari esperando WebKit), **nunca** pausa a WebKit — detecta esta dependencia automáticamente.
   - Si oscila mucho entre perfiles (agresivo ↔ conservador) se auto-bloquea en modo "balanceado" por 5 minutos para no causar daño.

### En una oración

> Apollo es un piloto automático que silenciosamente identifica qué apps están desperdiciando los recursos de tu Mac y las pone a dormir o las ralentiza, sin que notes nada, para que la app que estás usando vaya más rápido.

### ¿Cuándo actúa y cuándo no?

| Tu Mac está... | Apollo hace... |
|---|---|
| Navegando y leyendo emails | Casi nada. Solo vigila. Tick cada 60 segundos. |
| Compilando un proyecto Rust/Swift | Congela agresivamente apps de fondo. Tick cada 2 segundos. |
| Con poca batería | Baja la tolerancia — actúa antes para ahorrar energía. |
| Calentándose | Reduce inmediatamente al modo conservador — no quiere empeorar el calor. |
| De noche (00:00–06:00) | Throttlea daemons de fondo inactivos para ahorrar batería. |
| Acabando de despertar (tras abrir la tapa) | Espera 60 segundos de gracia antes de intervenir. |

---

## 1. Diagrama General del Sistema (ASCII)

```
+================================================================================================+
|                                   APOLLO OPTIMIZER SYSTEM                                      |
+================================================================================================+
|                                                                                                |
|  Binarios:                                                                                     |
|  +------------------+    +------------------------+    +--------------------+                  |
|  | apollo-optimizer |    | apollo-optimizerd      |    | apollo-optimizerctl |                 |
|  | (CLI: snapshot,  |    | (Daemon long-running,  |    | (Client IPC:       |                  |
|  |  optimize, etc.) |    |  launchd-managed)      |    |  status, profile,  |                  |
|  +------------------+    +----------+-------------+    |  set-override)     |                  |
|                                     |                  +--------+-----------+                  |
|          Unix socket IPC  <---------+---------------------------+                              |
|          root:     /var/run/apollo-optimizer.sock                                               |
|          non-root: /tmp/apollo-optimizer.sock                                                  |
|                                     |                                                          |
+=====================================|==========================================================+
                                      |
                                      v
+================================================================================================+
|                           DAEMON INTERNAL ARCHITECTURE                                         |
+================================================================================================+
|                                                                                                |
|  [1] ========================= TELEMETRÍA Y SENSORES =====================================    |
|  |                                                                                        |   |
|  |  +-------------------+  +-------------------+  +--------------------+                  |   |
|  |  | iokit_sensors.rs  |  | smc_direct.rs     |  | kqueue_pressure.rs |                  |   |
|  |  | GPU%, core watts, |  | CPU temp (Tc0P),  |  | EVFILT_VM:         |                  |   |
|  |  | temps, fan RPM,   |  | fan speed, board  |  |  NOTE_VM_PRESSURE  |                  |   |
|  |  | battery mWh       |  | temp, battery     |  | Darwin Notifications:                |   |
|  |  +-------------------+  +-------------------+  |  thermal, spawn,   |                  |   |
|  |                                                 |  power source      |                  |   |
|  |  +-------------------+  +-------------------+  +--------------------+                  |   |
|  |  | silicon_probe.rs  |  | amx_detector.rs   |                                          |   |
|  |  | SiliconInfo:      |  | Detecta si el     |  +--------------------+                  |   |
|  |  |  chip gen (M1/2/3)|  | AMX coprocessor   |  | host_vm_info       |                  |   |
|  |  |  core count,      |  | está activo (IA   |  | (Mach host_stat64: |                  |   |
|  |  |  RAM total,       |  | workloads de      |  |  free, active,     |                  |   |
|  |  |  LSE, RNDR caps   |  | CoreML/PyTorch)   |  |  compressor, swap) |                  |   |
|  |  +-------------------+  +-------------------+  +--------------------+                  |   |
|  |                                                                                        |   |
|  |  +-------------------+  +-------------------+                                          |   |
|  |  | lse_counters.rs   |  | kpc_counters.rs   |                                          |   |
|  |  | ARM64 Large Sys.  |  | PMC performance   |                                          |   |
|  |  | Extension atomics |  | counters (EL0)    |                                          |   |
|  |  +-------------------+  +-------------------+                                          |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [2] ========================= EFFECTIVE PRESSURE ========================================    |
|  |                                                                                        |   |
|  |  effective_pressure.rs — "la presión autoritativa"                                     |   |
|  |                                                                                        |   |
|  |  effective = base_kernel_pressure                                                      |   |
|  |            + hardware_boost     (Warning=+0.15, Critical=+0.30)                        |   |
|  |            + battery_boost      (Normal=+0.04, LowPower=+0.10, Critical=+0.18)         |   |
|  |            + thermal_boost      (Phase1=+0.07, Phase2=+0.15, Phase3=+0.25, Phase4=+0.40)|  |
|  |            + llm_workload_boost (ollama/llama detected=+0.20)                          |   |
|  |            + charging_stress    (>8W while charging=+0.06)                             |   |
|  |            + battery_low        (TTE <20min=+0.08)                                     |   |
|  |            + memory_bandwidth   (AMC >80% saturated=+0.10)                             |   |
|  |            + smc_thermal        (≥80°C=+0.05, ≥90°C=+0.15, ≥100°C=+0.30)              |   |
|  |            + battery_overheat   (flag=+0.12)                                           |   |
|  |                                                                                        |   |
|  |  Resultado: clamp(0.0, 1.0)                                                            |   |
|  |  Ejemplo: base=0.60 + hw=0.15 + batt=0.04 + thermal=0.07 = 0.86 → BackgroundPressure  |   |
|  |           (sin los boosts, 0.60 sería InteractiveFocus y no haría nada)                |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [3] ======================== DOMAIN-GROUPED SHARED STATE ================================    |
|  |  (Patrón Strangler Fig — 44 campos planos → 6+ grupos de dominio)                     |   |
|  |                                                                                        |   |
|  |  +----------------+  +----------------+  +------------------+  +---------------+       |   |
|  |  | HardwareState  |  | ProcessState   |  | UsageDomainState |  | MetricsState  |       |   |
|  |  | last_hw_snap   |  | frozen: HashSet|  | UsageModel       |  | RuntimeMetrics|       |   |
|  |  | hw_status      |  | frozen_since   |  | EffectTracker    |  | 50+ counters  |       |   |
|  |  | sysctl_status  |  | wake_state     |  | OverflowGuard    |  | cycle_durations|      |   |
|  |  +----------------+  | last_blockers  |  +------------------+  +---------------+       |   |
|  |                       +----------------+                                                |   |
|  |  +----------------+  +----------------+  +--------------------------------------+      |   |
|  |  | PolicyState    |  | LlmDomainState |  | (otros campos: AdaptiveGovernor,     |      |   |
|  |  | ProfileGov     |  | llm_cfg        |  |  WorkloadClassifier, UserProfile,    |      |   |
|  |  | latency_target |  | llm_state      |  |  MachQoS, LearnedPolicy, LV state,  |      |   |
|  |  | governor_state |  | 5 paths        |  |  RL agent, signal_intelligence)      |      |   |
|  |  | overflow_guard |  +----------------+  +--------------------------------------+      |   |
|  |  +----------------+                                                                    |   |
|  |                                                                                        |   |
|  |  Impacto: el hot-path (reactor) ya no bloquea las rutinas de reporte de métricas      |   |
|  |  ni las peticiones API concurrentes. Cada domain se lockea independientemente.          |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [4] ==================== CLASIFICACIÓN DE PROCESOS ======================================    |
|  |                                                                                        |   |
|  |  process_classifier.rs — 8 tiers de clasificación heurística                           |   |
|  |                                                                                        |   |
|  |  +-- ProcessTier (mayor → menor importancia) ---------------------------------+        |   |
|  |  |                                                                            |        |   |
|  |  |  SystemEssential   (launchd, kernel_task, WindowServer, coreaudiod, ...)   |        |   |
|  |  |  ActiveForeground  (GUI + interacción < 30s)                               |        |   |
|  |  |  BackgroundVisible (GUI + sin interacción reciente)                        |        |   |
|  |  |  AppHelper         (Chrome Helper, WebKit, Electron, plugin-container)     |        |   |
|  |  |  SilentDaemon      (sin GUI, CPU > 0.5%, wakeups > 1/s)                   |        |   |
|  |  |  Stale             (sin GUI, CPU < 0.5%, wakeups < 1/s, idle > 300s)      |        |   |
|  |  |  Telemetry         (DiagnosticReporter, analyticsd, Siri, rapportd, ...)   |        |   |
|  |  |  ZombieOrphan      (is_zombie || padre muerto && ppid != 1)               |        |   |
|  |  |                                                                            |        |   |
|  |  +------------------------------------------------------------------------ ---+        |   |
|  |                                                                                        |   |
|  |  Scores por proceso:                                                                   |   |
|  |    utility_score [0,1] = f(GUI, interacción, red, CPU, wakeups, Rosetta, idle)         |   |
|  |    waste_score   [0,1] = f(tier, wakeups, RSS >200MB, idle >1h)                        |   |
|  |                                                                                        |   |
|  |  Ejemplo de utility_score:                                                             |   |
|  |    GUI + interacción<10s + red activa = 0.50+0.25+0.20+0.05 = 1.00                    |   |
|  |    sin GUI + idle >1h + wakeups>50 — penalización = 0.50-0.40-0.20 = -0.10 → 0.0      |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [5] ================== ZOMBIE HUNTER ====================================================    |
|  |                                                                                        |   |
|  |  zombie_hunter.rs — 5 clases de "peso muerto"                                         |   |
|  |                                                                                        |   |
|  |  Clase 1: TrueZombie    (kernel SZOMB)            → Kill inmediato                     |   |
|  |  Clase 2: Orphan        (padre muerto, ppid≠1)    → Kill inmediato                     |   |
|  |  Clase 3: GhostHelper   (host ausente >24h)       → Suspend tras 3 ciclos confirm.     |   |
|  |  Clase 4: WakeupBurner  (>20 wakeups/s, sin GUI)  → NiceToMax tras 3 ciclos confirm.   |   |
|  |  Clase 5: MemoryHoarder (>256MB RSS, idle >30min) → Suspend tras 3 ciclos confirm.     |   |
|  |                                                                                        |   |
|  |  confirmation_cycles = 3: las reglas blandas (3-5) requieren 3 observaciones           |   |
|  |  consecutivas antes de actuar. Previene falsos positivos por picos momentáneos.        |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [6] ============ ADAPTIVE GOVERNOR ("El Cerebro") =======================================    |
|  |                                                                                        |   |
|  |  adaptive_governor.rs — toma la decisión final por proceso                             |   |
|  |                                                                                        |   |
|  |  Inputs: ProcessSnapshot + HuntSnapshot + foreground_app + hour_of_day + HwFeatures    |   |
|  |                                                                                        |   |
|  |  Cascada de decisión (se evalúa en orden, la primera que matchea gana):                |   |
|  |                                                                                        |   |
|  |   1. ZombieOrphan?                        → Kill                                      |   |
|  |   2. SystemEssential o ActiveForeground?  → Allow (protegido absoluto)                 |   |
|  |   3. Uptime < 8 segundos?                 → Allow (efímero XPC, se va solo)            |   |
|  |   4. AppHelper con audio/video/red?       → Allow (romperlo crashea el tab)            |   |
|  |   5. AppHelper inactivo?                  → Throttle (nunca Freeze)                    |   |
|  |   6. Telemetry?                           → Throttle (Freeze si workload pesado)       |   |
|  |   7. Mach ports > 80?                     → Allow (IPC hub — throttle causa beachball) |   |
|  |   8. LLM cargado (ollama >1GB RSS)?       → Allow si idle<12h (reload cuesta 30s+)    |   |
|  |   9. I/O activo (pageins>50K, CPU>5%)?    → Allow (backup/encode en curso)             |   |
|  |  10. SilentDaemon idle (CPU<0.5%, fg>1h)? → Freeze si Rosetta o RSS>1GB, Throttle si no|  |
|  |  11. Idle graduado (sin GUI):                                                          |   |
|  |      - > 6h sin foreground → Throttle                                                 |   |
|  |      - > 12h sin foreground → Freeze                                                  |   |
|  |  12. Helper del foreground activo?         → Allow (Safari→WebKit, Chrome→Chrome Helper)|  |
|  |  13. Modo nocturno (00:00-06:00)?          → Throttle daemons idle>15min               |   |
|  |  14. Stale + utility < 0.05?               → Freeze                                   |   |
|  |  15. Render pipeline (GPU buffer/faults)?  → Allow (throttle causa frame drops)        |   |
|  |  16. Waste override (waste ≥ 0.90)?        → Throttle si utility < 0.60                |   |
|  |  17. Swarm (>30 procs, waste≥0.30)?        → Throttle (Freeze si Rosetta)              |   |
|  |  18. Wakeup hog (>100 wakeups/s, sin GUI)? → Throttle                                 |   |
|  |  19. utility < 0.05?                       → Freeze                                   |   |
|  |  20. utility < 0.20?                       → Throttle                                  |   |
|  |  21. else                                  → Allow                                     |   |
|  |                                                                                        |   |
|  |  Config calibrada por hardware al inicio:                                              |   |
|  |    M1 8GB:  waste_override = 0.80  (más agresivo — RAM escasa)                         |   |
|  |    M3 Max:  waste_override = 0.90  (más tolerante — RAM abundante)                     |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [7] ================= PROFILE GOVERNOR ==================================================    |
|  |                                                                                        |   |
|  |  profile_governor.rs — máquina de estados para el perfil global del daemon             |   |
|  |                                                                                        |   |
|  |  Tres perfiles: SafeRoot ↔ BalancedRoot ↔ AggressiveRoot                               |   |
|  |                                                                                        |   |
|  |  Fórmula de presión:                                                                   |   |
|  |    score = 0.35×cpu + 0.35×ram + 0.20×interactive_wait + 0.10×reactor_events           |   |
|  |          + swap_boost (min(swap_GB/2, 1.0) × 0.12)                                     |   |
|  |                                                                                        |   |
|  |  Crisis override: ram≥0.60 && swap≥1.5GB →                                             |   |
|  |    crisis_score = 0.60 + clamp(swap-1.5, 0, 1.5)/1.5 × 0.25                           |   |
|  |    score = max(base, crisis_score) — garantiza cruzar 0.72                             |   |
|  |                                                                                        |   |
|  |  Transiciones:                                                                         |   |
|  |    BalancedRoot → AggressiveRoot:  score≥0.72 × 3 ciclos (2 en Build mode)             |   |
|  |    BalancedRoot → SafeRoot:        score≤0.28 × 6 ciclos (4 en Idle mode)              |   |
|  |    AggressiveRoot → BalancedRoot:  score≤0.55 × 6 ciclos                               |   |
|  |    SafeRoot → BalancedRoot:        score≥0.40 × 3 ciclos                               |   |
|  |                                                                                        |   |
|  |  Overrides (prioridad descendente):                                                    |   |
|  |    1. ManualOverride con TTL (vía apolloctl set-override)                              |   |
|  |    2. thermal_constrained → cap en BalancedRoot                                        |   |
|  |    3. anti-thrash lock (>4 transiciones/10min → Balanced 5min)                         |   |
|  |       ↳ se rompe si ram≥0.60 && swap≥2GB (crisis real)                                 |   |
|  |    4. workload_onset (cargo/rustc detecado) → AggressiveRoot proactivo                 |   |
|  |    5. context_switch_burst (3+ cambios/5min, ram<0.70) → AggressiveRoot                |   |
|  |    6. dev/interactive_floor → mínimo BalancedRoot                                      |   |
|  |                                                                                        |   |
|  |  Throttle levels (lectura):                                                            |   |
|  |    score ≥ 0.72 → "high"  | 0.40..0.72 → "medium"  | < 0.40 → "low"                  |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [8] ================= OVERFLOW GUARD ====================================================    |
|  |                                                                                        |   |
|  |  overflow_guard.rs — aprendizaje adaptativo para prevenir OOM                          |   |
|  |                                                                                        |   |
|  |  Thresholds base:                                                                      |   |
|  |    bg_pressure = 0.78, critical = 0.88, extreme = 0.90                                 |   |
|  |                                                                                        |   |
|  |  Ajustes aditivos:                                                                     |   |
|  |    + overflow_offset   (cada overflow: -5pp, piso -20pp, half-life 8h)                 |   |
|  |    + workload_bonus    (Idle: +3pp, Interactive: +1pp, Build: -3pp, HeavyBuild: -5pp)  |   |
|  |    + rl_adjustment     (Q-learning Phase 4: corrección aprendida online)               |   |
|  |    + device_offset     (≤8GB: -5pp, ≤16GB: 0pp, >16GB: +5pp)                          |   |
|  |                                                                                        |   |
|  |  Deduplicación: ventana 60s entre eventos del mismo overflow.                          |   |
|  |  Persistencia: overflow_history.json sobrevive reboots (máx 20 eventos).               |   |
|  |  Build mode: ≥2 herramientas de compilación activas (rustc, cargo, clang, swift, etc.) |   |
|  |  Pattern matching: resembles_past_overflow() compara apps actuales con historial.      |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [9] =========== LOTKA-VOLTERRA (Modelo Ecológico de RAM) ================================    |
|  |                                                                                        |   |
|  |  lotka_volterra.rs — dinámica competitiva de procesos por RAM                          |   |
|  |                                                                                        |   |
|  |  Modelo de competencia interespecífica (Volterra, 1926):                                |   |
|  |                                                                                        |   |
|  |    dx/dt = r₁·x·(1 - (x + α₁₂·y)/K)    ← proceso dominante                           |   |
|  |    dy/dt = r₂·y·(1 - (y + α₂₁·x)/K)    ← resto del sistema                           |   |
|  |                                                                                        |   |
|  |    x,y  = fracciones de RAM [0,1]                                                      |   |
|  |    K    = 1.0 (normalizado)                                                            |   |
|  |    rᵢ   = growth rate (EWMA α=0.2 del cambio de RSS/dt)                               |   |
|  |    αᵢⱼ  = coef. competencia (aprendido: si x↑ y y↓ → α↑)                              |   |
|  |                                                                                        |   |
|  |  monopoly_risk() [0,1] = raíz cúbica(share × growth × competition)                    |   |
|  |    Media geométrica: los tres factores deben ser altos para alarma.                    |   |
|  |    growth_risk normalizado: 0.01/s=moderado, 0.05/s=rápido.                            |   |
|  |                                                                                        |   |
|  |  simulate_forward(horizon_secs): Euler explícito, paso 1s, máx 120 pasos.             |   |
|  |    Predice la fracción de RAM del dominante en `horizon` segundos.                     |   |
|  |                                                                                        |   |
|  |  Simplificación clave: solo 2 "especies" (dominante vs resto), no N (O(N²)).           |   |
|  |  Resetea growth tracking al cambiar de proceso dominante.                              |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [10] ================= EJECUCIÓN Y SEGURIDAD ============================================    |
|  |                                                                                        |   |
|  |  safety.rs + capabilities.rs + process_identity.rs + execute_actions.rs                |   |
|  |                                                                                        |   |
|  |  Procesos ABSOLUTAMENTE protegidos (nunca throttle/freeze/kill):                       |   |
|  |    kernel_task  launchd  WindowServer  loginwindow  configd  securityd                 |   |
|  |    tccd  syspolicyd  notifyd  hidd  UserEventAgent                                     |   |
|  |    Spotlight  mds  mds_stores  mdworker  mdworker_shared                               |   |
|  |                                                                                        |   |
|  |  Background crítico (solo throttle ligero):                                            |   |
|  |    Contenedores: podman, docker, colima, qemu-system                                   |   |
|  |    Bases de datos: postgres, mysqld, redis-server, mongod                              |   |
|  |    Runtimes: node, python, java, nginx, go, ruby, php                                  |   |
|  |    Compilación: rustc, cargo                                                           |   |
|  |                                                                                        |   |
|  |  Render pipeline (nunca throttle — causa frame drops):                                 |   |
|  |    VDCAssistant, coreservicesd, com.apple.gpu, MTLCompilerService, mediaserverd         |   |
|  |                                                                                        |   |
|  |  Budgets por ciclo por perfil:                                                         |   |
|  |    +--------------------+--------+-----------+--------+----------+                     |   |
|  |    | Perfil             | Boosts | Throttles | Freezes| Cooldown |                     |   |
|  |    +--------------------+--------+-----------+--------+----------+                     |   |
|  |    | AggressiveRoot     |     10 |        20 |      8 |      10s |                     |   |
|  |    | BalancedRoot       |      6 |        12 |      4 |      20s |                     |   |
|  |    | SafeRoot           |      3 |         6 |      2 |      45s |                     |   |
|  |    +--------------------+--------+-----------+--------+----------+                     |   |
|  |                                                                                        |   |
|  |  13 invariantes de seguridad:                                                          |   |
|  |                                                                                        |   |
|  |    1. Nunca congelar protected_processes().                                            |   |
|  |    2. Nunca congelar critical_background_processes().                                  |   |
|  |    3. Comandos vía std::process::Command — sin shell injection.                        |   |
|  |    4. Sysctl solo sobre allowlist de 16 claves exactas.                                |   |
|  |    5. Cooldown 90s entre transiciones de perfil.                                       |   |
|  |    6. Anti-thrash: >4 transiciones/10min → BalancedRoot lock 5min.                     |   |
|  |    7. Dev floor: sesión activa → nunca bajar a SafeRoot.                               |   |
|  |    8. Gracia post-wake: 60s de agresión suprimida.                                     |   |
|  |    9. LLM patterns saneados: max 80 chars, sin newlines, confianza≥0.80.              |   |
|  |   10. PIDs congelados en frozen_state.json → descongelados al reiniciar.               |   |
|  |   11. PID identity check: start_sec/start_usec — previene A-B-A recycling.            |   |
|  |   12. AppHelper = throttle-only (nunca freeze) — Chromium watchdog crashea tab.        |   |
|  |   13. IPC hubs (>80 Mach ports) = Allow siempre — throttle causa beachballs.           |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [11] =================== LEARNING PIPELINE ==============================================    |
|  |                                                                                        |   |
|  |  learning_pipeline.rs — coordinador de mini-batch para 3 subsistemas                   |   |
|  |                                                                                        |   |
|  |  LearningObservation:                                                                  |   |
|  |    { process_name, skill_name?, pre_pressure, post_pressure, workload, cycle }         |   |
|  |    effective() = (pre - post) >= 0.01                                                  |   |
|  |                                                                                        |   |
|  |  batch_size = 8 (default) — acumula, ordena por process_name (cache locality),         |   |
|  |  luego fan-out a los 3 subsistemas + cross-feed + sync al EffectivenessTracker.        |   |
|  |                                                                                        |   |
|  |  ┌──────────────────────────────────────────────────────────────────────┐               |   |
|  |  │  SUBSISTEMA 1: OutcomeTracker (outcome_tracker.rs)                 │               |   |
|  |  │  Bayesian per-process weights: (effective+1)/(throttle+2) Laplace  │               |   |
|  |  │  → Mantiene co-occurrence matrix + experience memory buffer        │               |   |
|  |  │  → effectiveness() por proceso                                     │               |   |
|  |  ├──────────────────────────────────────────────────────────────────────┤               |   |
|  |  │  SUBSISTEMA 2: CausalGraph (causal_graph.rs)                       │               |   |
|  |  │  Edges: (cause, effect) → CausalEdge                               │               |   |
|  |  │    confidence = EMA α=0.10 (Bayesian update)                       │               |   |
|  |  │    avg_delta  = EMA α=0.15 (magnitud del pressure drop)            │               |   |
|  |  │  eval_delay = 3 ciclos (espera antes de evaluar resultado)         │               |   |
|  |  │  pending queue ≤ 200 entradas                                      │               |   |
|  |  │  is_solid: confidence > 0.7 && evidence ≥ 5                        │               |   |
|  |  │  is_weak:  confidence < 0.25 && evidence ≥ 5                       │               |   |
|  |  │  impact_score = confidence × avg_delta (ranking real-world)         │               |   |
|  |  ├──────────────────────────────────────────────────────────────────────┤               |   |
|  |  │  SUBSISTEMA 3: SkillRegistry (optimization_skills.rs)              │               |   |
|  |  │  OptimizationSkill = receta aprendida:                              │               |   |
|  |  │    { name, min_pressure, workload_hint, throttle_targets,          │               |   |
|  |  │      success_rate, apply_count, success_count }                    │               |   |
|  |  │  Individual: aprendido de throttles directos                        │               |   |
|  |  │  Induced (group:/batch:): generado por rule_inducer de co-ocurrencia│              |   |
|  |  │  is_reliable: apply_count ≥ 5 && success_rate ≥ 0.60              │               |   |
|  |  │  should_retire: (≥10 apps, <35%) || (≥20 apps, <50% "zombie")     │               |   |
|  |  │  adapt_pressure: EMA α=0.20 (auto-calibra min_pressure)           │               |   |
|  |  │  next_trial_skill: round-robin exploration de skills unproven      │               |   |
|  |  │  purge_unexecutable: elimina si todos los targets son protegidos   │               |   |
|  |  └──────────────────────────────────────────────────────────────────────┘               |   |
|  |                                                                                        |   |
|  |  Cross-feed rules (al flush del batch):                                                |   |
|  |                                                                                        |   |
|  |    A. OutcomeTracker → SkillRegistry:                                                  |   |
|  |       Si effectiveness > 0.7 (≥3 throttles) → boost skill.success_count +1             |   |
|  |       (acelera convergencia de skills nuevos con evidencia bayesiana fuerte)            |   |
|  |                                                                                        |   |
|  |    B. CausalGraph → SkillRegistry:                                                     |   |
|  |       Si borde sólido (conf>0.7, ≥5 evidencia) && skill.success_rate < 0.5             |   |
|  |       → +1 éxito artificial (corrige trials con failures anómalos)                     |   |
|  |                                                                                        |   |
|  |    C. SkillRegistry → OutcomeTracker:                                                  |   |
|  |       Si skill.success_rate > 0.8 (≥20 apps) → siembra el prior bayesiano             |   |
|  |       (sabiduría persistente sobrevive reinicios del daemon)                            |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [12] ============ EFFECTIVENESS TRACKER (F3 Blend) ======================================    |
|  |                                                                                        |   |
|  |  effectiveness_tracker.rs — número autoritativo único por proceso                      |   |
|  |                                                                                        |   |
|  |  Basado en Thompson Sampling con multi-source Beta posteriors (Russo 2018)             |   |
|  |                                                                                        |   |
|  |  Fórmula:                                                                              |   |
|  |    cred_bayesian = min(bayesian_obs / 20, 1.0)    ← satura a 20 obs                   |   |
|  |    cred_causal   = min(causal_obs / 5, 1.0)       ← satura a 5 (Pearl dominance)      |   |
|  |    cred_skill    = min(skill_obs / 10, 1.0)       ← satura a 10                       |   |
|  |                                                                                        |   |
|  |    blended = (cred_b×bayes + cred_c×causal + cred_s×skill)                            |   |
|  |            / (cred_b + cred_c + cred_s)                                                |   |
|  |                                                                                        |   |
|  |    Cold start (0 obs) → 0.5 (neutral). NaN guard + clamp [0,1].                       |   |
|  |                                                                                        |   |
|  |  Ejemplo de dominancia causal:                                                         |   |
|  |    Causal: 5 obs → cred=1.0, conf=0.90                                                |   |
|  |    Bayes:  2 obs → cred=0.10, eff=0.30                                                |   |
|  |    Score = (0.10×0.30 + 1.0×0.90) / 1.10 = 0.845 ← causal gana                       |   |
|  |                                                                                        |   |
|  |  Interpretación:                                                                       |   |
|  |    ≥ 0.6 → objetivo fiable de throttling                                              |   |
|  |    0.4–0.6 → neutral / datos insuficientes                                            |   |
|  |    < 0.4 → throttling históricamente inefectivo                                       |   |
|  |                                                                                        |   |
|  |  GC: cada ~500 ciclos elimina entradas con age > max_stale_cycles && obs < min_obs.   |   |
|  |  Persistence: snapshot() / restore_from_map() para LearnedState.                       |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [13] =================== APOLLO INTELLIGENCE SCORE (AIS) ================================    |
|  |                                                                                        |   |
|  |  intelligence_score.rs — métrica compuesta [0, 100] con 6 dimensiones                  |   |
|  |                                                                                        |   |
|  |  AIS = Σ wᵢ × Dᵢ(x) × 100                                                            |   |
|  |                                                                                        |   |
|  |  +----------------------------+------+----------------------------------------------+  |   |
|  |  | Dimensión                  | Peso | Qué mide                                    |  |   |
|  |  +----------------------------+------+----------------------------------------------+  |   |
|  |  | D1: Decision Precision     | 0.25 | F1 sobre: preserved=40%, noise=30%, int=30% |  |   |
|  |  | D2: Signal Quality         | 0.20 | Kalman RMSE, CUSUM Fβ(β=2), Hazard calib.  |  |   |
|  |  | D3: Learning Velocity      | 0.20 | RL speed, Q-var, causal depth, skill rate   |  |   |
|  |  | D4: Resource Efficiency    | 0.15 | P75 cycle<100ms, skip-rate ~40%, habituation|  |   |
|  |  | D5: Safety Compliance      | 0.12 | 0 frozen_critical (=0 o score=0), kills,    |  |   |
|  |  |                            |      | survival acts, failures, overflows          |  |   |
|  |  | D6: Adaptability           | 0.08 | Profile switch acc, workload class, regime  |  |   |
|  |  +----------------------------+------+----------------------------------------------+  |   |
|  |                                                                                        |   |
|  |  Pareto balanced: todas las dimensiones ≥ 0.30 → ninguna puede mejorar               |   |
|  |  sin degradar otra.                                                                    |   |
|  |                                                                                        |   |
|  |  Grades: S(≥90) A(≥80) B(≥70) C(≥60) D(≥50) F(<50)                                   |   |
|  |  Regression floor: score ≥ 87.0 en runtime benchmark (daemon M1 estable)               |   |
|  |                                                                                        |   |
|  |  D5 Safety: frozen_critical > 0 → score = 0.0 (hard kill switch).                     |   |
|  |  D2 Kalman: threshold = √P* = 0.0884 (Riccati steady-state, Welch & Bishop 2006).     |   |
|  |  D2 CUSUM: Fβ con β=2 (recall 4× más importante que precision).                       |   |
|  |  D3 RL: post-convergence (total_ticks ≥ max_ticks) → speed = 1.0 (stability reward).  |   |
|  |  D4 Budget: pressure≥0.55 → budget_score=1.0 (running all subsystems is correct).     |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                      |                                                         |
|  [14] ================= REACTOR KQUEUE =========================== ========================   |
|  |                                                                                        |   |
|  |  Hilo separado que escucha eventos del kernel en tiempo real:                          |   |
|  |                                                                                        |   |
|  |  Evento 1: EVFILT_VM / NOTE_VM_PRESSURE → re-optimización inmediata                   |   |
|  |  Evento 2: Darwin Notif thermal (com.apple.system.thermalpressurelevel)                |   |
|  |  Evento 3: Darwin Notif spawn (com.apple.launchd.spawn)                                |   |
|  |  Evento 4: Darwin Notif power (com.apple.system.powersources.source)                   |   |
|  |                                                                                        |   |
|  |  Ante cualquier evento:                                                                |   |
|  |    fast_tick_until = now + fast_tick_duration                                           |   |
|  |    reactor_event_weight += 1 (alimenta pressure_score del gobernador)                  |   |
|  |    → dispara ciclo de optimización inmediato                                           |   |
|  |                                                                                        |   |
|  |  Tick rate adaptivo:                                                                   |   |
|  |    Idle normal:          60s                                                            |   |
|  |    Workload pro:         15s                                                            |   |
|  |    Post-evento kqueue:   2s (durante fast_tick_duration)                                |   |
|  |                                                                                        |   |
|  =========================================================================================    |
|                                                                                                |
+================================================================================================+
```

---

## 2. El Ciclo de Vida del Reactor (Execution Flow Detallado)

Cada "tick" del daemon ejecuta este pipeline completo:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  TICK START                                                                  │
│                                                                              │
│  1. RECOLECCIÓN (~30ms)                                                      │
│     ├── sysinfo → CPU%, memoria por proceso, PIDs                           │
│     ├── host_stat64 → RAM kernel {free, active, inactive, compressor, swap} │
│     ├── iokit_sensors → temperatura, watts, GPU%, ventilador                │
│     ├── smc_direct → CPU temp (Tc0P), board temp, battery flag              │
│     └── kqueue_pressure → último evento VM pressure                         │
│                                                                              │
│  2. PRESIÓN EFECTIVA (~1ms)                                                  │
│     └── effective_pressure::compute() → presión autoritativa [0,1]          │
│                                                                              │
│  3. OVERFLOW GUARD (~1ms)                                                    │
│     ├── tick_decay(pressure, compressor) → RL agent tick + offset decay     │
│     └── thresholds(workload_mode) → bg/critical/extreme dinámicos           │
│                                                                              │
│  4. CLASIFICACIÓN Y DECISIÓN (~5ms)                                         │
│     ├── ProcessClassifier::classify_all() → 8 tiers + utility + waste       │
│     ├── ZombieHunter::evaluate_all() → dead weight detection                │
│     ├── AdaptiveGovernor::decide_all_with_hw() → vecor de ProcessDecision   │
│     ├── LotkaVolterra::update() + monopoly_risk()                           │
│     └── ProfileGovernor::evaluate() → perfil efectivo + throttle_level      │
│                                                                              │
│  5. SEGURIDAD Y FILTRADO (~1ms)                                             │
│     ├── Filtrar protected_processes()                                        │
│     ├── Filtrar critical_background_processes()                              │
│     ├── Wait-graph: si foreground espera al candidato → VETO               │
│     ├── Budget check: no exceder max boosts/throttles/freezes por ciclo     │
│     └── Process identity check: start_sec/start_usec match                  │
│                                                                              │
│  6. EJECUCIÓN (~10ms)                                                        │
│     ├── Boost: taskpolicy LATENCY_QOS_TIER_0                                │
│     ├── Throttle: renice +10 / taskpolicy THROUGHPUT                        │
│     ├── Freeze: SIGSTOP + registro en frozen_state.json                     │
│     ├── Kill: SIGKILL (solo zombies confirmados)                            │
│     └── Journal append: journal.jsonl con estado before/after               │
│                                                                              │
│  7. APRENDIZAJE (~2ms)                                                       │
│     ├── LearningPipeline::push(observation)                                 │
│     │   └── Si batch.len() >= 8 → flush:                                    │
│     │       ├── Fan-out a OutcomeTracker, CausalGraph, SkillRegistry        │
│     │       ├── Cross-feed rules A, B, C                                    │
│     │       └── Sync EffectivenessTracker (F3 blend)                        │
│     ├── CausalGraph::evaluate(current_pressure, cycle) → pending resolved   │
│     └── MetricsReporter::update() → runtime_metrics.json                    │
│                                                                              │
│  8. ESTADO Y PERSISTENCIA (~5ms, no cada tick)                              │
│     ├── governor_state.json (solo en transición)                             │
│     ├── frozen_state.json (solo si cambió set congelados)                    │
│     ├── optimization_skills.json (en flush del pipeline)                     │
│     ├── learned_state.json (periódico)                                       │
│     └── runtime_metrics.json (cada tick)                                     │
│                                                                              │
│  TICK END — sleep hasta próximo tick (2s–60s según modo)                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Persistencia y Estado

| Archivo | Formato | Cuándo se escribe | Contenido |
|---------|---------|-------------------|-----------|
| `journal.jsonl` | JSONL append | Cada acción | Acción, PID, nombre, before/after |
| `runtime_metrics.json` | JSON | Cada tick | 50+ contadores, cycle_durations ring |
| `governor_state.json` | JSON | En transición de perfil | Perfil, cooldown, override |
| `profile_timeline.jsonl` | JSONL append | En transición | from, to, reason, score |
| `frozen_state.json` | JSON | Cambio en set congelados | PIDs + start times |
| `wake_state.json` | JSON | Eventos sleep/wake | Estado y timestamps |
| `learned_state.json` | JSON | Periódico | Kalman, CUSUM, Hazard, OutcomeTracker |
| `overflow_history.json` | JSON | En overflow | Últimos 20 eventos + offset |
| `optimization_skills.json` | JSON | Flush del pipeline | Map de skills con rates |
| `rl_threshold.json` | JSON | Cada 50 ticks | Q-table + current_adjustment |

**Ubicación:** `/var/lib/apollo/` (root) | `/tmp/` (non-root)
**Técnica:** Write-then-Rename — archivos JSON nunca quedan corruptos ante crash.
**Startup recovery:** Si el daemon crasheó con PIDs congelados, los descongela al reiniciar.

---

## 4. Glosario Técnico Completo

| Término | Significado | Archivo fuente |
|---------|-------------|----------------|
| **AIS** | Apollo Intelligence Score — 6 dimensiones × pesos → [0,100] | `intelligence_score.rs` |
| **F3 Blend** | Mezcla ponderada por credibilidad de 3 fuentes (Bayes + Causal + Skill) | `effectiveness_tracker.rs` |
| **Solid edge** | Borde causal con confidence > 0.7 && evidence ≥ 5 | `causal_graph.rs` |
| **Impact score** | confidence × avg_delta — ranking por efecto real | `causal_graph.rs` |
| **EMA** | Exponential Moving Average — promedio ponderado exponencial | Múltiples |
| **OverflowGuard** | Módulo que baja thresholds tras overflows (half-life 8h) | `overflow_guard.rs` |
| **Strangler Fig** | Patrón de refactorización incremental del monolito SharedState | `daemon_state.rs` |
| **Cross-feed** | Reglas que transfieren conocimiento entre los 3 subsistemas | `learning_pipeline.rs` |
| **workload_onset** | Build detectado → escala proactivamente a AggressiveRoot | `profile_governor.rs` |
| **anti-thrash lock** | Bloqueo BalancedRoot por 5min ante >4 oscilaciones/10min | `profile_governor.rs` |
| **P-Cores / E-Cores** | Firestorm (rendimiento) / Icestorm (eficiencia) del Apple Silicon | `silicon_probe.rs` |
| **Lotka-Volterra** | Modelo ecológico de competencia por RAM entre procesos | `lotka_volterra.rs` |
| **monopoly_risk** | Score [0,1] de riesgo de que un proceso acapare toda la RAM | `lotka_volterra.rs` |
| **ZombieClass** | 5 tipos de procesos inútiles detectados por ZombieHunter | `zombie_hunter.rs` |
| **ProcessTier** | 8 niveles de clasificación heurística de procesos | `process_classifier.rs` |
| **utility_score** | [0,1] — qué tan valioso es el proceso para el usuario ahora | `process_classifier.rs` |
| **waste_score** | [0,1] — qué tan despilfarrador es el proceso | `process_classifier.rs` |
| **PressureComponents** | Desglose de los 9 boosts que componen la presión efectiva | `effective_pressure.rs` |
| **RL threshold** | Q-learning agent que ajusta umbrales online (Phase 4) | `rl_threshold.rs` |
| **device_offset** | Ajuste de thresholds por RAM del dispositivo (±5pp) | `overflow_guard.rs` |
| **LearnedPolicy** | Patrones aprendidos via LLM teacher (max 80 chars, conf≥0.80) | `llm.rs` |
| **CUSUM** | Cumulative Sum — detecta cambios de régimen (Page, 1954) | `cusum.rs` |
| **Kalman** | Filtro 1D para suavizar ruido de presión de memoria | `kalman.rs` |
| **Hazard model** | Cox proportional hazards — predice probabilidad de OOM | `hazard_model.rs` |
| **Dyna-Q** | Model-based RL que usa transiciones simuladas (Sutton, 1991) | `rl_threshold.rs` |

---

## 5. Referencias Académicas Citadas en el Código

| Referencia | Dónde se usa |
|---|---|
| Pearl (2009) "Causality: Models, Reasoning and Inference" | `causal_graph.rs`, `effectiveness_tracker.rs` |
| Thompson (1933) "On the likelihood that one unknown probability exceeds another" | `effectiveness_tracker.rs` |
| Russo et al. (2018) "A Tutorial on Thompson Sampling" arXiv:1707.02038 | `effectiveness_tracker.rs` |
| Auer et al. (2002) "Finite-time Analysis of the Multiarmed Bandit Problem" | `effectiveness_tracker.rs` |
| Shannon (1948) Information Theory | `intelligence_score.rs` |
| Bellman (1957) Optimality Principle | `intelligence_score.rs` |
| Volterra (1926) Competitive species dynamics | `lotka_volterra.rs` |
| Sutton & Barto (2018) "Reinforcement Learning" §6.3, §6.5 | `intelligence_score.rs` |
| Welch & Bishop (2006) Kalman filter performance (Riccati P*) | `intelligence_score.rs` |
| Page (1954) "CUSUM schemes" | `intelligence_score.rs` |
| Cox (1972) "Regression Models and Life Tables" | `intelligence_score.rs` |
| Hellerstein (2004) "Feedback Control of Computing Systems" | `intelligence_score.rs` |
| Jain (1991) "Art of Computer Systems Performance Analysis" | `intelligence_score.rs` |
| Jaynes (2003) "Probability Theory" (MaxEnt neutral prior) | `intelligence_score.rs` |

---

## 6. Jerarquía Cognitiva de Aprendizaje (Módulos Post-v1.0)

> Los módulos `[1]`–`[14]` del diagrama ASCII cubren el reactor operativo.
> Las secciones 6–12 documentan la **capa cognitiva** que se construyó encima:
> el sistema que hace que Apollo *aprenda a aprender*.

### 6.1 NestedLearner — Coordinador de 3 Niveles

`nested_learner.rs` — Inspirado en [Google Nested Learning 2025].

Flujo de contexto bidireccional entre tres frecuencias de aprendizaje:

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

| Nivel | Frecuencia | Subsistemas | Qué controla |
|-------|------------|-------------|--------------|
| **L0** | Cada ciclo | `SignalIntelligence` (Kalman, CUSUM, Entropía, Hazard) | `l0_quality` EMA [0,1]. Gate de L1. |
| **L1** | Por outcome | `OutcomeTracker`, `CausalGraph` | `l1_aggregate` EMA. Ponderado por `l0_quality`. |
| **L2** | Cada 20 flushes de L1 | `LearningPipeline`, `ReptileMeta` | `l2_context`. Alimenta meta-learning rate. |

Retroalimentación L2→L0 [Google NL 2025 §6.2]:
- `l2_meta_velocity` = EMA de |Δl2_context| por flush.
- `dynamic_l1_gate = L1_GATE_THRESHOLD + L2_VELOCITY_GATE_SCALE × l2_meta_velocity`
- Clamped a `[0.25, 0.60]`. Si el meta-aprendizaje oscila → exige señales más limpias.

Persistencia: serializado dentro de `learned_state.json`. Sobrevive reinicios.

---

### 6.2 TeacherConsolidator — Consolidación S2 → S1

`teacher_consolidation.rs` — [McGaugh 2004], [Yerkes-Dodson 1908], [Kahneman 2011].

El LLM (Gemma 4) opera como "System 2". Sus sugerencias se compilan en reflejos "System 1"
(pattern_weights + NARS beliefs) mediante consolidación afectiva:

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
            │   fuera de banda → dampening (no consolidar en crisis extrema)
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
            │   │   → effectiveness baja automáticamente por Laplace
            │   ├─ NARS: observe_salient(proc, success=false, salience)
            │   └─ GemmaTrust[categoría] ← EMA hacia 0.0
            │
            └─ Deadband: |causal_effect| < 0.015 → BELOW_DEADBAND, sin update
               (calibrado contra noise floor de ~0.01 en M1 8GB)
```

GemmaTrust por categoría:
- `Interactive`, `Noise`, `Protected`, `Profile`, `Latency`
- `is_reliable()`: count ≥ 3 && trust ≥ 0.70
- Apollo puede ignorar sugerencias de una categoría con bajo trust.

Benchmark: `consolidate()` < 100µs/call (hot-path safe).

---

### 6.3 NARS Belief System — Creencias No-Axiomáticas

`nars_belief.rs` — [Wang 2013 NARS §3.3.3], [McGaugh 2004].

Implementa un sistema de creencias con **affective salience weighting**:

**TruthValue** = `{ frequency ∈ [0,1], confidence ∈ [0,1] }`
- `frequency` = fracción de éxitos observados (Bayesian: (pos+1)/(pos+neg+2))
- `confidence` = certeza (decay × función del total de evidencia)

**Salience** = `{ arousal ∈ [0,1], valence ∈ [-1,1] }`
- `arousal` = qué tan importante es este evento (pressure × p_oom × swap)
- `valence` = positivo (éxito) o negativo (fracaso)
- Arousal alta → evidencia pesa 4× más (via `evidence_weight()`)

**DriftDetector** — Detección de concept drift:
- Rastrea frecuencia de éxito por proceso. Si la frecuencia cae significativamente
  respecto al `prior_frequency` → señal de drift.
- `drift_score` = media de |frequency - prior| ponderada por confianza.
- `needs_recalibration()`: drift_score > 0.05 && drifted_count ≥ 2.
- `acknowledge_recalibration()`: resetea priors al estado actual.

**ArousalState** — Estado de activación global:
- EMA de arousal y valence promedio.
- Alimenta al `Neuromodulator` y a la dimensión D3 del UCHS.

---

### 6.4 FreezeIntelligence — NARS Aplicado a Freeze/Thaw

`freeze_intelligence.rs` — [Wang 2013], [Altmann & Trafton 2002].

Capa cognitiva universal para decisiones de congelamiento. Reemplaza la lógica per-app
hardcodeada con creencias NARS por **categoría de proceso**:

| Categoría | Ejemplos | Default confidence |
|-----------|----------|-------------------|
| `chromium-renderer` | Brave Helper (Renderer), Slack Helper (Renderer) | 0.70 |
| `chromium-gpu` | Code Helper (GPU), Brave Helper (GPU) | 0.70 |
| `ide-lsp` | sourcekit-lsp, clangd, rust-analyzer | 0.70 |
| `xpc-service` | *.XPCService | 0.70 |
| `media-helper` | Spotify Helper, Music Helper | 0.70 |
| `app-helper` | SomeApp Helper (plain) | 0.70 |
| `generic` | Todo lo demás | 0.70 |

- `observe(process_name, success, salience)` → actualiza la belief de su categoría.
- `should_freeze(name)` → `false` si confidence < 0.35 (MIN_FREEZE_CONFIDENCE).
- `pre_thaw_hint(predicted_app)` → categorías que deben thaw antes de un switch.
  Ej: "Brave Browser" → `["chromium-renderer", "chromium-gpu"]`.
- Failures en una categoría **no afectan** a otra (aislamiento por diseño).

---

## 7. MetaCognición y Auto-Evaluación

### 7.1 MetaCognition — Calibración de Segundo Orden

`meta_cognition.rs` — [Guo 2017 ECE], [Lakshminarayanan 2017].

Rastrea la brecha entre lo que cada subsistema **predice** y lo que **realmente ocurre**:

Subsistemas rastreados: `RlAgent`, `LinUcb`, `NarsBelief`, `CausalGraph`, `SignalKalman`, `FreezeIntelligence`.

```
  AccuracyEMA por subsistema:
    predicted_ema = EMA(α=0.05) de confianzas predichas
    actual_ema    = EMA(α=0.05) de resultados reales
    calibration_gap = EMA(α=0.05) de |predicted - actual|

  Aggregate ECE:
    calibration_error = Σ(gap_i × √obs_i) / Σ(√obs_i)
    meta_confidence   = 1.0 - calibration_error
```

**Humble Mode** (calibration_error > 0.20 && ≥10 observaciones):
- Duración: 50 ciclos mínimo.
- `humble_exploration_mult()` = 2.0× (duplica ε-greedy y LinUCB α).
- `humble_freeze_confidence_floor()` = 0.45 (vs 0.35 normal).
- Sale solo si: ciclos expirados AND calibration_error < 0.20.

Miscalibración direccional:
- `predicted > actual` → overconfident → Apollo "cree que sabe más de lo que sabe".
- `predicted < actual` → underconfident → Apollo descarta señal buena.

---

### 7.2 SelfRewardingEvaluator — Auto-Recompensa Retroactiva

`self_reward.rs` — [Yuan 2024 DR-ZERO], [Pearl 2009].

Problema: entre eventos OOM, no hay señal de aprendizaje (sparse reward).
Solución: Apollo juzga sus propias decisiones pasadas usando el CausalGraph como oráculo.

```
  1. log_decision(cycle, "throttle:Firefox", predicted=0.80, pressure=0.75)
  2. ... esperan EVAL_DELAY_CYCLES=10 ciclos ...
  3. evaluate_past(current_cycle, current_pressure, causal_confidence_fn):
       causal_conf = CausalGraph.confidence("throttle:Firefox")
       pressure_improvement = (pressure_at_decision - current_pressure).max(0)
       JuicyScore = causal_conf × pressure_improvement / (cycles × 0.1 + 1.0)
       prediction_error = JuicyScore - predicted_score
```

- `reward_ema` = EMA de JuicyScore (calidad general de decisiones).
- `self_eval_accuracy` = EMA de |prediction_error| (qué tan bien se auto-evalúa).
- `is_well_calibrated()`: ≥10 evals informativas && accuracy < 0.20.
- `evaluator_trust()` [0,1]: combina calibración (60%) + volumen (40%).

Alimenta: `CognitiveRewardBus` y dimensión D4 del UCHS.

---

### 7.3 AdversarialProbe — Stress Testing Sintético

`adversarial_probe.rs` — [Madry 2018 adversarial robustness], [Yuan 2024 §4.2].

Cada 500 ciclos, ejecuta 4 escenarios de "peor caso" sobre **copias** del estado cognitivo
(zero side effects en producción):

| Escenario | Expectation | Qué prueba |
|-----------|-------------|------------|
| Presión 0.98, P(OOM)=0.95, protegidos=[kernel_task, WindowServer, Claude] | `NoFreezeProtected` | Jamás congelar procesos protegidos, ni bajo presión extrema |
| Presión 0.99, P(OOM)=0.99 | `SafetyFloorRespected` | RL threshold ≥ 0.45 (piso de seguridad) |
| Inject drift en DriftDetector (15 obs negativas con crisis salience) | `NarsDriftRecovery` | Debe detectar AND recuperarse en ≤20 ciclos |
| Incertidumbre máxima en todas las dimensiones | `EpistemicBlocksAggressive` | Composite uncertainty > 0.70 → DEBE bloquear acciones agresivas |

- `pass_rate_ema` (EMA α=0.10) alimenta dimensión D6 del UCHS.
- `safety_alert = true` cuando pass_rate < 0.75.
- Failure log de hasta 20 entradas (newest first vía `recent_failures()`).

---

## 8. Neuromodulador — Modulación Bio-Inspirada de Parámetros

`neuromodulator.rs` — Adaptado de memoria-core.

4 señales "neuroquímicas" modulan parámetros del sistema en tiempo real.
Costo: ~50ns/ciclo, 0 allocations, 0 dependencias. Leaky integration con τ≈10 ticks.

| Señal | Inputs | Parámetro derivado | Rango |
|-------|--------|--------------------|-------|
| **Dopamine** (recompensa) | pressure_drop, outcome_penalty, !overflow | `alpha_multiplier` (RL learning rate) | [0.5, 1.5] |
| **Noradrenaline** (estrés) | urgency, regime_shift_up, pressure_velocity, thermal_emergency | `dyna_steps` (Dyna-Q planning) | [4, 20] |
| **Serotonin** (estabilidad) | low_pressure_streak, !urgency, regime_shift_down, !overflow | `serotonin_shift` (zone threshold shift) | [-0.05, +0.05] |
| **Acetylcholine** (novedad) | process_churn, entropy_anomaly, rl_exploring | `epsilon_bonus` (exploración) | [0.0, 0.05] |

Baseline (todos en 0.5): parámetros derivados igualan los valores hardcodeados originales.

---

## 9. Incertidumbre Epistémica

`epistemic.rs` — [Lakshminarayanan 2017 §3].

Composite uncertainty de 4 fuentes independientes:

```
  composite = 0.30 × rl_q_variance        ← Q-values muy dispersos
            + 0.30 × linucb_exploration    ← brazo con pocas observaciones
            + 0.25 × nars_confidence_spread ← alguna creencia NARS con confianza baja
            + 0.15 × drift_score           ← modelo diverge de realidad
```

| Composite | Modo | Efecto |
|-----------|------|--------|
| < 0.40 | LOW | Operación normal |
| 0.40–0.70 | MODERATE | Sin restricciones |
| 0.70–0.85 | HIGH | Bloquea freezes agresivos, SIGSTOP vetado |
| > 0.85 | OBSERVE-ONLY | Fuerza brazo "Observe" (zero side effects) |

`dominant_source()`: identifica qué componente contribuye más (para debug).

---

## 10. Meta-Learning (Reptile) y PredictiveAgent (LinUCB)

### 10.1 ReptileMeta — Adaptación Rápida entre Workloads

`reptile_meta.rs` — [Nichol 2018], [Finn 2017 MAML simplificado].

Mantiene θ_slow (global) + θ_fast (por workload fingerprint):

```
  On workload change (fingerprint A → B):
    1. Save θ_current → workload_params[A]
    2. Reptile update: θ_slow ← θ_slow + ε × (θ_current - θ_slow)   [ε = 0.01]
    3. Si B es conocido: θ_current = θ_slow + 0.5 × (θ_fast[B] - θ_slow)
       Si B es nuevo:    θ_current = θ_slow  (warm start desde experiencia global)
```

MetaParams (biases, no copias completas — memoria pequeña):
- `rl_q_bias[48]`: corrección aditiva por estado de Q-table.
- `linucb_arm_biases[5]`: corrección por brazo LinUCB.
- `nars_confidence_adj`: ajuste al piso de confianza NARS [−0.15, +0.30].

Cache: máx 16 workload fingerprints. Eviction por LRU. Stale > 10,000 ciclos → prune.

### 10.2 PredictiveAgent — Contextual Bandit (LinUCB)

`predictive_agent.rs` — [Li 2010 LinUCB], [Auer 2002].

5 brazos para intervención proactiva de memoria, seleccionados con LinUCB
sobre un vector de contexto de 12 dimensiones:

| Brazo | Acción proactiva |
|-------|-----------------|
| 0: Observe | Solo observar, no actuar |
| 1: TightenThresholds | Bajar bg_pressure -5pp |
| 2: SuggestAggressive | Recomendar perfil AggressiveRoot |
| 3: PreemptiveThrottle | Throttle top-3 waste processes |
| 4: WarnUser | Emitir alerta de presión |

Contexto = [pressure, pressure_velocity, p_oom, compressor_ratio, swap_gb,
            lv_monopoly_risk, lv_predicted, hour_sin, hour_cos,
            workload_encoded, profile_encoded, effectiveness_top3]

---

## 11. Unified Cognitive Health Score (UCHS)

`cognitive_health.rs` — [Doncieux 2018 Open-ended Learning], [Yuan 2024 §5].

**UCHS** mide *qué tan bien aprende Apollo* (vs AIS que mide qué tan bien *optimiza*).

```
  UCHS = Σ wᵢ × Dᵢ     (composite ∈ [0, 1])

  +----------------------------+------+------------------------------------------+
  | Dimensión                  | Peso | Fuente                                   |
  +----------------------------+------+------------------------------------------+
  | D1: Calibration            | 0.20 | MetaCognition.meta_confidence            |
  | D2: Reward Quality         | 0.20 | CognitiveRewardBus.signal_to_noise       |
  |                            |      | (normalizado: tanh(SNR/3))               |
  | D3: Belief Stability       | 0.15 | 1 - DriftDetector.drift_score            |
  | D4: Self-Awareness         | 0.20 | SelfRewardingEvaluator.evaluator_trust   |
  | D5: Adaptability           | 0.10 | ReptileMeta.adaptation_quality           |
  | D6: Safety                 | 0.15 | AdversarialProbe.pass_rate_ema           |
  +----------------------------+------+------------------------------------------+
```

Grades: S+(≥0.95) S(≥0.90) A(≥0.80) B(≥0.70) C(≥0.60) D(≥0.40) F(<0.40)

**Recovery Mode** (composite < 0.40):
- Pausa todo el aprendizaje por 10 ciclos.
- Sale solo si: ciclos expirados AND composite ≥ 0.40.
- Previene que el sistema "aprenda basura" cuando su cognición está degradada.

`weakest_dimension()`: identifica la dimensión más baja para mejora dirigida.

---

## 12. Pipeline de Tick Cognitivo (2 Stages)

El daemon ejecuta dos pipelines acoplados por ciclo. El reactor (§2) es Stage 1.
Después viene el pipeline cognitivo:

### Stage 1: learning_tick (learning_tick.rs)
```
  1. NestedLearner::tick_l0(signal_quality)  → gating de outcomes
  2. OutcomeTracker feed + CausalGraph evaluate
  3. NestedLearner::tick_l1(effectiveness)   → acumula para L2
  4. Si L2 gate period alcanzado → NestedLearner::flush_l2()
  5. LearningPipeline flush (fan-out + cross-feed)
  6. ReptileMeta::apply_learning_delta()
  7. TeacherConsolidator::consolidate() (si hay SuggestionOutcome pendiente)
```

### Stage 2: cognitive_tick (cognitive_tick.rs)
```
  1. CognitiveRewardBus::collect_rewards()   → normalización PPO-style
  2. MetaCognition::observe() + tick()       → ECE, humble mode
  3. SelfRewardingEvaluator::evaluate_past() → dense reward signal
  4. EpistemicUncertainty::update()          → action gating
  5. CognitiveHealthScore::update()          → UCHS composite
  6. AdversarialProbe (si should_probe)      → safety invariants
  7. Neuromodulator::tick()                  → parameter modulation
  8. StabilityOracle::record_*()            → instability penalty para RL
```

### StabilityOracle (stability_oracle.rs)

Agrega 5 señales de estabilidad perceptual en un score compuesto [0,1]:

| Señal | Fuente | Normalización |
|-------|--------|---------------|
| Display jank | DisplayTurbo deactivate | 0/1 evento |
| Zombie rate | heuristic_stats.zombies | count/5, cap 1 |
| Swap spike | Δswap ≥ 512MB/ciclo | 0/1 evento |
| VM thrashing | VmRate.thrashing_score | score/5000, cap 1 |
| CPU stall | ContentionTracker.stall_fraction | [0,1] directo |

- `stability_score = 1 - mean(5 EMAs)`
- `instability_penalty_attenuated(uptime)`: dampener lineal durante primeros 300s
  post-boot (Spotlight reindexing, launchd warmup no son culpa de Apollo).
- Penalty inyectado al RL vía `NeuroSignals::outcome_penalty`.

---

## 13. Nuevas Referencias Académicas (Capa Cognitiva)

| Referencia | Dónde se usa |
|---|---|
| Google (2025) "Nested Learning" (context flow between frequency levels) | `nested_learner.rs` |
| Hochreiter & Schmidhuber (1997) LSTM multi-timescale memory | `nested_learner.rs` |
| McGaugh (2004) "Amygdala modulates consolidation" | `teacher_consolidation.rs`, `nars_belief.rs` |
| Yerkes & Dodson (1908) Inverted-U arousal/learning | `teacher_consolidation.rs` |
| Kahneman (2011) "Thinking, Fast and Slow" (S1/S2) | `teacher_consolidation.rs` |
| Rubin (1974) Potential Outcomes (counterfactual) | `teacher_consolidation.rs` |
| Wang (2013) "Non-Axiomatic Reasoning System" §3.3.3 | `nars_belief.rs`, `freeze_intelligence.rs` |
| Guo (2017) "On Calibration of Modern Neural Networks" ICML | `meta_cognition.rs` |
| Lakshminarayanan (2017) "Predictive Uncertainty" NeurIPS §2-3 | `meta_cognition.rs`, `epistemic.rs` |
| Yuan (2024) "Self-Rewarding Language Models" arXiv:2401.10020 §3-4 | `self_reward.rs`, `adversarial_probe.rs` |
| Madry (2018) "Towards DL Models Resistant to Adversarial Attacks" ICLR | `adversarial_probe.rs` |
| Nichol (2018) "On First-Order Meta-Learning Algorithms" arXiv:1803.02999 | `reptile_meta.rs` |
| Finn (2017) "Model-Agnostic Meta-Learning" ICML (MAML) | `reptile_meta.rs` |
| Li (2010) "A Contextual-Bandit Approach to Personalized News" WWW | `predictive_agent.rs` |
| Doncieux (2018) "Open-ended Learning" Front. Neurorobotics | `cognitive_health.rs` |
| Schulman (2017) "Proximal Policy Optimization" | `cognitive_bus.rs`, `stability_oracle.rs` |
| Altmann & Trafton (2002) Pre-activation before task switch | `freeze_intelligence.rs` |
| Kuncheva (2004) Non-stationary signal EMA tracking | `stability_oracle.rs` |
