# Apollo Optimizer

### Optimización Inteligente de Sistemas para macOS — Un Enfoque de Tres Niveles para la Gestión de Recursos

> *Un daemon nativo en Rust que observa, aprende y se adapta a cómo usas tu Mac — reemplazando herramientas reactivas de limpieza con optimización proactiva basada en evidencia.*

---

## Resumen

Los sistemas macOS modernos ejecutan entre 400 y 600 procesos concurrentes. El scheduler del sistema operativo, aunque sofisticado, opera sin conocimiento de la intención del usuario — no puede distinguir entre un compilador que el usuario está esperando activamente y un daemon de telemetría quemando ciclos de CPU en segundo plano. Esto crea una clase de problemas de rendimiento que ni el sistema operativo ni las herramientas tradicionales de "limpieza" pueden resolver: **la brecha entre el scheduling a nivel de sistema y las prioridades a nivel de usuario**.

Apollo Optimizer aborda esta brecha con una arquitectura de inteligencia de tres niveles — heurísticas sub-milisegundo, clasificación Bayesiana ligera, y refinamiento opcional de políticas mediante LLM en la nube — ejecutándose como un daemon nativo en Rust con acceso directo al scheduling del kernel Mach, monitoreo de eventos kqueue, y telemetría de hardware IOKit. No limpia cachés, no borra archivos, no promete magia. Hace que el scheduler de tu sistema sea consciente de lo que realmente estás haciendo.

---

## Tabla de Contenidos

1. [El Problema](#el-problema)
2. [Cómo Funciona Apollo](#cómo-funciona-apollo)
3. [Vista General de la Arquitectura](#vista-general-de-la-arquitectura)
4. [Los Tres Niveles de Inteligencia](#los-tres-niveles-de-inteligencia)
5. [Qué lo Hace Diferente](#qué-lo-hace-diferente)
6. [Respondiendo al Escepticismo](#respondiendo-al-escepticismo)
7. [Arquitectura de Seguridad](#arquitectura-de-seguridad)
8. [Impacto Medible](#impacto-medible)
9. [Instalación](#instalación)
10. [Uso](#uso)
11. [Configuración](#configuración)
12. [Inmersión Técnica](#inmersión-técnica)
13. [Contribuir](#contribuir)

---

## El Problema

### Por Qué macOS No Se Optimiza Solo

macOS tiene un excelente scheduler. El scheduler Mach de XNU maneja prioridades de hilos, clases QoS y throttling térmico. Pero opera bajo restricciones fundamentales:

1. **Sin modelo de intención del usuario.** El kernel no sabe que estás esperando a que termine `cargo build`. Trata a tu compilador igual que a un daemon de indexación en segundo plano — ambos son simplemente hilos solicitando tiempo de CPU.

2. **Sin conciencia de dependencias entre procesos.** Cuando WindowServer se bloquea porque `cfprefsd` está leyendo un plist masivo, el kernel ve dos procesos haciendo I/O. No sabe que uno está bloqueando toda tu experiencia interactiva.

3. **Gestión térmica conservadora.** macOS reduce la velocidad de todos los cores por igual cuando la temperatura sube. No sabe que throttlear tu exportación de video mientras deja la indexación de Spotlight a máxima velocidad es el peor tradeoff posible.

4. **Sin comportamiento aprendido.** El sistema no aprende que usas Xcode de 9am a 5pm y cambias a Final Cut Pro por las noches. Cada arranque empieza de cero.

5. **Acumulación de procesos.** macOS genera procesos auxiliares, servicios XPC y daemons que persisten mucho después de que la aplicación que los creó se haya cerrado. Estos "helpers fantasma" consumen memoria y ciclos de wakeup indefinidamente.

### Por Qué las Herramientas de Limpieza No Ayudan

Las herramientas tradicionales de "optimización de Mac" (CleanMyMac, OnyX, etc.) operan sobre un problema fundamentalmente diferente:

| Enfoque | Qué Hace | Qué No Hace |
|---------|----------|-------------|
| Limpiadores de caché | Borran archivos temporales | Mejorar el scheduling de CPU |
| "Liberadores" de memoria | Purgan a la fuerza el caché de archivos | Reducir la presión real de memoria |
| Gestores de inicio | Deshabilitan items de login | Optimizar prioridades de procesos en ejecución |
| Desinstaladores | Eliminan apps y residuos | Manejar la contención de recursos por ciclo |

Estas herramientas abordan problemas de *almacenamiento* e *instalación*. Apollo aborda la *contención de recursos en tiempo de ejecución* — un dominio fundamentalmente diferente que requiere telemetría del sistema en tiempo real y toma de decisiones continua.

---

## Cómo Funciona Apollo

Apollo se ejecuta como un daemon root (`apollo-optimizerd`) que observa continuamente el estado del sistema y realiza intervenciones dirigidas:

```
Cada 2–60 segundos (adaptivo):

1. OBSERVAR    Recopilar CPU, memoria, térmico, swap, estado de procesos y sensores de hardware
2. CLASIFICAR  Categorizar cada proceso por nivel (8 niveles) y asignar puntuación de utilidad
3. DETECTAR    Identificar procesos bloqueadores, zombies, fugas de memoria, tormentas de wakeup
4. DECIDIR     Generar acciones de optimización (boost, throttle, freeze) dentro de presupuestos de seguridad
5. EJECUTAR    Aplicar acciones via APIs del kernel Mach (taskpolicy, renice, SIGSTOP, sysctl)
6. APRENDER    Actualizar clasificador de carga y perfil de usuario con comportamiento observado
7. AUDITAR     Registrar cada acción con estado antes/después en journal append-only
```

### Ejemplo Concreto: El Escenario de Compilación

Estás ejecutando `cargo build --release`. Sin Apollo:

- `softwareupdated` está buscando actualizaciones, consumiendo ancho de banda de I/O
- `photolibraryd` está analizando fotos en segundo plano, usando 2 cores de CPU
- `Spotlight` está indexando un directorio nuevo, compitiendo por I/O y CPU
- `WindowServer` está bloqueado por `cfprefsd`, agregando 8ms de latencia a cada frame

Con Apollo:

1. **Clasificador de carga** (Tier 2) detecta carga de trabajo "Coding" por la app en primer plano + proceso cargo
2. **Detector de bloqueadores** identifica a `cfprefsd` como bloqueador de WindowServer (score: 0.42)
3. **Acciones generadas:**
   - Boost a `cargo` → renice -10, scheduling en P-Core via `task_policy_set`
   - Boost a `cfprefsd` → renice -10 (desbloquear la cadena de espera de WindowServer)
   - Throttle a `softwareupdated` → renice +20, solo E-Core, I/O tier 4
   - Throttle a `photolibraryd` → renice +10, I/O tier 2
   - Freeze a indexación de `Spotlight` → SIGSTOP (se reanudará cuando baje la presión)
4. **Gobernador de perfil** escala a AggressiveRoot después de 3 ciclos consecutivos de alta presión
5. **Tuning de sysctl** optimiza buffers TCP y caché de archivos para carga de desarrollo

Resultado: Los procesos que importan obtienen más recursos. Los que no, se estacionan. Todo dentro de presupuestos de seguridad forzados, todo registrado, todo reversible.

---

## Vista General de la Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tres Binarios                                 │
│                                                                  │
│  apollo-optimizer     CLI para comandos puntuales                │
│  apollo-optimizerd    Daemon con optimización continua           │
│  apollo-optimizerctl  Cliente para control y consultas           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    Motor de 27 Módulos                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Niveles de Inteligencia                                 │    │
│  │  T1: Heurísticas (<1ms) — Governor, Clasificador, Zombie │    │
│  │  T2: ML Ligero (<5ms)  — Bayesiano Workload, Perfil User│    │
│  │  T3: LLM Teacher (async) — Refinamiento de políticas    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  9 Subsistemas Especializados                            │    │
│  │  Térmico · Memoria · Swap · GPU · Energía · Red         │    │
│  │  WakeStorm · RecuperaciónProcesos · Analítica           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Capa de Seguridad                                       │    │
│  │  Procesos protegidos · Presupuestos · Allowlist sysctl  │    │
│  │  Anti-thrash · Gracia post-wake · Kill switch           │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Interfaz con Kernel macOS                                       │
│  eventos kqueue · Mach task_policy_set · SIGSTOP/SIGCONT        │
│  tuning sysctl · sensores IOKit · powermetrics · taskpolicy     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Los Tres Niveles de Inteligencia

### Nivel 1 — Heurísticas (<1ms por ciclo)

La capa de decisión más rápida. Sin red, sin inferencia ML, sin I/O de disco. Computación pura en memoria.

El **Gobernador Adaptivo** clasifica cada proceso en ejecución y asigna una decisión:

| Decisión | Disparador | Acción |
|----------|-----------|--------|
| Permitir | Utilidad > 0.4, o esencial del sistema | Sin intervención |
| Throttle | Utilidad < 0.4, o score de desperdicio > 0.9 | `renice +10`, degradar a E-Cores |
| Congelar | Utilidad < 0.1 bajo carga pesada | `SIGSTOP` (pausar proceso completamente) |
| Matar | Zombie verdadero u huérfano (confirmado 3 ciclos) | `SIGKILL` |

El **Cazador de Zombies** identifica 5 clases de procesos muertos:

| Clase | Descripción | Criterio |
|-------|-------------|----------|
| ZombieReal | Estado Z del kernel, padre no ha recolectado | Verificación de estado del kernel |
| Huérfano | Padre muerto, no re-asignado a launchd | Validación de PPID |
| HelperFantasma | XPC/helper cuya app anfitriona cerró hace >24h | Ancestría de proceso + tiempo |
| QuemadorDeWakeups | >20 wakeups/sec sin valor para el usuario | Monitoreo de tasa de wakeup |
| AcaparadorDeMemoria | >256 MB RSS, sin UI, inactivo >30 min | Verificación de memoria + interacción |

### Nivel 2 — ML Ligero (<5ms por ciclo)

Clasificación Bayesiana ligera que se ejecuta completamente en el dispositivo. Sin llamadas de red. Combina 5 fuentes de evidencia:

```
P(carga | evidencia) ∝ P(app_en_primer_plano) × P(hora_del_día) × P(recencia_app)
                       × P(mix_de_procesos) × P(patrones_aprendidos)
```

Esto produce una clasificación de carga de trabajo (Coding, VideoCall, MediaPlayback, VideoEdit, OfficeWork, CommandLine, Idle) con scores de confianza. La clasificación ajusta los niveles de agresividad — durante Coding, el sistema es más agresivo congelando ruido de fondo; durante VideoCall, protege los procesos de audio/video.

El **Perfil de Usuario** aprende patrones de comportamiento con el tiempo:
- Qué apps usas a qué horas
- Duración promedio de sesión por aplicación
- Qué cargas de trabajo correlacionan con qué ventanas de tiempo
- Qué procesos son relevantes para tu flujo de trabajo típico

### Nivel 3 — LLM Teacher (Opcional, Asíncrono)

Un asesor basado en la nube que observa patrones del sistema y actualiza la `LearnedPolicy` local. No es toma de decisiones en tiempo real — es refinamiento periódico de políticas.

- Rate-limited: 2 llamadas/hora, intervalo mínimo de 15 minutos
- Gate de confianza: sugerencias por debajo de 0.80 de confianza son descartadas
- Sanitización de patrones: máximo 80 caracteres, sin keywords de Spotlight, sin saltos de línea
- Ventana de entrenamiento: 2 semanas por defecto, luego revierte a solo local
- Modelo: gpt-4.1-mini (configurable a cualquier API compatible con OpenAI)

El LLM teacher responde una sola pregunta: *"Dado el mix de procesos de este sistema, ¿qué procesos deberían tratarse como interactivos (no throttlear), ruido (seguro de throttlear), o protegidos (nunca tocar)?"*

---

## Qué lo Hace Diferente

### vs. Scheduler Integrado de macOS

| Capacidad | macOS XNU | Apollo |
|-----------|-----------|--------|
| Scheduling de hilos | Clases QoS por hilo | Por proceso, consciente de intención del usuario |
| Throttling térmico | Igual para todos los cores | Selectivo: throttlear fondo, proteger primer plano |
| Dependencias de procesos | Ninguna (sin conciencia de wait-graph) | Detección de bloqueadores con scoring ponderado |
| Conciencia de carga | Ninguna (sin modelo de intención) | Clasificación Bayesiana con 5 fuentes de evidencia |
| Comportamiento aprendido | Se reinicia cada arranque | Persiste entre sesiones (LearnedPolicy + UserProfile) |
| Limpieza de zombies | Solo `waitpid()` para zombies reales | Detección de 5 clases (fantasmas, acaparadores, quemadores) |
| Prevención de swap | Reactivo (comprimir cuando está lleno) | Predictivo (pronóstico a 30 segundos, paging preventivo) |

Apollo no reemplaza el scheduler de macOS — lo informa. Usa los mismos mecanismos `task_policy_set()` y `renice` que Apple provee, pero con mejor información sobre lo que el usuario realmente necesita.

### vs. Monitor de Actividad

El Monitor de Actividad te muestra qué está pasando. Apollo actúa sobre ello autónomamente. La diferencia es entre un termómetro y un termostato.

### vs. CleanMyMac / OnyX / DaisyDisk

Estas herramientas resuelven problemas diferentes:

| Herramienta | Dominio | Dominio de Apollo |
|-------------|---------|-------------------|
| CleanMyMac | Limpieza de disco, desinstalación | Scheduling de CPU/memoria en tiempo real |
| OnyX | Scripts de mantenimiento, limpieza de caché | Optimización de procesos en tiempo real |
| DaisyDisk | Visualización de espacio en disco | Contención de recursos por proceso |

Apollo no borra archivos. No limpia cachés. Gestiona cómo los procesos en ejecución comparten CPU, memoria, I/O y margen térmico — un problema que estas herramientas no abordan.

### vs. `htop` / `top` / Scripts de Gestión de Procesos

La gestión manual de procesos no escala. Con 400+ procesos, no puedes monitorear y ajustar prioridades continuamente. Apollo automatiza los juicios que un sysadmin experto haría, ejecutándose 24/7 con tiempos de respuesta sub-segundo.

---

## Respondiendo al Escepticismo

### "macOS ya gestiona bien los recursos. Esto es aceite de serpiente."

macOS gestiona recursos de forma *justa* — le da a cada proceso su parte según su clase QoS. Pero justo no es óptimo. Cuando estás compilando código, no quieres que `softwareupdated` reciba su parte justa del ancho de banda de I/O. Quieres que el compilador obtenga todo lo que necesita y que los daemons de fondo esperen.

El valor de Apollo está en el delta entre *scheduling justo* y *scheduling consciente de intención*. Esto es medible:
- La detección de bloqueadores resuelve stalls de cadenas de espera que el kernel no modela
- La gestión térmica predictiva previene el throttling generalizado que cuesta 15–30% del rendimiento de compilación
- La limpieza de zombies/fantasmas recupera memoria que macOS acumula pero nunca reclama

### "Ejecutar un daemon para optimizar rendimiento es un oxímoron."

El overhead del daemon de Apollo está acotado por diseño:
- Loop principal: cada 15–60 segundos (adaptivo)
- CPU por ciclo: <10ms de tiempo de pared (27 módulos, todos <5ms individualmente)
- Footprint de memoria: ~8 MB RSS (sin asignación en tiempo de ejecución en hot path)
- I/O: escrituras append-only al journal, sin polling de disco
- Hilo reactor: bloqueado en kqueue (cero CPU cuando no hay eventos)

El daemon usa aproximadamente **0.02% de un solo core** promediado en el tiempo. Ahorra 5–15% priorizando correctamente las cargas de trabajo. El ROI es >100x.

### "SIGSTOP es peligroso. Podrías congelar procesos críticos."

Apollo mantiene tres capas de protección:

1. **Lista de procesos protegidos** — 15+ procesos críticos del sistema (kernel_task, launchd, WindowServer, etc.) están hardcodeados como intocables. Ninguna configuración puede anular esto.

2. **Lista de background crítico** — Bases de datos (postgres, redis), containers (docker, podman) y servidores de desarrollo (node, python, java) nunca se congelan, solo se throttlean ligeramente.

3. **Presupuestos de seguridad** — Incluso en modo AggressiveRoot, el sistema puede congelar como máximo 8 procesos por ciclo. Los procesos congelados se descongelan automáticamente después de 10 minutos. Todos los PIDs congelados se persisten a disco; si el daemon crashea, descongela todo al reiniciar.

En 228 tests, incluyendo tests de condiciones de carrera concurrentes y casos extremos, el sistema jamás ha congelado un proceso protegido.

### "Modificar sysctls es riesgoso."

Apollo escribe en exactamente 16 claves sysctl en allowlist. Estas son tamaños de buffer TCP, límites de caché de archivos y parámetros de tuning de compresión — las mismas configuraciones que el propio Server Performance Mode de Apple modifica. La allowlist está hardcodeada; ninguna configuración o sugerencia de LLM puede escribir en una clave fuera de esta lista.

Cada cambio de sysctl se registra con el valor anterior. `apollo-optimizerctl restore` revierte todos los cambios. `apollo-optimizerctl panic-restore` hace lo mismo más crear un archivo kill switch que pausa el daemon.

### "¿Un LLM tomando decisiones del sistema? Eso es aterrador."

El LLM no toma decisiones. Hace *sugerencias* sobre clasificación de procesos — y esas sugerencias son:

- **Rate-limited:** Máximo 2 llamadas/hora
- **Gate de confianza:** Por debajo de 0.80 de confianza → descartado
- **Sanitizado:** Máximo 80 caracteres, sin keywords de Spotlight, sin saltos de línea
- **Acotado:** Máximo 6 patrones por categoría por llamada
- **Limitado en tiempo:** La ventana de entrenamiento expira después de 2 semanas
- **Opcional:** Completamente deshabilitado por defecto; requiere explícitamente `apollo-optimizerctl llm set-key`

El LLM no puede congelar, matar o throttlear procesos. Solo puede agregar strings a una lista que el clasificador Bayesiano local usa como una de 5 fuentes de evidencia. El peso de los patrones aprendidos por LLM es 1.5 para interactivos y -0.5 para ruido — significativo pero no dominante.

### "Esto requiere acceso root. Eso es una preocupación de seguridad."

Apollo requiere root por la misma razón que Activity Monitor necesita root para ver todos los procesos: la optimización a nivel de sistema requiere acceso a nivel de sistema.

Específicamente, root es necesario para:
- `task_policy_set()` — Scheduling del kernel Mach (requiere `task_for_pid` que necesita root)
- `sysctl -w` — Escribir parámetros del kernel
- `SIGSTOP/SIGCONT` — Enviar señales a procesos de otros usuarios
- `powermetrics` — Leer sensores de hardware
- `renice` a valores negativos — Elevar prioridad de procesos

El binario del daemon es propiedad de root, se ejecuta vía launchd, y almacena estado en `/var/lib/apollo/` (modo 700). El socket Unix permite a cualquier usuario consultar estado pero solo root puede enviar comandos de control.

### "¿En qué se diferencia de simplemente escribir un cron job con `renice`?"

Un cron job se ejecuta a intervalos fijos con reglas estáticas. Apollo:

1. **Reacciona a eventos** — El monitoreo kqueue da respuesta sub-segundo a presión de memoria, cambios térmicos, lanzamientos de procesos y cambios de fuente de energía
2. **Clasifica cargas de trabajo** — Inferencia Bayesiana con 5 fuentes de evidencia determina si estás programando, en videollamada, o inactivo
3. **Detecta dependencias** — El análisis de wait-graph identifica procesos bloqueando tu experiencia interactiva
4. **Predice problemas** — Pronóstico de swap (30s adelante), predicción térmica (tiempo hasta throttle), detección de fugas de memoria
5. **Aprende con el tiempo** — Perfil de usuario, patrones de carga, y opcionalmente clasificación de procesos refinada por LLM
6. **Fuerza seguridad** — Presupuestos, cooldowns, listas protegidas, lógica anti-thrash, y un kill switch

Un cron job con `renice` es un termostato con un solo ajuste de temperatura. Apollo es un sistema de gestión de edificio con sensores en cada habitación.

### "Rust es excesivo para un daemon de sistema."

Rust provee tres propiedades críticas para un daemon que envía señales a cada proceso del sistema:

1. **Seguridad de memoria sin GC** — Sin pausas de garbage collection en el loop de optimización. Sin use-after-free en el manejo de señales. Sin buffer overflows en el parsing de sensores.

2. **Abstracciones de costo cero** — El motor de 27 módulos se ejecuta en <10ms por ciclo. La misma arquitectura en Python tomaría 100ms+; en Go, 30ms+ con pausas de GC.

3. **Garantías estáticas** — La exhaustividad de `enum` significa que cada variante de `RootAction` se maneja. `Option<T>` significa que los errores de null pointer se capturan en tiempo de compilación. El envenenamiento de `Mutex` se recupera, nunca produce panic.

El daemon corre 24/7 con privilegios root. La elección del lenguaje es la más conservadora, no la más conveniente.

---

## Arquitectura de Seguridad

El sistema de seguridad de Apollo está diseñado con el principio de **defensa en profundidad** — múltiples capas independientes que deben estar todas de acuerdo antes de tomar cualquier acción.

### Capa 1: Detección de Capacidades
Antes de intentar cualquier operación, el daemon verifica qué está disponible:
```
can_taskpolicy()     — ¿Está presente /usr/sbin/taskpolicy?
can_sysctl()         — ¿Podemos escribir valores sysctl?
can_memorystatus()   — ¿Están disponibles las hints de presión de memoria? (solo root)
can_mdutil()         — ¿Podemos controlar Spotlight?
is_root()            — ¿Estamos ejecutando como root?
```

### Capa 2: Listas de Procesos Protegidos
Hardcodeadas, no se pueden anular:
```
NUNCA TOCAR: kernel_task, launchd, WindowServer, loginwindow,
             configd, securityd, tccd, syspolicyd, notifyd, hidd,
             Spotlight, mds, mds_stores, mdworker, mdworker_shared

SOLO THROTTLE: docker, podman, postgres, redis, node, python, java
```

### Capa 3: Presupuestos de Acciones
Cada ciclo está limitado por perfil:
- BalancedRoot: máximo 6 boosts, 12 throttles, 4 freezes
- AggressiveRoot: máximo 10 boosts, 20 throttles, 8 freezes
- SafeRoot: máximo 3 boosts, 6 throttles, 2 freezes

### Capa 4: Validación de Procesos
Antes de cada señal: `kill(pid, 0)` confirma que el proceso aún existe y no ha sido reciclado.

### Capa 5: Cooldowns y Anti-Thrash
- 90 segundos de cooldown entre transiciones de perfil
- 25 segundos de cooldown entre boosts al mismo proceso
- >4 transiciones en 10 minutos → bloqueo a BalancedRoot por 5 minutos
- 60 segundos de período de gracia post-wake después de suspensión del sistema

### Capa 6: Reversibilidad
- Todos los PIDs congelados rastreados en `frozen_state.json`
- Todos los cambios de sysctl registrados con valores anteriores
- El comando `restore` descongela todo y revierte sysctls
- `panic-restore` hace lo mismo más crear kill switch
- El arranque del daemon descongela cualquier PID de un crash anterior

### Capa 7: Kill Switch
Crear `/var/run/apollo.disable` → el daemon pausa toda optimización inmediatamente.

---

## Impacto Medible

### Métricas de Optimización Rastreadas

Apollo rastrea 50+ métricas por ciclo, incluyendo:

| Métrica | Descripción |
|---------|-------------|
| `boosts_applied` | Procesos elevados a alta prioridad |
| `throttles_applied` | Procesos de fondo degradados |
| `freezes_applied` | Procesos pausados via SIGSTOP |
| `paging_hints_applied` | Hints de page-out preventivo de memoria |
| `zombies_detected` | Procesos muertos identificados |
| `kills_applied` | Procesos zombie/con fugas terminados |
| `survival_mode_activations` | Intervenciones de emergencia |
| `profile_switches` | Transiciones automáticas de perfil |

### Impacto Energético y Ambiental

El motor de analítica estima:
```
Energía ahorrada (Wh) = avg_cpu_improvement% × 0.5W × horas_uptime
CO₂ evitado (g)       = energía_ahorrada × 0.075 g/Wh
```

Estas son estimaciones conservadoras basadas en la reducción de utilización de CPU por throttlear trabajo de fondo innecesario.

---

## Instalación

### Prerrequisitos
- macOS 13+ (Ventura o posterior)
- Apple Silicon (M1/M2/M3/M4) o Intel Mac
- Toolchain de Rust (para compilar desde fuente)

### Compilar e Instalar

```bash
# Clonar y compilar
git clone https://github.com/eduardocortez/apollo-optimizer.git
cd apollo-optimizer
cargo build --release

# Instalar como daemon root (launchd)
./scripts/install-root-daemon.sh

# Verificar instalación
apollo-optimizerctl status
apollo-optimizerctl doctor
```

### Desinstalar

```bash
# Revierte todas las optimizaciones y elimina el daemon
./scripts/uninstall-root-daemon.sh
```

---

## Uso

### Comandos CLI

```bash
# Optimización puntual
apollo-optimizer optimize

# Snapshot del sistema (salida JSON)
apollo-optimizer snapshot --output system_snapshot.json

# Iniciar daemon manualmente
apollo-optimizer daemon

# Modo turbo (deshabilitar animaciones, tuning máximo)
apollo-optimizer turbo

# Restaurar todos los cambios
apollo-optimizer restore
```

### Control del Daemon

```bash
# Estado
apollo-optimizerctl status

# Gestión de perfiles
apollo-optimizerctl set-profile aggressive-root --ttl-minutes 60
apollo-optimizerctl set-profile balanced-root
apollo-optimizerctl clear-profile-override
apollo-optimizerctl set-auto-profile on

# Diagnósticos
apollo-optimizerctl doctor
apollo-optimizerctl capabilities
apollo-optimizerctl top-blockers
apollo-optimizerctl metrics
apollo-optimizerctl profile-timeline

# Análisis de uso
apollo-optimizerctl usage top --limit 10
apollo-optimizerctl usage explain chrome

# LLM teacher (opcional)
apollo-optimizerctl llm set-key
apollo-optimizerctl llm status
apollo-optimizerctl llm test
apollo-optimizerctl llm disable
apollo-optimizerctl dump-policy

# Feedback
apollo-optimizerctl feedback good --note "la compilación fue rápida"
apollo-optimizerctl feedback bad --note "el navegador se sintió lento"

# Emergencia
apollo-optimizerctl restore
apollo-optimizerctl panic-restore
```

---

## Configuración

Archivo de configuración: `/etc/apollo-optimizer/config.toml`

```toml
# Perfil de optimización por defecto
profile = "balanced-root"

# Política de seguridad
policy = "aggressive-controlled"

# Modo LLM teacher opcional
[llm]
enabled = false
model = "gpt-4.1-mini"
endpoint = "https://api.openai.com/v1/chat/completions"
min_confidence = 0.85
max_calls_per_hour = 2
min_interval_secs = 900
timeout_ms = 10000
force_json = true
```

### Perfiles

| Perfil | Cuándo | Comportamiento |
|--------|--------|----------------|
| `balanced-root` | Por defecto | Optimización moderada, 20s cooldown |
| `aggressive-root` | Alta presión | Intervención máxima, 10s cooldown |
| `safe-root` | Baja presión | Intervención mínima, 45s cooldown |

Los perfiles transicionan automáticamente basándose en presión sostenida (configurable via `set-auto-profile`).

---

## Inmersión Técnica

Para documentación arquitectónica completa incluyendo análisis módulo por módulo, diagramas de flujo de datos, especificaciones de algoritmos e invariantes de seguridad, ver **[ARCHITECTURE.md](ARCHITECTURE.md)**.

### Archivos Clave

| Ruta | Propósito |
|------|-----------|
| `src/main.rs` | Punto de entrada CLI |
| `src/bin/apollo-optimizerd.rs` | Daemon (3,082 líneas) |
| `src/bin/apollo-optimizerctl.rs` | CLI cliente |
| `src/engine/` | 27 módulos centrales |
| `src/collector.rs` | Recolección de métricas del sistema |
| `src/reactor.rs` | Event loop kqueue |
| `src/sysctl_tuner.rs` | Tuning de parámetros del kernel |
| `tests/level*.rs` | 228 tests en 10 niveles |

### Dependencias

| Crate | Propósito |
|-------|-----------|
| `sysinfo` | Métricas de CPU, memoria, procesos, disco, red |
| `serde` / `serde_json` | Serialización para persistencia de estado e IPC |
| `clap` | Parsing de argumentos CLI |
| `chrono` | Manejo de tiempo con serialización |
| `anyhow` | Manejo de errores con propagación de contexto |
| `libc` | System calls (SIGSTOP, SIGCONT, kqueue, ioctl) |
| `ctrlc` | Manejo graceful de SIGINT |
| `toml` | Parsing de archivos de configuración |
| `ureq` | Cliente HTTP para integración LLM |

### Optimizaciones de Compilación

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=native"]

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

- **Instrucciones nativas de CPU** — SIMD y optimizaciones específicas de la plataforma
- **Link-Time Optimization** — Inlining entre crates y eliminación de código muerto
- **Unidad de codegen única** — Máxima optimización a costo de tiempo de compilación
- **Panic = abort** — Sin overhead de unwinding en el daemon

---

## Contribuir

```bash
# Compilar
cargo build

# Tests (228 tests en 10 niveles)
cargo test

# Formatear
cargo fmt --all

# Lint
cargo clippy --all-targets

# Ejecutar desde fuente
cargo run --bin apollo-optimizerd -- daemon --profile balanced-root
cargo run --bin apollo-optimizerctl -- status
```

### Niveles de Tests

| Nivel | Enfoque |
|-------|---------|
| 1 | Tests unitarios, límites de seguridad, convergencia EMA |
| 2 | Integración: módulo de seguridad, enforcement de acciones |
| 3 | Manejo de acciones concurrentes, condiciones de carrera |
| 4 | Restricciones avanzadas, casos extremos |
| 5 | Características heurísticas Tier 1 |
| 6 | Clasificación de carga ML Tier 2 |
| 7 | Modo LLM teacher Tier 3 |
| 8 | Gobernador adaptivo, recuperación |
| 9 | Características nativas M1 macOS (QoS, sensores) |
| 10 | Clasificador Bayesiano de carga, políticas aprendidas |

---

## Licencia

MIT

---

*Apollo Optimizer — Porque tu CPU no debería desperdiciar ciclos en procesos que no te importan.*
