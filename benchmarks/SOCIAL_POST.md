# Construí un optimizador de sistema para macOS Apple Silicon en Rust. Aquí están los benchmarks honestos.

**TL;DR**: en un MacBook Air M1 con 8 GB de RAM, mi optimizador entrega **+18% throughput multi-thread**, **+128% IOPS aleatorio en disco**, y **−27% uso de swap bajo presión de memoria**, sin sacrificar performance single-thread. El costo es +30% más consumo de power durante sustained CPU stress (pero también +18% más trabajo terminado).

---

## El problema que estoy resolviendo

Tengo un MacBook Air M1 con 8 GB de RAM. Es la base del lineup Apple Silicon: poderoso pero ajustado en memoria. Con Brave abierto con docenas de pestañas, Cursor corriendo, Spotify reproduciendo música, y ocasional ollama haciendo inference local, el sistema empieza a sentirse pegajoso. Beachballs en Finder, lag al cambiar de pestaña, builds de cargo que tardan más de lo esperado.

macOS hace su propio housekeeping (compressor, jetsam, App Nap), pero está optimizado para máquinas mainstream. En M1 8GB con workloads de developer, hay margen de mejora.

Construí Apollo Optimizer: un daemon de Rust que se ejecuta como root y aplica optimizaciones específicas:

- Throttling de procesos background con SIGSTOP/SIGCONT (analyticsd, cloudphotod, mds_stores, etc.)
- Tuning de 16 sysctls del kernel (TCP buffers, vnode cache, compressor)
- QoS routing per-thread entre P-cluster y E-cluster
- LLM teacher local (Gemma 4) para policy learning adaptativo
- 8 subsistemas predictivos (Kalman, CUSUM, Hazard model, Holt-Winters, MPC, Markov)
- 3 lazos de aprendizaje (RL Q-table, Bayesian outcome tracker, Causal graph)

Es ~110K líneas de Rust, edición 2021, deployed via launchd como root daemon.

## Los benchmarks

Para validar que Apollo realmente entrega valor, construí un suite comprehensive que mide siete dimensiones:

1. CoreMark single-thread (CPU integer arithmetic standard)
2. sysbench CPU multi-thread (8 hilos sobre 8 cores)
3. sysbench memory bandwidth (read + write)
4. fio disk I/O (sequential 1MB + random 4KB)
5. stress-ng memory pressure (4GB allocation)
6. powermetrics sustained power (60s sample window)
7. mixed workload (CPU + IO + VM concurrent)

Cada test corre dos veces: una con Apollo activo, otra con Apollo pausado vía kill switch (`/var/run/apollo.disable`). Los resultados se comparan punto a punto.

## Resultados honestos

| Categoría | Test específico | Apollo CON | Apollo SIN | Δ |
|---|---|---|---|---|
| Multi-thread CPU | sysbench 8t prime-20K events/s | **60.7M** | 51.4M | **+18.1%** |
| Multi-thread CPU | sysbench 8t total events/30s | 1.82B | 1.54B | +279M |
| Disk I/O | fio random 4KB IOPS | **38,450** | 16,704 | **+130%** |
| Disk I/O | fio random 4KB throughput | **153 MB/s** | 67 MB/s | **+128%** |
| Disk I/O | fio random 4KB latency mean | **26 µs** | 59 µs | **−56%** |
| Disk I/O | fio sequential 1MB throughput | 3,017 MB/s | 2,944 MB/s | +2.5% |
| Memory pressure | swap usage post 4GB stress | **2.97 GB** | 4.09 GB | **−27%** (1.12 GB menos) |
| Memory bandwidth | sysbench read MiB/s | 61,669 | 59,769 | +3.2% |
| Memory bandwidth | sysbench write MiB/s | 61,112 | 59,328 | +3.0% |
| Single-thread CPU | CoreMark avg iter/s | 29,787 | 28,800 | +3.4% (within noise) |
| Sustained power | 60s 8-thread avg combined | 10,067 mW | 7,753 mW | +30% |
| Sustained power | GPU power avg | **42 mW** | 334 mW | **−87%** |
| Mixed workload | 30s completion | 30.29 s | 30.41 s | identical |

**Lo que me sorprendió más**: el random disk I/O. +130% IOPS, latencia cortada a la mitad. Eso no era el target original del proyecto; surgió como side effect de fixes de I/O cleanup (Apollo estaba escribiendo audit logs muy seguido y excediendo el límite de Resource Coalition de macOS, lo que hacía que macOS throttleara el SSD del daemon, y ese throttle se filtraba a procesos del usuario).

**Lo que NO sorprendió**: single-thread CPU es neutral. Apollo no acelera el silicon de M1, solo libera contención. Cualquier optimizador que afirme acelerar single-thread CoreMark sin tocar el código está mintiendo o midiendo mal.

**El trade-off honesto**: Apollo consume +30% más power durante sustained 8-thread CPU stress. Pero también termina +18% más trabajo en el mismo tiempo. Si optimizas por wall clock (terminar el build más rápido), gana Apollo. Si optimizas por battery life en idle, pausa Apollo durante esa sesión específica.

## Por qué esto importa

Para un usuario M1 8GB con workload de desarrollo:

- Build de Rust de 10 minutos pasa a ~8.5 minutos (+18% multi-thread CPU)
- Tab switching en Brave se siente notoriamente más rápido (+130% IOPS aleatorio en disco)
- Bajo presión de memoria, menos swap = menos SSD wear acumulado

Para un usuario M1 8GB con workload light:

- Single-thread CPU igual (Apollo neutral)
- Idle power similar (Apollo no impacta cuando no hay nada que optimizar)

Para un caso adversarial (sustained ML training de horas en batería):

- Apollo cuesta más power. Pausarlo durante esa sesión específica es la opción correcta.

## El proceso de construcción

Apollo no es código que escribí en un weekend. Tiene un historial de aproximadamente 9 meses con 700+ commits, decisiones que documenté con citas académicas (Hellerstein 2004 para feedback control, Cox 1972 para hazard models, Pei Wang 2013 para NARS belief revision), y un peer-review continuo con NotebookLM como adversarial reviewer.

Hubo bugs gloriosos en el camino. Mi favorito: descubrí hace dos días que Apollo apagaba Spotlight automáticamente cuando detectaba LLM inference activo, lo cual causaba beachballs en Finder porque el off→on edge invalidaba el cache de metadata. Lo arreglé removing toggling automático.

Otro: durante el Sprint 4 reciente, NotebookLM detectó que mi ActionAccumulator commit había olvidado wirear los counters al runtime metrics, los datos atómicos se incrementaban pero nunca llegaban al JSON. Sprint 3 había repetido el mismo bug class. Esa lección ahora está en CLAUDE.md como regla obligatoria de telemetry sync chain audit.

El día del benchmark coincidió con el cierre de Sprint 5 Mes 0, donde split-eé el monolito de 110K líneas en un Cargo workspace con un crate `apollo-engine` separado de los binarios. El cargo test pasó de 20 minutos a 8.4 segundos en cold build (+143× speedup), lo cual desbloquea iteración futura.

## Caveats explícitos

- M1 8GB únicamente. M2/M3/M4 con más RAM y más cores tendría perfil distinto.
- Workload mío específico (Brave + Cursor + Spotify + occasional ollama). Tu workload puede dar números distintos.
- Sample windows de 20-30 segundos por test, 60 segundos para sustained. Tests de horas podrían surgir thermal throttling diferenciado.
- No medí real-world latencia (jank perception, page load times). Eso requeriría harness de aplicaciones reales.
- No medí battery life real over 8 horas. Solo power instantáneo.

## Lo que esto NO es

Apollo no es un Geekbench killer. No es un competidor de turbo modes hardware. No es magia.

Apollo es **ingeniería específica para un caso específico**: máquinas Apple Silicon con RAM constrained, workload de developer, que sufren contención entre el trabajo activo del usuario y los daemons de macOS. Para ese caso, los números muestran beneficio medible. Para otros casos, Apollo es neutral o irrelevante.

## Reproducibilidad

```bash
git clone https://github.com/<your-handle>/apollo-optimizer
cd apollo-optimizer
brew install sysbench stress-ng fio jq

# Setup CoreMark
./scripts/setup_apple_benchmark.sh

# Build + install Apollo
cargo build --release
./scripts/install-root-daemon.sh

# Run benchmark
./benchmarks/run_full_suite.sh con-apollo results-con-apollo
sudo touch /var/run/apollo.disable
./benchmarks/run_full_suite.sh no-apollo results-no-apollo
sudo rm /var/run/apollo.disable
```

Toda la data cruda y los scripts están en el repo. Los logs de fio son JSON. Los logs de powermetrics tienen timestamp por sample. Cualquiera con un M1 puede correr los mismos tests y comparar.

## Lo que aprendí construyendo esto

1. **Optimizadores adaptativos son siempre trade-offs.** No existe "optimizar todo simultáneamente". Apollo gana en multi-thread + memory + disk a costa de power en sustained CPU. Aceptar que cada decisión técnica tiene un costo es el primer paso para diseñar honestamente.

2. **Los bugs más caros son invisibles.** El bug del audit log saturando macOS Resource Coalition vivió tres días antes de que lo notara, manifestándose solo como "el sistema se siente raro a veces". Solo lo encontré porque insistí en verificación mecánica con powermetrics y diag reports.

3. **Adversarial review tiene precio real.** Cada vez que un reviewer (humano o AI) encuentra un bug en código que pasó self-review, eso es N horas que NO gasté debuggeando en producción. Vale el costo.

4. **Apple Silicon es asombroso.** M1 con 8 GB de RAM puede aguantar workloads que harían sufrir a máquinas Intel con 16-32 GB. La pregunta es: ¿cómo extraer ese performance latente bajo restricciones de RAM? Eso es Apollo.

---

Si tenés un M1/M2 con 8 GB y querés probar: el repo está abierto. Si tenés feedback técnico sobre el diseño o los benchmarks, abre un issue. Si pensás que esto es over-engineering para un side project: probablemente tenés razón, pero también construí un Mac que se siente medibleemente más rápido y eficiente, y aprendí más sobre kernel de macOS y sistemas adaptativos que en cualquier curso.

**Repo**: github.com/&lt;your-handle&gt;/apollo-optimizer
**Resultados completos**: `benchmarks/RESULTS.md`
**Suite de benchmark**: `benchmarks/run_full_suite.sh`

Stack: Rust 2021 edition, sysinfo, libc, mach APIs, IOReport, SMC direct reads, kqueue, sysctl, mdutil. Sin shell injection, sin unsafe global state, mutex poisoning recovery por convención, write-then-rename para crash safety, persistencia atómica via JSON.
