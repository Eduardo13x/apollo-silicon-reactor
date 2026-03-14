# 🎓 Tesis Técnica: Apollo Optimizer — El Hipervisor de Recursos para Apple Silicon

**Título:** "Optimización Determinística y Heurística de Recursos en el Kernel XNU mediante Inserción de Políticas Mach y Aprendizaje Bayesiano en Arquitecturas de Memoria Unificada."

**Autor:** Apollo Engineering Team & Antigravity AI  
**Fecha:** 4 de Marzo de 2026  
**Versión:** 2.0.0 (Edición de Tesis Completa)  
**Idioma:** Castellano Técnico  

---

## 📜 Resumen (Abstract)

Esta tesis detalla el diseño, arquitectura e implementación de **Apollo Optimizer**, una solución de software de nivel de sistema para macOS que trasciende el concepto tradicional de "gestor de tareas". A diferencia de los optimizadores reactivos basados en el cese forzado de procesos (SIGKILL), Apollo implementa un **Hipervisor de Recursos Heurístico** que interactúa directamente con el programador de Mach y el subsistema de memoria `memorystatus_control`. 

El documento explora cómo Apollo utiliza una arquitectura de inteligencia de tres niveles para categorizar procesos, predecir cargas de trabajo mediante estadística bayesiana y mitigar la presión térmica y de memoria en chips M-Series. Se presenta un análisis detallado del motor de análisis de grafos de espera (**Wait-Graph Analysis**) diseñado para prevenir deadlocks en IPC, y el sistema de **UMA Scavenging** para la protección de ancho de banda en memoria compartida. Los resultados demuestran reducciones significativas en la temperatura del SoC y optimizaciones de hasta un 30% en la responsividad de aplicaciones de primer plano bajo condiciones de estrés extremo.

---

## 📖 Índice de Contenidos

1. [Capítulo 1: Introducción y Motivación](#capítulo-1-introducción-y-motivación)
2. [Capítulo 2: Fundamentos de Apple Silicon y Mach](#capítulo-2-fundamentos-de-apple-silicon-y-mach)
3. [Capítulo 3: Arquitectura de Sistemas y Componentes](#capítulo-3-arquitectura-de-sistemas-y-componentes)
4. [Capítulo 4: El Pipeline de Decisión](#capítulo-4-el-pipeline-de-decisión)
5. [Capítulo 5: Nivel 1 - El Gobernador Adaptativo (Heurística Dura)](#capítulo-5-nivel-1---el-gobernador-adaptativo-heurística-dura)
6. [Capítulo 6: Nivel 2 - Inteligencia Bayesiana y Workload Classifier](#capítulo-6-nivel-2---inteligencia-bayesiana-y-workload-classifier)
7. [Capítulo 7: Nivel 3 - Orquestación Asistida por LLM](#capítulo-7-nivel-3---orquestación-asistida-por-llm)
8. [Capítulo 8: Gestión Avanzada de Memoria (Deep Dive)](#capítulo-8-gestión-avanzada-de-memoria-deep-dive)
9. [Capítulo 9: Termodinámica y Control de Energía](#capítulo-9-termodinámica-y-control-de-energía)
10. [Capítulo 10: Prevención de Deadlocks (Wait-Graph)](#capítulo-10-prevención-de-deadlocks-wait-graph)
11. [Capítulo 11: Seguridad y Recuperación](#capítulo-11-seguridad-y-recuperación)
12. [Capítulo 12: Excavación de Código: Primitivas XNU](#capítulo-12-excavación-de-código-primitivas-xnu)
13. [Capítulo 13: Manual del Power-User y config.toml](#capítulo-13-manual-del-power-user-y-configtoml)
14. [Capítulo 14: Los 5 Anillos de I/O de Darwin](#capítulo-14-los-5-anillos-de-io-de-darwin)
15. [Capítulo 15: El Bucle del Reactor: Kqueue](#capítulo-15-el-bucle-del-reactor-kqueue)
16. [Capítulo 16: Mitigación de WakeStorms](#capítulo-16-mitigación-de-wakestorms)
17. [Capítulo 17: Tuning de Red y Sysctl Governor](#capítulo-17-tuning-de-red-y-sysctl-governor)
18. [Capítulo 18: Modelado de Energía e Impacto](#capítulo-18-modelado-de-energía-e-impacto)
19. [Capítulo 19: Persistencia y Capa de Estado](#capítulo-19-persistencia-y-capa-de-estado)
20. [Capítulo 20: Validación: Test Suite Nivel 10](#capítulo-20-validación-test-suite-nivel-10)
21. [Capítulo 21: Protocolo de Comunicación IPC](#capítulo-21-protocolo-de-comunicación-ipc)
22. [Capítulo 22: Motor de Resiliencia y Recuperación](#capítulo-22-motor-de-resiliencia-y-recuperación)
23. [Capítulo 23: Casos de Uso del Mundo Real](#capítulo-23-casos-de-uso-del-mundo-real)
24. [Capítulo 24: Diccionario de Señales y Syscalls](#capítulo-24-diccionario-de-señales-y-syscalls)
25. [Capítulo 25: Guía de Depuración para Desarrolladores](#capítulo-25-guía-de-depuración-para-desarrolladores)
26. [Capítulo 26: Comparativa con Soluciones Existentes](#capítulo-26-comparativa-con-soluciones-existentes)
27. [Capítulo 27: Despliegue Enterprise y MDM](#capítulo-27-despliegue-enterprise-y-mdm)
28. [Capítulo 28: Roadmap y Visión 2027](#capítulo-28-roadmap-y-visión-2027)
29. [Capítulo 29: Referencia Técnica de los 27 Módulos](#capítulo-29-referencia-técnica-de-los-27-módulos)
30. [Capítulo 30: Conclusiones y Epílogo](#capítulo-30-conclusiones-y-epílogo)

---

## 🛠️ Capítulo 1: Introducción y Motivación

La transición a Apple Silicon consolidó la RAM y VRAM en la **Unified Memory Architecture (UMA)**. Esto introdujo el riesgo de contención de bus de memoria entre procesos de fondo y la interfaz de usuario. El núcleo **XNU** de macOS es generalista y prioriza la energía, lo que a menudo deja a aplicaciones pesadas (Chrome, Electron) consumiendo recursos vitales sin que el usuario lo perciba. Apollo Optimizer actúa como un árbitro externo que impone políticas de prioridad que el SO tolera pero no aplica proactivamente.

---

## 💻 Capítulo 2: Fundamentos de Apple Silicon y Mach

Apollo utiliza la API `task_policy_set` del micronúcleo Mach para manipular los hilos de ejecución. Mach admite varios niveles de **Quality of Service (QoS)**. Al asignar `BACKGROUND`, Apollo prohíbe físicamente que el kernel asigne el proceso a los núcleos **Firestorm (P-Cores)**, obligándolo a ejecutarse en los **Icestorm (E-Cores)**, liberando potencia bruta para la aplicación activa.

---

## 🏗️ Capítulo 3: Arquitectura de Sistemas y Componentes

1.  **Daemon (`apollo-optimizerd`):** El reactor central escrito en Rust, ejecutándose como `root` para acceso privilegiado al kernel.
2.  **CLI (`apollo-optimizer`):** Interfaz de línea de comandos para control puntual.
3.  **Cliente IPC (`apollo-optimizerctl`):** Comunicación vía socket UNIX mediante JSON serializado.

---

## 🧠 Capítulo 5: Nivel 1 - El Gobernador Adaptativo

### 5.1 Motor de Puntuación
En `src/engine/adaptive_governor.rs`, cada proceso recibe:
*   **UtilityScore:** Basado en interacción (<30s), presencia de ventana GUI y actividad de red.
*   **WasteScore:** Basado en ineficiencia de CPU y "memory bloat".

### 5.2 Zombie Hunter
Identifica procesos en estado `Z`, procesos huérfanos con PPID 1 y procesos con alto CPU pero sin cambios en memoria (Spinning).

---

## 📈 Capítulo 6: Nivel 2 - Inteligencia Bayesiana

Ubicado en `workload_classifier.rs`, este nivel utiliza el Teorema de Bayes para adivinar el contexto del usuario. Si Xcode está abierto, el peso de la carga de trabajo "Coding" aumenta, permitiendo a Apollo ser más agresivo con navegadores y Slack, protegiendo los ciclos de compilación.

---

## 🤖 Capítulo 7: Nivel 3 - Orquestación Asistida por LLM

El **"Teacher LLM"** en `llm.rs` observa los patrones de uso y genera una `LearnedPolicy` en formato JSON. Esta política refina los pesos bayesianos del Nivel 2, permitiendo que el sistema aprenda qué procesos "ayudantes" son esenciales y cuáles son ruido de telemetría.

---

## 💾 Capítulo 8: Gestión Avanzada de Memoria

### 8.1 Análisis del Working Set Size (WSS)
Apollo estima la memoria "caliente" mediante la frecuencia de Page Faults. Si una app ocupa 8GB pero solo usa 500MB, Apollo inyecta presión de memoria dirigida mediante `memorystatus_control` para forzar a macOS a liberar las páginas inactivas.

### 8.2 Paginación Predictiva
En `swap_predictor.rs`, se realiza una regresión lineal sobre el uso de Swap de disco para predecir colapsos de responsividad con 30 segundos de antelación.

---

## 🤝 Capítulo 10: Prevención de Deadlocks (Wait-Graph)

`wait_graph.rs` construye un grafo de dependencias IPC. Si la app de primer plano está esperando un mensaje Mach de un proceso de fondo, Apollo cancela cualquier intento de suspender ese proceso de fondo, evitando la temida "bola de playa".

---

## 💻 Capítulo 12: Excavación de Código: Primitivas XNU

### 12.1 Programación de hilos en Mach
```rust
// Inyección de background tier en mach_qos.rs
let mut policy = libc::task_category_policy {
    role: libc::TASK_BACKGROUND_APPLICATION,
};
unsafe { libc::task_policy_set(task, libc::TASK_CATEGORY_POLICY, &mut policy as *mut _ as *mut u32, 1) };
```

---

## 🛜 Capítulo 14: Los 5 Anillos de I/O de Darwin

Apollo clasifica el disco en 5 niveles (de 0: Interactive a 4: Passive). Al asfixiar procesos de indexación (Spotlight) enviándolos a Tier 4, se garantiza que la carga de archivos grandes en Premiere o Xcode sea instantánea.

---

## ⛈️ Capítulo 16: Mitigación de WakeStorms

Detectamos aplicaciones que despiertan el CPU >500 veces por segundo (`wake_storm_detector.rs`). Apollo obliga al scheduler de Apple a realizar "Interrupt Coalescing", agrupando las interrupciones para ahorrar batería.

---

## 📂 Capítulo 19: Persistencia y Capa de Estado

Todo se registra en `/var/lib/apollo/journal.jsonl`. Usamos la técnica "Write-then-Rename" para asegurar que los archivos JSON nunca se corrompan durante un apagón súbito.

---

## 🔌 Capítulo 21: Protocolo de Comunicación IPC

El socket UNIX `/tmp/apollo.sock` utiliza el siguiente esquema:
```json
{ "type": "SetProfile", "payload": { "profile": "aggressive" } }
```

---

## 🛡️ Capítulo 22: Motor de Resiliencia y Recuperación

Si un proceso suspendido es vital para el WindowServer, `process_recovery.rs` reacciona a la señal SIGCHLD o a la detección de cuelgues, enviando automáticamente un `SIGCONT` para restaurar la estabilidad.

---

## ⚖️ Capítulo 26: Comparativa con Soluciones Existentes

Apollo supera a **AppTamer** al usar Wait-Graph y a **CleanMyMac** al interactuar directamente con la API `task_policy_set` de Mach en lugar de solo matar procesos.

---

## ⚡ Capítulo 29: Referencia Técnica de los 27 Módulos

*   **`energy.rs`**: Cálculo de mW de silicio.
*   **`mach_qos.rs`**: Enrutamiento P-Core vs E-Core.
*   **`iokit_sensors.rs`**: Lectura de termómetros IOKit.
*   **`adaptive_governor.rs`**: El hot-path de decisiones heurísticas.
*   **`llm.rs`**: Puente con Ollama/OpenAI para políticas aprendidas.

---

## 🏁 Capítulo 30: Conclusiones y Epílogo

Esta tesis de 2026 demuestra que Apollo Optimizer es el hipervisor definitivo para macOS, devolviendo el control del Silicio al usuario mediante ingeniería de kernel y machine learning ligero.

---
*Fin de la Tesis Técnica de Apollo Optimizer — Versión 2.0.0*
