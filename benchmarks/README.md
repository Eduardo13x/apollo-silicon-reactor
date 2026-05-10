# 🍏 Apollo System Optimizer - Benchmark Suite

Este directorio contiene una suite de pruebas de rendimiento estandarizadas para evaluar el impacto de **Apollo Optimizer** en hardware de Apple (Apple Silicon).

## 🛠 Herramientas Incluidas

### 1. CoreMark (EEMBC Standard)
Ubicación: `benchmarks/coremark`
*   **¿Qué es?**: El estándar de la industria para medir el rendimiento de un solo núcleo y multi-núcleo de un CPU.
*   **Uso**: Genera una carga de trabajo pura de CPU para verificar si el optimizador mejora la eficiencia o reduce el estrangulamiento térmico (throttling).

### 2. Apple Native Metrics (powermetrics)
Ubicación: `benchmarks/run_apple_test.sh`
*   **¿Qué es?**: La herramienta oficial de Apple para telemetría de bajo nivel.
*   **Métricas**: Consumo de energía (mW), frecuencia de los núcleos (P-cores y E-cores), temperatura y uso de GPU.

## 🚀 Cómo ejecutar las pruebas

Para preparar el entorno (si no lo has hecho):
```bash
bash scripts/setup_apple_benchmark.sh
```

Para ejecutar el benchmark oficial de Apple Silicon:
```bash
./benchmarks/run_apple_test.sh
```

## 📊 Interpretación de Resultados

- **`coremark_results.txt`**: Verás un puntaje final (CoreMark/MHz). Compara este puntaje con y sin Apollo activado.
- **`apple_silicon_metrics.txt`**: Este archivo contiene el historial de lo que pasó dentro del procesador. Busca la sección `Thermal Pressure` para ver si Apollo evitó que el sistema bajara la velocidad por calor.

---

## 🔬 Otras Herramientas de Apple recomendadas

Si quieres ir más allá, puedes usar las herramientas de perfilado que vienen con Xcode:

1. **`xctrace`**: Versión CLI de Instruments.
   ```bash
   xctrace record --template "System Trace" --output my_trace.trace --attach apollo-optimizerd
   ```
2. **`os_signpost`**: (Usado internamente en el código) para marcar eventos específicos en el Timeline de Instruments.
