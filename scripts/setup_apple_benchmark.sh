#!/usr/bin/env bash

# Apple Silicon Benchmark Suite for Apollo Optimizer
# Utiliza herramientas nativas de Apple (powermetrics) y CoreMark (estándar de la industria)

set -e

BENCHMARK_DIR="$(pwd)/benchmarks"
COREMARK_DIR="$BENCHMARK_DIR/coremark"

mkdir -p "$BENCHMARK_DIR"

echo "========================================================="
echo "🍏 Preparando Apollo Apple Silicon Benchmark Suite"
echo "========================================================="

# 1. Descargar CoreMark si no existe
if [ ! -d "$COREMARK_DIR" ]; then
    echo "📥 Descargando CoreMark (EEMBC Standard CPU Benchmark)..."
    git clone https://github.com/eembc/coremark.git "$COREMARK_DIR"
else
    echo "✅ CoreMark ya está descargado."
fi

# 2. Compilar CoreMark
echo "🔨 Compilando CoreMark para arquitectura nativa..."
cd "$COREMARK_DIR"
make compile > /dev/null
cd ../../

# 3. Script de ejecución
cat << 'EOF' > "$BENCHMARK_DIR/run_apple_test.sh"
#!/usr/bin/env bash

echo "🚀 Iniciando prueba de carga con powermetrics..."
echo "⚠️  Se requiere SUDO para leer métricas de hardware de Apple (powermetrics)."

# Ejecutar powermetrics en segundo plano para recopilar datos de CPU, GPU y temperatura
sudo powermetrics --samplers cpu_power,thermal -n 10 -i 1000 > apple_silicon_metrics.txt &
POWERMETRICS_PID=$!

echo "🏃 Ejecutando CoreMark (estrés de CPU)..."
cd coremark
make run > ../coremark_results.txt

echo "🛑 Deteniendo métricas..."
sudo kill $POWERMETRICS_PID

echo "✅ Prueba finalizada."
echo "📊 Resultados de rendimiento guardados en: benchmarks/coremark_results.txt"
echo "🌡️ Métricas de hardware (Apple) guardadas en: benchmarks/apple_silicon_metrics.txt"
EOF

chmod +x "$BENCHMARK_DIR/run_apple_test.sh"

echo "✅ Listo. Puedes ejecutar la prueba corriendo: ./benchmarks/run_apple_test.sh"
