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
