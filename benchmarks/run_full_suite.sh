#!/usr/bin/env bash
# Apollo Optimizer — Comprehensive benchmark suite
# Apple Silicon M1 8GB, runs both CON Apollo and SIN Apollo, generates
# comparison data suitable for README / blog / portfolio.

set -e

MODE="${1:-con-apollo}"
RESULTS_DIR="${2:-results-$MODE}"
mkdir -p "$RESULTS_DIR"

echo "================================================="
echo "Apollo Benchmark Suite — mode: $MODE"
echo "Results: $RESULTS_DIR"
echo "================================================="

# Hardware info (one-time per mode for reproducibility)
{
    echo "=== Hardware ==="
    sysctl -n machdep.cpu.brand_string
    sysctl -n hw.physicalcpu hw.logicalcpu
    sysctl -n hw.memsize | awk '{printf "RAM: %.1f GB\n", $1/1024/1024/1024}'
    sw_vers
    date -u +"%Y-%m-%dT%H:%M:%SZ"
} > "$RESULTS_DIR/hardware.txt"

# Apollo state
{
    echo "=== Apollo state ==="
    sudo apollo-optimizerctl is-paused 2>&1 || echo "is-paused command failed"
    sudo cat /var/lib/apollo/runtime_metrics.json 2>/dev/null | python3 -c "
import json, sys
m = json.load(sys.stdin)
print(f'cycles={m.get(\"cycles\")} pressure={m.get(\"memory_pressure\"):.2f} swap_gb={m.get(\"swap_used_bytes\",0)/1e9:.1f} failures={m.get(\"failures\")}')" 2>/dev/null || echo "metrics unreadable"
} > "$RESULTS_DIR/apollo_state.txt"

# Capture initial vm_stat (baseline)
vm_stat > "$RESULTS_DIR/vm_stat_pre.txt"

# ---------------------------------------------------
echo "[1/7] CoreMark (single-thread CPU)..."
cd coremark
make run > /dev/null 2>&1 || true
cp run1.log "../$RESULTS_DIR/coremark_perf.log"
cp run2.log "../$RESULTS_DIR/coremark_validation.log"
cd ..

# ---------------------------------------------------
echo "[2/7] sysbench CPU multi-thread (8 threads, 30s)..."
sysbench cpu --threads=8 --time=30 --cpu-max-prime=20000 run > "$RESULTS_DIR/sysbench_cpu.txt" 2>&1

# ---------------------------------------------------
echo "[3/7] sysbench memory (read+write, 1MB block, 4 threads)..."
sysbench memory --threads=4 --time=20 --memory-block-size=1M --memory-total-size=10G --memory-oper=read run > "$RESULTS_DIR/sysbench_mem_read.txt" 2>&1
sysbench memory --threads=4 --time=20 --memory-block-size=1M --memory-total-size=10G --memory-oper=write run > "$RESULTS_DIR/sysbench_mem_write.txt" 2>&1

# ---------------------------------------------------
echo "[4/7] fio disk I/O (sequential + random read, 30s each)..."
mkdir -p /tmp/apollo-bench-fio
fio --name=seq_read --filename=/tmp/apollo-bench-fio/test --size=512M --bs=1M --rw=read --runtime=20 --time_based --output-format=json > "$RESULTS_DIR/fio_seq_read.json" 2>&1 || true
fio --name=rand_read --filename=/tmp/apollo-bench-fio/test --size=512M --bs=4K --rw=randread --runtime=20 --time_based --output-format=json > "$RESULTS_DIR/fio_rand_read.json" 2>&1 || true
rm -rf /tmp/apollo-bench-fio

# ---------------------------------------------------
echo "[5/7] Sustained power (60s sysbench cpu --threads=8)..."
# Run sysbench in background, sample powermetrics 60 times
sysbench cpu --threads=8 --time=60 --cpu-max-prime=20000 run > "$RESULTS_DIR/sustained_cpu_score.txt" 2>&1 &
SYSBENCH_PID=$!
sleep 2
sudo powermetrics --samplers cpu_power,thermal -n 30 -i 2000 > "$RESULTS_DIR/sustained_power.txt" 2>&1 &
POWER_PID=$!
wait $SYSBENCH_PID
sudo kill $POWER_PID 2>/dev/null || true

# ---------------------------------------------------
echo "[6/7] Memory pressure response (stress-ng vm 4G for 20s)..."
# Capture vm_stat before, during, after
vm_stat > "$RESULTS_DIR/mem_pressure_pre.txt"
stress-ng --vm 1 --vm-bytes 4G --timeout 20s > "$RESULTS_DIR/stress_ng_mem.txt" 2>&1 &
STRESS_PID=$!
sleep 5
vm_stat > "$RESULTS_DIR/mem_pressure_during.txt"
sudo cat /var/lib/apollo/runtime_metrics.json 2>/dev/null | python3 -c "
import json, sys
m = json.load(sys.stdin)
print(f'pressure={m.get(\"memory_pressure\"):.3f} swap_gb={m.get(\"swap_used_bytes\",0)/1e9:.2f} swap_delta_bps={m.get(\"swap_delta_bps\",0):.0f}')" > "$RESULTS_DIR/apollo_during_mem_pressure.txt" 2>/dev/null
wait $STRESS_PID
sleep 5
vm_stat > "$RESULTS_DIR/mem_pressure_post.txt"

# ---------------------------------------------------
echo "[7/7] Mixed workload (CPU + IO + memory simultaneously, 30s)..."
sudo powermetrics --samplers cpu_power,thermal -n 15 -i 2000 > "$RESULTS_DIR/mixed_workload_power.txt" 2>&1 &
POWER_PID=$!
stress-ng --cpu 4 --io 2 --vm 1 --vm-bytes 1G --timeout 30s > "$RESULTS_DIR/mixed_workload_stress.txt" 2>&1
sudo kill $POWER_PID 2>/dev/null || true

# Final apollo state
sudo cat /var/lib/apollo/runtime_metrics.json 2>/dev/null | python3 -c "
import json, sys
m = json.load(sys.stdin)
print(f'cycles={m.get(\"cycles\")} pressure={m.get(\"memory_pressure\"):.2f} swap_gb={m.get(\"swap_used_bytes\",0)/1e9:.1f} failures={m.get(\"failures\")}')" > "$RESULTS_DIR/apollo_state_final.txt" 2>/dev/null || true

echo ""
echo "================================================="
echo "✅ Suite complete — $RESULTS_DIR"
echo "================================================="
ls -la "$RESULTS_DIR/"
