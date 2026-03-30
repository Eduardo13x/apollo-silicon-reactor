#!/usr/bin/env python3
"""Simulate causal graph learning with real Apollo data.
Run: sudo python3 scripts/causal_sim.py
"""
import json, subprocess, time

def get_status():
    out = subprocess.check_output(["/usr/local/bin/apollo-optimizerctl", "status"], text=True)
    return json.loads(out)

# Snapshot current state
d = get_status()
m = d.get("metrics", {})

print("=== Apollo Causal Graph Simulation ===")
print(f"Cycles: {m.get('cycles', 0)}")
print(f"Pressure: {m.get('memory_pressure', 0):.3f}")
print(f"p95: {m.get('p95_cycle_ms', 0)}ms")
print(f"Throttles applied: {m.get('throttles_applied', 0)}")
print(f"Freezes applied: {m.get('freezes_applied', 0)}")
print()

# Read last actions summary
summary = m.get("last_actions_summary", "")
print(f"Last actions: {summary}")
print()

# Check HRPO groups
print("=== HRPO Groups ===")
hrpo = d.get("hrpo_groups", {})
if not hrpo:
    # Try from file
    try:
        with open("/var/lib/apollo/hrpo_groups.json") as f:
            hrpo = json.load(f)
    except:
        pass

for name, g in sorted(hrpo.items(), key=lambda x: x[1].get("throttle_count", 0), reverse=True):
    tc = g.get("throttle_count", 0)
    eff = g.get("effective_count", 0) / max(tc, 1)
    print(f"  {name}: throttled={tc} effectiveness={eff:.1%}")

print()

# Read frozen state
try:
    with open("/var/lib/apollo/frozen_state.json") as f:
        frozen = json.load(f)
    print(f"=== Frozen ({len(frozen)} processes) ===")
    for pid, info in list(frozen.items())[:10]:
        print(f"  PID {pid}: frozen_at={info.get('frozen_at', '?')}")
except:
    print("=== Frozen: could not read ===")

print()

# Simulate causal learning with 5 snapshots 10s apart
print("=== Live Causal Observation (5 samples, 10s apart) ===")
samples = []
for i in range(5):
    d = get_status()
    m = d.get("metrics", {})
    pressure = m.get("memory_pressure", 0)
    throttles = m.get("throttles_applied", 0)
    cycles = m.get("cycles", 0)
    p95 = m.get("p95_cycle_ms", 0)
    samples.append({"pressure": pressure, "throttles": throttles, "cycles": cycles, "p95": p95})
    print(f"  [{i+1}/5] cycle={cycles} pressure={pressure:.3f} throttles={throttles} p95={p95}ms")
    if i < 4:
        time.sleep(10)

# Analyze deltas
print()
print("=== Pressure Deltas ===")
for i in range(1, len(samples)):
    dp = samples[i]["pressure"] - samples[i-1]["pressure"]
    dt = samples[i]["throttles"] - samples[i-1]["throttles"]
    dc = samples[i]["cycles"] - samples[i-1]["cycles"]
    arrow = "v" if dp < -0.01 else ("^" if dp > 0.01 else "=")
    print(f"  {arrow} delta_pressure={dp:+.3f} new_throttles={dt} cycles={dc}")

# Estimate how many causal edges would form
total_throttles = samples[-1]["throttles"]
print()
print(f"=== Causal Graph Estimate ===")
print(f"Total throttles since boot: {total_throttles}")
print(f"Estimated unique (action, process) pairs: ~{min(total_throttles, 200)}")
print(f"Edges needing >=5 obs to be actionable: ~{min(total_throttles // 5, 100)}")
print(f"At current rate, graph becomes useful in: ~{max(0, 500 - samples[-1]['cycles'])} more cycles")
