#!/bin/bash
# Purge false-positive overflow events (pressure < 0.60) and reset RL state.
# Run with: sudo bash scripts/purge-false-overflows.sh

set -e

OVERFLOW_FILE="/var/lib/apollo/overflow_history.json"
RL_FILE="/var/lib/apollo/rl_threshold_state.json"

echo "=== Apollo Overflow Purge ==="

python3 << 'PYEOF'
import json

with open("/var/lib/apollo/overflow_history.json") as f:
    data = json.load(f)

before = len(data["events"])
data["events"] = [e for e in data["events"] if e["memory_pressure"] >= 0.60]
after = len(data["events"])

data["threshold_offset"] = max(-0.20, -0.05 * after)

print(f"Events: {before} -> {after} (purged {before - after} false positives)")
print(f"Remaining pressures: {[round(e['memory_pressure'], 2) for e in data['events']]}")
print(f"New threshold_offset: {data['threshold_offset']}")

with open("/var/lib/apollo/overflow_history.json", "w") as f:
    json.dump(data, f, indent=2)

print("overflow_history.json updated")
PYEOF

if [ -f "$RL_FILE" ]; then
    rm "$RL_FILE"
    echo "RL state reset (was trained on false positives)"
else
    echo "No RL state file found (already clean)"
fi

echo ""
echo "=== Done. Changes take effect on next daemon cycle (no restart needed). ==="
