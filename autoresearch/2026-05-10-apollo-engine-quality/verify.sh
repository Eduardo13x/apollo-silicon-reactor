#!/usr/bin/env bash
# Autoresearch verify for apollo-optimizer. Higher score = better.
# Composite: prod p95/failures/errors + apollo-engine LoC complexity penalty.
#
# Workflow:
# 1. Build daemon release
# 2. Deploy via bootout/bootstrap (preserves /usr/local/libexec/apollo-optimizerd.bak.*)
# 3. Soak 60s for daemon to accumulate cycles
# 4. Read /var/lib/apollo/runtime_metrics.json
# 5. Emit single integer score on stdout
#
# If build/deploy fails OR daemon crashes → score 0 (forces revert).

set -e

SOAK_SECS="${1:-60}"
LOG_FILE="autoresearch/2026-05-10-apollo-engine-quality/iter.log"
echo "=== verify start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG_FILE"

# Step 1: build
if ! cargo build --release --bin apollo-optimizerd >> "$LOG_FILE" 2>&1; then
    echo "BUILD_FAIL" >> "$LOG_FILE"
    echo "0"
    exit 0
fi

# Step 2: deploy
if ! sudo cp target/release/apollo-optimizerd /usr/local/libexec/apollo-optimizerd 2>> "$LOG_FILE"; then
    echo "DEPLOY_CP_FAIL" >> "$LOG_FILE"
    echo "0"
    exit 0
fi

if ! sudo launchctl bootout system/com.eduardocortez.systemoptimizerd 2>> "$LOG_FILE"; then
    echo "BOOTOUT_FAIL" >> "$LOG_FILE"
fi
sleep 2
if ! sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist 2>> "$LOG_FILE"; then
    echo "BOOTSTRAP_FAIL" >> "$LOG_FILE"
    echo "0"
    exit 0
fi

# Step 3: soak
sleep "$SOAK_SECS"

# Step 4: read prod metrics
METRICS=$(sudo cat /var/lib/apollo/runtime_metrics.json 2>/dev/null)
if [ -z "$METRICS" ]; then
    echo "METRICS_UNREADABLE" >> "$LOG_FILE"
    echo "0"
    exit 0
fi

P95=$(echo "$METRICS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('p95_cycle_ms', 9999))")
CYCLES=$(echo "$METRICS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('cycles', 0))")
FAILURES=$(echo "$METRICS" | python3 -c "import json,sys;print(json.load(sys.stdin).get('failures', 9999))")
LAST_ERR=$(echo "$METRICS" | python3 -c "import json,sys;m=json.load(sys.stdin);print(0 if m.get('last_error') is None else 1)")

# Crash detection: if cycles=0 after 60s soak, daemon never started.
if [ "$CYCLES" -lt 5 ]; then
    echo "DAEMON_CRASH cycles=$CYCLES" >> "$LOG_FILE"
    echo "0"
    exit 0
fi

# Step 5: LoC complexity
LOC=$(find crates/apollo-engine/src -name "*.rs" -not -path "*/target/*" -exec cat {} + | wc -l | awk '{print $1}')

# Composite score (higher = better):
#  base 50000 — anchor
#  -p95*10  : 100ms p95 → -1000
#  -failures*1000: 1 failure → -1000
#  -last_err*5000: any error → -5000
#  -LoC/100  : 110000 LoC → -1100
SCORE=$(python3 -c "print(int(50000 - $P95 * 10 - $FAILURES * 1000 - $LAST_ERR * 5000 - $LOC / 100))")
echo "score=$SCORE p95=$P95 cycles=$CYCLES failures=$FAILURES last_err=$LAST_ERR loc=$LOC" >> "$LOG_FILE"
echo "$SCORE"
