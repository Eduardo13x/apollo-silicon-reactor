#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Apollo AutoResearch — Production Stability Validator
# ══════════════════════════════════════════════════════════════════════════════
# Usage: sudo ./scripts/autoresearch-stability.sh [minutes]
#
# Rewards REAL production metrics, not simulations:
#   - STABILITY:  cycles keep incrementing without hangs
#   - EFFICIENCY: cycle time stays under budget (150ms)
#   - FLUIDITY:   no blocking calls, ctl responds
#   - INTELLIGENCE: actions are being applied (freezes, throttles, boosts)
#
# Relies on watch-deploy.sh being active for the deploy step.
# Output: /tmp/apollo-autoresearch-report.txt
set -uo pipefail

REPORT="/tmp/apollo-autoresearch-report.txt"
CTL="/usr/local/bin/apollo-optimizerctl"
DURATION_MIN="${1:-3}"  # default 3 minutes of observation
POLL_INTERVAL=5         # seconds between polls

echo "" > "$REPORT"
chmod 644 "$REPORT"

r() { echo "$1" >> "$REPORT"; }

r "══════════════════════════════════════════════════════════════════"
r "  AUTORESEARCH — Production Stability Report"
r "  $(date '+%Y-%m-%d %H:%M:%S')  |  Observing ${DURATION_MIN}m"
r "══════════════════════════════════════════════════════════════════"
r ""

# ── Step 1: Trigger rebuild via watcher ──────────────────────────────────
if [ -f /tmp/apollo-trigger ] || [ -f /tmp/apollo-pipeline-report.txt ]; then
    r "[deploy] Triggering rebuild..."
    touch /tmp/apollo-trigger
    # Wait for watcher to pick up and complete
    for i in $(seq 1 60); do
        if grep -q "STATUS: DONE" /tmp/apollo-pipeline-report.txt 2>/dev/null; then
            r "[deploy] Build + deploy complete."
            break
        fi
        if grep -q "STATUS: BUILD_FAILED" /tmp/apollo-pipeline-report.txt 2>/dev/null; then
            r "[deploy] BUILD FAILED — aborting."
            r ""
            r "SCORE: 0"
            r "VERDICT: FAIL (build error)"
            exit 1
        fi
        sleep 2
    done
else
    r "[deploy] No watcher detected, assuming daemon already running."
fi

sleep 3

# ── Step 2: Verify daemon alive ──────────────────────────────────────────
PID=$(pgrep -x apollo-optimizerd 2>/dev/null | head -1)
if [ -z "$PID" ]; then
    r "[verify] DAEMON NOT RUNNING"
    r ""
    r "SCORE: 0"
    r "VERDICT: FAIL (daemon dead)"
    exit 1
fi
r "[verify] Daemon alive PID=$PID"
r ""

# ── Step 3: Observe production cycles ────────────────────────────────────
r "── OBSERVATION (${DURATION_MIN} minutes) ──"

CYCLES_START=""
CYCLES_END=""
CYCLE_TIMES=()
CTL_FAILS=0
CTL_OK=0
HANG_DETECTED=0
PREV_CYCLES=0
STUCK_COUNT=0
ACTIONS_SEEN=0
MAX_CYCLE_MS=0

TOTAL_POLLS=$((DURATION_MIN * 60 / POLL_INTERVAL))

for i in $(seq 1 "$TOTAL_POLLS"); do
    STATUS=$("$CTL" status 2>/dev/null || echo '{"error":"no response"}')

    if echo "$STATUS" | grep -q '"error"'; then
        CTL_FAILS=$((CTL_FAILS + 1))
    else
        CTL_OK=$((CTL_OK + 1))

        CYCLES=$(echo "$STATUS" | grep -oE '"cycles": [0-9]+' | grep -oE '[0-9]+' || echo 0)

        if [ -z "$CYCLES_START" ]; then
            CYCLES_START="$CYCLES"
        fi
        CYCLES_END="$CYCLES"

        # Detect hang: same cycle count for 2+ consecutive polls
        if [ "$CYCLES" = "$PREV_CYCLES" ] && [ "$PREV_CYCLES" != "0" ]; then
            STUCK_COUNT=$((STUCK_COUNT + 1))
            if [ "$STUCK_COUNT" -ge 3 ]; then
                HANG_DETECTED=1
                r "  [HANG] Cycle stuck at $CYCLES for ${STUCK_COUNT} polls"
            fi
        else
            STUCK_COUNT=0
        fi
        PREV_CYCLES="$CYCLES"

        # Extract cycle time from stderr log
        LAST_TIME=$(grep -oE 'COMPLETE \([0-9]+ms\)' /var/log/apollo-optimizer.err.log 2>/dev/null | tail -1 | grep -oE '[0-9]+')
        if [ -n "$LAST_TIME" ]; then
            CYCLE_TIMES+=("$LAST_TIME")
            if [ "$LAST_TIME" -gt "$MAX_CYCLE_MS" ]; then
                MAX_CYCLE_MS="$LAST_TIME"
            fi
        fi
    fi

    sleep "$POLL_INTERVAL"
done

# ── Step 4: Final status snapshot ────────────────────────────────────────
FINAL_STATUS=$("$CTL" status 2>/dev/null || echo '{"error":"timeout"}')
r ""
r "── FINAL STATUS ──"
r "$FINAL_STATUS"

# ── Step 5: Score calculation ────────────────────────────────────────────
r ""
r "── METRICS ──"

# Cycles progressed
if [ -n "$CYCLES_START" ] && [ -n "$CYCLES_END" ]; then
    CYCLES_DELTA=$((CYCLES_END - CYCLES_START))
else
    CYCLES_DELTA=0
fi
r "Cycles: $CYCLES_START → $CYCLES_END (+$CYCLES_DELTA)"

# Expected cycles (3s interval = ~20 cycles/min)
EXPECTED=$((DURATION_MIN * 20))
if [ "$EXPECTED" -eq 0 ]; then EXPECTED=1; fi

# Stability score (0-25): cycles / expected
STAB_RAW=$((CYCLES_DELTA * 25 / EXPECTED))
if [ "$STAB_RAW" -gt 25 ]; then STAB_RAW=25; fi
if [ "$HANG_DETECTED" -eq 1 ]; then STAB_RAW=$((STAB_RAW / 2)); fi
r "Stability: $STAB_RAW/25 (expected ~$EXPECTED cycles, hang=$HANG_DETECTED)"

# Efficiency score (0-20): avg cycle time under 150ms
if [ ${#CYCLE_TIMES[@]} -gt 0 ]; then
    SUM=0
    for t in "${CYCLE_TIMES[@]}"; do SUM=$((SUM + t)); done
    AVG=$((SUM / ${#CYCLE_TIMES[@]}))
    if [ "$AVG" -le 80 ]; then
        EFF_RAW=20
    elif [ "$AVG" -le 100 ]; then
        EFF_RAW=16
    elif [ "$AVG" -le 150 ]; then
        EFF_RAW=10
    else
        EFF_RAW=5
    fi
    r "Efficiency: $EFF_RAW/20 (avg=${AVG}ms, max=${MAX_CYCLE_MS}ms, samples=${#CYCLE_TIMES[@]})"
else
    AVG=0
    EFF_RAW=0
    r "Efficiency: 0/20 (no cycle times captured)"
fi

# Fluidity score (0-15): ctl connectivity
TOTAL_CTL=$((CTL_OK + CTL_FAILS))
if [ "$TOTAL_CTL" -gt 0 ]; then
    FLUID_RAW=$((CTL_OK * 15 / TOTAL_CTL))
else
    FLUID_RAW=0
fi
r "Fluidity: $FLUID_RAW/15 (ctl ok=$CTL_OK, fail=$CTL_FAILS)"

# Intelligence score (0-30): daemon is doing smart work across all subsystems
FREEZES=$(echo "$FINAL_STATUS" | grep -oE '"freezes_applied": [0-9]+' | grep -oE '[0-9]+' || echo 0)
THROTTLES=$(echo "$FINAL_STATUS" | grep -oE '"throttles_applied": [0-9]+' | grep -oE '[0-9]+' || echo 0)
BOOSTS=$(echo "$FINAL_STATUS" | grep -oE '"boosts_applied": [0-9]+' | grep -oE '[0-9]+' || echo 0)
UNFREEZES=$(echo "$FINAL_STATUS" | grep -oE '"unfreezes_applied": [0-9]+' | grep -oE '[0-9]+' || echo 0)
PAGING=$(echo "$FINAL_STATUS" | grep -oE '"paging_hints_applied": [0-9]+' | grep -oE '[0-9]+' || echo 0)
DEEP_SCANS=$(echo "$FINAL_STATUS" | grep -oE '"deep_scan_count": [0-9]+' | grep -oE '[0-9]+' || echo 0)
SYSCTL=$(echo "$FINAL_STATUS" | grep -oE '"sysctl_applied": [0-9]+' | grep -oE '[0-9]+' || echo 0)
FAILURES=$(echo "$FINAL_STATUS" | grep -oE '"failures": [0-9]+' | head -1 | grep -oE '[0-9]+' || echo 0)

INTEL_RAW=0
# Core actions (0-15)
[ "${FREEZES:-0}" -gt 0 ] && INTEL_RAW=$((INTEL_RAW + 3))
[ "${THROTTLES:-0}" -gt 0 ] && INTEL_RAW=$((INTEL_RAW + 3))
[ "${BOOSTS:-0}" -gt 0 ] && INTEL_RAW=$((INTEL_RAW + 3))
[ "${SYSCTL:-0}" -gt 0 ] && INTEL_RAW=$((INTEL_RAW + 3))
# 0 failures = +3
[ "${FAILURES:-0}" -eq 0 ] && INTEL_RAW=$((INTEL_RAW + 3))
# Advanced subsystems (0-15)
[ "${UNFREEZES:-0}" -gt 0 ] && INTEL_RAW=$((INTEL_RAW + 5))
[ "${PAGING:-0}" -gt 0 ] && INTEL_RAW=$((INTEL_RAW + 5))
[ "${DEEP_SCANS:-0}" -gt 0 ] && INTEL_RAW=$((INTEL_RAW + 5))
r "Intelligence: $INTEL_RAW/30"
r "  Core: freezes=$FREEZES throttles=$THROTTLES boosts=$BOOSTS sysctl=$SYSCTL failures=$FAILURES"
r "  Advanced: unfreezes=$UNFREEZES paging_hints=$PAGING deep_scans=$DEEP_SCANS"

# Total (stability:25 + efficiency:20 + fluidity:15 + intelligence:30 = 90 base + 10 bonus)
# Bonus: resource efficiency
PRESSURE=$(echo "$FINAL_STATUS" | grep -oE '"memory_pressure": [0-9.]+' | head -1 | grep -oE '[0-9.]+' || echo 0)
P95=$(echo "$FINAL_STATUS" | grep -oE '"p95_cycle_ms": [0-9.]+' | grep -oE '[0-9.]+' || echo 999)
# Bonus for low p95 and reasonable pressure
BONUS=0
# p95 under 100ms = +5
P95_INT=$(echo "$P95" | cut -d. -f1)
[ "${P95_INT:-999}" -le 100 ] && BONUS=$((BONUS + 5))
# Active causal analysis = +5
CAUSAL=$(echo "$FINAL_STATUS" | grep -oE '"causal_effect_avg": [0-9.]+' | grep -oE '[0-9.]+' || echo 0)
[ "$CAUSAL" != "0" ] && BONUS=$((BONUS + 5))
r "Bonus: $BONUS/10 (p95=${P95}ms, causal=$CAUSAL)"

TOTAL=$((STAB_RAW + EFF_RAW + FLUID_RAW + INTEL_RAW + BONUS))
r ""
r "══════════════════════════════════════════════════════════════════"
r "  SCORE: $TOTAL / 100"

if [ "$TOTAL" -ge 80 ]; then
    VERDICT="EXCELLENT"
elif [ "$TOTAL" -ge 60 ]; then
    VERDICT="GOOD"
elif [ "$TOTAL" -ge 40 ]; then
    VERDICT="NEEDS WORK"
else
    VERDICT="UNSTABLE"
fi

r "  VERDICT: $VERDICT"
if [ "$HANG_DETECTED" -eq 1 ]; then
    r "  WARNING: Hang detected during observation!"
    # Dump last stderr for diagnosis
    r ""
    r "── STDERR (last 20 lines) ──"
    tail -20 /var/log/apollo-optimizer.err.log 2>/dev/null >> "$REPORT"
fi
r "══════════════════════════════════════════════════════════════════"

echo ""
echo "AutoResearch complete. Score: $TOTAL/100 ($VERDICT)"
echo "Full report: $REPORT"
