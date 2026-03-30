#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Apollo Watch & Deploy + Dr. Zero Self-Evolution
# ══════════════════════════════════════════════════════════════════════════════
# Usage: sudo ./scripts/watch-deploy.sh
#
# Triggers:
#   /tmp/apollo-trigger  → rebuild + deploy + warmup + autoresearch
#   /tmp/apollo-observe  → observe-only (no rebuild, daemon keeps state)
#   /tmp/apollo-watch-stop → stop watcher
#
# Dr. Zero (arxiv:2601.07055):
#   Proposer: generates progressively harder synthetic load (levels 0-3)
#   Solver: daemon handles the load; score feeds back as curriculum signal
#   HRPO: hop-grouped process learning inside the daemon
set -uo pipefail

cd "$(dirname "$0")/.."
REPORT="/tmp/apollo-pipeline-report.txt"
TRIGGER="/tmp/apollo-trigger"
OBSERVE="/tmp/apollo-observe"
STOP="/tmp/apollo-watch-stop"
LABEL="com.eduardocortez.systemoptimizerd"
DAEMON="/usr/local/libexec/apollo-optimizerd"
CTL="/usr/local/bin/apollo-optimizerctl"
DZ_STATE="/tmp/apollo-dr-zero-state.json"
DZ_FEEDBACK="/tmp/apollo-dr-zero-feedback.json"

rm -f "$STOP" "$TRIGGER" "$OBSERVE"

write_report() {
    echo "$1" >> "$REPORT"
}

get_src_hash() {
    find src/ -name '*.rs' -newer /tmp/.apollo-last-build 2>/dev/null | head -1
}

# ── Dr. Zero Proposer: spawn synthetic load ──────────────────────────────────
dz_spawn_challenge() {
    local level=$1
    DZ_PIDS=""
    case "$level" in
        0)
            write_report "CHALLENGE: baseline (no synthetic load)"
            ;;
        1)
            write_report "CHALLENGE: light (200MB alloc + 10 sleeper processes)"
            python3 -c "x=bytearray(200*1024*1024); import time; time.sleep(300)" &
            DZ_PIDS="$!"
            for _s in $(seq 1 10); do sleep 300 & DZ_PIDS="$DZ_PIDS $!"; done
            ;;
        2)
            write_report "CHALLENGE: medium (400MB alloc + CPU stress + 20 processes)"
            python3 -c "x=bytearray(400*1024*1024); import time; time.sleep(300)" &
            DZ_PIDS="$!"
            for _c in $(seq 1 2); do
                python3 -c "import time; t=time.time()
while time.time()-t<300: sum(range(10000))" &
                DZ_PIDS="$DZ_PIDS $!"
            done
            for _s in $(seq 1 20); do sleep 300 & DZ_PIDS="$DZ_PIDS $!"; done
            ;;
        3)
            write_report "CHALLENGE: heavy (CPU×2 + 40 processes, no alloc)"
            for _c in $(seq 1 2); do
                python3 -c "import time; t=time.time()
while time.time()-t<300: sum(range(10000))" &
                DZ_PIDS="${DZ_PIDS:-} $!"
            done
            for _s in $(seq 1 40); do sleep 300 & DZ_PIDS="${DZ_PIDS:-} $!"; done
            ;;
    esac
}

dz_cleanup() {
    if [ -n "${DZ_PIDS:-}" ]; then
        for pid in $DZ_PIDS; do kill "$pid" 2>/dev/null; done
        wait 2>/dev/null
        DZ_PIDS=""
    fi
}

# ── Autoresearch: observe + score ────────────────────────────────────────────
run_autoresearch() {
    local warmup_secs=${1:-0}

    # Load Dr. Zero state
    local DZ_LEVEL=0 DZ_STREAK=0
    if [ -f "$DZ_STATE" ]; then
        DZ_LEVEL=$(grep -oE '"level": [0-9]+' "$DZ_STATE" | grep -oE '[0-9]+' || echo 0)
        DZ_STREAK=$(grep -oE '"streak": [0-9]+' "$DZ_STATE" | grep -oE '[0-9]+' || echo 0)
    fi

    write_report ""
    write_report "── DR.ZERO PROPOSER (level=$DZ_LEVEL, streak=$DZ_STREAK) ──"
    dz_spawn_challenge "$DZ_LEVEL"

    # Warmup: let daemon accumulate state (unfreezes, causal, HRPO)
    if [ "$warmup_secs" -gt 0 ]; then
        write_report ""
        write_report "── WARMUP (${warmup_secs}s) ──"
        echo "  [warmup] ${warmup_secs}s..."
        sleep "$warmup_secs"
        # Verify daemon survived warmup
        if ! pgrep -x apollo-optimizerd >/dev/null 2>&1; then
            write_report "DAEMON DIED DURING WARMUP"
            dz_cleanup
            write_report "STATUS: DAEMON_DEAD"
            return
        fi
        write_report "WARMUP: daemon alive after ${warmup_secs}s"
    fi

    # ── 90s observation ──
    write_report ""
    write_report "── AUTORESEARCH (90s observation) ──"
    local INIT_STATUS
    INIT_STATUS=$("$CTL" status 2>/dev/null || echo '{"error":"no response"}')
    local AR_CYCLES_START
    AR_CYCLES_START=$(echo "$INIT_STATUS" | grep -oE '"cycles": [0-9]+' | grep -oE '[0-9]+' || echo 0)
    local AR_CTL_OK=0 AR_CTL_FAIL=0 AR_HANG=0 AR_PREV_CYCLES=0 AR_STUCK=0
    local AR_MAX_MS=0 AR_SUM_MS=0 AR_SAMPLES=0

    for _poll in $(seq 1 18); do  # 18 × 5s = 90s
        local AR_STATUS
        AR_STATUS=$("$CTL" status 2>/dev/null || echo '{"error":"no response"}')
        if echo "$AR_STATUS" | grep -q '"error"'; then
            AR_CTL_FAIL=$((AR_CTL_FAIL + 1))
            echo "  [poll $_poll/18] daemon: NO RESPONSE"
        else
            AR_CTL_OK=$((AR_CTL_OK + 1))
            local AR_CUR AR_P95_LIVE AR_PRESS_LIVE
            AR_CUR=$(echo "$AR_STATUS" | grep -oE '"cycles": [0-9]+' | grep -oE '[0-9]+' || echo 0)
            AR_P95_LIVE=$(echo "$AR_STATUS" | grep -oE '"p95_cycle_ms": [0-9.]+' | grep -oE '[0-9]+' | head -1 || echo "?")
            AR_PRESS_LIVE=$(echo "$AR_STATUS" | grep -oE '"memory_pressure": [0-9.]+' | grep -oE '[0-9.]+' | head -1 || echo "?")
            echo "  [poll $_poll/18] cycles=$AR_CUR p95=${AR_P95_LIVE}ms pressure=${AR_PRESS_LIVE}"
            if [ "$AR_CUR" = "$AR_PREV_CYCLES" ] && [ "$AR_PREV_CYCLES" != "0" ]; then
                AR_STUCK=$((AR_STUCK + 1))
                [ "$AR_STUCK" -ge 3 ] && AR_HANG=1 && echo "  [HANG DETECTED]"
            else
                AR_STUCK=0
            fi
            AR_PREV_CYCLES="$AR_CUR"
        fi
        local LT
        LT=$(echo "$AR_STATUS" | grep -oE '"p95_cycle_ms": [0-9.]+' | grep -oE '[0-9]+' | head -1)
        if [ -n "$LT" ] && [ "$LT" -gt 0 ]; then
            AR_SUM_MS=$((AR_SUM_MS + LT))
            AR_SAMPLES=$((AR_SAMPLES + 1))
            [ "$LT" -gt "$AR_MAX_MS" ] && AR_MAX_MS="$LT"
        fi
        sleep 5
    done

    # Final snapshot
    local AR_FINAL
    AR_FINAL=$("$CTL" status 2>/dev/null || echo '{}')
    local AR_CYCLES_END
    AR_CYCLES_END=$(echo "$AR_FINAL" | grep -oE '"cycles": [0-9]+' | grep -oE '[0-9]+' || echo 0)
    local AR_DELTA=$((AR_CYCLES_END - AR_CYCLES_START))

    # Stability (0-25)
    local AR_EXPECTED=30
    local AR_STAB=$((AR_DELTA * 25 / (AR_EXPECTED > 0 ? AR_EXPECTED : 1)))
    [ "$AR_STAB" -gt 25 ] && AR_STAB=25
    [ "$AR_HANG" -eq 1 ] && AR_STAB=$((AR_STAB / 2))

    # Efficiency (0-20)
    local AR_AVG=0 AR_EFF=0
    if [ "$AR_SAMPLES" -gt 0 ]; then
        AR_AVG=$((AR_SUM_MS / AR_SAMPLES))
        if [ "$AR_AVG" -le 80 ]; then AR_EFF=20
        elif [ "$AR_AVG" -le 100 ]; then AR_EFF=16
        elif [ "$AR_AVG" -le 150 ]; then AR_EFF=10
        else AR_EFF=5; fi
    fi

    # Fluidity (0-15)
    local AR_TOTAL_CTL=$((AR_CTL_OK + AR_CTL_FAIL))
    local AR_FLUID=0
    if [ "$AR_TOTAL_CTL" -gt 0 ]; then
        AR_FLUID=$((AR_CTL_OK * 15 / AR_TOTAL_CTL))
    fi

    # Intelligence (0-30)
    ar_get() { echo "$AR_FINAL" | grep -oE "\"$1\": [0-9]+" | grep -oE '[0-9]+' || echo 0; }
    local AR_FREEZES AR_THROTTLES AR_BOOSTS AR_SYSCTL AR_FAILURES
    local AR_UNFREEZES AR_PAGING AR_DEEPSCANS
    AR_FREEZES=$(ar_get freezes_applied)
    AR_THROTTLES=$(ar_get throttles_applied)
    AR_BOOSTS=$(ar_get boosts_applied)
    AR_SYSCTL=$(ar_get sysctl_applied)
    AR_FAILURES=$(echo "$AR_FINAL" | grep -oE '"failures": [0-9]+' | head -1 | grep -oE '[0-9]+' || echo 0)
    AR_UNFREEZES=$(ar_get unfreezes_applied)
    AR_PAGING=$(ar_get paging_hints_applied)
    AR_DEEPSCANS=$(ar_get deep_scan_count)

    local AR_INTEL=0
    [ "${AR_FREEZES:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 3))
    [ "${AR_THROTTLES:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 3))
    [ "${AR_BOOSTS:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 3))
    [ "${AR_SYSCTL:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 3))
    [ "${AR_FAILURES:-0}" -eq 0 ] && AR_INTEL=$((AR_INTEL + 3))
    [ "${AR_UNFREEZES:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 5))
    [ "${AR_PAGING:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 5))
    [ "${AR_DEEPSCANS:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 5))

    # Bonus (0-10)
    local AR_P95 AR_CAUSAL AR_BONUS=0
    AR_P95=$(echo "$AR_FINAL" | grep -oE '"p95_cycle_ms": [0-9.]+' | tail -1 | grep -oE '[0-9.]+' || echo 999)
    AR_CAUSAL=$(echo "$AR_FINAL" | grep -oE '"causal_effect_avg": [0-9.]+' | tail -1 | grep -oE '[0-9.]+' || echo 0)
    local AR_P95_INT
    AR_P95_INT=$(echo "$AR_P95" | cut -d. -f1 | tr -d '[:space:]')
    [ "${AR_P95_INT:-999}" -le 120 ] && AR_BONUS=$((AR_BONUS + 5))
    [ "$AR_CAUSAL" != "0" ] && AR_BONUS=$((AR_BONUS + 5))

    # Dr. Zero HRPO bonus (0-5): reward hop-group learning
    local AR_DZ_GROUPS
    AR_DZ_GROUPS=$(echo "$AR_FINAL" | grep -c '"dr_zero_groups"' || echo 0)
    local AR_DZ_CHALLENGE
    AR_DZ_CHALLENGE=$(echo "$AR_FINAL" | grep -oE '"dr_zero_self_challenge": [0-9.]+' | grep -oE '[0-9.]+' || echo 1)
    local AR_DZ_BONUS=0
    # Low self-challenge score = well-calibrated solver
    local AR_DZ_INT
    AR_DZ_INT=$(echo "$AR_DZ_CHALLENGE" | cut -d. -f1)
    if [ "${AR_DZ_INT:-1}" -eq 0 ] && echo "$AR_FINAL" | grep -q '"dr_zero_groups": \['; then
        local DZ_GROUP_COUNT
        DZ_GROUP_COUNT=$(echo "$AR_FINAL" | grep -oE '\(eff=' | wc -l | tr -d ' ')
        [ "${DZ_GROUP_COUNT:-0}" -ge 2 ] && AR_DZ_BONUS=5
    fi

    local AR_SCORE=$((AR_STAB + AR_EFF + AR_FLUID + AR_INTEL + AR_BONUS + AR_DZ_BONUS))
    local AR_VERDICT
    if [ "$AR_SCORE" -ge 90 ]; then AR_VERDICT="EXCELLENT"
    elif [ "$AR_SCORE" -ge 70 ]; then AR_VERDICT="GOOD"
    elif [ "$AR_SCORE" -ge 50 ]; then AR_VERDICT="NEEDS WORK"
    else AR_VERDICT="UNSTABLE"; fi

    write_report "STABILITY: $AR_STAB/25 (cycles=$AR_DELTA, expected=$AR_EXPECTED, hang=$AR_HANG)"
    write_report "EFFICIENCY: $AR_EFF/20 (avg=${AR_AVG}ms, max=${AR_MAX_MS}ms)"
    write_report "FLUIDITY: $AR_FLUID/15 (ok=$AR_CTL_OK, fail=$AR_CTL_FAIL)"
    write_report "INTELLIGENCE: $AR_INTEL/30 (freeze=$AR_FREEZES throt=$AR_THROTTLES boost=$AR_BOOSTS sysctl=$AR_SYSCTL fail=$AR_FAILURES unfreeze=$AR_UNFREEZES paging=$AR_PAGING deepscan=$AR_DEEPSCANS)"
    write_report "BONUS: $AR_BONUS/10 (p95=${AR_P95_INT}ms, causal=$AR_CAUSAL)"
    write_report "DR.ZERO_BONUS: $AR_DZ_BONUS/5 (self_challenge=$AR_DZ_CHALLENGE, groups=$DZ_GROUP_COUNT)"
    write_report ""
    write_report "AUTORESEARCH_SCORE: $AR_SCORE/105 ($AR_VERDICT)"
    echo ""
    echo "══ SCORE: $AR_SCORE/105 ($AR_VERDICT) | p95=${AR_AVG}ms | pressure=${AR_PRESS_LIVE} | Dr.Zero L${DZ_LEVEL} ══"

    # Cleanup synthetic load
    dz_cleanup
    [ -n "${DZ_PIDS:-}" ] || write_report "DR.ZERO: synthetic load cleaned up"

    # Dr. Zero curriculum: adjust difficulty
    if [ "$AR_SCORE" -ge 85 ]; then
        DZ_STREAK=$((DZ_STREAK + 1))
        if [ "$DZ_STREAK" -ge 2 ] && [ "$DZ_LEVEL" -lt 3 ]; then
            DZ_LEVEL=$((DZ_LEVEL + 1))
            DZ_STREAK=0
            write_report "DR.ZERO: LEVEL UP → $DZ_LEVEL (2 wins)"
        else
            write_report "DR.ZERO: win streak=$DZ_STREAK (need 2 to level up)"
        fi
    elif [ "$AR_SCORE" -lt 60 ]; then
        DZ_STREAK=0
        if [ "$DZ_LEVEL" -gt 0 ]; then
            DZ_LEVEL=$((DZ_LEVEL - 1))
            write_report "DR.ZERO: LEVEL DOWN → $DZ_LEVEL (score < 60)"
        fi
    else
        DZ_STREAK=0
        write_report "DR.ZERO: hold level=$DZ_LEVEL (score 60-84)"
    fi

    # Persist Dr. Zero state + feedback for daemon
    cat > "$DZ_STATE" <<DZEOF
{"level": $DZ_LEVEL, "streak": $DZ_STREAK, "last_score": $AR_SCORE, "last_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
DZEOF
    chmod 644 "$DZ_STATE"

    # Feedback file: daemon can read this to self-adjust
    cat > "$DZ_FEEDBACK" <<FBEOF
{"score": $AR_SCORE, "stability": $AR_STAB, "efficiency": $AR_EFF, "fluidity": $AR_FLUID, "intelligence": $AR_INTEL, "bonus": $AR_BONUS, "dz_bonus": $AR_DZ_BONUS, "level": $DZ_LEVEL, "at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
FBEOF
    chmod 644 "$DZ_FEEDBACK"

    write_report ""
    write_report "STATUS: DONE"
    echo "[$(date '+%H:%M:%S')] Score: $AR_SCORE/105 ($AR_VERDICT) [Dr.Zero L$DZ_LEVEL]"
}

# ══════════════════════════════════════════════════════════════════════════════

echo "══ Apollo Watcher active ══"
echo "  Rebuild:  touch $TRIGGER"
echo "  Observe:  touch $OBSERVE"
echo "  Stop:     touch $STOP"
echo ""

# ctl binary synced during trigger deploy (not on startup to avoid codesign hang)

touch /tmp/.apollo-last-build

while true; do
    if [ -f "$STOP" ]; then
        echo "Stop signal received."
        rm -f "$STOP"
        exit 0
    fi

    # ── OBSERVE-ONLY: no rebuild, daemon keeps state ──────────────────
    if [ -f "$OBSERVE" ]; then
        rm -f "$OBSERVE"
        echo "" > "$REPORT"
        chmod 644 "$REPORT"
        write_report "═══ OBSERVE @ $(date '+%H:%M:%S') ═══"
        write_report ""
        write_report "── DAEMON STATUS ──"
        STATUS=$("$CTL" status 2>/dev/null || echo '{"error":"no response"}')
        CYCLES=$(echo "$STATUS" | grep -oE '"cycles": [0-9]+' | grep -oE '[0-9]+' || echo 0)
        PRESSURE=$(echo "$STATUS" | grep -oE '"memory_pressure": [0-9.]+' | grep -oE '[0-9.]+' || echo "?")
        write_report "CYCLES: $CYCLES"
        write_report "PRESSURE: $PRESSURE"
        write_report "MODE: observe-only (no restart)"

        run_autoresearch 0  # no warmup needed, daemon already warm
        sleep 3
        continue
    fi

    # ── REBUILD + DEPLOY ─────────────────────────────────────────────
    CHANGED=$(get_src_hash)
    if [ -f "$TRIGGER" ] || [ -n "$CHANGED" ]; then
        rm -f "$TRIGGER"
        echo "" > "$REPORT"
        chmod 644 "$REPORT"

        TS=$(date '+%H:%M:%S')
        write_report "═══ REBUILD @ $TS ═══"

        # Build
        write_report ""
        write_report "── BUILD ──"
        BUILD_OUT=$(cargo build --release 2>&1)
        BUILD_RC=$?
        write_report "$BUILD_OUT"
        if [ $BUILD_RC -ne 0 ]; then
            write_report "STATUS: BUILD_FAILED"
            sleep 3
            continue
        fi
        write_report "BUILD: OK"
        touch /tmp/.apollo-last-build

        # Deploy
        write_report ""
        write_report "── DEPLOY ──"
        killall apollo-optimizerd 2>/dev/null || true
        sleep 1
        if pgrep -x apollo-optimizerd >/dev/null 2>&1; then
            killall -9 apollo-optimizerd 2>/dev/null || true
            sleep 1
        fi

        cp -f target/release/apollo-optimizerd "$DAEMON"
        chown root:wheel "$DAEMON"
        chmod 755 "$DAEMON"
        timeout 8 codesign --force --sign - "$DAEMON" 2>&1 || true
        MD5=$(md5 -q "$DAEMON")
        write_report "BINARY: installed (md5=$MD5)"

        if [ -f target/release/apollo-optimizerctl ]; then
            cp -f target/release/apollo-optimizerctl "$CTL"
            chown root:wheel "$CTL"
            chmod 755 "$CTL"
            timeout 8 codesign --force --sign - "$CTL" 2>&1 || true
        fi

        truncate -s 0 /var/log/apollo-optimizer.out.log /var/log/apollo-optimizer.err.log 2>/dev/null || true

        launchctl kickstart -k system/$LABEL 2>/dev/null || \
            launchctl kickstart system/$LABEL 2>/dev/null || \
            launchctl load /Library/LaunchDaemons/$LABEL.plist 2>/dev/null || true
        write_report "RESTART: issued"

        # Verify
        write_report ""
        write_report "── VERIFY (waiting 15s) ──"
        sleep 15

        PID=$(pgrep -x apollo-optimizerd 2>/dev/null | head -1)
        if [ -n "$PID" ]; then
            USR=$(ps -o user= -p "$PID" 2>/dev/null)
            write_report "PROCESS: alive PID=$PID user=$USR"
        else
            write_report "PROCESS: NOT RUNNING"
            write_report "── STDERR ──"
            write_report "$(tail -20 /var/log/apollo-optimizer.err.log 2>/dev/null)"
            write_report "STATUS: DAEMON_DEAD"
            sleep 3
            continue
        fi

        write_report ""
        write_report "── STDERR (last 10) ──"
        write_report "$(tail -10 /var/log/apollo-optimizer.err.log 2>/dev/null)"

        RESTARTS=$(grep -c 'predictive-agent: loaded' /var/log/apollo-optimizer.err.log 2>/dev/null || echo 0)
        if [ "$RESTARTS" -gt 3 ]; then
            write_report "CRASH_LOOP: YES ($RESTARTS restarts)"
        else
            write_report "CRASH_LOOP: NO"
        fi

        # Warmup 180s after deploy, then observe
        run_autoresearch 180
    fi

    sleep 3
done
