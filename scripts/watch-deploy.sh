#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Apollo Watch & Deploy — runs as root, rebuilds on file change
# ══════════════════════════════════════════════════════════════════════════════
# Usage: sudo ./scripts/watch-deploy.sh
#
# Claude edits code → this script detects changes → rebuilds → redeploys
# → writes report to /tmp/apollo-pipeline-report.txt for Claude to read.
#
# Touch /tmp/apollo-trigger to force a rebuild without file changes.
# Touch /tmp/apollo-watch-stop to stop the watcher.
set -uo pipefail

cd "$(dirname "$0")/.."
REPORT="/tmp/apollo-pipeline-report.txt"
TRIGGER="/tmp/apollo-trigger"
STOP="/tmp/apollo-watch-stop"
LABEL="com.eduardocortez.systemoptimizerd"
DAEMON="/usr/local/libexec/apollo-optimizerd"
CTL="/usr/local/bin/apollo-optimizerctl"

rm -f "$STOP" "$TRIGGER"

write_report() {
    echo "$1" >> "$REPORT"
}

get_src_hash() {
    find src/ -name '*.rs' -newer /tmp/.apollo-last-build 2>/dev/null | head -1
}

echo "══ Apollo Watcher active ══"
echo "  Claude: edit code, then write to $TRIGGER"
echo "  Stop:   touch $STOP"
echo ""

# Sync ctl binary on startup (in case it's outdated)
if [ -f target/release/apollo-optimizerctl ]; then
    INSTALLED_MD5=$(md5 -q "$CTL" 2>/dev/null || echo "none")
    BUILT_MD5=$(md5 -q target/release/apollo-optimizerctl)
    if [ "$INSTALLED_MD5" != "$BUILT_MD5" ]; then
        echo "  Syncing ctl binary (outdated)..."
        cp -f target/release/apollo-optimizerctl "$CTL"
        chown root:wheel "$CTL"
        chmod 755 "$CTL"
        codesign --force --sign - "$CTL" 2>/dev/null
        echo "  ctl updated."
    fi
fi

# Initial marker
touch /tmp/.apollo-last-build

while true; do
    # Check stop signal
    if [ -f "$STOP" ]; then
        echo "Stop signal received."
        rm -f "$STOP"
        exit 0
    fi

    # Check for trigger or source changes
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

        # Kill + Copy + Sign
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
        codesign --force --sign - "$DAEMON" 2>&1
        MD5=$(md5 -q "$DAEMON")
        write_report "BINARY: installed (md5=$MD5)"

        if [ -f target/release/apollo-optimizerctl ]; then
            cp -f target/release/apollo-optimizerctl "$CTL"
            chown root:wheel "$CTL"
            chmod 755 "$CTL"
            codesign --force --sign - "$CTL" 2>&1
        fi

        # Truncate logs + restart
        truncate -s 0 /var/log/apollo-optimizer.out.log /var/log/apollo-optimizer.err.log 2>/dev/null || true

        launchctl kickstart -k system/$LABEL 2>/dev/null || \
            launchctl kickstart system/$LABEL 2>/dev/null || \
            launchctl load /Library/LaunchDaemons/$LABEL.plist 2>/dev/null || true
        write_report "RESTART: issued"

        # Wait for daemon to cycle
        write_report ""
        write_report "── VERIFY (waiting 15s) ──"
        sleep 15

        # Process check
        PID=$(pgrep -x apollo-optimizerd 2>/dev/null | head -1)
        if [ -n "$PID" ]; then
            USR=$(ps -o user= -p "$PID" 2>/dev/null)
            write_report "PROCESS: alive PID=$PID user=$USR"
        else
            write_report "PROCESS: NOT RUNNING"
            write_report ""
            write_report "── STDOUT ──"
            write_report "$(tail -10 /var/log/apollo-optimizer.out.log 2>/dev/null)"
            write_report ""
            write_report "── STDERR ──"
            write_report "$(tail -20 /var/log/apollo-optimizer.err.log 2>/dev/null)"
            write_report ""
            write_report "STATUS: DAEMON_DEAD"
            sleep 3
            continue
        fi

        # Logs
        write_report ""
        write_report "── STDOUT (last 10) ──"
        write_report "$(tail -10 /var/log/apollo-optimizer.out.log 2>/dev/null)"
        write_report ""
        write_report "── STDERR (last 80) ──"
        write_report "$(tail -80 /var/log/apollo-optimizer.err.log 2>/dev/null)"

        # Status from daemon
        write_report ""
        write_report "── DAEMON STATUS ──"
        STATUS=$("$CTL" status 2>/dev/null || echo '{"error":"no response"}')
        CYCLES=$(echo "$STATUS" | grep -oE '"cycles": [0-9]+' | grep -oE '[0-9]+' || echo 0)
        LAST=$(echo "$STATUS" | grep -oE '"last_cycle_at": "[^"]+"' | cut -d'"' -f4 || echo "?")
        PRESSURE=$(echo "$STATUS" | grep -oE '"memory_pressure": [0-9.]+' | grep -oE '[0-9.]+' || echo "?")
        write_report "CYCLES: $CYCLES"
        write_report "LAST_CYCLE: $LAST"
        write_report "PRESSURE: $PRESSURE"

        # Crash loop check
        RESTARTS=$(grep -c 'predictive-agent: loaded' /var/log/apollo-optimizer.err.log 2>/dev/null || echo 0)
        if [ "$RESTARTS" -gt 3 ]; then
            write_report "CRASH_LOOP: YES ($RESTARTS restarts)"
        else
            write_report "CRASH_LOOP: NO"
        fi

        # ── AUTORESEARCH: 90s stability observation ──────────────────
        write_report ""
        write_report "── AUTORESEARCH (90s observation) ──"
        AR_CYCLES_START="$CYCLES"
        AR_CTL_OK=0
        AR_CTL_FAIL=0
        AR_HANG=0
        AR_PREV_CYCLES=0
        AR_STUCK=0
        AR_MAX_MS=0
        AR_SUM_MS=0
        AR_SAMPLES=0

        for _poll in $(seq 1 18); do  # 18 polls × 5s = 90s
            AR_STATUS=$("$CTL" status 2>/dev/null || echo '{"error":"no response"}')
            if echo "$AR_STATUS" | grep -q '"error"'; then
                AR_CTL_FAIL=$((AR_CTL_FAIL + 1))
            else
                AR_CTL_OK=$((AR_CTL_OK + 1))
                AR_CUR=$(echo "$AR_STATUS" | grep -oE '"cycles": [0-9]+' | grep -oE '[0-9]+' || echo 0)
                if [ "$AR_CUR" = "$AR_PREV_CYCLES" ] && [ "$AR_PREV_CYCLES" != "0" ]; then
                    AR_STUCK=$((AR_STUCK + 1))
                    [ "$AR_STUCK" -ge 3 ] && AR_HANG=1
                else
                    AR_STUCK=0
                fi
                AR_PREV_CYCLES="$AR_CUR"
            fi
            # Cycle time from status JSON (p95_cycle_ms)
            LT=$(echo "$AR_STATUS" | grep -oE '"p95_cycle_ms": [0-9.]+' | grep -oE '[0-9]+' | head -1)
            if [ -n "$LT" ] && [ "$LT" -gt 0 ]; then
                AR_SUM_MS=$((AR_SUM_MS + LT))
                AR_SAMPLES=$((AR_SAMPLES + 1))
                [ "$LT" -gt "$AR_MAX_MS" ] && AR_MAX_MS="$LT"
            fi
            sleep 5
        done

        # Final snapshot for intelligence scoring
        AR_FINAL=$("$CTL" status 2>/dev/null || echo '{}')
        AR_CYCLES_END=$(echo "$AR_FINAL" | grep -oE '"cycles": [0-9]+' | grep -oE '[0-9]+' || echo 0)
        AR_DELTA=$((AR_CYCLES_END - AR_CYCLES_START))

        # Stability (0-25)
        AR_EXPECTED=30  # ~20 cycles/min × 1.5min
        AR_STAB=$((AR_DELTA * 25 / (AR_EXPECTED > 0 ? AR_EXPECTED : 1)))
        [ "$AR_STAB" -gt 25 ] && AR_STAB=25
        [ "$AR_HANG" -eq 1 ] && AR_STAB=$((AR_STAB / 2))

        # Efficiency (0-20)
        if [ "$AR_SAMPLES" -gt 0 ]; then
            AR_AVG=$((AR_SUM_MS / AR_SAMPLES))
            if [ "$AR_AVG" -le 80 ]; then AR_EFF=20
            elif [ "$AR_AVG" -le 100 ]; then AR_EFF=16
            elif [ "$AR_AVG" -le 150 ]; then AR_EFF=10
            else AR_EFF=5; fi
        else
            AR_AVG=0; AR_EFF=0
        fi

        # Fluidity (0-15)
        AR_TOTAL_CTL=$((AR_CTL_OK + AR_CTL_FAIL))
        if [ "$AR_TOTAL_CTL" -gt 0 ]; then
            AR_FLUID=$((AR_CTL_OK * 15 / AR_TOTAL_CTL))
        else
            AR_FLUID=0
        fi

        # Intelligence (0-30)
        ar_get() { echo "$AR_FINAL" | grep -oE "\"$1\": [0-9]+" | grep -oE '[0-9]+' || echo 0; }
        AR_FREEZES=$(ar_get freezes_applied)
        AR_THROTTLES=$(ar_get throttles_applied)
        AR_BOOSTS=$(ar_get boosts_applied)
        AR_SYSCTL=$(ar_get sysctl_applied)
        AR_FAILURES=$(echo "$AR_FINAL" | grep -oE '"failures": [0-9]+' | head -1 | grep -oE '[0-9]+' || echo 0)
        AR_UNFREEZES=$(ar_get unfreezes_applied)
        AR_PAGING=$(ar_get paging_hints_applied)
        AR_DEEPSCANS=$(ar_get deep_scan_count)

        AR_INTEL=0
        [ "${AR_FREEZES:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 3))
        [ "${AR_THROTTLES:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 3))
        [ "${AR_BOOSTS:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 3))
        [ "${AR_SYSCTL:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 3))
        [ "${AR_FAILURES:-0}" -eq 0 ] && AR_INTEL=$((AR_INTEL + 3))
        [ "${AR_UNFREEZES:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 5))
        [ "${AR_PAGING:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 5))
        [ "${AR_DEEPSCANS:-0}" -gt 0 ] && AR_INTEL=$((AR_INTEL + 5))

        # Bonus (0-10)
        AR_P95=$(echo "$AR_FINAL" | grep -oE '"p95_cycle_ms": [0-9.]+' | grep -oE '[0-9.]+' || echo 999)
        AR_CAUSAL=$(echo "$AR_FINAL" | grep -oE '"causal_effect_avg": [0-9.]+' | grep -oE '[0-9.]+' || echo 0)
        AR_BONUS=0
        AR_P95_INT=$(echo "$AR_P95" | cut -d. -f1)
        [ "${AR_P95_INT:-999}" -le 120 ] && AR_BONUS=$((AR_BONUS + 5))
        [ "$AR_CAUSAL" != "0" ] && AR_BONUS=$((AR_BONUS + 5))

        AR_SCORE=$((AR_STAB + AR_EFF + AR_FLUID + AR_INTEL + AR_BONUS))
        if [ "$AR_SCORE" -ge 80 ]; then AR_VERDICT="EXCELLENT"
        elif [ "$AR_SCORE" -ge 60 ]; then AR_VERDICT="GOOD"
        elif [ "$AR_SCORE" -ge 40 ]; then AR_VERDICT="NEEDS WORK"
        else AR_VERDICT="UNSTABLE"; fi

        write_report "STABILITY: $AR_STAB/25 (cycles=$AR_DELTA, expected=$AR_EXPECTED, hang=$AR_HANG)"
        write_report "EFFICIENCY: $AR_EFF/20 (avg=${AR_AVG}ms, max=${AR_MAX_MS}ms)"
        write_report "FLUIDITY: $AR_FLUID/15 (ok=$AR_CTL_OK, fail=$AR_CTL_FAIL)"
        write_report "INTELLIGENCE: $AR_INTEL/30 (freeze=$AR_FREEZES throt=$AR_THROTTLES boost=$AR_BOOSTS sysctl=$AR_SYSCTL fail=$AR_FAILURES unfreeze=$AR_UNFREEZES paging=$AR_PAGING deepscan=$AR_DEEPSCANS)"
        write_report "BONUS: $AR_BONUS/10 (p95=${AR_P95}ms, causal=$AR_CAUSAL)"
        write_report ""
        write_report "AUTORESEARCH_SCORE: $AR_SCORE/100 ($AR_VERDICT)"

        write_report ""
        write_report "STATUS: DONE"
        echo "[$TS] Deploy + autoresearch complete. Score: $AR_SCORE/100 ($AR_VERDICT)"
    fi

    sleep 3
done
