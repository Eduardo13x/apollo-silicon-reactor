#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Apollo Pipeline — Build → Test → Deploy → Verify
# ══════════════════════════════════════════════════════════════════════════════
# Usage: sudo ./scripts/pipeline.sh [--skip-test] [--skip-deploy]
#
# Exit codes:
#   0 = all green
#   1 = build failed
#   2 = tests failed
#   3 = deploy failed
#   4 = daemon not cycling
set -uo pipefail

SKIP_TEST=false
SKIP_DEPLOY=false
for arg in "$@"; do
    case "$arg" in
        --skip-test) SKIP_TEST=true ;;
        --skip-deploy) SKIP_DEPLOY=true ;;
    esac
done

cd "$(dirname "$0")/.."

# Write all output to a file Claude can read
REPORT="/tmp/apollo-pipeline-report.txt"
exec > >(tee "$REPORT") 2>&1
chmod 644 "$REPORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }

TOTAL_START=$(date +%s)

# ── 1. BUILD ─────────────────────────────────────────────────────────────────
echo "═══ 1/4 BUILD ═══"
if cargo build --release 2>&1 | tail -3; then
    ok "cargo build --release"
else
    fail "build failed"
    exit 1
fi

# ── 2. TEST ──────────────────────────────────────────────────────────────────
echo ""
echo "═══ 2/4 TEST ═══"
if $SKIP_TEST; then
    warn "skipped (--skip-test)"
else
    # Clippy
    CLIPPY_OUT=$(cargo clippy --all-targets 2>&1)
    CLIPPY_WARNS=$(echo "$CLIPPY_OUT" | grep -c 'warning\[' || true)
    if [ "$CLIPPY_WARNS" -eq 0 ]; then
        ok "clippy clean"
    else
        fail "clippy: $CLIPPY_WARNS warnings"
    fi

    # Unit + lib tests
    TEST_OUT=$(cargo test --lib --bins 2>&1)
    TEST_PASS=$(echo "$TEST_OUT" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | head -1)
    TEST_FAIL=$(echo "$TEST_OUT" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' | head -1)
    TEST_PASS=${TEST_PASS:-0}
    TEST_FAIL=${TEST_FAIL:-0}
    if [ "$TEST_FAIL" -eq 0 ] && [ "$TEST_PASS" -gt 0 ]; then
        ok "tests: $TEST_PASS passed"
    else
        fail "tests: $TEST_PASS passed, $TEST_FAIL failed"
        exit 2
    fi

    # Scenario benchmarks (all prepare_* files)
    SCENARIO_TOTAL=0
    SCENARIO_PASS=0
    for testfile in prepare prepare_latency prepare_memory prepare_actions prepare_classifier prepare_signals prepare_rl; do
        OUT=$(cargo test --test "$testfile" 2>&1 || true)
        P=$(echo "$OUT" | grep -cE 'test scenarios::.* ok$' || echo 0)
        T=$(echo "$OUT" | grep -cE 'test scenarios::' || echo 0)
        SCENARIO_PASS=$((SCENARIO_PASS + P))
        SCENARIO_TOTAL=$((SCENARIO_TOTAL + T))
    done
    if [ "$SCENARIO_PASS" -eq "$SCENARIO_TOTAL" ] && [ "$SCENARIO_TOTAL" -gt 0 ]; then
        ok "scenarios: $SCENARIO_PASS/$SCENARIO_TOTAL"
    else
        fail "scenarios: $SCENARIO_PASS/$SCENARIO_TOTAL"
        exit 2
    fi
fi

# ── 3. DEPLOY ────────────────────────────────────────────────────────────────
echo ""
echo "═══ 3/4 DEPLOY ═══"
if $SKIP_DEPLOY; then
    warn "skipped (--skip-deploy)"
else
    # Need root for deploy
    if [ "$(id -u)" -ne 0 ]; then
        fail "deploy requires root — run with sudo"
        exit 3
    fi

    # Kill all instances
    killall apollo-optimizerd 2>/dev/null || true
    sleep 1

    # Verify dead
    if pgrep -x apollo-optimizerd >/dev/null 2>&1; then
        killall -9 apollo-optimizerd 2>/dev/null || true
        sleep 1
    fi

    # Copy + sign
    cp -f target/release/apollo-optimizerd /usr/local/libexec/apollo-optimizerd
    chown root:wheel /usr/local/libexec/apollo-optimizerd
    chmod 755 /usr/local/libexec/apollo-optimizerd
    codesign --force --sign - /usr/local/libexec/apollo-optimizerd
    ok "binary installed + signed"

    # Also update ctl
    if [ -f target/release/apollo-optimizerctl ]; then
        cp -f target/release/apollo-optimizerctl /usr/local/bin/apollo-optimizerctl
        chown root:wheel /usr/local/bin/apollo-optimizerctl
        chmod 755 /usr/local/bin/apollo-optimizerctl
        codesign --force --sign - /usr/local/bin/apollo-optimizerctl
        ok "ctl binary updated"
    fi

    # Truncate logs
    truncate -s 0 /var/log/apollo-optimizer.out.log /var/log/apollo-optimizer.err.log 2>/dev/null || true
    ok "logs truncated"

    # Restart via launchd
    launchctl kickstart -k system/com.eduardocortez.systemoptimizerd 2>/dev/null || \
      launchctl kickstart system/com.eduardocortez.systemoptimizerd 2>/dev/null || \
      launchctl load /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist 2>/dev/null || true
    ok "launchd restart issued"
fi

# ── 4. VERIFY ────────────────────────────────────────────────────────────────
echo ""
echo "═══ 4/4 VERIFY ═══"
if $SKIP_DEPLOY; then
    warn "skipped (no deploy)"
else
    echo "  waiting 15s for daemon..."
    sleep 15

    # Check process
    DAEMON_PID=$(pgrep -x apollo-optimizerd 2>/dev/null | head -1)
    if [ -n "$DAEMON_PID" ]; then
        DAEMON_USER=$(ps -o user= -p "$DAEMON_PID" 2>/dev/null)
        ok "process alive: PID=$DAEMON_PID user=$DAEMON_USER"
    else
        fail "daemon not running"
        echo ""
        echo "── STDERR ──"
        tail -20 /var/log/apollo-optimizer.err.log 2>/dev/null
        echo ""
        echo "── STDOUT ──"
        tail -10 /var/log/apollo-optimizer.out.log 2>/dev/null
        exit 4
    fi

    # Check cycling
    STATUS=$(/usr/local/bin/apollo-optimizerctl status 2>/dev/null || echo '{}')
    CYCLES=$(echo "$STATUS" | grep -oE '"cycles": [0-9]+' | grep -oE '[0-9]+' || echo 0)
    LAST_CYCLE=$(echo "$STATUS" | grep -oE '"last_cycle_at": "[^"]+"' | cut -d'"' -f4 || echo "unknown")
    PRESSURE=$(echo "$STATUS" | grep -oE '"memory_pressure": [0-9.]+' | grep -oE '[0-9.]+' || echo "?")
    URGENCY=$(echo "$STATUS" | grep -oE '"si_urgency": [0-9.]+' | grep -oE '[0-9.]+' || echo "?")

    if [ "$CYCLES" -gt 0 ]; then
        ok "cycling: $CYCLES cycles"
        ok "last_cycle: $LAST_CYCLE"
        ok "pressure: $PRESSURE | urgency: $URGENCY"
    else
        warn "cycles=$CYCLES — may still be starting up"
        echo ""
        echo "── STDERR ──"
        tail -20 /var/log/apollo-optimizer.err.log 2>/dev/null
        echo ""
        echo "── STDOUT ──"
        tail -10 /var/log/apollo-optimizer.out.log 2>/dev/null
    fi

    # Check for crash loop
    ERR_LINES=$(wc -l < /var/log/apollo-optimizer.err.log 2>/dev/null || echo 0)
    RESTART_COUNT=$(grep -c 'predictive-agent: loaded' /var/log/apollo-optimizer.err.log 2>/dev/null || echo 0)
    if [ "$RESTART_COUNT" -gt 3 ]; then
        fail "crash loop detected: $RESTART_COUNT restarts in stderr"
        echo ""
        echo "── STDERR (last 20) ──"
        tail -20 /var/log/apollo-optimizer.err.log 2>/dev/null
    elif [ "$RESTART_COUNT" -gt 1 ]; then
        warn "$RESTART_COUNT restarts detected"
    else
        ok "no crash loop"
    fi
fi

# ── SUMMARY ──────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
ELAPSED=$((TOTAL_END - TOTAL_START))
echo ""
echo "═══ DONE ═══ (${ELAPSED}s)"
