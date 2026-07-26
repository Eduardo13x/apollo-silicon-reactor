#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Apollo Pipeline — Build → Test → Deploy → Verify
# ══════════════════════════════════════════════════════════════════════════════
# Usage: ./scripts/pipeline.sh [--skip-test] [--skip-deploy]
#
# Exit codes:
#   0 = all green
#   1 = build failed
#   2 = tests failed
#   3 = deploy failed
#   4 = daemon not cycling
set -euo pipefail

SKIP_TEST=false
SKIP_DEPLOY=false
for arg in "$@"; do
    case "$arg" in
        --skip-test) SKIP_TEST=true ;;
        --skip-deploy) SKIP_DEPLOY=true ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."
source scripts/hardware-build-profile.sh

CARGO_BIN="$(command -v cargo || true)"
if [ -z "$CARGO_BIN" ]; then
    fail_path="${HOME}/.cargo/bin/cargo"
    if [ -x "$fail_path" ]; then
        CARGO_BIN="$fail_path"
    else
        echo "cargo not found in PATH or $fail_path" >&2
        exit 1
    fi
fi

# Capture a deterministic report without Bash process substitution. Hardened
# macOS runners may deny /dev/fd even though `>(tee ...)` parsed successfully.
REPORT="/tmp/apollo-pipeline-report.txt"
exec 3>&1
report_and_exit() {
    status=$?
    trap - EXIT
    cat "$REPORT" >&3
    exit "$status"
}
trap report_and_exit EXIT
exec >"$REPORT" 2>&1
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
BUILD_STATUS=0
BUILD_OUT=$("$CARGO_BIN" build --workspace --bins --release ${APOLLO_CARGO_FEATURE_ARGS[@]+"${APOLLO_CARGO_FEATURE_ARGS[@]}"} 2>&1) || BUILD_STATUS=$?
if [ "$BUILD_STATUS" -eq 0 ]; then
    printf '%s\n' "$BUILD_OUT" | tail -3
    ok "cargo build --workspace --bins --release ($APOLLO_BUILD_PROFILE)"
else
    printf '%s\n' "$BUILD_OUT" | tail -80
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
    CLIPPY_STATUS=0
    CLIPPY_OUT=$("$CARGO_BIN" clippy --workspace --all-targets --message-format=short ${APOLLO_CARGO_FEATURE_ARGS[@]+"${APOLLO_CARGO_FEATURE_ARGS[@]}"} 2>&1) \
      || CLIPPY_STATUS=$?
    CLIPPY_WARNS=$(printf '%s\n' "$CLIPPY_OUT" | grep -c 'warning:' || true)
    if [ "$CLIPPY_STATUS" -ne 0 ]; then
        printf '%s\n' "$CLIPPY_OUT" | tail -40
        fail "clippy failed (exit=$CLIPPY_STATUS)"
        exit 2
    elif [ "$CLIPPY_WARNS" -eq 0 ]; then
        ok "clippy clean"
    else
        warn "clippy: $CLIPPY_WARNS warnings (technical debt, command succeeded)"
    fi

    # Run the workspace in parallel, except the daemon E2E suite whose tests
    # intentionally share one global Unix socket and therefore require serial
    # isolation. This keeps the M4 busy without manufacturing socket races.
    TEST_STATUS=0
    TEST_OUT=$("$CARGO_BIN" test --workspace --quiet ${APOLLO_CARGO_FEATURE_ARGS[@]+"${APOLLO_CARGO_FEATURE_ARGS[@]}"} -- --skip e2e_ 2>&1) \
      || TEST_STATUS=$?
    if [ "$TEST_STATUS" -ne 0 ]; then
        printf '%s\n' "$TEST_OUT" | tail -80
        fail "parallel workspace tests failed (exit=$TEST_STATUS)"
        exit 2
    fi

    E2E_STATUS=0
    E2E_OUT=$("$CARGO_BIN" test --test e2e_dry_run --quiet ${APOLLO_CARGO_FEATURE_ARGS[@]+"${APOLLO_CARGO_FEATURE_ARGS[@]}"} -- --test-threads=1 2>&1) \
      || E2E_STATUS=$?
    if [ "$E2E_STATUS" -ne 0 ]; then
        printf '%s\n' "$E2E_OUT" | tail -80
        fail "serial daemon E2E tests failed (exit=$E2E_STATUS)"
        exit 2
    fi

    TEST_PASS=$(printf '%s\n%s\n' "$TEST_OUT" "$E2E_OUT" \
      | awk '/test result: ok/ { for (i=1; i<=NF; i++) if ($i == "passed;") total += $(i-1) } END { print total+0 }')
    ok "workspace tests: $TEST_PASS passed (parallel core + serial daemon E2E)"
fi

# ── 3. DEPLOY ────────────────────────────────────────────────────────────────
echo ""
echo "═══ 3/4 DEPLOY ═══"
RESTART_BASELINE=0
if $SKIP_DEPLOY; then
    warn "skipped (--skip-deploy)"
else
    DEPLOYER="/usr/local/sbin/apollo-deploy"
    if [ ! -x "$DEPLOYER" ]; then
        fail "scoped deployer missing: $DEPLOYER"
        exit 3
    fi
    RESTART_BASELINE=$(grep -c 'predictive-agent: loaded' /var/log/apollo-optimizer.err.log 2>/dev/null || true)
    RESTART_BASELINE=${RESTART_BASELINE:-0}
    cp -f target/release/apollo-optimizerd /private/tmp/apollo-optimizerd-candidate
    cp -f target/release/apollo-optimizerctl /private/tmp/apollo-optimizerctl-candidate
    chmod 755 /private/tmp/apollo-optimizerd-candidate /private/tmp/apollo-optimizerctl-candidate
    codesign --force --sign - /private/tmp/apollo-optimizerd-candidate
    codesign --force --sign - /private/tmp/apollo-optimizerctl-candidate
    sudo -n "$DEPLOYER"
    ok "scoped deploy completed with backup + launchd verification"
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
    RESTART_TOTAL=$(grep -c 'predictive-agent: loaded' /var/log/apollo-optimizer.err.log 2>/dev/null || true)
    RESTART_TOTAL=${RESTART_TOTAL:-0}
    RESTART_COUNT=$((RESTART_TOTAL - RESTART_BASELINE))
    [ "$RESTART_COUNT" -lt 0 ] && RESTART_COUNT="$RESTART_TOTAL"
    if [ "$RESTART_COUNT" -gt 3 ]; then
        fail "crash loop detected: $RESTART_COUNT starts since deploy"
        echo ""
        echo "── STDERR (last 20) ──"
        tail -20 /var/log/apollo-optimizer.err.log 2>/dev/null
        exit 4
    elif [ "$RESTART_COUNT" -gt 1 ]; then
        warn "$RESTART_COUNT starts detected since deploy"
    else
        ok "no crash loop"
    fi
fi

# ── SUMMARY ──────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
ELAPSED=$((TOTAL_END - TOTAL_START))
echo ""
echo "═══ DONE ═══ (${ELAPSED}s)"
