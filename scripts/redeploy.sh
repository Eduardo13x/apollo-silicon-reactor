#!/bin/bash
# Quick redeploy: build, copy, sign, restart daemon, show logs.
# Usage: sudo ./scripts/redeploy.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "── Build release..."
cargo build --release 2>&1 | tail -3

echo "── Kill old daemon..."
killall apollo-optimizerd 2>/dev/null || true
sleep 1

echo "── Copy binary..."
cp -f target/release/apollo-optimizerd /usr/local/libexec/apollo-optimizerd
codesign --force --sign - /usr/local/libexec/apollo-optimizerd
echo "   MD5: $(md5 -q /usr/local/libexec/apollo-optimizerd)"

echo "── Truncate logs..."
truncate -s 0 /var/log/apollo-optimizer.out.log /var/log/apollo-optimizer.err.log

echo "── Restart daemon..."
launchctl kickstart -k system/com.eduardocortez.systemoptimizerd 2>/dev/null || \
  launchctl kickstart system/com.eduardocortez.systemoptimizerd 2>/dev/null || \
  echo "   kickstart failed, trying load..." && \
  launchctl load /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist 2>/dev/null || true

echo "── Waiting 12s for daemon to cycle..."
sleep 12

echo ""
echo "══ PROCESS ══"
ps aux | grep apollo-optimizerd | grep -v grep || echo "  NOT RUNNING"

echo ""
echo "══ STDOUT (last 10) ══"
tail -10 /var/log/apollo-optimizer.out.log 2>/dev/null || echo "  (empty)"

echo ""
echo "══ STDERR (last 20) ══"
tail -20 /var/log/apollo-optimizer.err.log 2>/dev/null || echo "  (empty)"

echo ""
echo "══ STATUS ══"
# Try to get status from daemon socket
timeout 5 /usr/local/bin/apollo-optimizerctl status 2>/dev/null | grep -E '"cycles"|last_cycle_at|running' || echo "  (no response)"
