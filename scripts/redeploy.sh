#!/bin/bash
# Quick compatibility entry point: build, atomically deploy the complete fabric,
# then show status. The privileged helper owns all system writes.
# Usage: ./scripts/redeploy.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/hardware-build-profile.sh

echo "── Build release..."
"$PWD/scripts/build-release.sh" 2>&1 | tail -4
apollo_verify_build_manifest

echo "── Deploy complete fabric..."
"$PWD/scripts/deploy.sh"

echo "── Waiting 3s for daemon to cycle..."
sleep 3

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
