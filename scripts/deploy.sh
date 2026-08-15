#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Apollo Quick Deploy — hot-swap daemon binary without full reinstall
# ══════════════════════════════════════════════════════════════════════════════
# Usage: sudo ./scripts/deploy.sh
#
# SECURITY: this invokes a root deployer. Never recommend a NOPASSWD sudoers
# rule for arbitrary daemon installation; that is equivalent to root code
# execution. Normal `sudo` is used at the trust boundary.
#
# Assumes install-root-daemon.sh was run at least once (plist, dirs, config
# already in place). Only rebuilds, signs, copies, and restarts.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOYER="/usr/local/sbin/apollo-deploy"
LABEL="com.eduardocortez.systemoptimizerd"

cd "$ROOT_DIR"
source scripts/hardware-build-profile.sh

# ── Build ──────────────────────────────────────────────────────────────────
echo "── Building release..."
"$ROOT_DIR/scripts/build-release.sh"
apollo_verify_build_manifest

echo "── Staging verified fabric..."
sudo install -o root -g wheel -m 0755 "$ROOT_DIR/scripts/apollo-deploy" "$DEPLOYER"
[[ "$(apollo_sha256_file "$DEPLOYER")" == \
    "$(apollo_sha256_file "$ROOT_DIR/scripts/apollo-deploy")" ]] \
    || { echo "scoped deployer install verification failed: $DEPLOYER" >&2; exit 1; }
cp "$APOLLO_RELEASE_DIR/apollo-optimizerd" /private/tmp/apollo-optimizerd-candidate
cp "$APOLLO_RELEASE_DIR/apollo-optimizerctl" /private/tmp/apollo-optimizerctl-candidate
cp "$APOLLO_RELEASE_DIR/apollo-context-agent" /private/tmp/apollo-context-agent-candidate
cp "$APOLLO_RELEASE_DIR/apollo-web-bridge" /private/tmp/apollo-web-bridge-candidate
cp "$ROOT_DIR/models/apollo-temporal-v1.mlmodel" /private/tmp/apollo-temporal-v1.mlmodel-candidate
cp "$ROOT_DIR/scripts/com.eduardocortez.apollo-context-agent.plist" \
    /private/tmp/com.eduardocortez.apollo-context-agent.plist-candidate
chmod 755 /private/tmp/apollo-optimizerd-candidate \
    /private/tmp/apollo-optimizerctl-candidate \
    /private/tmp/apollo-context-agent-candidate \
    /private/tmp/apollo-web-bridge-candidate
codesign --force --sign - /private/tmp/apollo-optimizerd-candidate
codesign --force --sign - /private/tmp/apollo-optimizerctl-candidate
codesign --force --sign - /private/tmp/apollo-context-agent-candidate
codesign --force --sign - /private/tmp/apollo-web-bridge-candidate

echo "── Deploying atomically..."
sudo "$DEPLOYER" deploy \
    "$(apollo_sha256_file /private/tmp/apollo-optimizerd-candidate)" \
    "$(apollo_sha256_file /private/tmp/apollo-optimizerctl-candidate)" \
    "$(apollo_sha256_file /private/tmp/apollo-context-agent-candidate)" \
    "$(apollo_sha256_file /private/tmp/apollo-web-bridge-candidate)" \
    "$(apollo_sha256_file /private/tmp/apollo-temporal-v1.mlmodel-candidate)" \
    "$(apollo_sha256_file /private/tmp/com.eduardocortez.apollo-context-agent.plist-candidate)"
"$ROOT_DIR/scripts/install-webflow-extension.sh"

sleep 2
if launchctl print system/$LABEL 2>/dev/null | grep -q 'state = running'; then
    echo "✓ Deployed and running"
else
    echo "✗ Daemon may not be running — check: tail -20 /var/log/apollo-optimizer.err.log"
    exit 1
fi
