#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Apollo Quick Deploy — hot-swap daemon binary without full reinstall
# ══════════════════════════════════════════════════════════════════════════════
# Usage: sudo ./scripts/deploy.sh
#
# Assumes install-root-daemon.sh was run at least once (plist, dirs, config
# already in place). Only rebuilds, signs, copies, and restarts.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DAEMON_DST="/usr/local/libexec/apollo-optimizerd"
CTL_DST="/usr/local/bin/apollo-optimizerctl"
LABEL="com.eduardocortez.systemoptimizerd"
ENTITLEMENTS="$ROOT_DIR/scripts/apollo-optimizerd.entitlements"

cd "$ROOT_DIR"

# ── Build ──────────────────────────────────────────────────────────────────
echo "── Building release..."
cargo build --release

# ── Sign & install (same logic as install-root-daemon.sh) ──────────────────
sign_binary() {
    local dst="$1"
    local src="$2"
    local use_entitlements="${3:-false}"

    cp "$src" "$dst"
    chown root:wheel "$dst"
    chmod 755 "$dst"

    if [[ -n "${APOLLO_SIGN_ID:-}" ]]; then
        if [[ "$use_entitlements" == "true" && -f "$ENTITLEMENTS" ]]; then
            codesign --force --options runtime \
                --entitlements "$ENTITLEMENTS" \
                --sign "$APOLLO_SIGN_ID" "$dst"
        else
            codesign --force --options runtime \
                --sign "$APOLLO_SIGN_ID" "$dst"
        fi
    else
        codesign --force --sign - "$dst"
    fi

    if ! codesign --verify --verbose "$dst" 2>/dev/null; then
        echo "ERROR: code signature verification failed for $dst" >&2
        exit 1
    fi
}

echo "── Installing binaries..."
sign_binary "$DAEMON_DST" "$ROOT_DIR/target/release/apollo-optimizerd" true
sign_binary "$CTL_DST"    "$ROOT_DIR/target/release/apollo-optimizerctl" false

# ── Restart daemon ─────────────────────────────────────────────────────────
echo "── Restarting daemon..."
launchctl kickstart -k system/$LABEL

sleep 2
if launchctl print system/$LABEL 2>/dev/null | grep -q 'state = running'; then
    echo "✓ Deployed and running"
else
    echo "✗ Daemon may not be running — check: tail -20 /var/log/apollo-optimizer.err.log"
    exit 1
fi
