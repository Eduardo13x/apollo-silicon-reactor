#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$ROOT_DIR/scripts/com.eduardocortez.systemoptimizerd.plist"
PLIST_DST="/Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist"
DAEMON_DST="/usr/local/libexec/apollo-optimizerd"
CTL_DST="/usr/local/bin/apollo-optimizerctl"
AGENT_DST="/usr/local/libexec/apollo-context-agent"
WEB_BRIDGE_DST="/usr/local/libexec/apollo-web-bridge"
MODEL_SRC="$ROOT_DIR/models/apollo-temporal-v1.mlmodel"
MODEL_DST="/usr/local/share/apollo/models/apollo-temporal-v1.mlmodel"
AGENT_PLIST_SRC="$ROOT_DIR/scripts/com.eduardocortez.apollo-context-agent.plist"
AGENT_PLIST_DST="/Library/LaunchAgents/com.eduardocortez.apollo-context-agent.plist"
DEPLOYER_SRC="$ROOT_DIR/scripts/apollo-deploy"
DEPLOYER_DST="/usr/local/sbin/apollo-deploy"
LABEL="com.eduardocortez.systemoptimizerd"

cd "$ROOT_DIR"
source scripts/hardware-build-profile.sh

echo "── Building release..."
"$ROOT_DIR/scripts/build-release.sh"
apollo_verify_build_manifest

# ── Code signing ────────────────────────────────────────────────────────────
# Apple Silicon requires valid code signature. cp invalidates cargo's ad-hoc
# signature, so we ALWAYS re-sign after install.
# If APOLLO_SIGN_ID is set, use that identity + entitlements for private APIs.
# Otherwise, ad-hoc sign (--sign -) which is sufficient for local execution.
ENTITLEMENTS="$ROOT_DIR/scripts/apollo-optimizerd.entitlements"
sign_binary() {
    local dst="$1"
    local src="$2"
    local use_entitlements="${3:-false}"

    sudo cp "$src" "$dst"
    sudo chown root:wheel "$dst"
    sudo chmod 755 "$dst"

    if [[ -n "${APOLLO_SIGN_ID:-}" ]]; then
        if [[ "$use_entitlements" == "true" && -f "$ENTITLEMENTS" ]]; then
            sudo codesign --force --options runtime \
                --entitlements "$ENTITLEMENTS" \
                --sign "$APOLLO_SIGN_ID" "$dst"
        else
            sudo codesign --force --options runtime \
                --sign "$APOLLO_SIGN_ID" "$dst"
        fi
    else
        sudo codesign --force --sign - "$dst"
    fi

    # Verify signature is valid before proceeding
    if ! sudo codesign --verify --verbose "$dst" 2>/dev/null; then
        echo "ERROR: code signature verification failed for $dst" >&2
        exit 1
    fi
}

echo "── Installing binaries..."
sudo mkdir -p /usr/local/libexec /usr/local/bin /usr/local/sbin \
  /usr/local/share/apollo/models /var/lib/apollo/backups /etc/apollo-optimizer /var/log

sign_binary "$DAEMON_DST" "$APOLLO_RELEASE_DIR/apollo-optimizerd" true
sign_binary "$CTL_DST"    "$APOLLO_RELEASE_DIR/apollo-optimizerctl" false
sign_binary "$AGENT_DST"  "$APOLLO_RELEASE_DIR/apollo-context-agent" false
sign_binary "$WEB_BRIDGE_DST" "$APOLLO_RELEASE_DIR/apollo-web-bridge" false
"$ROOT_DIR/scripts/install-webflow-extension.sh"
sudo install -o root -g wheel -m 0644 "$MODEL_SRC" "$MODEL_DST"
sudo install -o root -g wheel -m 0644 "$AGENT_PLIST_SRC" "$AGENT_PLIST_DST"
sudo install -o root -g wheel -m 0755 "$DEPLOYER_SRC" "$DEPLOYER_DST"

sudo cp "$PLIST_SRC" "$PLIST_DST"
sudo chown root:wheel "$PLIST_DST"
sudo chmod 644 "$PLIST_DST"
sudo chmod 700 /var/lib/apollo /etc/apollo-optimizer

# Backup config on each install run (defensive, root-only).
if sudo test -f /etc/apollo-optimizer/config.toml; then
  TS="$(date +%Y%m%d-%H%M%S)"
  sudo cp /etc/apollo-optimizer/config.toml "/etc/apollo-optimizer/config.toml.bak.$TS" || true
fi

if ! sudo test -f /etc/apollo-optimizer/config.toml; then
  cat <<'CFG' | sudo tee /etc/apollo-optimizer/config.toml >/dev/null
# Apollo Optimizer — daemon configuration
# Location: /etc/apollo-optimizer/config.toml
# Changes take effect on next daemon restart.

# Optimization profile: "balanced-root" | "aggressive-root" | "safe-root"
#   balanced-root  — default; adapts to pressure, conservative on foreground
#                    apps and developer tools.
#   aggressive-root — more aggressive background freezing; good for builds.
#   safe-root      — minimal intervention; safest option.
profile = "balanced-root"

# Safety policy: "aggressive-controlled" | "conservative"
#   aggressive-controlled — freeze background processes when pressure > 0.55.
#   conservative          — only freeze processes idle > 10 min at > 0.75.
policy = "aggressive-controlled"

# Additional processes to never freeze/throttle (substring match against name).
# The default protected set (Claude, Brave, rustc, cargo, etc.) is always active.
#protected_extra = ["my-app", "postgres"]

# Foreground latency target.
#   low    — 16 ms budget (real-time audio / gaming)
#   normal — 50 ms budget (default)
#   max    — 150 ms budget (batch workloads dominant)
#latency_target = "normal"

# Log level for structured daemon output to /var/log/apollo-optimizer.err.log
# Override at runtime: APOLLO_LOG=debug apollo-optimizerd
# Values: "error" | "warn" | "info" | "debug" | "trace"
#log_level = "info"

# Reversible low-latency lane. It observes for 500 healthy cycles before its
# admission decisions replace the legacy acceleration admission.
[reflex]
enabled = true
shadow_cycles = 500

CFG
fi

sudo chmod 600 /etc/apollo-optimizer/config.toml

sudo touch /var/log/apollo-optimizer.out.log /var/log/apollo-optimizer.err.log
sudo chown root:wheel /var/log/apollo-optimizer.out.log /var/log/apollo-optimizer.err.log

# ── Launch daemon ───────────────────────────────────────────────────────────
echo "── Starting daemon..."
sudo rm -f /var/run/apollo-optimizer.sock
sudo launchctl bootout system/$LABEL 2>/dev/null || true
sleep 2

sudo launchctl bootstrap system "$PLIST_DST"
sudo launchctl kickstart -k system/$LABEL

CONSOLE_UID="$(stat -f %u /dev/console 2>/dev/null || true)"
if [[ "$CONSOLE_UID" =~ ^[0-9]+$ ]] && [[ "$CONSOLE_UID" -gt 0 ]]; then
  sudo launchctl bootout "gui/$CONSOLE_UID/com.eduardocortez.apollo-context-agent" 2>/dev/null || true
  sudo launchctl bootstrap "gui/$CONSOLE_UID" "$AGENT_PLIST_DST"
fi

# Wait and verify the daemon is actually running (not crash-looping).
sleep 3
if sudo launchctl print system/$LABEL 2>/dev/null | grep -q 'state = running'; then
    echo "✓ Daemon is running"
elif pgrep -f apollo-optimizerd >/dev/null 2>&1; then
    echo "✓ Daemon process found"
else
    echo "✗ Daemon may not be running — check: sudo tail -20 /var/log/apollo-optimizer.err.log"
    echo "  Code signature: sudo codesign -vv $DAEMON_DST"
    echo "  System log:     log show --predicate 'eventMessage contains \"apollo\"' --last 1m"
    exit 1
fi

echo ""
echo "Installed and started: $LABEL"
echo "Try: $CTL_DST doctor"
