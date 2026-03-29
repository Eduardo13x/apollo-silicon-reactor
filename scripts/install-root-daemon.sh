#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$ROOT_DIR/scripts/com.eduardocortez.systemoptimizerd.plist"
PLIST_DST="/Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist"
DAEMON_DST="/usr/local/libexec/apollo-optimizerd"
CTL_DST="/usr/local/bin/apollo-optimizerctl"
LABEL="com.eduardocortez.systemoptimizerd"

cd "$ROOT_DIR"

echo "── Building release..."
cargo build --release

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
sudo mkdir -p /usr/local/libexec /usr/local/bin /var/lib/apollo /etc/apollo-optimizer /var/log

sign_binary "$DAEMON_DST" "$ROOT_DIR/target/release/apollo-optimizerd" true
sign_binary "$CTL_DST"    "$ROOT_DIR/target/release/apollo-optimizerctl" false

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
profile = "balanced-root"
policy = "aggressive-controlled"

# Optional LLM teacher mode (requires `apollo-optimizerctl llm set-key`).
#[llm]
#enabled = false
#model = "gpt-4.1-mini"
#endpoint = "https://api.openai.com/v1/chat/completions"
#min_confidence = 0.85
#max_calls_per_hour = 2
#min_interval_secs = 900
#timeout_ms = 10000
#force_json = true
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
