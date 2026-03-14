#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$ROOT_DIR/scripts/com.eduardocortez.systemoptimizerd.plist"
PLIST_DST="/Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist"
DAEMON_DST="/usr/local/libexec/apollo-optimizerd"
CTL_DST="/usr/local/bin/apollo-optimizerctl"
LABEL="com.eduardocortez.systemoptimizerd"

cd "$ROOT_DIR"

cargo build --release

# ── Code signing (optional, requires Apple Developer certificate) ─────────────
# If APOLLO_SIGN_ID is set, sign with entitlements to unlock private APIs.
# Example: APOLLO_SIGN_ID="Developer ID Application: Tu Nombre (TEAMID)" ./install-root-daemon.sh
ENTITLEMENTS="$ROOT_DIR/scripts/apollo-optimizerd.entitlements"
if [[ -n "${APOLLO_SIGN_ID:-}" ]]; then
    echo "Signing with identity: $APOLLO_SIGN_ID"
    codesign --force --options runtime \
        --entitlements "$ENTITLEMENTS" \
        --sign "$APOLLO_SIGN_ID" \
        "$ROOT_DIR/target/release/apollo-optimizerd"
    codesign --force --options runtime \
        --sign "$APOLLO_SIGN_ID" \
        "$ROOT_DIR/target/release/apollo-optimizerctl"
fi

sudo mkdir -p /usr/local/libexec /usr/local/bin /var/lib/apollo /etc/apollo-optimizer /var/log
sudo cp "$ROOT_DIR/target/release/apollo-optimizerd" "$DAEMON_DST"
sudo cp "$ROOT_DIR/target/release/apollo-optimizerctl" "$CTL_DST"
sudo cp "$PLIST_SRC" "$PLIST_DST"

sudo chown root:wheel "$DAEMON_DST" "$CTL_DST" "$PLIST_DST"
sudo chmod 755 "$DAEMON_DST" "$CTL_DST"
sudo chmod 644 "$PLIST_DST"
sudo chmod 700 /var/lib/apollo /etc/apollo-optimizer

# IMPORTANT: `/etc/apollo-optimizer` is chmod 700, so a non-root `test -f` will fail
# even if the file exists. Always check existence via sudo to avoid clobbering.

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

# Unload existing service and wait for launchd to finish processing it.
sudo launchctl bootout system/$LABEL 2>/dev/null || true
sleep 2

sudo launchctl bootstrap system "$PLIST_DST"
sudo launchctl kickstart -k system/$LABEL
sudo launchctl print system/$LABEL | sed -n '1,120p'

echo "Installed and started: $LABEL"
echo "Try: $CTL_DST doctor"
