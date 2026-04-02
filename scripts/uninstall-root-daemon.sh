#!/bin/bash
set -euo pipefail

LABEL="com.eduardocortez.systemoptimizerd"
PLIST_DST="/Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist"

# ── 1. Graceful stop: ctl sends panic-restore (unfreezes + reverts sysctls) ──
if command -v /usr/local/bin/apollo-optimizerctl >/dev/null 2>&1; then
    sudo /usr/local/bin/apollo-optimizerctl panic-restore 2>/dev/null || true
fi
sudo launchctl bootout system/$LABEL >/dev/null 2>&1 || true
sleep 1  # give daemon cleanup path a moment to finish

# ── 2. Emergency SIGCONT for any still-frozen processes ──────────────────────
# Reads frozen_state.json; sends SIGCONT to each listed PID.  Safe if already
# running — SIGCONT to a running process is a no-op.
FROZEN_STATE="/var/lib/apollo/frozen_state.json"
if sudo test -f "$FROZEN_STATE"; then
    echo "Sending SIGCONT to any frozen processes..."
    sudo awk -F'[,:{} "]+' '
        /\"pid"/ { for(i=1;i<=NF;i++) if($i=="pid") { pid=$(i+1)+0; if(pid>0) print pid } }
    ' "$FROZEN_STATE" | while read -r pid; do
        sudo kill -CONT "$pid" 2>/dev/null && echo "  SIGCONT → PID $pid" || true
    done
fi

# ── 3. Revert sysctl changes to captured defaults ────────────────────────────
SYSCTL_DEFAULTS="/var/lib/apollo/sysctl_defaults.json"
if sudo test -f "$SYSCTL_DEFAULTS"; then
    echo "Reverting sysctl changes..."
    sudo awk -F'[":,{}]+' '
        NF>=3 && $2 ~ /^[a-z]/ {
            gsub(/^[ \t]+|[ \t]+$/, "", $2)
            gsub(/^[ \t]+|[ \t]+$/, "", $3)
            if ($2 != "" && $3 != "") print $2 "=" $3
        }
    ' "$SYSCTL_DEFAULTS" | while IFS='=' read -r key val; do
        sudo sysctl -w "${key}=${val}" 2>/dev/null && echo "  restored: $key=$val" || true
    done
fi

# ── 4. Remove binaries and launchd artifacts ──────────────────────────────────
sudo rm -f "$PLIST_DST"
sudo rm -f /usr/local/libexec/apollo-optimizerd /usr/local/bin/apollo-optimizerctl
sudo rm -f /var/run/apollo-optimizer.sock /var/run/apollo.disable

# ── 5. Remove log files ───────────────────────────────────────────────────────
sudo rm -f /var/log/apollo-optimizer.out.log /var/log/apollo-optimizer.err.log

# ── 6. Optional: remove state directory ──────────────────────────────────────
echo ""
read -r -p "Remove all Apollo state data (/var/lib/apollo/)? [y/N] " REPLY
if [[ "${REPLY:-N}" =~ ^[Yy]$ ]]; then
    sudo rm -rf /var/lib/apollo/
    echo "State data removed."
else
    echo "State data preserved at /var/lib/apollo/"
fi

# ── 7. Optional: remove config ────────────────────────────────────────────────
read -r -p "Remove config (/etc/apollo-optimizer/)? [y/N] " REPLY
if [[ "${REPLY:-N}" =~ ^[Yy]$ ]]; then
    sudo rm -rf /etc/apollo-optimizer/
    echo "Config removed."
else
    echo "Config preserved at /etc/apollo-optimizer/"
fi

echo ""
echo "Uninstalled $LABEL"
