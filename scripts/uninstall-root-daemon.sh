#!/bin/bash
set -euo pipefail

LABEL="com.eduardocortez.systemoptimizerd"
PLIST_DST="/Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist"

if command -v /usr/local/bin/apollo-optimizerctl >/dev/null 2>&1; then
  sudo /usr/local/bin/apollo-optimizerctl panic-restore || true
fi

sudo launchctl bootout system/$LABEL >/dev/null 2>&1 || true
sudo rm -f "$PLIST_DST"
sudo rm -f /usr/local/libexec/apollo-optimizerd /usr/local/bin/apollo-optimizerctl
sudo rm -f /var/run/apollo-optimizer.sock /var/run/apollo.disable

echo "Uninstalled $LABEL"
