#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
TMP_HOME=$(mktemp -d /private/tmp/apollo-webflow-install-test.XXXXXX)
trap 'rm -rf "$TMP_HOME"' EXIT

OUTPUT=$(HOME="$TMP_HOME" "$ROOT/scripts/install-webflow-extension.sh" --dry-run)

printf '%s\n' "$OUTPUT" | grep -q 'BraveSoftware/Brave-Browser/NativeMessagingHosts'
printf '%s\n' "$OUTPUT" | grep -q 'Google/Chrome/NativeMessagingHosts'
printf '%s\n' "$OUTPUT" | grep -q 'Microsoft Edge/NativeMessagingHosts'
printf '%s\n' "$OUTPUT" | grep -q 'Chromium/NativeMessagingHosts'
printf '%s\n' "$OUTPUT" | grep -q 'mhagiddoeecedoknmhdlhghdnglglbhp'

if find "$TMP_HOME" -type f | grep -q .; then
    echo "dry-run wrote files" >&2
    exit 1
fi

grep -q '"path": "/usr/local/libexec/apollo-web-bridge"' \
    "$ROOT/extensions/apollo-webflow-chromium/native-host.json"
grep -q 'chrome-extension://mhagiddoeecedoknmhdlhghdnglglbhp/' \
    "$ROOT/extensions/apollo-webflow-chromium/native-host.json"
