#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
SOURCE="$ROOT/extensions/apollo-webflow-chromium/native-host.json"
EXTENSION="$ROOT/extensions/apollo-webflow-chromium"
EXTENSION_ID="mhagiddoeecedoknmhdlhghdnglglbhp"
DRY_RUN=false

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
elif [ "$#" -ne 0 ]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

[ -f "$SOURCE" ] || { echo "missing native host manifest: $SOURCE" >&2; exit 1; }

install_for() {
    product=$1
    destination="$HOME/Library/Application Support/$product/NativeMessagingHosts"
    target="$destination/com.eduardocortez.apollo_webflow.json"
    echo "$target"
    if [ "$DRY_RUN" = false ]; then
        install -d -m 0700 "$destination"
        install -m 0600 "$SOURCE" "$target"
    fi
}

install_for "BraveSoftware/Brave-Browser"
install_for "Google/Chrome"
install_for "Microsoft Edge"
install_for "Chromium"

echo "extension-id: $EXTENSION_ID"
echo "load-unpacked: $EXTENSION"
echo "After loading it, click the Apollo WebFlow toolbar action to grant optional site metrics."
