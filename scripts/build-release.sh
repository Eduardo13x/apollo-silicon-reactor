#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
source scripts/hardware-build-profile.sh

CARGO_BIN="$(command -v cargo 2>/dev/null || true)"
if [ -z "$CARGO_BIN" ] && [ -x "$HOME/.cargo/bin/cargo" ]; then
    CARGO_BIN="$HOME/.cargo/bin/cargo"
fi
[ -n "$CARGO_BIN" ] || { echo "cargo not found" >&2; exit 1; }

echo "Building Apollo profile=$APOLLO_BUILD_PROFILE target=$CARGO_TARGET_DIR"
RUSTFLAGS="-C target-cpu=$APOLLO_RUST_TARGET_CPU" \
"$CARGO_BIN" build --workspace --bins --release \
    ${APOLLO_CARGO_FEATURE_ARGS[@]+"${APOLLO_CARGO_FEATURE_ARGS[@]}"}
apollo_write_build_manifest
apollo_verify_build_manifest
echo "Build manifest verified: $APOLLO_BUILD_MANIFEST"
