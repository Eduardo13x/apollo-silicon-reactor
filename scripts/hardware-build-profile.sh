#!/usr/bin/env bash
# Source from Apollo build/deploy scripts. Every Apple Silicon build uses the
# M1 instruction baseline and contains runtime-selected multicore support.

APOLLO_PROFILE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APOLLO_SOURCE_ROOT="$(cd "$APOLLO_PROFILE_SCRIPT_DIR/.." && pwd)"
APOLLO_REPO_ROOT="${APOLLO_REPO_ROOT:-$(cd "$APOLLO_PROFILE_SCRIPT_DIR/.." && pwd)}"
APOLLO_CARGO_FEATURE_ARGS=()
APOLLO_BUILD_PROFILE="sequential"
APOLLO_ADAPTIVE_FEATURE=0
APOLLO_CPU_BASELINE="apple-m1"
APOLLO_RUST_TARGET_CPU="apple-a14"
APOLLO_DEPLOYMENT_MODE="${APOLLO_DEPLOYMENT_MODE:-portable}"
APOLLO_MINIMUM_MACOS="13.0"
case "$APOLLO_DEPLOYMENT_MODE" in
    portable|heterogeneous-required) ;;
    *) echo "invalid APOLLO_DEPLOYMENT_MODE: $APOLLO_DEPLOYMENT_MODE" >&2; return 1 2>/dev/null || exit 1 ;;
esac

apollo_uint_or_zero() {
    case "${1:-}" in
        ''|*[!0-9]*) printf '0\n' ;;
        *) printf '%s\n' "$1" ;;
    esac
}

APOLLO_UNAME="${APOLLO_UNAME_OVERRIDE:-$(uname -s 2>/dev/null || true)}"
APOLLO_ARCH="${APOLLO_ARCH_OVERRIDE:-$(uname -m 2>/dev/null || true)}"
if [ "$APOLLO_UNAME" = "Darwin" ]; then
    if [ "$APOLLO_ARCH" = "arm64" ]; then
        APOLLO_CARGO_FEATURE_ARGS=(--features adaptive-multicore)
        APOLLO_BUILD_PROFILE="adaptive-multicore"
        APOLLO_ADAPTIVE_FEATURE=1
    fi
    APOLLO_HW_CORES="$(apollo_uint_or_zero "${APOLLO_HW_CORES_OVERRIDE:-$(sysctl -n hw.ncpu 2>/dev/null || true)}")"
    APOLLO_P_CORES="$(apollo_uint_or_zero "${APOLLO_P_CORES_OVERRIDE:-$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || true)}")"
    APOLLO_E_CORES="$(apollo_uint_or_zero "${APOLLO_E_CORES_OVERRIDE:-$(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || true)}")"
    if [ "$APOLLO_HW_CORES" -eq 0 ] \
        || [ "$APOLLO_P_CORES" -eq 0 ] \
        || [ "$APOLLO_E_CORES" -eq 0 ]; then
        APOLLO_HARDWARE_TEXT="${APOLLO_SYSTEM_PROFILER_TEXT_OVERRIDE:-$(LC_ALL=C system_profiler SPHardwareDataType 2>/dev/null || true)}"
        APOLLO_PROFILE_HW_CORES=$(printf '%s\n' "$APOLLO_HARDWARE_TEXT" \
            | /usr/bin/sed -nE 's/.*Total Number of Cores:[[:space:]]*([0-9]+).*/\1/p' \
            | /usr/bin/head -n 1)
        APOLLO_PROFILE_P_CORES=$(printf '%s\n' "$APOLLO_HARDWARE_TEXT" \
            | /usr/bin/sed -nE 's/.*\(([0-9]+)[[:space:]]+Performance.*/\1/p' \
            | /usr/bin/head -n 1)
        APOLLO_PROFILE_E_CORES=$(printf '%s\n' "$APOLLO_HARDWARE_TEXT" \
            | /usr/bin/sed -nE 's/.*and[[:space:]]+([0-9]+)[[:space:]]+Efficiency.*/\1/p' \
            | /usr/bin/head -n 1)
        [ "$APOLLO_HW_CORES" -gt 0 ] \
            || APOLLO_HW_CORES="$(apollo_uint_or_zero "$APOLLO_PROFILE_HW_CORES")"
        [ "$APOLLO_P_CORES" -gt 0 ] \
            || APOLLO_P_CORES="$(apollo_uint_or_zero "$APOLLO_PROFILE_P_CORES")"
        [ "$APOLLO_E_CORES" -gt 0 ] \
            || APOLLO_E_CORES="$(apollo_uint_or_zero "$APOLLO_PROFILE_E_CORES")"
    fi
fi

if [ "$APOLLO_DEPLOYMENT_MODE" = "heterogeneous-required" ]; then
    [ "$APOLLO_ADAPTIVE_FEATURE" = 1 ] \
        || { echo "heterogeneous-required needs an Apple Silicon portable build" >&2; return 1 2>/dev/null || exit 1; }
    [ -f "$APOLLO_SOURCE_ROOT/crates/apollo-engine/native/gpu_imagination_bridge.mm" ] \
        || { echo "heterogeneous-required needs the Metal source lane" >&2; return 1 2>/dev/null || exit 1; }
fi

CARGO_TARGET_DIR="$APOLLO_REPO_ROOT/target/apollo-$APOLLO_BUILD_PROFILE"
APOLLO_RELEASE_DIR="$CARGO_TARGET_DIR/release"
APOLLO_BUILD_MANIFEST="$CARGO_TARGET_DIR/apollo-build-manifest-v2"
export CARGO_TARGET_DIR APOLLO_RELEASE_DIR APOLLO_BUILD_MANIFEST
export APOLLO_BUILD_PROFILE APOLLO_ADAPTIVE_FEATURE
export APOLLO_CPU_BASELINE APOLLO_RUST_TARGET_CPU APOLLO_DEPLOYMENT_MODE
export APOLLO_MINIMUM_MACOS

apollo_sha256_file() {
    /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'
}

apollo_target_triple() {
    local rustc_bin
    rustc_bin="$(command -v rustc 2>/dev/null || true)"
    if [ -z "$rustc_bin" ] && [ -x "$HOME/.cargo/bin/rustc" ]; then
        rustc_bin="$HOME/.cargo/bin/rustc"
    fi
    [ -n "$rustc_bin" ] || return 1
    "$rustc_bin" -vV | /usr/bin/awk '/^host:/ {print $2}'
}

apollo_manifest_value() {
    /usr/bin/awk -F= -v key="$2" \
        '$1 == key { print substr($0, length(key) + 2); exit }' "$1"
}

apollo_write_build_manifest() {
    local daemon ctl context_agent web_bridge coreml_model context_plist target_triple temporary metal_source metal_mode metal_hash feature_set
    daemon="$APOLLO_RELEASE_DIR/apollo-optimizerd"
    ctl="$APOLLO_RELEASE_DIR/apollo-optimizerctl"
    context_agent="$APOLLO_RELEASE_DIR/apollo-context-agent"
    web_bridge="$APOLLO_RELEASE_DIR/apollo-web-bridge"
    coreml_model="$APOLLO_SOURCE_ROOT/models/apollo-temporal-v1.mlmodel"
    context_plist="$APOLLO_SOURCE_ROOT/scripts/com.eduardocortez.apollo-context-agent.plist"
    [ -x "$daemon" ] || { echo "missing build artifact: $daemon" >&2; return 1; }
    [ -x "$ctl" ] || { echo "missing build artifact: $ctl" >&2; return 1; }
    [ -x "$context_agent" ] || { echo "missing build artifact: $context_agent" >&2; return 1; }
    [ -x "$web_bridge" ] || { echo "missing build artifact: $web_bridge" >&2; return 1; }
    [ -f "$coreml_model" ] || { echo "missing Core ML artifact: $coreml_model" >&2; return 1; }
    [ -f "$context_plist" ] || { echo "missing context-agent plist: $context_plist" >&2; return 1; }
    target_triple="$(apollo_target_triple)" \
        || { echo "unable to determine Rust target triple" >&2; return 1; }
    metal_source="$APOLLO_SOURCE_ROOT/crates/apollo-engine/native/gpu_imagination_bridge.mm"
    if [ -f "$metal_source" ]; then
        metal_mode=source
        metal_hash="$(apollo_sha256_file "$metal_source")"
    else
        metal_mode=unavailable
        metal_hash=absent
    fi
    if [ "$APOLLO_ADAPTIVE_FEATURE" = 1 ]; then
        feature_set=adaptive-multicore
    else
        feature_set=none
    fi
    mkdir -p "$CARGO_TARGET_DIR"
    temporary="$(mktemp "$APOLLO_BUILD_MANIFEST.XXXXXX")"
    umask 077
    {
        printf 'schema=4\n'
        printf 'profile=%s\n' "$APOLLO_BUILD_PROFILE"
        printf 'adaptive_feature=%s\n' "$APOLLO_ADAPTIVE_FEATURE"
        printf 'target_triple=%s\n' "$target_triple"
        printf 'cpu_baseline=%s\n' "$APOLLO_CPU_BASELINE"
        printf 'rust_target_cpu=%s\n' "$APOLLO_RUST_TARGET_CPU"
        printf 'minimum_macos=%s\n' "$APOLLO_MINIMUM_MACOS"
        printf 'deployment_mode=%s\n' "$APOLLO_DEPLOYMENT_MODE"
        printf 'feature_set=%s\n' "$feature_set"
        printf 'metal_source_mode=%s\n' "$metal_mode"
        printf 'metal_source_sha256=%s\n' "$metal_hash"
        printf 'daemon_name=apollo-optimizerd\n'
        printf 'daemon_sha256=%s\n' "$(apollo_sha256_file "$daemon")"
        printf 'ctl_name=apollo-optimizerctl\n'
        printf 'ctl_sha256=%s\n' "$(apollo_sha256_file "$ctl")"
        printf 'context_agent_name=apollo-context-agent\n'
        printf 'context_agent_sha256=%s\n' "$(apollo_sha256_file "$context_agent")"
        printf 'web_bridge_name=apollo-web-bridge\n'
        printf 'web_bridge_sha256=%s\n' "$(apollo_sha256_file "$web_bridge")"
        printf 'coreml_model_name=apollo-temporal-v1.mlmodel\n'
        printf 'coreml_model_sha256=%s\n' "$(apollo_sha256_file "$coreml_model")"
        printf 'context_plist_name=com.eduardocortez.apollo-context-agent.plist\n'
        printf 'context_plist_sha256=%s\n' "$(apollo_sha256_file "$context_plist")"
    } > "$temporary"
    mv -f "$temporary" "$APOLLO_BUILD_MANIFEST"
}

apollo_verify_build_manifest() {
    local daemon ctl context_agent web_bridge coreml_model context_plist expected_triple manifest_profile manifest_feature expected_feature metal_source
    daemon="$APOLLO_RELEASE_DIR/apollo-optimizerd"
    ctl="$APOLLO_RELEASE_DIR/apollo-optimizerctl"
    context_agent="$APOLLO_RELEASE_DIR/apollo-context-agent"
    web_bridge="$APOLLO_RELEASE_DIR/apollo-web-bridge"
    coreml_model="$APOLLO_SOURCE_ROOT/models/apollo-temporal-v1.mlmodel"
    context_plist="$APOLLO_SOURCE_ROOT/scripts/com.eduardocortez.apollo-context-agent.plist"
    [ -f "$APOLLO_BUILD_MANIFEST" ] \
        || { echo "build manifest missing: $APOLLO_BUILD_MANIFEST" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" schema)" = "4" ] \
        || { echo "build manifest schema mismatch" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" cpu_baseline)" = "$APOLLO_CPU_BASELINE" ] \
        || { echo "unsupported CPU baseline" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" rust_target_cpu)" = "$APOLLO_RUST_TARGET_CPU" ] \
        || { echo "Rust target CPU mismatch" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" deployment_mode)" = "$APOLLO_DEPLOYMENT_MODE" ] \
        || { echo "deployment mode mismatch" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" minimum_macos)" = "$APOLLO_MINIMUM_MACOS" ] \
        || { echo "minimum macOS mismatch" >&2; return 1; }
    if [ "$APOLLO_ADAPTIVE_FEATURE" = 1 ]; then
        expected_feature=adaptive-multicore
    else
        expected_feature=none
    fi
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" feature_set)" = "$expected_feature" ] \
        || { echo "feature set mismatch" >&2; return 1; }
    case "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" metal_source_mode)" in
        source)
            metal_source="$APOLLO_SOURCE_ROOT/crates/apollo-engine/native/gpu_imagination_bridge.mm"
            [ -f "$metal_source" ] \
                && [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" metal_source_sha256)" = "$(apollo_sha256_file "$metal_source")" ] \
                || { echo "Metal source hash mismatch" >&2; return 1; }
            ;;
        unavailable)
            [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" metal_source_sha256)" = absent ] \
                || { echo "invalid unavailable Metal source hash" >&2; return 1; }
            ;;
        *) echo "invalid Metal source mode" >&2; return 1 ;;
    esac
    manifest_profile="$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" profile)"
    [ "$manifest_profile" = "$APOLLO_BUILD_PROFILE" ] \
        || { echo "build profile mismatch: expected $APOLLO_BUILD_PROFILE, got $manifest_profile" >&2; return 1; }
    manifest_feature="$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" adaptive_feature)"
    [ "$manifest_feature" = "$APOLLO_ADAPTIVE_FEATURE" ] \
        || { echo "adaptive feature mismatch for $APOLLO_BUILD_PROFILE" >&2; return 1; }
    expected_triple="$(apollo_target_triple)" \
        || { echo "unable to determine Rust target triple" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" target_triple)" = "$expected_triple" ] \
        || { echo "build target triple mismatch" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" daemon_name)" = "apollo-optimizerd" ] \
        || { echo "daemon manifest entry malformed" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" ctl_name)" = "apollo-optimizerctl" ] \
        || { echo "ctl manifest entry malformed" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" context_agent_name)" = "apollo-context-agent" ] \
        || { echo "context-agent manifest entry malformed" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" web_bridge_name)" = "apollo-web-bridge" ] \
        || { echo "web bridge manifest entry malformed" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" coreml_model_name)" = "apollo-temporal-v1.mlmodel" ] \
        || { echo "Core ML manifest entry malformed" >&2; return 1; }
    [ "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" context_plist_name)" = "com.eduardocortez.apollo-context-agent.plist" ] \
        || { echo "context plist manifest entry malformed" >&2; return 1; }
    [ -x "$daemon" ] && [ "$(apollo_sha256_file "$daemon")" = \
        "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" daemon_sha256)" ] \
        || { echo "daemon artifact missing or stale" >&2; return 1; }
    [ -x "$ctl" ] && [ "$(apollo_sha256_file "$ctl")" = \
        "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" ctl_sha256)" ] \
        || { echo "ctl artifact missing or stale" >&2; return 1; }
    [ -x "$context_agent" ] && [ "$(apollo_sha256_file "$context_agent")" = \
        "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" context_agent_sha256)" ] \
        || { echo "context-agent artifact missing or stale" >&2; return 1; }
    [ -x "$web_bridge" ] && [ "$(apollo_sha256_file "$web_bridge")" = \
        "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" web_bridge_sha256)" ] \
        || { echo "web bridge artifact missing or stale" >&2; return 1; }
    [ -f "$coreml_model" ] && [ "$(apollo_sha256_file "$coreml_model")" = \
        "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" coreml_model_sha256)" ] \
        || { echo "Core ML artifact missing or stale" >&2; return 1; }
    [ -f "$context_plist" ] && [ "$(apollo_sha256_file "$context_plist")" = \
        "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" context_plist_sha256)" ] \
        || { echo "context plist missing or stale" >&2; return 1; }
}
