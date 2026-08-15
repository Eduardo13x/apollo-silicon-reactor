#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILE_SCRIPT="$REPO_ROOT/scripts/hardware-build-profile.sh"
ACCEPT_GATE="$REPO_ROOT/scripts/apollo-accept-gate.sh"
PIPELINE="$REPO_ROOT/scripts/pipeline.sh"
TEST_ROOT="$(mktemp -d /private/tmp/apollo-build-profile.XXXXXX)"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
    printf 'not ok - %s\n' "$*" >&2
    exit 1
}

assert_eq() {
    [ "$1" = "$2" ] || fail "$3: expected '$1', got '$2'"
}

(
    export APOLLO_REPO_ROOT="$TEST_ROOT"
    export APOLLO_UNAME_OVERRIDE=Darwin
    export APOLLO_HW_CORES_OVERRIDE=10
    export APOLLO_P_CORES_OVERRIDE=4
    export APOLLO_E_CORES_OVERRIDE=6
    source "$PROFILE_SCRIPT"
    assert_eq adaptive-multicore "$APOLLO_BUILD_PROFILE" "M4 profile"
    assert_eq "$TEST_ROOT/target/apollo-adaptive-multicore" "$CARGO_TARGET_DIR" \
        "M4 target directory"
    assert_eq "$CARGO_TARGET_DIR/release" "$APOLLO_RELEASE_DIR" "release directory"
    [ "${APOLLO_CARGO_FEATURE_ARGS[*]}" = "--features adaptive-multicore" ] \
        || fail "M4 adaptive feature missing"
)

(
    export APOLLO_REPO_ROOT="$TEST_ROOT"
    export APOLLO_UNAME_OVERRIDE=Darwin
    export APOLLO_HW_CORES_OVERRIDE=8
    export APOLLO_P_CORES_OVERRIDE=4
    export APOLLO_E_CORES_OVERRIDE=4
    source "$PROFILE_SCRIPT"
    assert_eq adaptive-multicore "$APOLLO_BUILD_PROFILE" "portable M1 profile"
    assert_eq "$TEST_ROOT/target/apollo-adaptive-multicore" "$CARGO_TARGET_DIR" \
        "M1 target directory"
    [ "${APOLLO_CARGO_FEATURE_ARGS[*]}" = "--features adaptive-multicore" ] \
        || fail "portable M1 artifact must contain runtime multicore support"
)

(
    export APOLLO_REPO_ROOT="$TEST_ROOT"
    export APOLLO_DEPLOYMENT_MODE=heterogeneous-required
    export APOLLO_UNAME_OVERRIDE=Darwin
    export APOLLO_HW_CORES_OVERRIDE=8
    export APOLLO_P_CORES_OVERRIDE=4
    export APOLLO_E_CORES_OVERRIDE=4
    source "$PROFILE_SCRIPT"
    assert_eq adaptive-multicore "$APOLLO_BUILD_PROFILE" \
        "heterogeneous-required portable M1 profile"
)

(
    export APOLLO_REPO_ROOT="$TEST_ROOT"
    export APOLLO_UNAME_OVERRIDE=Darwin
    export APOLLO_HW_CORES_OVERRIDE=0
    export APOLLO_P_CORES_OVERRIDE=0
    export APOLLO_E_CORES_OVERRIDE=0
    export APOLLO_SYSTEM_PROFILER_TEXT_OVERRIDE='Hardware:
      Chip: Apple M4
      Total Number of Cores: 10 (4 Performance and 6 Efficiency)'
    source "$PROFILE_SCRIPT"
    assert_eq adaptive-multicore "$APOLLO_BUILD_PROFILE" \
        "M4 system_profiler fallback profile"
    assert_eq 10 "$APOLLO_HW_CORES" "fallback total cores"
    assert_eq 4 "$APOLLO_P_CORES" "fallback performance cores"
    assert_eq 6 "$APOLLO_E_CORES" "fallback efficiency cores"
)

(
    export APOLLO_REPO_ROOT="$TEST_ROOT"
    export APOLLO_UNAME_OVERRIDE=Darwin
    export APOLLO_HW_CORES_OVERRIDE=10
    export APOLLO_P_CORES_OVERRIDE=4
    export APOLLO_E_CORES_OVERRIDE=6
    source "$PROFILE_SCRIPT"
    mkdir -p "$APOLLO_RELEASE_DIR"
    printf daemon > "$APOLLO_RELEASE_DIR/apollo-optimizerd"
    printf ctl > "$APOLLO_RELEASE_DIR/apollo-optimizerctl"
    printf agent > "$APOLLO_RELEASE_DIR/apollo-context-agent"
    printf bridge > "$APOLLO_RELEASE_DIR/apollo-web-bridge"
    chmod 755 "$APOLLO_RELEASE_DIR/apollo-optimizerd" "$APOLLO_RELEASE_DIR/apollo-optimizerctl" \
        "$APOLLO_RELEASE_DIR/apollo-context-agent" "$APOLLO_RELEASE_DIR/apollo-web-bridge"
    apollo_write_build_manifest
    apollo_verify_build_manifest
    assert_eq 4 "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" schema)" \
        "manifest schema"
    assert_eq apollo-web-bridge \
        "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" web_bridge_name)" \
        "web bridge manifest name"
    assert_eq apple-m1 "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" cpu_baseline)" \
        "portable CPU baseline"
    assert_eq portable "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" deployment_mode)" \
        "deployment mode"
    assert_eq source "$(apollo_manifest_value "$APOLLO_BUILD_MANIFEST" metal_source_mode)" \
        "Metal source mode"
    printf tampered >> "$APOLLO_RELEASE_DIR/apollo-optimizerd"
    if apollo_verify_build_manifest >/dev/null 2>&1; then
        fail "stale daemon hash was accepted"
    fi
)

(
    export APOLLO_REPO_ROOT="$TEST_ROOT"
    export APOLLO_UNAME_OVERRIDE=Darwin
    export APOLLO_HW_CORES_OVERRIDE=10
    export APOLLO_P_CORES_OVERRIDE=4
    export APOLLO_E_CORES_OVERRIDE=6
    source "$PROFILE_SCRIPT"
    mkdir -p "$APOLLO_RELEASE_DIR"
    printf daemon > "$APOLLO_RELEASE_DIR/apollo-optimizerd"
    printf ctl > "$APOLLO_RELEASE_DIR/apollo-optimizerctl"
    printf agent > "$APOLLO_RELEASE_DIR/apollo-context-agent"
    printf bridge > "$APOLLO_RELEASE_DIR/apollo-web-bridge"
    chmod 755 "$APOLLO_RELEASE_DIR/apollo-optimizerd" "$APOLLO_RELEASE_DIR/apollo-optimizerctl" \
        "$APOLLO_RELEASE_DIR/apollo-context-agent" "$APOLLO_RELEASE_DIR/apollo-web-bridge"
    apollo_write_build_manifest
    /usr/bin/sed -i '' 's/cpu_baseline=apple-m1/cpu_baseline=native/' "$APOLLO_BUILD_MANIFEST"
    if apollo_verify_build_manifest >/dev/null 2>&1; then
        fail "native CPU baseline was accepted as portable"
    fi
)

grep -q 'apollo_verify_build_manifest.*risky_fail' "$ACCEPT_GATE" \
    || fail "risky acceptance gate must reject an invalid build manifest"
grep -q 'scripts/build-release.sh' "$PIPELINE" \
    || fail "pipeline must use the portable release builder"

printf 'ok - hardware build profiles and manifest verification\n'
